//! Shared URL sanitization for inbound and outbound media references.

use base64::Engine;
use reqwest::redirect::Policy;
use sha2::{Digest, Sha256};
use std::net::IpAddr;
use std::time::Duration;

const MAX_MEDIA_URL_BYTES: usize = 16_384;
const INVALID_MEDIA_URL: &str = "[invalid-media-url]";
pub(crate) const MAX_VISION_IMAGE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SanitizedMediaUrl {
    pub storage_url: String,
    pub identity_hash: String,
    pub requires_cache: bool,
}

/// Parse and sanitize a remote media URL. Temporary credentials are removed
/// from the persisted form, while the identity hash is stable across query
/// ordering and credential rotation.
pub(crate) fn sanitize_remote_media_url(
    raw: &str,
    require_https: bool,
) -> Option<SanitizedMediaUrl> {
    if raw.is_empty() || raw.len() > MAX_MEDIA_URL_BYTES {
        return None;
    }
    let mut url = reqwest::Url::parse(raw).ok()?;
    let scheme_allowed = if require_https {
        url.scheme() == "https"
    } else {
        matches!(url.scheme(), "http" | "https")
    };
    if !scheme_allowed || url.host_str().is_none() {
        return None;
    }
    if !url.username().is_empty() || url.password().is_some() {
        return None;
    }

    url.set_fragment(None);
    let mut requires_cache = false;
    let stable_pairs = url
        .query_pairs()
        .filter_map(|(key, value)| {
            if is_sensitive_query_key(&key) {
                requires_cache = true;
                None
            } else {
                Some((key.into_owned(), value.into_owned()))
            }
        })
        .collect::<Vec<_>>();

    let mut storage_url = url.clone();
    replace_query(&mut storage_url, &stable_pairs);

    let mut identity_pairs = stable_pairs;
    identity_pairs.sort_unstable();
    let mut identity_url = url;
    replace_query(&mut identity_url, &identity_pairs);
    let digest = Sha256::digest(identity_url.as_str().as_bytes());
    let identity_hash = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    Some(SanitizedMediaUrl {
        storage_url: storage_url.to_string(),
        identity_hash,
        requires_cache,
    })
}

pub(crate) fn redact_url_for_storage(raw: &str) -> String {
    sanitize_remote_media_url(raw, false)
        .map(|url| url.storage_url)
        .unwrap_or_else(|| INVALID_MEDIA_URL.to_string())
}

/// Download a temporary/signed image without forwarding its credentials to an
/// LLM provider. The returned payload is bounded and suitable for a base64
/// image content block. This intentionally refuses redirects and private IPs.
pub(crate) async fn fetch_image_data(raw: &str, timeout_ms: u64) -> Option<(String, String)> {
    let parsed = sanitize_remote_media_url(raw, true)?;
    if !parsed.requires_cache {
        return None;
    }
    let url = reqwest::Url::parse(raw).ok()?;
    let host = url.host_str()?.to_owned();
    let port = url.port_or_known_default()?;
    let addresses = tokio::net::lookup_host((host.as_str(), port))
        .await
        .ok()?
        .filter(|address| is_public_media_address(address.ip()))
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return None;
    }
    let client = reqwest::Client::builder()
        .redirect(Policy::none())
        .timeout(Duration::from_millis(timeout_ms.clamp(1_000, 30_000)))
        .resolve_to_addrs(&host, &addresses)
        .build()
        .ok()?;
    let mut response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_VISION_IMAGE_BYTES as u64)
    {
        return None;
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(normalize_image_content_type)
        .or_else(|| infer_image_content_type(raw))?;
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.ok()? {
        if bytes.len().saturating_add(chunk.len()) > MAX_VISION_IMAGE_BYTES {
            return None;
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.is_empty() {
        return None;
    }
    Some((
        content_type.to_string(),
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

fn normalize_image_content_type(value: &str) -> Option<&'static str> {
    let value = value.split(';').next()?.trim().to_ascii_lowercase();
    match value.as_str() {
        "image/jpeg" | "image/jpg" => Some("image/jpeg"),
        "image/png" => Some("image/png"),
        "image/webp" => Some("image/webp"),
        "image/gif" => Some("image/gif"),
        _ => None,
    }
}

fn infer_image_content_type(raw: &str) -> Option<&'static str> {
    let path = reqwest::Url::parse(raw).ok()?.path().to_ascii_lowercase();
    if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if path.ends_with(".png") {
        Some("image/png")
    } else if path.ends_with(".webp") {
        Some("image/webp")
    } else if path.ends_with(".gif") {
        Some("image/gif")
    } else {
        None
    }
}

fn is_public_media_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            let [first, second, third, _] = address.octets();
            !address.is_private()
                && !address.is_loopback()
                && !address.is_link_local()
                && !address.is_unspecified()
                && !address.is_multicast()
                && !address.is_broadcast()
                && !address.is_documentation()
                && first != 0
                && !(first == 100 && (64..=127).contains(&second))
                && !(first == 192 && second == 0)
                && !(first == 192 && second == 88 && third == 99)
                && !(first == 198 && matches!(second, 18 | 19))
                && first < 240
        }
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return is_public_media_address(IpAddr::V4(mapped));
            }
            let segments = address.segments();
            !(address.is_unspecified()
                || address.is_loopback()
                || address.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8))
        }
    }
}

fn replace_query(url: &mut reqwest::Url, pairs: &[(String, String)]) {
    url.set_query(None);
    if pairs.is_empty() {
        return;
    }
    let mut query = url.query_pairs_mut();
    for (key, value) in pairs {
        query.append_pair(key, value);
    }
}

fn is_sensitive_query_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.contains("token")
        || key.contains("secret")
        || key.contains("rkey")
        || key.contains("signature")
        || key == "sig"
        || key == "auth"
        || key.ends_with("_key")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qq_signed_url_is_redacted_and_keeps_stable_identity() {
        let first = sanitize_remote_media_url(
            "https://multimedia.nt.qq.com.cn/download?spec=0&rkey=old&appid=1407&fileid=abc#fragment",
            true,
        )
        .expect("QQ media URL should parse");
        let second = sanitize_remote_media_url(
            "https://multimedia.nt.qq.com.cn/download?fileid=abc&appid=1407&rkey=new&spec=0",
            true,
        )
        .expect("rotated QQ media URL should parse");

        assert!(first.requires_cache);
        assert!(!first.storage_url.contains("rkey"));
        assert!(!first.storage_url.contains("fragment"));
        assert!(first.storage_url.contains("fileid=abc"));
        assert_eq!(first.identity_hash, second.identity_hash);
    }

    #[test]
    fn credentials_and_non_remote_sources_are_rejected() {
        assert!(sanitize_remote_media_url("https://user:pass@example.test/a.png", true).is_none());
        assert!(sanitize_remote_media_url("http://example.test/a.png", true).is_none());
        assert!(sanitize_remote_media_url("file:///tmp/a.png", false).is_none());
        assert_eq!(redact_url_for_storage("not a URL"), INVALID_MEDIA_URL);
    }

    #[test]
    fn stable_public_url_remains_sendable() {
        let media =
            sanitize_remote_media_url("https://example.test/a.png?size=large&format=png", true)
                .expect("public media URL should parse");
        assert!(!media.requires_cache);
        assert_eq!(
            media.storage_url,
            "https://example.test/a.png?size=large&format=png"
        );
    }
}
