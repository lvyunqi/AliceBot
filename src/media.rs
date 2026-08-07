//! Shared URL sanitization for inbound and outbound media references.

use sha2::{Digest, Sha256};

const MAX_MEDIA_URL_BYTES: usize = 16_384;
const INVALID_MEDIA_URL: &str = "[invalid-media-url]";

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
