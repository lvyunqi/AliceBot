//! Incoming URL media collection.
use sha2::{Digest, Sha256};

/// Compatibility entry point for callers without policy metadata.
pub async fn maybe_collect(image_url: &str, context: &str) -> bool {
    maybe_collect_with_metadata(image_url, context, "unknown", "", "", 1.0)
        .await
        .is_some()
}

/// Apply deterministic sampling and insert or touch a sticker record.
pub async fn maybe_collect_with_metadata(
    image_url: &str,
    context: &str,
    protocol: &str,
    source_user: &str,
    source_session: &str,
    probability: f32,
) -> Option<i64> {
    let media = crate::media::sanitize_remote_media_url(image_url, true)?;
    if probability <= 0.0 {
        return None;
    }
    if probability < 1.0
        && deterministic_sample(&media.identity_hash) >= probability.clamp(0.0, 1.0)
    {
        return None;
    }

    let database = crate::pipeline::try_db()?;
    let now = chrono::Utc::now().timestamp_millis();
    let requires_cache = i32::from(media.requires_cache);
    let cache_status = if media.requires_cache {
        "required"
    } else {
        "remote"
    };
    let sticker_id = {
        let connection = database.conn.lock().ok()?;
        let existing = connection
            .query_row(
                "SELECT id FROM stickers WHERE url_hash = ?1 LIMIT 1",
                rusqlite::params![media.identity_hash],
                |row| row.get::<_, i64>(0),
            )
            .ok();
        if let Some(id) = existing {
            connection
                .execute(
                    "UPDATE stickers
                     SET usage_count = usage_count + 1, last_used = ?1, updated_at = ?1,
                         media_url = CASE
                             WHEN url_requires_cache = 1 AND ?2 = 0 THEN ?3
                             ELSE media_url
                         END,
                         url_requires_cache = MIN(url_requires_cache, ?2),
                         cache_status = CASE WHEN ?2 = 0 THEN 'remote' ELSE cache_status END
                     WHERE id = ?4",
                    rusqlite::params![now, requires_cache, media.storage_url, id],
                )
                .ok()?;
            id
        } else {
            connection
                .execute(
                    "INSERT INTO stickers
                     (protocol, kind, media_url, url_hash, url_requires_cache, cache_status,
                      source_user, source_session, usage_count, last_used, created_at, updated_at)
                     VALUES (?1, 'image', ?2, ?3, ?4, ?5, ?6, ?7, 1, ?8, ?8, ?8)",
                    rusqlite::params![
                        protocol,
                        media.storage_url,
                        media.identity_hash,
                        requires_cache,
                        cache_status,
                        source_user,
                        source_session,
                        now
                    ],
                )
                .ok()?;
            connection.last_insert_rowid()
        }
    };

    let tags = tags_from_context(context);
    if let Ok(connection) = database.conn.lock() {
        for tag in tags {
            let _ = connection.execute(
                "INSERT INTO sticker_tags (sticker_id, tag, weight) VALUES (?1, ?2, 1)
                 ON CONFLICT(sticker_id, tag) DO UPDATE SET weight = weight + 1",
                rusqlite::params![sticker_id, tag],
            );
        }
    }
    Some(sticker_id)
}

pub fn deterministic_sample(value: &str) -> f32 {
    let digest = Sha256::digest(value.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes) as f64 as f32 / u64::MAX as f32
}

fn tags_from_context(context: &str) -> Vec<String> {
    let known_tags = [
        "开心", "难过", "生气", "哈哈", "无语", "震惊", "可爱", "猫", "狗", "谢谢", "恭喜", "问号",
    ];
    let mut tags = vec!["image".to_string()];
    for tag in known_tags {
        if context.contains(tag) {
            tags.push(tag.to_string());
        }
    }
    tags.sort();
    tags.dedup();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_deterministic_and_bounded() {
        let first = deterministic_sample("https://example.test/a.png");
        assert_eq!(first, deterministic_sample("https://example.test/a.png"));
        assert!((0.0..=1.0).contains(&first));
    }

    #[test]
    fn invalid_media_is_not_collected() {
        assert!(tags_from_context("今天哈哈开心").contains(&"哈哈".to_string()));
        assert!(
            crate::media::sanitize_remote_media_url("http://example.test/a.png", true).is_none()
        );
    }

    #[test]
    fn credential_rotation_keeps_collection_sampling_stable() {
        let first = crate::media::sanitize_remote_media_url(
            "https://example.test/a.png?fileid=1&rkey=old",
            true,
        )
        .unwrap();
        let second = crate::media::sanitize_remote_media_url(
            "https://example.test/a.png?rkey=new&fileid=1",
            true,
        )
        .unwrap();
        assert_eq!(first.identity_hash, second.identity_hash);
        assert_eq!(
            deterministic_sample(&first.identity_hash),
            deterministic_sample(&second.identity_hash)
        );
    }
}
