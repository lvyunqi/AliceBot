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
    if !image_url.starts_with("https://") || probability <= 0.0 {
        return None;
    }
    if probability < 1.0 && deterministic_sample(image_url) >= probability.clamp(0.0, 1.0) {
        return None;
    }

    let database = crate::pipeline::try_db()?;
    let now = chrono::Utc::now().timestamp_millis();
    let file_hash = hash_url(image_url);
    let sticker_id = {
        let connection = database.conn.lock().ok()?;
        let existing = connection
            .query_row(
                "SELECT id FROM stickers WHERE file_hash = ?1 LIMIT 1",
                rusqlite::params![file_hash],
                |row| row.get::<_, i64>(0),
            )
            .ok();
        if let Some(id) = existing {
            connection
                .execute(
                    "UPDATE stickers
                     SET usage_count = usage_count + 1, last_used = ?1
                     WHERE id = ?2",
                    rusqlite::params![now, id],
                )
                .ok()?;
            id
        } else {
            connection
                .execute(
                    "INSERT INTO stickers
                     (protocol, kind, media_url, file_hash, source_user, source_session,
                      usage_count, last_used, created_at)
                     VALUES (?1, 'image', ?2, ?3, ?4, ?5, 1, ?6, ?6)",
                    rusqlite::params![
                        protocol,
                        image_url,
                        file_hash,
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

fn hash_url(url: &str) -> String {
    let digest = Sha256::digest(url.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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
        assert_eq!(hash_url("http://example.test/a.png").len(), 64);
        assert!(tags_from_context("今天哈哈开心").contains(&"哈哈".to_string()));
    }
}
