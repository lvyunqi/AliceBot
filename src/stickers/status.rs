//! Factual collection status and bounded sticker-library queries.
//!
//! The collection event record intentionally stores only a normalized hash,
//! never a raw or signed media URL. Live cache state is read from `stickers`
//! when someone asks whether a recently sent image was collected.

use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::db::Database;
use crate::pipeline::{InMessage, MediaRef};
use crate::stickers::collect::CollectionResult;

const RECENT_COLLECTION_WINDOW_MILLIS: i64 = 30 * 60 * 1_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CollectionStatus {
    pub(crate) outcome: String,
    pub(crate) cache_status: Option<String>,
    pub(crate) url_requires_cache: bool,
}

/// Persist one collection decision after the collection transaction has
/// completed. Queue and cache state remain on the sticker row and are joined
/// at read time so the answer reflects the worker's latest outcome.
pub(crate) fn record_attempt(
    message: &InMessage,
    media: &MediaRef,
    media_index: usize,
    result: &CollectionResult,
) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    record_attempt_in_database(
        &database,
        message,
        media,
        media_index,
        result.outcome_code(),
        result.sticker_id(),
        chrono::Utc::now().timestamp_millis(),
    );
}

fn record_attempt_in_database(
    database: &Database,
    message: &InMessage,
    media: &MediaRef,
    media_index: usize,
    outcome: &str,
    sticker_id: Option<i64>,
    now: i64,
) {
    let url_hash = media_identity_hash(&media.url);
    let Ok(media_index) = i64::try_from(media_index) else {
        return;
    };
    let Ok(connection) = database.conn.lock() else {
        return;
    };
    let _ = connection.execute(
        "INSERT INTO sticker_collection_events
         (source_event_key, media_index, protocol, session_type, session_id, source_user,
          url_hash, outcome, sticker_id, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
         ON CONFLICT(source_event_key, media_index) DO UPDATE SET
            outcome = excluded.outcome,
            sticker_id = excluded.sticker_id,
            created_at = excluded.created_at",
        params![
            message.event_key,
            media_index,
            message.protocol,
            message.session_type,
            message.session_id,
            message.sender_id,
            url_hash,
            outcome,
            sticker_id,
            now,
        ],
    );
}

/// A status question is intentionally narrow. More general sticker discovery
/// remains available to the native tool loop.
pub(crate) fn is_collection_status_query(content: &str) -> bool {
    let compact = content
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    [
        "收藏了吗",
        "收藏了没",
        "收藏没",
        "收到了吗",
        "收到了没",
        "收图了吗",
        "收表情包了吗",
        "收藏表情包了吗",
        "保存了吗",
        "存下来了吗",
    ]
    .iter()
    .any(|marker| compact.contains(marker))
}

/// Return a deterministic response for an explicit @-mention status query.
pub(crate) fn reply_for_status_query(
    message: &InMessage,
    stickers_enabled: bool,
    auto_collect: bool,
) -> Option<&'static str> {
    if !message.at_me || !is_collection_status_query(&message.content) {
        return None;
    }
    if !stickers_enabled {
        return Some("表情包功能没开。");
    }
    if !auto_collect {
        return Some("自动收藏没开，这张图没有自动收藏。");
    }
    Some(match recent_status(message) {
        Some(status) => reply_for_status(&status),
        None => "没找到你刚发的图片，不能确认收藏状态。",
    })
}

pub(crate) fn recent_status(message: &InMessage) -> Option<CollectionStatus> {
    let database = crate::pipeline::try_db()?;
    recent_status_in_database(
        &database,
        &message.protocol,
        &message.session_type,
        &message.session_id,
        &message.sender_id,
        chrono::Utc::now().timestamp_millis(),
    )
}

fn recent_status_in_database(
    database: &Database,
    protocol: &str,
    session_type: &str,
    session_id: &str,
    source_user: &str,
    now: i64,
) -> Option<CollectionStatus> {
    let connection = database.conn.lock().ok()?;
    connection
        .query_row(
            "SELECT event.outcome, sticker.cache_status,
                    COALESCE(sticker.url_requires_cache, 0)
             FROM sticker_collection_events AS event
             LEFT JOIN stickers AS sticker ON sticker.id = event.sticker_id
             WHERE event.protocol = ?1
               AND event.session_type = ?2
               AND event.session_id = ?3
               AND event.source_user = ?4
               AND event.created_at >= ?5
             ORDER BY event.created_at DESC, event.media_index DESC
             LIMIT 1",
            params![
                protocol,
                session_type,
                session_id,
                source_user,
                now.saturating_sub(RECENT_COLLECTION_WINDOW_MILLIS),
            ],
            |row| {
                Ok(CollectionStatus {
                    outcome: row.get(0)?,
                    cache_status: row.get(1)?,
                    url_requires_cache: row.get::<_, i64>(2)? != 0,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
}

/// Execute the read-only native tool for a user's recent collection result.
pub(crate) fn sticker_status_tool(current: &InMessage) -> Value {
    let Some(status) = recent_status(current) else {
        return json!({
            "found": false,
            "instruction": "当前发言者在本会话最近三十分钟内没有可确认的收藏记录；不要声称已经收藏。"
        });
    };
    json!({
        "found": true,
        "outcome": status.outcome,
        "state": status_label(&status),
        "cache_required": status.url_requires_cache,
        "instruction": "只根据这个状态回答，不要杜撰收藏、缓存或发送结果，也不要复述图片 URL。"
    })
}

/// Confirm whether this exact media identity already belongs to the current
/// protocol's sticker library. This never returns the URL or internal ID.
pub(crate) fn media_is_collected(protocol: &str, media: &MediaRef) -> bool {
    let Some(database) = crate::pipeline::try_db() else {
        return false;
    };
    media_is_collected_in_database(&database, protocol, media)
}

fn media_is_collected_in_database(database: &Database, protocol: &str, media: &MediaRef) -> bool {
    let url_hash = media_identity_hash(&media.url);
    let Ok(connection) = database.conn.lock() else {
        return false;
    };
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1
                 FROM stickers AS sticker
                 LEFT JOIN sticker_sources AS source ON source.sticker_id = sticker.id
                 WHERE sticker.protocol = ?1
                   AND (sticker.url_hash = ?2 OR source.url_hash = ?2)
                   AND sticker.cache_status <> 'invalid'
             )",
            params![protocol, url_hash],
            |row| row.get::<_, i64>(0),
        )
        .map(|exists| exists != 0)
        .unwrap_or(false)
}

/// Execute the read-only native tool for a small, redacted sticker search.
pub(crate) fn search_stickers_tool(arguments: &Value, current: &InMessage) -> Value {
    let query = arguments
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .chars()
        .filter(|character| !character.is_control())
        .take(80)
        .collect::<String>();
    let limit = arguments
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(4)
        .clamp(1, 6) as i64;
    let Some(database) = crate::pipeline::try_db() else {
        return json!({"items": [], "available": false});
    };
    let Ok(rows) = search_sticker_rows(
        &database,
        &current.protocol,
        &current.session_id,
        &query,
        limit,
    ) else {
        return json!({"items": [], "available": false});
    };

    let items = render_search_items(rows);
    json!({
        "available": true,
        "query": query,
        "items": items,
        "instruction": "结果只表示可检索的收藏，不代表已经发送；没有匹配项时不要假装发送了图片。"
    })
}

type StickerSearchRow = (Option<String>, bool, String);

fn search_sticker_rows(
    database: &Database,
    protocol: &str,
    session_id: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<StickerSearchRow>, ()> {
    let connection = database.conn.lock().map_err(|_| ())?;
    let pattern = format!("%{}%", escape_like(query));
    connection
        .prepare(
            "SELECT sticker.cache_status,
                    COALESCE(sticker.url_requires_cache, 0),
                    COALESCE(GROUP_CONCAT(DISTINCT tag.tag), '') AS tags
             FROM stickers AS sticker
             LEFT JOIN sticker_tags AS tag ON tag.sticker_id = sticker.id
             WHERE sticker.protocol = ?1
               AND (?2 = '' OR sticker.media_url LIKE ?3 ESCAPE '\\'
                    OR EXISTS (
                        SELECT 1 FROM sticker_tags AS matching_tag
                        WHERE matching_tag.sticker_id = sticker.id
                          AND matching_tag.tag LIKE ?3 ESCAPE '\\'
                    ))
             GROUP BY sticker.id
             ORDER BY (sticker.source_session = ?4) DESC,
                      COALESCE(sticker.last_used, 0) ASC,
                      sticker.id ASC
             LIMIT ?5",
        )
        .and_then(|mut statement| {
            let rows = statement.query_map(
                params![protocol, query, pattern, session_id, limit],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, i64>(1)? != 0,
                        row.get::<_, String>(2)?,
                    ))
                },
            )?;
            rows.collect::<Result<Vec<_>, _>>()
        })
        .map_err(|_| ())
}

fn render_search_items(rows: Vec<StickerSearchRow>) -> Vec<Value> {
    rows.into_iter()
        .map(|(cache_status, url_requires_cache, tags)| {
            let status = CollectionStatus {
                outcome: "collected".to_string(),
                cache_status,
                url_requires_cache,
            };
            let tags = tags
                .split(',')
                .filter(|tag| !tag.is_empty())
                .take(8)
                .collect::<Vec<_>>();
            json!({"state": status_label(&status), "tags": tags})
        })
        .collect()
}

fn media_identity_hash(raw_url: &str) -> String {
    crate::media::sanitize_remote_media_url(raw_url, true)
        .map(|media| media.identity_hash)
        .unwrap_or_else(|| {
            let digest = Sha256::digest(raw_url.as_bytes());
            format!(
                "invalid:{}",
                digest
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            )
        })
}

fn status_label(status: &CollectionStatus) -> &'static str {
    match status.outcome.as_str() {
        "collected" => match status.cache_status.as_deref() {
            Some("queued" | "caching" | "required") => "已收藏，缓存中",
            Some("cached") => "已收藏，已缓存",
            Some("failed") => "已收藏，缓存失败",
            Some("quota_exceeded") => "已收藏，缓存空间不足",
            Some("invalid") => "收藏记录不可用",
            _ => "已收藏",
        },
        "skipped_sensitive" => "安全规则跳过",
        "skipped_daily_limit" => "今日收藏额度已满",
        "skipped_sampling" => "收藏采样跳过",
        "skipped_low_signal" => "未识别为可收藏图片",
        "skipped_invalid_media" => "图片链接未通过校验",
        _ => "收藏状态未知",
    }
}

fn reply_for_status(status: &CollectionStatus) -> &'static str {
    match status.outcome.as_str() {
        "collected" => match status.cache_status.as_deref() {
            Some("queued" | "caching" | "required") => "收到了，已经收藏，正在缓存。",
            Some("cached") => "收到了，已经收藏并缓存好了。",
            Some("failed") => "收到了，收藏记录还在，但缓存失败了。",
            Some("quota_exceeded") => "收到了，已经收藏，但缓存空间满了。",
            Some("invalid") => "这张图的收藏记录不可用了。",
            _ => "收到了，已经收藏。",
        },
        "skipped_sensitive" => "这张图按安全规则没收藏。",
        "skipped_daily_limit" => "今天的收藏额度用完了，这张没收藏。",
        "skipped_sampling" => "这张图这次没有收藏。",
        "skipped_low_signal" => "这张图没被识别为可收藏内容。",
        "skipped_invalid_media" => "这张图没通过链接校验，没收藏。",
        _ => "这次收藏处理没完成，不能说已经收藏。",
    }
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(sender_id: &str, session_id: &str) -> InMessage {
        InMessage {
            event_key: format!("onebot11:{sender_id}:image"),
            protocol: "onebot11".to_string(),
            bot_account_id: "bot".to_string(),
            session_type: "group".to_string(),
            session_id: session_id.to_string(),
            sender_id: sender_id.to_string(),
            sender_name: "tester".to_string(),
            message_id: "image".to_string(),
            reply_to_id: String::new(),
            content: String::new(),
            media: Vec::new(),
            has_media: true,
            at_me: true,
            timestamp: 1,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn collection_status_query_is_specific_to_collection_questions() {
        assert!(is_collection_status_query("收藏了吗"));
        assert!(is_collection_status_query("  收藏表情包了吗  "));
        assert!(!is_collection_status_query("来个表情包"));
    }

    #[test]
    fn recent_status_is_scoped_to_sender_and_session() {
        let database = Database::open(":memory:").unwrap();
        let media = MediaRef {
            url: "https://example.test/one.png".to_string(),
            media_type: "image/png".to_string(),
            requires_cache: false,
        };
        let first = message("member-1", "group-1");
        let now = chrono::Utc::now().timestamp_millis();
        record_attempt_in_database(
            &database,
            &first,
            &media,
            0,
            "skipped_daily_limit",
            None,
            now,
        );

        let status =
            recent_status_in_database(&database, "onebot11", "group", "group-1", "member-1", now)
                .expect("matching member should see their status");
        assert_eq!(status.outcome, "skipped_daily_limit");
        assert!(
            recent_status_in_database(&database, "onebot11", "group", "group-1", "member-2", now,)
                .is_none()
        );
        assert!(
            recent_status_in_database(&database, "onebot11", "group", "group-2", "member-1", now,)
                .is_none()
        );
    }

    #[test]
    fn sticker_search_result_never_contains_media_url() {
        let database = Database::open(":memory:").unwrap();
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_session, created_at, updated_at)
                 VALUES ('onebot11', 'https://example.test/private.png?token=hidden', 'one',
                         0, 'remote', 'group-1', 1, 1)",
                [],
            )
            .unwrap();
        let sticker_id = connection.last_insert_rowid();
        connection
            .execute(
                "INSERT INTO sticker_tags (sticker_id, tag, weight) VALUES (?1, 'image', 1)",
                [sticker_id],
            )
            .unwrap();
        drop(connection);

        let rows = search_sticker_rows(&database, "onebot11", "group-1", "image", 4)
            .expect("matching tag should be searchable");
        let rendered = Value::Array(render_search_items(rows)).to_string();
        assert!(!rendered.contains("example.test"));
        assert!(!rendered.contains("token"));
        assert!(rendered.contains("image"));
    }

    #[test]
    fn exact_media_identity_confirms_existing_collection_without_exposing_url() {
        let database = Database::open(":memory:").unwrap();
        let media = MediaRef {
            url: "https://example.test/collected.png".to_string(),
            media_type: "image/png".to_string(),
            requires_cache: false,
        };
        let hash = media_identity_hash(&media.url);
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_session, created_at, updated_at)
                 VALUES ('onebot11', ?1, ?2, 0, 'remote', 'group-1', 1, 1)",
                params![media.url, hash],
            )
            .unwrap();
        assert!(media_is_collected_in_database(
            &database, "onebot11", &media
        ));
        assert!(!media_is_collected_in_database(
            &database,
            "qq-official",
            &media
        ));
    }
}
