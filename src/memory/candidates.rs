use rusqlite::{OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use crate::db::Database;
use crate::pipeline::InMessage;

const ACTIVE_CONFIDENCE: i32 = 85;
const REINFORCEMENT_STEP: i32 = 15;
const MAX_INPUT_CHARS: usize = 500;
const MAX_VALUE_CHARS: usize = 40;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtractedCandidate {
    normalized_key: String,
    content: String,
    kind: &'static str,
    importance: i32,
    confidence: i32,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct UpdateStats {
    inserted: usize,
    reinforced: usize,
    promoted: usize,
    blocked_by_tombstone: usize,
    duplicate_sources: usize,
}

struct ExistingMemory {
    id: i64,
    content: String,
    importance: i32,
    confidence: i32,
    status: String,
    version: i32,
}

pub(super) async fn observe(msg: &InMessage) {
    let candidates = extract(msg);
    if candidates.is_empty() {
        return;
    }
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    if let Err(error) = apply(&database, msg, &candidates) {
        log::debug!("[AliceBot] memory candidate update failed: {error}");
    }
}

fn apply(
    database: &Database,
    msg: &InMessage,
    candidates: &[ExtractedCandidate],
) -> Result<UpdateStats, String> {
    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "memory database lock failed".to_string())?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let now = msg.timestamp;
    let mut stats = UpdateStats::default();

    for candidate in candidates {
        if has_forgotten_tombstone(&transaction, &candidate.normalized_key, msg)? {
            stats.blocked_by_tombstone += 1;
            continue;
        }

        let existing = latest_memory(&transaction, &candidate.normalized_key, msg)?;
        match existing {
            Some(existing) if existing.content == candidate.content => {
                let source_inserted = insert_source(&transaction, existing.id, msg, now)?;
                if !source_inserted {
                    stats.duplicate_sources += 1;
                    continue;
                }
                stats.reinforced += 1;
                let confidence = existing
                    .confidence
                    .max(candidate.confidence)
                    .saturating_add(REINFORCEMENT_STEP)
                    .clamp(0, 100);
                let status = if existing.status == "active" || confidence >= ACTIVE_CONFIDENCE {
                    "active"
                } else {
                    "candidate"
                };
                transaction
                    .execute(
                        "UPDATE long_memory
                         SET importance = ?1, confidence = ?2, status = ?3,
                             is_active = ?4, updated_at = ?5
                         WHERE id = ?6",
                        params![
                            existing.importance.max(candidate.importance),
                            confidence,
                            status,
                            i32::from(status == "active"),
                            now,
                            existing.id,
                        ],
                    )
                    .map_err(|error| error.to_string())?;
                if existing.status != "active" && status == "active" {
                    supersede_other_versions(
                        &transaction,
                        &candidate.normalized_key,
                        msg,
                        existing.id,
                        now,
                    )?;
                    stats.promoted += 1;
                }
            }
            existing => {
                let version = existing
                    .as_ref()
                    .map(|memory| memory.version.saturating_add(1))
                    .unwrap_or(1);
                let status = if candidate.confidence >= ACTIVE_CONFIDENCE {
                    "active"
                } else {
                    "candidate"
                };
                transaction
                    .execute(
                        "INSERT INTO long_memory
                         (normalized_key, protocol, session_type, scope, session_id, subject_id,
                          content, kind, importance, confidence, privacy, status, version,
                          is_active, created_at, updated_at)
                         VALUES (?1, ?2, ?3, 'user_session', ?4, ?5, ?6, ?7, ?8, ?9,
                                 'normal', ?10, ?11, ?12, ?13, ?13)",
                        params![
                            candidate.normalized_key,
                            msg.protocol,
                            msg.session_type,
                            msg.session_id,
                            msg.sender_id,
                            candidate.content,
                            candidate.kind,
                            candidate.importance.clamp(0, 100),
                            candidate.confidence.clamp(0, 100),
                            status,
                            version,
                            i32::from(status == "active"),
                            now,
                        ],
                    )
                    .map_err(|error| error.to_string())?;
                let memory_id = transaction.last_insert_rowid();
                insert_source(&transaction, memory_id, msg, now)?;
                stats.inserted += 1;
                if status == "active" {
                    supersede_other_versions(
                        &transaction,
                        &candidate.normalized_key,
                        msg,
                        memory_id,
                        now,
                    )?;
                    stats.promoted += 1;
                }
            }
        }
    }

    transaction.commit().map_err(|error| error.to_string())?;
    Ok(stats)
}

fn has_forgotten_tombstone(
    transaction: &Transaction<'_>,
    normalized_key: &str,
    msg: &InMessage,
) -> Result<bool, String> {
    transaction
        .query_row(
            "SELECT 1 FROM long_memory
             WHERE normalized_key = ?1 AND status = 'forgotten'
               AND (
                    (protocol = ?2 AND session_type = ?3 AND session_id = ?4)
                    OR protocol = 'legacy'
               )
             LIMIT 1",
            params![
                normalized_key,
                msg.protocol,
                msg.session_type,
                msg.session_id
            ],
            |_| Ok(()),
        )
        .optional()
        .map(|row| row.is_some())
        .map_err(|error| error.to_string())
}

fn latest_memory(
    transaction: &Transaction<'_>,
    normalized_key: &str,
    msg: &InMessage,
) -> Result<Option<ExistingMemory>, String> {
    transaction
        .query_row(
            "SELECT id, content, importance, confidence, status, version
             FROM long_memory
             WHERE normalized_key = ?1 AND protocol = ?2
               AND session_type = ?3 AND session_id = ?4
             ORDER BY version DESC LIMIT 1",
            params![
                normalized_key,
                msg.protocol,
                msg.session_type,
                msg.session_id
            ],
            |row| {
                Ok(ExistingMemory {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    importance: row.get(2)?,
                    confidence: row.get(3)?,
                    status: row.get(4)?,
                    version: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(|error| error.to_string())
}

fn insert_source(
    transaction: &Transaction<'_>,
    memory_id: i64,
    msg: &InMessage,
    now: i64,
) -> Result<bool, String> {
    transaction
        .execute(
            "INSERT OR IGNORE INTO memory_sources
             (memory_id, source_type, source_id, evidence_weight, created_at)
             VALUES (?1, 'message', ?2, 1, ?3)",
            params![memory_id, msg.event_key, now],
        )
        .map(|changed| changed > 0)
        .map_err(|error| error.to_string())
}

fn supersede_other_versions(
    transaction: &Transaction<'_>,
    normalized_key: &str,
    msg: &InMessage,
    active_id: i64,
    now: i64,
) -> Result<(), String> {
    transaction
        .execute(
            "UPDATE long_memory
             SET status = 'superseded', is_active = 0, superseded_by = ?1,
                 updated_at = ?2
             WHERE normalized_key = ?3 AND id <> ?1
               AND protocol = ?4 AND session_type = ?5 AND session_id = ?6
               AND status IN ('candidate', 'active')",
            params![
                active_id,
                now,
                normalized_key,
                msg.protocol,
                msg.session_type,
                msg.session_id
            ],
        )
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn extract(msg: &InMessage) -> Vec<ExtractedCandidate> {
    let text = msg.content.trim();
    if text.is_empty()
        || text.chars().count() > MAX_INPUT_CHARS
        || contains_sensitive_content(text)
        || text.contains("http://")
        || text.contains("https://")
        || text.ends_with(['?', '？'])
        || text.contains("不要叫我")
        || text.contains("别叫我")
    {
        return Vec::new();
    }

    let patterns = [
        ("我的名字是", "identity:name", "称呼", "fact", 85, 90, true),
        ("以后请叫我", "identity:name", "称呼", "fact", 85, 90, true),
        ("我叫", "identity:name", "称呼", "fact", 85, 90, true),
        ("叫我", "identity:name", "称呼", "fact", 80, 88, true),
        (
            "我不喜欢",
            "preference:dislike",
            "不喜欢",
            "preference",
            65,
            75,
            false,
        ),
        (
            "我喜欢",
            "preference:like",
            "喜欢",
            "preference",
            60,
            70,
            false,
        ),
        (
            "我是",
            "identity:description",
            "自述",
            "fact",
            55,
            65,
            false,
        ),
    ];
    let subject = subject_digest(msg);
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for (marker, slot, label, kind, importance, confidence, fixed_slot) in patterns {
        let Some(value) = value_after_marker(text, marker) else {
            continue;
        };
        if slot == "identity:description"
            && ["说", "觉得", "想", "在"]
                .iter()
                .any(|prefix| value.starts_with(prefix))
        {
            continue;
        }
        let value_key = normalized_value(&value);
        if value_key.is_empty() {
            continue;
        }
        let normalized_key = if fixed_slot {
            format!("{subject}:{slot}")
        } else {
            format!("{subject}:{slot}:{}", digest_prefix(value_key.as_bytes()))
        };
        if !seen.insert(normalized_key.clone()) {
            continue;
        }
        result.push(ExtractedCandidate {
            normalized_key,
            content: format!("{label}：{value}"),
            kind,
            importance,
            confidence,
        });
    }
    result
}

fn value_after_marker(text: &str, marker: &str) -> Option<String> {
    let start = text.find(marker)? + marker.len();
    let value = text[start..]
        .split(['。', '！', '!', '？', '?', '，', ',', '；', ';', '\n', '\r'])
        .next()
        .unwrap_or_default()
        .trim()
        .trim_matches(['"', '\'', '“', '”', '‘', '’', ':', '：'])
        .trim();
    let count = value.chars().count();
    if count == 0 || count > MAX_VALUE_CHARS {
        return None;
    }
    Some(value.to_string())
}

fn normalized_value(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && !character.is_ascii_punctuation())
        .flat_map(char::to_lowercase)
        .collect()
}

fn subject_digest(msg: &InMessage) -> String {
    let mut hasher = Sha256::new();
    hasher.update(msg.protocol.as_bytes());
    hasher.update([0]);
    hasher.update(msg.sender_id.as_bytes());
    format!("subject:{}", hex_prefix(&hasher.finalize()))
}

fn digest_prefix(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    hex_prefix(&digest)
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn contains_sensitive_content(text: &str) -> bool {
    [
        "密码",
        "验证码",
        "身份证",
        "手机号",
        "电话号码",
        "邮箱",
        "email",
        "住址",
        "地址是",
        "生日",
        "出生",
        "护照",
        "社保",
        "银行卡",
        "api key",
        "apikey",
        "token",
        "secret",
    ]
    .iter()
    .any(|marker| text.to_ascii_lowercase().contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(event: &str, content: &str, timestamp: i64) -> InMessage {
        InMessage {
            event_key: event.to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "memory-group".to_string(),
            sender_id: "anonymous-user".to_string(),
            sender_name: String::new(),
            message_id: event.to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn explicit_preferences_are_extracted_but_sensitive_content_is_rejected() {
        let candidates = extract(&message("memory:extract", "我叫小林，我喜欢爵士乐。", 10));
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.content == "称呼：小林")
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.content == "喜欢：爵士乐")
        );
        assert!(extract(&message("memory:secret", "我的密码是 123456", 11)).is_empty());
        assert!(extract(&message("memory:url", "我喜欢 https://example.com", 12)).is_empty());
        assert!(extract(&message("memory:question", "你知道我叫什么吗？", 13)).is_empty());
        assert!(extract(&message("memory:negative", "不要叫我小林。", 14)).is_empty());
        assert!(extract(&message("memory:birth", "我是 1990 年出生。", 15)).is_empty());
    }

    #[test]
    fn distinct_sources_promote_candidate_and_duplicate_source_does_not_reinforce() {
        let database = Database::open(":memory:").unwrap();
        let first = message("memory:preference:1", "我喜欢咖啡。", 10);
        let second = message("memory:preference:2", "我喜欢咖啡。", 20);
        let candidate = extract(&first);

        assert_eq!(apply(&database, &first, &candidate).unwrap().inserted, 1);
        let duplicate = apply(&database, &first, &candidate).unwrap();
        assert_eq!(duplicate.duplicate_sources, 1);
        let reinforced = apply(&database, &second, &extract(&second)).unwrap();
        assert_eq!(reinforced.reinforced, 1);
        assert_eq!(reinforced.promoted, 1);

        let connection = database.conn.lock().unwrap();
        let row: (String, i32, i32, i64) = connection
            .query_row(
                "SELECT status, confidence, is_active,
                        (SELECT COUNT(*) FROM memory_sources WHERE memory_id = long_memory.id)
                 FROM long_memory",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(row, ("active".to_string(), 85, 1, 2));
    }

    #[test]
    fn candidates_with_identical_ids_remain_route_scoped() {
        let database = Database::open(":memory:").unwrap();
        let onebot = message("memory:route:onebot", "我叫小林。", 10);
        let mut official = message("memory:route:official", "我叫小林。", 20);
        official.protocol = "qq-official".to_string();
        let mut private = message("memory:route:private", "我叫小林。", 30);
        private.session_type = "private".to_string();

        for item in [&onebot, &official, &private] {
            assert_eq!(apply(&database, item, &extract(item)).unwrap().inserted, 1);
        }

        let connection = database.conn.lock().unwrap();
        let routes = connection
            .prepare(
                "SELECT protocol, session_type FROM long_memory
                 ORDER BY protocol, session_type",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            routes,
            vec![
                ("onebot11".to_string(), "group".to_string()),
                ("onebot11".to_string(), "private".to_string()),
                ("qq-official".to_string(), "group".to_string()),
            ]
        );
    }

    #[test]
    fn high_confidence_name_change_supersedes_previous_version() {
        let database = Database::open(":memory:").unwrap();
        let first = message("memory:name:1", "我叫小林。", 10);
        let second = message("memory:name:2", "我叫小周。", 20);
        apply(&database, &first, &extract(&first)).unwrap();
        apply(&database, &second, &extract(&second)).unwrap();

        let connection = database.conn.lock().unwrap();
        let rows = connection
            .prepare(
                "SELECT content, status, version, superseded_by FROM long_memory ORDER BY version",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i32>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].1, "superseded");
        assert_eq!(rows[0].2, 1);
        assert!(rows[0].3.is_some());
        assert_eq!(rows[1].0, "称呼：小周");
        assert_eq!(rows[1].1, "active");
        assert_eq!(rows[1].2, 2);
    }

    #[test]
    fn forgotten_tombstone_blocks_automatic_reactivation() {
        let database = Database::open(":memory:").unwrap();
        let first = message("memory:forgotten:1", "我喜欢薄荷。", 10);
        let second = message("memory:forgotten:2", "我喜欢薄荷。", 20);
        let candidate = extract(&first);
        apply(&database, &first, &candidate).unwrap();
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE long_memory SET status = 'forgotten', is_active = 0",
                [],
            )
            .unwrap();

        let stats = apply(&database, &second, &extract(&second)).unwrap();
        assert_eq!(stats.blocked_by_tombstone, 1);
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE long_memory SET protocol = 'legacy', session_type = 'legacy'",
                [],
            )
            .unwrap();
        let third = message("memory:forgotten:3", "我喜欢薄荷。", 30);
        let legacy_stats = apply(&database, &third, &extract(&third)).unwrap();
        assert_eq!(legacy_stats.blocked_by_tombstone, 1);
        let connection = database.conn.lock().unwrap();
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM long_memory", [], |row| row.get(0))
            .unwrap();
        let source_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM memory_sources", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(source_count, 1);
    }
}
