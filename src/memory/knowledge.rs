//! Conservative session knowledge candidates with independent-source evidence.

use rusqlite::{OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};

use crate::db::Database;
use crate::pipeline::InMessage;

const ACTIVE_CONFIDENCE: i32 = 85;
const INITIAL_CONFIDENCE: i32 = 70;
const ACTOR_REINFORCEMENT: i32 = 15;
const MAX_INPUT_CHARS: usize = 500;
const MAX_SUBJECT_CHARS: usize = 40;
const MAX_VALUE_CHARS: usize = 120;

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtractedKnowledge {
    normalized_key: String,
    subject: String,
    content: String,
    category: &'static str,
    confidence: i32,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct UpdateStats {
    inserted: usize,
    reinforced: usize,
    promoted: usize,
    duplicate_sources: usize,
    blocked_by_tombstone: usize,
}

struct ExistingKnowledge {
    id: i64,
    confidence: i32,
    status: String,
}

pub(super) async fn observe(msg: &InMessage) {
    if !crate::pipeline::current_config().memories.knowledge_enabled {
        return;
    }
    let Some(candidate) = extract(msg) else {
        return;
    };
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    if let Err(error) = apply(&database, msg, &candidate) {
        log::debug!("[AliceBot] knowledge candidate update failed: {error}");
    }
}

fn apply(
    database: &Database,
    msg: &InMessage,
    candidate: &ExtractedKnowledge,
) -> Result<UpdateStats, String> {
    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "knowledge database lock failed".to_string())?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let mut stats = UpdateStats::default();

    if has_forgotten_tombstone(&transaction, &candidate.normalized_key)? {
        stats.blocked_by_tombstone += 1;
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(stats);
    }

    if let Some(existing) =
        matching_knowledge(&transaction, &candidate.normalized_key, &candidate.content)?
    {
        if !insert_source(&transaction, existing.id, msg)? {
            stats.duplicate_sources += 1;
            transaction.commit().map_err(|error| error.to_string())?;
            return Ok(stats);
        }
        stats.reinforced += 1;
        let distinct_actors = distinct_source_actors(&transaction, existing.id)?;
        let evidence_confidence = candidate.confidence.saturating_add(
            i32::try_from(distinct_actors.saturating_sub(1))
                .unwrap_or(i32::MAX)
                .saturating_mul(ACTOR_REINFORCEMENT),
        );
        let confidence = existing.confidence.max(evidence_confidence).clamp(0, 100);
        let status = if existing.status == "active" || confidence >= ACTIVE_CONFIDENCE {
            "active"
        } else {
            "candidate"
        };
        transaction
            .execute(
                "UPDATE knowledge
                 SET confidence = ?1, status = ?2, is_active = ?3,
                     updated_at = MAX(COALESCE(updated_at, ?4), ?4)
                 WHERE id = ?5",
                params![
                    confidence,
                    status,
                    i32::from(status == "active"),
                    msg.timestamp,
                    existing.id
                ],
            )
            .map_err(|error| error.to_string())?;
        if existing.status != "active" && status == "active" {
            supersede_other_versions(
                &transaction,
                &candidate.normalized_key,
                existing.id,
                msg.timestamp,
            )?;
            stats.promoted += 1;
        }
    } else {
        let version = next_version(&transaction, &candidate.normalized_key)?;
        transaction
            .execute(
                "INSERT INTO knowledge
                 (normalized_key, subject, content, category, scope, protocol,
                  session_type, session_id, source, confidence, status, version,
                  is_active, access_count, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, 'session', ?5, ?6, ?7, 'message',
                         ?8, 'candidate', ?9, 0, 0, ?10, ?10)",
                params![
                    candidate.normalized_key,
                    candidate.subject,
                    candidate.content,
                    candidate.category,
                    msg.protocol,
                    msg.session_type,
                    msg.session_id,
                    candidate.confidence.clamp(0, 100),
                    version,
                    msg.timestamp,
                ],
            )
            .map_err(|error| error.to_string())?;
        let knowledge_id = transaction.last_insert_rowid();
        insert_source(&transaction, knowledge_id, msg)?;
        stats.inserted += 1;
    }

    transaction.commit().map_err(|error| error.to_string())?;
    Ok(stats)
}

fn has_forgotten_tombstone(
    transaction: &Transaction<'_>,
    normalized_key: &str,
) -> Result<bool, String> {
    transaction
        .query_row(
            "SELECT 1 FROM knowledge
             WHERE normalized_key = ?1 AND status = 'forgotten' LIMIT 1",
            params![normalized_key],
            |_| Ok(()),
        )
        .optional()
        .map(|row| row.is_some())
        .map_err(|error| error.to_string())
}

fn matching_knowledge(
    transaction: &Transaction<'_>,
    normalized_key: &str,
    content: &str,
) -> Result<Option<ExistingKnowledge>, String> {
    transaction
        .query_row(
            "SELECT id, confidence, status FROM knowledge
             WHERE normalized_key = ?1 AND content = ?2
               AND status IN ('candidate', 'active')
             ORDER BY version DESC LIMIT 1",
            params![normalized_key, content],
            |row| {
                Ok(ExistingKnowledge {
                    id: row.get(0)?,
                    confidence: row.get(1)?,
                    status: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(|error| error.to_string())
}

fn next_version(transaction: &Transaction<'_>, normalized_key: &str) -> Result<i32, String> {
    transaction
        .query_row(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM knowledge WHERE normalized_key = ?1",
            params![normalized_key],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())
}

fn insert_source(
    transaction: &Transaction<'_>,
    knowledge_id: i64,
    msg: &InMessage,
) -> Result<bool, String> {
    transaction
        .execute(
            "INSERT OR IGNORE INTO knowledge_sources
             (knowledge_id, source_type, source_id, source_subject_id,
              evidence_weight, created_at)
             VALUES (?1, 'message', ?2, ?3, 1, ?4)",
            params![knowledge_id, msg.event_key, msg.sender_id, msg.timestamp],
        )
        .map(|changed| changed > 0)
        .map_err(|error| error.to_string())
}

fn distinct_source_actors(transaction: &Transaction<'_>, knowledge_id: i64) -> Result<i64, String> {
    transaction
        .query_row(
            "SELECT COUNT(DISTINCT source_subject_id) FROM knowledge_sources
             WHERE knowledge_id = ?1 AND source_type = 'message'",
            params![knowledge_id],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())
}

fn supersede_other_versions(
    transaction: &Transaction<'_>,
    normalized_key: &str,
    active_id: i64,
    now: i64,
) -> Result<(), String> {
    transaction
        .execute(
            "UPDATE knowledge
             SET status = 'superseded', is_active = 0, superseded_by = ?1,
                 updated_at = MAX(COALESCE(updated_at, ?2), ?2)
             WHERE normalized_key = ?3 AND id <> ?1
               AND status IN ('candidate', 'active')",
            params![active_id, now, normalized_key],
        )
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn extract(msg: &InMessage) -> Option<ExtractedKnowledge> {
    let text = msg.content.trim();
    if msg.session_type == "private"
        || text.is_empty()
        || text.chars().count() > MAX_INPUT_CHARS
        || text.ends_with(['?', '？'])
        || text.contains("http://")
        || text.contains("https://")
        || super::candidates::contains_sensitive_content(text)
        || ["听说", "可能", "大概", "也许", "好像"]
            .iter()
            .any(|marker| text.contains(marker))
    {
        return None;
    }

    let (category, label, body, require_pair) = if let Some(body) =
        strip_any(text, &["群规：", "群规:", "本群规定：", "本群规定:"])
    {
        ("group_rule", "群规", body, false)
    } else if let Some(body) = strip_any(text, &["群公告：", "群公告:", "公告：", "公告:"])
    {
        ("announcement", "群公告", body, false)
    } else if let Some(body) = strip_any(text, &["群信息：", "群信息:", "群资料：", "群资料:"])
    {
        ("group_fact", "群信息", body, true)
    } else if let Some(body) = text
        .strip_prefix("本群的")
        .or_else(|| text.strip_prefix("本群"))
    {
        ("group_fact", "群信息", body, true)
    } else {
        return None;
    };

    let pair = parse_pair(body);
    if require_pair && pair.is_none() {
        return None;
    }
    let (subject, value, key_material, content) = match pair {
        Some((subject, value)) => {
            let content = format!("{label}：{subject}：{value}");
            (subject.clone(), value, subject, content)
        }
        None => {
            let statement = clean_value(body, MAX_VALUE_CHARS).unwrap_or_default();
            let content = format!("{label}：{statement}");
            (label.to_string(), statement.clone(), statement, content)
        }
    };
    if subject.is_empty() || value.is_empty() {
        return None;
    }
    let normalized_subject = normalize(&key_material);
    if normalized_subject.is_empty() {
        return None;
    }
    let normalized_key = knowledge_key(msg, category, &normalized_subject);
    Some(ExtractedKnowledge {
        normalized_key,
        subject,
        content,
        category,
        confidence: INITIAL_CONFIDENCE,
    })
}

fn strip_any<'a>(text: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes
        .iter()
        .find_map(|prefix| text.strip_prefix(prefix))
        .map(str::trim)
}

fn parse_pair(body: &str) -> Option<(String, String)> {
    for separator in ["=", "：", "是"] {
        let Some(index) = body.find(separator) else {
            continue;
        };
        let subject = clean_value(&body[..index], MAX_SUBJECT_CHARS)?;
        let value = clean_value(&body[index + separator.len()..], MAX_VALUE_CHARS)?;
        if !subject.is_empty() && !value.is_empty() {
            return Some((subject, value));
        }
    }
    None
}

fn clean_value(value: &str, max_chars: usize) -> Option<String> {
    let value = value
        .trim()
        .trim_matches(|character: char| "，。！？!?；;：:,、= '“”\"".contains(character))
        .chars()
        .filter(|character| !character.is_control())
        .take(max_chars + 1)
        .collect::<String>();
    let count = value.chars().count();
    (count > 0 && count <= max_chars).then_some(value)
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && !character.is_ascii_punctuation())
        .flat_map(char::to_lowercase)
        .collect()
}

fn knowledge_key(msg: &InMessage, category: &str, normalized_subject: &str) -> String {
    let mut hasher = Sha256::new();
    for value in [
        msg.protocol.as_str(),
        msg.session_type.as_str(),
        msg.session_id.as_str(),
        category,
        normalized_subject,
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    format!(
        "knowledge:{category}:{}",
        digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(event: &str, sender: &str, content: &str, timestamp: i64) -> InMessage {
        InMessage {
            event_key: event.to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            sender_id: sender.to_string(),
            sender_name: sender.to_string(),
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
    fn extraction_requires_explicit_group_fact_syntax_and_rejects_sensitive_text() {
        let fact = extract(&message(
            "event-1",
            "user-1",
            "群信息：活动时间=周五晚上",
            1,
        ))
        .unwrap();
        assert_eq!(fact.subject, "活动时间");
        assert_eq!(fact.content, "群信息：活动时间：周五晚上");
        let first_rule = extract(&message("rule-1", "user-1", "群规：禁止广告", 2)).unwrap();
        let second_rule = extract(&message("rule-2", "user-1", "群规：禁止刷屏", 3)).unwrap();
        assert_eq!(first_rule.content, "群规：禁止广告");
        assert_ne!(first_rule.normalized_key, second_rule.normalized_key);
        assert!(extract(&message("event-2", "user-1", "活动时间是周五", 2)).is_none());
        assert!(
            extract(&message(
                "event-3",
                "user-1",
                "群信息：管理员密码=123456",
                3,
            ))
            .is_none()
        );
    }

    #[test]
    fn two_distinct_actors_promote_but_one_actor_repetition_does_not() {
        let database = Database::open(":memory:").unwrap();
        let first = message("event-1", "user-1", "群信息：活动时间=周五晚上", 1);
        let repeated = message("event-2", "user-1", "群信息：活动时间=周五晚上", 2);
        let corroborated = message("event-3", "user-2", "群信息：活动时间=周五晚上", 3);
        let candidate = extract(&first).unwrap();

        assert_eq!(apply(&database, &first, &candidate).unwrap().inserted, 1);
        assert_eq!(
            apply(&database, &repeated, &candidate).unwrap().reinforced,
            1
        );
        let before: (String, i32) = database
            .conn
            .lock()
            .unwrap()
            .query_row("SELECT status, confidence FROM knowledge", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!(before, ("candidate".to_string(), INITIAL_CONFIDENCE));

        let promoted = apply(&database, &corroborated, &candidate).unwrap();
        assert_eq!(promoted.promoted, 1);
        let after: (String, i32, i64) = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT status, confidence,
                        (SELECT COUNT(*) FROM knowledge_sources WHERE knowledge_id = knowledge.id)
                 FROM knowledge",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(after, ("active".to_string(), ACTIVE_CONFIDENCE, 3));
    }

    #[test]
    fn corroborated_conflict_supersedes_the_previous_active_version() {
        let database = Database::open(":memory:").unwrap();
        let friday_one = message("event-1", "user-1", "群信息：活动时间=周五", 1);
        let friday_two = message("event-2", "user-2", "群信息：活动时间=周五", 2);
        let saturday_one = message("event-3", "user-1", "群信息：活动时间=周六", 3);
        let saturday_two = message("event-4", "user-3", "群信息：活动时间=周六", 4);
        for msg in [&friday_one, &friday_two, &saturday_one, &saturday_two] {
            let candidate = extract(msg).unwrap();
            apply(&database, msg, &candidate).unwrap();
        }

        let rows = database
            .conn
            .lock()
            .unwrap()
            .prepare(
                "SELECT content, status, version, superseded_by
                 FROM knowledge ORDER BY version",
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
        assert_eq!(rows[1].1, "active");
        assert_eq!(rows[1].2, 2);
        assert!(rows[0].3.is_some());
    }

    #[test]
    fn forgotten_tombstone_blocks_automatic_knowledge_recreation() {
        let database = Database::open(":memory:").unwrap();
        let first = message("event-1", "user-1", "群信息：活动时间=周五", 1);
        let candidate = extract(&first).unwrap();
        apply(&database, &first, &candidate).unwrap();
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE knowledge SET status = 'forgotten', is_active = 0",
                [],
            )
            .unwrap();

        let second = message("event-2", "user-2", "群信息：活动时间=周五", 2);
        let stats = apply(&database, &second, &extract(&second).unwrap()).unwrap();
        assert_eq!(stats.blocked_by_tombstone, 1);
        let count: i64 = database
            .conn
            .lock()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM knowledge", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }
}
