//! 事件 journal 的隐私保留清理。

use crate::config::PrivacyConfig;
use crate::db::{Database, RetentionReport};

/// 按当前配置执行一次保留清理；失败只返回稳定错误分类。
pub(crate) fn run(config: &PrivacyConfig) -> Result<RetentionReport, String> {
    let database =
        crate::pipeline::try_db().ok_or_else(|| "database is not initialized".to_string())?;
    apply_in(
        &database,
        config.store_raw_events,
        config.raw_event_retention_days,
        config.message_retention_days,
        chrono::Utc::now().timestamp_millis(),
    )
}

fn apply_in(
    database: &Database,
    store_raw_events: bool,
    raw_event_retention_days: u32,
    message_retention_days: u32,
    now: i64,
) -> Result<RetentionReport, String> {
    database
        .apply_retention(
            store_raw_events,
            raw_event_retention_days,
            message_retention_days,
            now,
        )
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Database, DecisionTrace, OutboundAttempt};
    use crate::pipeline::InMessage;

    const DAY: i64 = 86_400_000;

    fn message(event_key: &str, timestamp: i64) -> InMessage {
        InMessage {
            event_key: event_key.to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "retention-group".to_string(),
            sender_id: "retention-user".to_string(),
            sender_name: "保留测试用户".to_string(),
            message_id: event_key.to_string(),
            reply_to_id: String::new(),
            content: format!("正文-{event_key}"),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp,
            safe_raw_json: format!(r#"{{"event":"{event_key}","token":"should-not-survive"}}"#),
        }
    }

    fn persist(database: &Database, event_key: &str, timestamp: i64, status: &str) {
        let message = message(event_key, timestamp);
        database.insert_message(&message).unwrap();
        if status != "recorded" {
            database
                .set_message_processing_status(event_key, status, None, timestamp)
                .unwrap();
        }
    }

    #[test]
    fn retention_scrubs_raw_payloads_and_deletes_only_unreferenced_completed_rows() {
        let database = Database::open(":memory:").unwrap();
        let now = 200 * DAY;
        persist(&database, "unreferenced", now - 40 * DAY, "processed");
        persist(&database, "protected", now - 40 * DAY, "processed");
        persist(&database, "pending", now - 40 * DAY, "processing");
        persist(&database, "fresh", now - 5 * DAY, "processed");

        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO long_memory
                 (normalized_key, protocol, session_type, scope, session_id, subject_id,
                  kind, content, importance, confidence, privacy, status, version,
                  is_active, created_at, updated_at)
                 VALUES ('retention-protected', 'onebot11', 'group', 'user_session',
                         'retention-group', 'retention-user', 'fact', '保留摘要',
                         70, 85, 'normal', 'active', 1, 1, ?1, ?1)",
                rusqlite::params![now - 40 * DAY],
            )
            .unwrap();
        let memory_id = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT id FROM long_memory WHERE normalized_key = 'retention-protected'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO memory_sources
                 (memory_id, source_type, source_id, evidence_weight, created_at)
                 VALUES (?1, 'message', 'protected', 1, ?2)",
                rusqlite::params![memory_id, now - 40 * DAY],
            )
            .unwrap();

        let report = apply_in(&database, true, 7, 30, now).unwrap();
        assert_eq!(
            report,
            RetentionReport {
                raw_events_redacted: 3,
                messages_deleted: 1,
            }
        );

        let connection = database.conn.lock().unwrap();
        let remaining = connection
            .prepare("SELECT event_key FROM messages ORDER BY event_key")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            remaining,
            vec![
                "fresh".to_string(),
                "pending".to_string(),
                "protected".to_string()
            ]
        );

        let protected: (String, Option<String>) = connection
            .query_row(
                "SELECT content, raw_json FROM messages WHERE event_key = 'protected'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(protected.0, "正文-protected");
        assert!(protected.1.is_none());

        let pending_status: String = connection
            .query_row(
                "SELECT processing_status FROM messages WHERE event_key = 'pending'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(pending_status, "processing");

        let fresh_raw: Option<String> = connection
            .query_row(
                "SELECT raw_json FROM messages WHERE event_key = 'fresh'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(fresh_raw.is_some());
    }

    #[test]
    fn raw_event_storage_can_be_disabled_before_retention_runs() {
        let database = Database::open(":memory:").unwrap();
        let disabled_message = message("disabled", 1);
        database
            .insert_message_with_raw_event_storage(&disabled_message, false)
            .unwrap();
        let existing = message("existing", 1);
        database
            .insert_message_with_raw_event_storage(&existing, true)
            .unwrap();

        let report = apply_in(&database, false, 7, 30, 2).unwrap();
        assert_eq!(
            report,
            RetentionReport {
                raw_events_redacted: 1,
                messages_deleted: 0,
            }
        );

        let raw_json: Option<String> = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT raw_json FROM messages WHERE event_key = 'disabled'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(raw_json.is_none());

        let existing_raw_json: Option<String> = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT raw_json FROM messages WHERE event_key = 'existing'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(existing_raw_json.is_none());
    }

    #[test]
    fn retention_preserves_events_referenced_by_knowledge_traces_or_outbound_audits() {
        let database = Database::open(":memory:").unwrap();
        let now = 200 * DAY;
        for event_key in ["knowledge", "trace", "outbound"] {
            persist(&database, event_key, now - 40 * DAY, "processed");
        }

        let knowledge_id = {
            let connection = database.conn.lock().unwrap();
            connection
                .execute(
                    "INSERT INTO knowledge
                     (normalized_key, subject, content, category, source, confidence, is_active,
                      created_at, updated_at, scope, status, version, protocol, session_type,
                      session_id)
                     VALUES ('retention-knowledge', 'retention-group', '保留摘要', 'group_rule',
                             'message', 85, 1, ?1, ?1, 'session', 'active', 1,
                             'onebot11', 'group', 'retention-group')",
                    rusqlite::params![now - 40 * DAY],
                )
                .unwrap();
            let knowledge_id = connection
                .query_row(
                    "SELECT id FROM knowledge WHERE normalized_key = 'retention-knowledge'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO knowledge_sources
                     (knowledge_id, source_type, source_id, source_subject_id,
                      evidence_weight, created_at)
                     VALUES (?1, 'message', 'knowledge', 'retention-user', 1, ?2)",
                    rusqlite::params![knowledge_id, now - 40 * DAY],
                )
                .unwrap();
            knowledge_id
        };
        assert!(knowledge_id > 0);

        database
            .insert_decision_trace(&DecisionTrace {
                event_key: "trace",
                session_id: "retention-group",
                policy_version: "retention-test",
                score: 0.0,
                threshold: 0.0,
                p_rule: 0.0,
                p_final: 0.0,
                random_value: 0.0,
                activity_ewma: 0.0,
                direct: false,
                outcome: "skip",
                reason: "retention_test",
                signals_json: "{}",
                coalesced_count: 1,
                created_at: now - 40 * DAY,
            })
            .unwrap();
        assert!(matches!(
            database
                .claim_outbound_attempt(
                    &OutboundAttempt {
                        action_key: "retention:outbound".to_string(),
                        source_event_key: Some("outbound".to_string()),
                        protocol: "onebot11".to_string(),
                        bot_account_id: String::new(),
                        session_type: "group".to_string(),
                        session_id: "retention-group".to_string(),
                        content: "审计引用".to_string(),
                        media_type: None,
                        media_url: None,
                    },
                    now - 40 * DAY,
                )
                .unwrap(),
            crate::db::OutboundClaim::Claimed(_)
        ));

        let report = apply_in(&database, true, 7, 30, now).unwrap();
        assert_eq!(
            report,
            RetentionReport {
                raw_events_redacted: 3,
                messages_deleted: 0,
            }
        );
        let connection = database.conn.lock().unwrap();
        let remaining = connection
            .prepare("SELECT event_key FROM messages ORDER BY event_key")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(remaining, vec!["knowledge", "outbound", "trace"]);
    }
}
