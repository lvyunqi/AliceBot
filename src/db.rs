//! SQLite 数据库封装
//!
//! 使用 rusqlite（bundled 模式，编译自带 SQLite）。WAL 模式提升并发。
//! 每条消息全量入库，定时由后台任务压缩/清理。

mod migrations;

use migrations::DatabaseError;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use std::path::Path;
use std::sync::Mutex;

/// 数据库连接（单连接，写操作串行化）
pub struct Database {
    pub conn: Mutex<Connection>,
    pub(crate) memory_search: crate::memory::search::SearchBackend,
}

/// 一次出站发送尝试的审计输入。只保存必要的路由和内容摘要，不保存凭据。
#[derive(Debug, Clone)]
pub struct OutboundAttempt {
    pub action_key: String,
    pub source_event_key: Option<String>,
    pub protocol: String,
    pub bot_account_id: String,
    pub session_type: String,
    pub session_id: String,
    pub content: String,
    pub media_type: Option<String>,
    pub media_url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutboundClaim {
    Claimed(i64),
    AlreadyAccepted,
    InFlightOrUncertain,
}

/// 一次保留期清理的脱敏和删除统计，不含任何用户正文。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetentionReport {
    pub raw_events_redacted: usize,
    pub messages_deleted: usize,
}

pub struct DecisionTrace<'a> {
    pub event_key: &'a str,
    pub session_id: &'a str,
    pub policy_version: &'a str,
    pub score: f32,
    pub threshold: f32,
    pub p_rule: f32,
    pub p_final: f32,
    pub random_value: f32,
    pub activity_ewma: f32,
    pub direct: bool,
    pub outcome: &'a str,
    pub reason: &'a str,
    pub signals_json: &'a str,
    pub coalesced_count: usize,
    pub created_at: i64,
}

impl Database {
    /// 打开/创建数据库
    pub fn open(path: &str) -> Result<Self, DatabaseError> {
        let path = Path::new(path);
        let had_existing_database = migrations::has_existing_database(path);
        let mut conn = Connection::open(path)?;
        let report = migrations::prepare_database(path, &mut conn, had_existing_database)?;
        let recovered = conn.execute(
            "UPDATE messages
             SET processing_status = 'record_only',
                 processing_error = 'interrupted_by_restart'
             WHERE direction = 'inbound'
               AND processing_status IN ('recorded', 'queued', 'processing')",
            [],
        )?;
        if recovered > 0 {
            log::warn!("[AliceBot] marked {recovered} interrupted inbound messages as record_only");
        }
        let memory_search = match crate::memory::search::initialize(&mut conn) {
            Ok(backend) => backend,
            Err(error) => {
                log::warn!(
                    "[AliceBot] FTS5 memory search initialization failed; using lexical fallback: {error}"
                );
                crate::memory::search::SearchBackend::Lexical
            }
        };

        if report.from < report.to {
            match report.backup_path {
                Some(backup_path) => log::info!(
                    "[AliceBot] database migrated v{} -> v{}; backup: {}",
                    report.from,
                    report.to,
                    backup_path.display()
                ),
                None => log::info!("[AliceBot] database initialized at schema v{}", report.to),
            }
        }

        Ok(Self {
            conn: Mutex::new(conn),
            memory_search,
        })
    }

    /// 插入一条消息
    pub fn insert_message(
        &self,
        msg: &crate::pipeline::InMessage,
    ) -> Result<bool, rusqlite::Error> {
        let store_raw_events = crate::pipeline::current_config().privacy.store_raw_events;
        self.insert_message_with_raw_event_storage(msg, store_raw_events)
    }

    /// 插入消息并按隐私配置决定是否保存已脱敏的原始事件 JSON。
    pub fn insert_message_with_raw_event_storage(
        &self,
        msg: &crate::pipeline::InMessage,
        store_raw_events: bool,
    ) -> Result<bool, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let media_type = msg.media.first().map(|media| media.media_type.as_str());
        let media_url = msg
            .media
            .first()
            .map(|media| crate::media::redact_url_for_storage(&media.url));
        let media_requires_cache = msg.media.first().is_some_and(|media| {
            crate::media::sanitize_remote_media_url(&media.url, false)
                .is_some_and(|media| media.requires_cache)
        });
        let raw_json = store_raw_events.then_some(msg.safe_raw_json.as_str());
        let changed = conn.execute(
            "INSERT OR IGNORE INTO messages
             (event_key, protocol, bot_account_id, direction, session_type, session_id, sender_id,
              sender_name, message_id, content, raw_json, has_media, media_type,
              media_url, media_requires_cache, reply_to_id, at_me, processing_status, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'inbound', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, 'recorded', ?17, ?17)",
            params![
                msg.event_key,
                msg.protocol,
                msg.bot_account_id,
                msg.session_type,
                msg.session_id,
                msg.sender_id,
                msg.sender_name,
                msg.message_id,
                msg.content,
                raw_json,
                msg.has_media as i32,
                media_type,
                media_url,
                media_requires_cache as i32,
                msg.reply_to_id,
                msg.at_me as i32,
                msg.timestamp,
            ],
        )?;
        Ok(changed > 0)
    }

    /// 清理超出保留期的事件详情，且不删除仍被派生数据或审计引用的 journal 行。
    pub fn apply_retention(
        &self,
        store_raw_events: bool,
        raw_event_retention_days: u32,
        message_retention_days: u32,
        now: i64,
    ) -> Result<RetentionReport, rusqlite::Error> {
        const MILLIS_PER_DAY: i64 = 86_400_000;
        let raw_days = i64::from(raw_event_retention_days.clamp(1, 3_650));
        let message_days = i64::from(message_retention_days.clamp(1, 3_650));
        let raw_cutoff = now.saturating_sub(raw_days.saturating_mul(MILLIS_PER_DAY));
        let message_cutoff = now.saturating_sub(message_days.saturating_mul(MILLIS_PER_DAY));
        let mut connection = self.conn.lock().unwrap();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let raw_events_redacted = transaction.execute(
            "UPDATE messages
             SET raw_json = NULL, updated_at = MAX(COALESCE(updated_at, ?1), ?1)
             WHERE raw_json IS NOT NULL
               AND (?2 = 0 OR created_at < ?3)",
            params![now, i32::from(store_raw_events), raw_cutoff],
        )?;
        let messages_deleted = transaction.execute(
            "DELETE FROM messages AS message
             WHERE message.direction = 'inbound'
               AND message.processing_status IN ('processed', 'record_only')
               AND message.created_at < ?1
               AND NOT EXISTS (
                   SELECT 1 FROM memory_sources AS source
                   WHERE source.source_type = 'message'
                     AND source.source_id = message.event_key
               )
               AND NOT EXISTS (
                   SELECT 1 FROM knowledge_sources AS source
                   WHERE source.source_type = 'message'
                     AND source.source_id = message.event_key
               )
               AND NOT EXISTS (
                   SELECT 1 FROM decision_traces AS trace
                   WHERE trace.event_key = message.event_key
               )
               AND NOT EXISTS (
                   SELECT 1 FROM outbound_messages AS outbound
                   WHERE outbound.source_event_key = message.event_key
               )",
            params![message_cutoff],
        )?;
        transaction.commit()?;

        Ok(RetentionReport {
            raw_events_redacted,
            messages_deleted,
        })
    }

    pub fn set_message_processing_status(
        &self,
        event_key: &str,
        status: &str,
        error: Option<&str>,
        now: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE messages
             SET processing_status = ?1, processing_error = ?2,
                 processed_at = CASE WHEN ?1 = 'processed' THEN ?3 ELSE processed_at END,
                 updated_at = MAX(COALESCE(updated_at, ?3), ?3)
             WHERE event_key = ?4 AND direction = 'inbound'",
            params![status, error.map(truncate_for_storage), now, event_key],
        )?;
        Ok(())
    }

    /// 原子认领一个动作。accepted/pending 动作不会再次提交给宿主。
    pub fn claim_outbound_attempt(
        &self,
        attempt: &OutboundAttempt,
        now: i64,
    ) -> Result<OutboundClaim, rusqlite::Error> {
        let mut conn = self.conn.lock().unwrap();
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing = transaction
            .query_row(
                "SELECT id, status FROM outbound_messages WHERE action_key = ?1",
                params![attempt.action_key],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        if let Some((id, status)) = existing {
            if status == "accepted" {
                transaction.commit()?;
                return Ok(OutboundClaim::AlreadyAccepted);
            }
            if status == "pending" || !matches!(status.as_str(), "rejected" | "invalid") {
                transaction.commit()?;
                return Ok(OutboundClaim::InFlightOrUncertain);
            }
            transaction.execute(
                "UPDATE outbound_messages
                 SET source_event_key = ?1, protocol = ?2, bot_account_id = ?3,
                     session_type = ?4, session_id = ?5, content = ?6,
                     media_type = ?7, media_url = ?8, status = 'pending',
                     host_status = NULL, error = NULL,
                     attempt_count = attempt_count + 1, updated_at = ?9
                 WHERE id = ?10",
                params![
                    attempt.source_event_key,
                    attempt.protocol,
                    attempt.bot_account_id,
                    attempt.session_type,
                    attempt.session_id,
                    truncate_for_storage(&attempt.content),
                    attempt.media_type,
                    attempt
                        .media_url
                        .as_deref()
                        .map(crate::media::redact_url_for_storage),
                    now,
                    id,
                ],
            )?;
            transaction.commit()?;
            return Ok(OutboundClaim::Claimed(id));
        }

        transaction.execute(
            "INSERT INTO outbound_messages
             (action_key, source_event_key, protocol, bot_account_id, session_type, session_id,
              content, media_type, media_url, status, attempt_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'pending', 1, ?10, ?10)",
            params![
                attempt.action_key,
                attempt.source_event_key,
                attempt.protocol,
                attempt.bot_account_id,
                attempt.session_type,
                attempt.session_id,
                truncate_for_storage(&attempt.content),
                attempt.media_type,
                attempt
                    .media_url
                    .as_deref()
                    .map(crate::media::redact_url_for_storage),
                now,
            ],
        )?;
        let id = transaction.last_insert_rowid();
        transaction.commit()?;
        Ok(OutboundClaim::Claimed(id))
    }

    /// 更新宿主接收结果；`host_status` 使用 Debug 名称，便于跨版本诊断。
    pub fn finish_outbound_attempt(
        &self,
        id: i64,
        status: &str,
        host_status: Option<&str>,
        error: Option<&str>,
        now: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE outbound_messages
             SET status = ?1, host_status = ?2, error = ?3, updated_at = ?4
             WHERE id = ?5 AND status = 'pending'",
            params![
                status,
                host_status,
                error.map(truncate_for_storage),
                now,
                id
            ],
        )?;
        Ok(())
    }

    /// 写入一条幂等决策 trace。
    pub fn insert_decision_trace(&self, trace: &DecisionTrace<'_>) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO decision_traces
             (event_key, session_id, policy_version, score, threshold, p_rule, p_final,
              random_value, activity_ewma, direct, outcome, reason, signals_json,
              coalesced_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                trace.event_key,
                trace.session_id,
                trace.policy_version,
                trace.score,
                trace.threshold,
                trace.p_rule,
                trace.p_final,
                trace.random_value,
                trace.activity_ewma,
                trace.direct as i32,
                trace.outcome,
                trace.reason,
                truncate_for_storage(trace.signals_json),
                trace.coalesced_count.min(i64::MAX as usize) as i64,
                trace.created_at,
            ],
        )?;
        Ok(())
    }

    /// 记录一次 provider 尝试开始；只存可观测指标，不存消息原文。
    pub fn begin_llm_call(
        &self,
        task: &str,
        provider_id: &str,
        protocol: &str,
        model: &str,
        attempt: u32,
        input_chars: usize,
        now: i64,
    ) -> Result<i64, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO llm_calls
             (task, provider_id, protocol, model, attempt, input_chars, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            params![
                task,
                provider_id,
                protocol,
                model,
                attempt,
                input_chars.min(i64::MAX as usize) as i64,
                now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// 完成一次 provider 尝试，`error_kind` 只允许错误分类而非原始响应。
    pub fn finish_llm_call(
        &self,
        id: i64,
        status: &str,
        error_kind: Option<&str>,
        output_chars: usize,
        latency_ms: u64,
        now: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE llm_calls
             SET status = ?1, error_kind = ?2, output_chars = ?3,
                 latency_ms = ?4, updated_at = ?5
             WHERE id = ?6",
            params![
                status,
                error_kind,
                output_chars.min(i64::MAX as usize) as i64,
                latency_ms.min(i64::MAX as u64) as i64,
                now,
                id,
            ],
        )?;
        Ok(())
    }

    pub fn begin_compaction_run(
        &self,
        run_key: &str,
        cursor_start: i64,
        cursor_end: i64,
        started_at: i64,
    ) -> Result<bool, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let changed = conn.execute(
            "INSERT OR IGNORE INTO compaction_runs
             (run_key, cursor_start, cursor_end, status, started_at)
             VALUES (?1, ?2, ?3, 'running', ?4)",
            params![run_key, cursor_start, cursor_end, started_at],
        )?;
        Ok(changed > 0)
    }

    pub fn finish_compaction_run(
        &self,
        run_key: &str,
        status: &str,
        processed_count: i64,
        error: Option<&str>,
        finished_at: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE compaction_runs
             SET status = ?1, processed_count = ?2, error = ?3, finished_at = ?4
             WHERE run_key = ?5",
            params![
                status,
                processed_count,
                error.map(truncate_for_storage),
                finished_at,
                run_key
            ],
        )?;
        Ok(())
    }

    /// 获取元数据
    pub fn get_meta(&self, key: &str) -> Result<Option<String>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1")?;
        let mut rows = stmt.query(params![key])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get::<_, String>(0)?)),
            None => Ok(None),
        }
    }
}

/// 初始化数据库（供 lib.rs 调用）
pub async fn init_database(path: &str) -> Result<Database, String> {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("数据库目录创建失败: {e}"))?;
    }
    Database::open(path).map_err(|e| format!("数据库打开失败: {}", e))
}

fn truncate_for_storage(value: &str) -> String {
    value.chars().take(16_384).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::InMessage;

    fn test_message(event_key: &str) -> InMessage {
        InMessage {
            event_key: event_key.to_string(),
            protocol: "qq-official".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            sender_id: "member-1".to_string(),
            sender_name: "测试用户".to_string(),
            message_id: "message-1".to_string(),
            reply_to_id: String::new(),
            content: "测试".to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp: 1,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn duplicate_event_key_is_ignored() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-db-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        let message = test_message("qq-official:message-1");

        assert!(
            database
                .insert_message(&message)
                .expect("first insert should work")
        );
        assert!(
            !database
                .insert_message(&message)
                .expect("duplicate insert should be ignored")
        );

        let count: i64 = database
            .conn
            .lock()
            .expect("database lock should work")
            .query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))
            .expect("count should work");
        assert_eq!(count, 1);

        database
            .set_message_processing_status(&message.event_key, "record_only", Some("queue_full"), 2)
            .expect("record-only status should update");
        let processing: (String, String) = database
            .conn
            .lock()
            .expect("database lock should work")
            .query_row(
                "SELECT processing_status, processing_error FROM messages
                 WHERE event_key = ?1",
                params![message.event_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("processing status should be queryable");
        assert_eq!(
            processing,
            ("record_only".to_string(), "queue_full".to_string())
        );

        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn signed_media_url_records_cache_requirement_without_persisting_credential() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-db-media-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        let mut message = test_message("qq-official:media-1");
        message.has_media = true;
        message.media = vec![crate::pipeline::MediaRef {
            url: "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=abc&rkey=temporary&spec=0"
                .to_string(),
            media_type: "image/jpeg".to_string(),
        }];

        assert!(
            database
                .insert_message(&message)
                .expect("message should insert")
        );
        let row: (String, i64) = database
            .conn
            .lock()
            .expect("database lock should work")
            .query_row(
                "SELECT media_url, media_requires_cache FROM messages WHERE event_key = ?1",
                params![message.event_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("media row should exist");
        assert!(!row.0.contains("rkey"));
        assert_eq!(row.1, 1);
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn outbound_attempt_is_audited_and_signed_url_is_redacted() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-outbound-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        let id = match database
            .claim_outbound_attempt(
                &OutboundAttempt {
                    action_key: "reply:test-event".to_string(),
                    source_event_key: Some("qq-official:test-event".to_string()),
                    protocol: "qq-official".to_string(),
                    bot_account_id: "bot-1".to_string(),
                    session_type: "group".to_string(),
                    session_id: "group-1".to_string(),
                    content: "hello".to_string(),
                    media_type: Some("image".to_string()),
                    media_url: Some("https://example.test/a.png?rkey=secret&spec=0".to_string()),
                },
                10,
            )
            .expect("outbound attempt should insert")
        {
            OutboundClaim::Claimed(id) => id,
            other => panic!("unexpected claim result: {other:?}"),
        };
        database
            .finish_outbound_attempt(id, "accepted", Some("Accepted"), None, 11)
            .expect("outbound attempt should finish");

        let connection = database.conn.lock().expect("database lock should work");
        let row: (String, String, String) = connection
            .query_row(
                "SELECT status, media_url, host_status FROM outbound_messages WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("outbound row should exist");
        assert_eq!(row.0, "accepted");
        assert_eq!(row.1, "https://example.test/a.png?spec=0");
        assert_eq!(row.2, "Accepted");

        drop(connection);
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn action_claim_is_at_most_once_and_retries_only_definite_rejections() {
        let database = Database::open(":memory:").unwrap();
        let attempt = OutboundAttempt {
            action_key: "reply:event-1:text".to_string(),
            source_event_key: Some("event-1".to_string()),
            protocol: "onebot11".to_string(),
            bot_account_id: "bot-1".to_string(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            content: "hello".to_string(),
            media_type: None,
            media_url: None,
        };

        let id = match database.claim_outbound_attempt(&attempt, 10).unwrap() {
            OutboundClaim::Claimed(id) => id,
            other => panic!("unexpected first claim: {other:?}"),
        };
        assert_eq!(
            database.claim_outbound_attempt(&attempt, 11).unwrap(),
            OutboundClaim::InFlightOrUncertain
        );
        database
            .finish_outbound_attempt(id, "accepted", Some("Accepted"), None, 12)
            .unwrap();
        assert_eq!(
            database.claim_outbound_attempt(&attempt, 13).unwrap(),
            OutboundClaim::AlreadyAccepted
        );

        let mut retry = attempt.clone();
        retry.action_key = "reply:event-2:text".to_string();
        let retry_id = match database.claim_outbound_attempt(&retry, 20).unwrap() {
            OutboundClaim::Claimed(id) => id,
            other => panic!("unexpected retry claim: {other:?}"),
        };
        database
            .finish_outbound_attempt(
                retry_id,
                "rejected",
                Some("QueueFull"),
                Some("host rejected enqueue"),
                21,
            )
            .unwrap();
        assert_eq!(
            database.claim_outbound_attempt(&retry, 22).unwrap(),
            OutboundClaim::Claimed(retry_id)
        );
        let attempt_count: i64 = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT attempt_count FROM outbound_messages WHERE id = ?1",
                params![retry_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(attempt_count, 2);
    }

    #[test]
    fn decision_trace_is_idempotent() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-decision-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        database
            .insert_decision_trace(&DecisionTrace {
                event_key: "event-1",
                session_id: "group-1",
                policy_version: "test-v1",
                score: 61.5,
                threshold: 60.0,
                p_rule: 0.72,
                p_final: 0.66,
                random_value: 0.2,
                activity_ewma: 0.3,
                direct: false,
                outcome: "reply",
                reason: "score_reached",
                signals_json: r#"{"question":true}"#,
                coalesced_count: 1,
                created_at: 10,
            })
            .expect("trace should insert");
        database
            .insert_decision_trace(&DecisionTrace {
                event_key: "event-1",
                session_id: "group-1",
                policy_version: "test-v1",
                score: 0.0,
                threshold: 60.0,
                p_rule: 0.0,
                p_final: 0.0,
                random_value: 0.5,
                activity_ewma: 0.0,
                direct: false,
                outcome: "skip",
                reason: "duplicate",
                signals_json: "{}",
                coalesced_count: 1,
                created_at: 11,
            })
            .expect("duplicate trace should be ignored");

        let connection = database.conn.lock().expect("database lock should work");
        let row: (i64, String, f64, String, f64, i64) = connection
            .query_row(
                "SELECT COUNT(*), outcome, score, policy_version, p_final, coalesced_count
                 FROM decision_traces WHERE event_key = 'event-1'",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .expect("trace should exist");
        assert_eq!(row.0, 1);
        assert_eq!(row.1, "reply");
        assert_eq!(row.2, 61.5);
        assert_eq!(row.3, "test-v1");
        assert!((row.4 - 0.66).abs() < 0.000_01);
        assert_eq!(row.5, 1);

        drop(connection);
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn llm_audit_stores_metrics_without_prompt_text() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-llm-audit-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        let id = database
            .begin_llm_call("group_reply", "primary", "openai", "mock-model", 1, 123, 10)
            .expect("llm call should insert");
        database
            .finish_llm_call(id, "error", Some("RateLimited"), 0, 42, 52)
            .expect("llm call should finish");

        let connection = database.conn.lock().expect("database lock should work");
        let row: (String, String, i64, i64, i64) = connection
            .query_row(
                "SELECT status, error_kind, input_chars, output_chars, latency_ms
                 FROM llm_calls WHERE id = ?1",
                params![id],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .expect("llm audit row should exist");
        assert_eq!(
            row,
            ("error".to_string(), "RateLimited".to_string(), 123, 0, 42)
        );

        drop(connection);
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }
}
