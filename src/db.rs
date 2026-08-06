//! SQLite 数据库封装
//!
//! 使用 rusqlite（bundled 模式，编译自带 SQLite）。WAL 模式提升并发。
//! 每条消息全量入库，定时由后台任务压缩/清理。

use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::Mutex;

/// 数据库连接（单连接，写操作串行化）
pub struct Database {
    pub conn: Mutex<Connection>,
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

impl Database {
    /// 打开/创建数据库
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;
        let db = Self {
            conn: Mutex::new(conn),
        };
        db.init_tables()?;
        Ok(db)
    }

    /// 初始化表结构
    fn init_tables(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        // 消息表
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key     TEXT,
                action_key    TEXT,
                protocol      TEXT NOT NULL,
                bot_account_id TEXT,
                direction     TEXT NOT NULL,
                session_type  TEXT NOT NULL,
                session_id    TEXT NOT NULL,
                sender_id     TEXT NOT NULL,
                sender_name   TEXT,
                message_id    TEXT,
                content       TEXT NOT NULL,
                raw_json      TEXT,
                has_media     INTEGER DEFAULT 0,
                media_type    TEXT,
                media_url     TEXT,
                reply_to_id   TEXT,
                at_me         INTEGER DEFAULT 0,
                sentiment     INTEGER,
                created_at    INTEGER NOT NULL,
                updated_at    INTEGER NOT NULL
            );",
        )?;
        ensure_column(&conn, "messages", "event_key", "TEXT")?;
        ensure_column(&conn, "messages", "action_key", "TEXT")?;
        ensure_column(&conn, "messages", "bot_account_id", "TEXT")?;
        ensure_column(&conn, "messages", "media_url", "TEXT")?;
        ensure_column(&conn, "messages", "updated_at", "INTEGER")?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_msg_session_time ON messages(session_id, created_at);",
        )?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_msg_sender_time ON messages(sender_id, created_at);",
        )?;
        conn.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_msg_event_direction
             ON messages(event_key, direction) WHERE event_key IS NOT NULL;",
        )?;

        // 出站消息和宿主发送尝试
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS outbound_messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                action_key      TEXT NOT NULL,
                source_event_key TEXT,
                protocol        TEXT NOT NULL,
                bot_account_id  TEXT,
                session_type    TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                content         TEXT NOT NULL DEFAULT '',
                media_type      TEXT,
                media_url       TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                host_status     TEXT,
                error           TEXT,
                attempt_count   INTEGER NOT NULL DEFAULT 1,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outbound_session_time
                ON outbound_messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_outbound_status_time
                ON outbound_messages(status, created_at);",
        )?;

        // 决策 trace：记录每条消息的可解释评分，支持离线回放和调参。
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS decision_traces (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key       TEXT NOT NULL UNIQUE,
                session_id      TEXT NOT NULL,
                score           REAL NOT NULL,
                threshold       REAL NOT NULL,
                direct          INTEGER NOT NULL,
                outcome         TEXT NOT NULL,
                reason          TEXT NOT NULL,
                signals_json    TEXT NOT NULL,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decision_session_time
                ON decision_traces(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_decision_outcome_time
                ON decision_traces(outcome, created_at);",
        )?;

        // 可恢复的压缩任务状态。
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS compaction_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_key         TEXT NOT NULL UNIQUE,
                cursor_start    INTEGER NOT NULL,
                cursor_end      INTEGER NOT NULL,
                status          TEXT NOT NULL,
                processed_count INTEGER NOT NULL DEFAULT 0,
                error           TEXT,
                started_at      INTEGER NOT NULL,
                finished_at     INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_compaction_status_time
                ON compaction_runs(status, started_at);",
        )?;

        // 长期记忆
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS long_memory (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT,
                subject_id    TEXT,
                content       TEXT NOT NULL,
                kind          TEXT DEFAULT 'fact',
                importance    INTEGER DEFAULT 50,
                is_active     INTEGER DEFAULT 1,
                access_count  INTEGER DEFAULT 0,
                last_access   INTEGER,
                created_at    INTEGER NOT NULL,
                updated_at    INTEGER
            );",
        )?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_mem_active_imp ON long_memory(is_active, importance);",
        )?;

        // 用户画像
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS personas (
                subject_id        TEXT PRIMARY KEY,
                protocol          TEXT,
                nickname          TEXT,
                first_seen        INTEGER,
                last_seen         INTEGER,
                interaction_count INTEGER DEFAULT 0,
                intimacy          INTEGER DEFAULT 0,
                relation          TEXT,
                traits            TEXT,
                preferences       TEXT,
                topics            TEXT,
                notes             TEXT
            );",
        )?;

        // 知识库
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                subject       TEXT,
                content       TEXT NOT NULL,
                category      TEXT,
                source        TEXT,
                confidence    INTEGER DEFAULT 60,
                is_active     INTEGER DEFAULT 1,
                access_count  INTEGER DEFAULT 0,
                created_at    INTEGER NOT NULL,
                updated_at    INTEGER
            );",
        )?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_know_subject ON knowledge(subject, is_active);",
        )?;

        // 表情包
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS stickers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol      TEXT,
                kind          TEXT DEFAULT 'image',
                media_url     TEXT NOT NULL,
                file_hash     TEXT,
                source_user   TEXT,
                source_session TEXT,
                usage_count   INTEGER DEFAULT 0,
                last_used     INTEGER,
                created_at    INTEGER NOT NULL
            );",
        )?;

        // 表情包标签
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sticker_tags (
                sticker_id    INTEGER NOT NULL,
                tag           TEXT NOT NULL,
                weight        INTEGER DEFAULT 1,
                PRIMARY KEY (sticker_id, tag)
            );",
        )?;

        // 表情包关联
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sticker_links (
                sticker_a     INTEGER NOT NULL,
                sticker_b     INTEGER NOT NULL,
                co_count      INTEGER DEFAULT 1,
                updated_at    INTEGER,
                PRIMARY KEY (sticker_a, sticker_b)
            );",
        )?;

        // 反思日志
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS reflection_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_by  TEXT,
                summary       TEXT,
                insights      TEXT,
                created_at    INTEGER NOT NULL
            );",
        )?;

        // 元数据
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );",
        )?;

        // 设置 schema 版本
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '3')",
            [],
        )?;
        conn.execute(
            "UPDATE meta SET value = '3' WHERE key = 'schema_version'",
            [],
        )?;

        Ok(())
    }

    /// 插入一条消息
    pub fn insert_message(
        &self,
        msg: &crate::pipeline::InMessage,
    ) -> Result<bool, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let media_type = msg.media.first().map(|media| media.media_type.as_str());
        let media_url = msg
            .media
            .first()
            .map(|media| redact_url_for_storage(&media.url));
        let changed = conn.execute(
            "INSERT OR IGNORE INTO messages
             (event_key, protocol, bot_account_id, direction, session_type, session_id, sender_id,
              sender_name, message_id, content, raw_json, has_media, media_type,
              media_url, reply_to_id, at_me, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'inbound', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?16)",
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
                msg.safe_raw_json,
                msg.has_media as i32,
                media_type,
                media_url,
                msg.reply_to_id,
                msg.at_me as i32,
                msg.timestamp,
            ],
        )?;
        Ok(changed > 0)
    }

    /// 记录一次发送前的 pending 尝试，并返回审计行 ID。
    pub fn begin_outbound_attempt(
        &self,
        attempt: &OutboundAttempt,
        now: i64,
    ) -> Result<i64, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
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
                attempt.media_url.as_deref().map(redact_url_for_storage),
                now,
            ],
        )?;
        Ok(conn.last_insert_rowid())
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
             WHERE id = ?5",
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
    pub fn insert_decision_trace(
        &self,
        event_key: &str,
        session_id: &str,
        score: f32,
        threshold: f32,
        direct: bool,
        outcome: &str,
        reason: &str,
        signals_json: &str,
        created_at: i64,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO decision_traces
             (event_key, session_id, score, threshold, direct, outcome, reason, signals_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                event_key,
                session_id,
                score,
                threshold,
                direct as i32,
                outcome,
                reason,
                truncate_for_storage(signals_json),
                created_at,
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

    /// 更新或插入用户画像
    pub fn upsert_persona(&self, persona: &crate::memory::Persona) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO personas (subject_id, protocol, nickname, first_seen, last_seen, interaction_count, intimacy, relation, traits, preferences, topics, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(subject_id) DO UPDATE SET
                last_seen = excluded.last_seen,
                interaction_count = excluded.interaction_count,
                intimacy = excluded.intimacy,
                nickname = excluded.nickname,
                traits = excluded.traits,
                preferences = excluded.preferences,
                topics = excluded.topics,
                notes = excluded.notes",
            params![
                persona.subject_id,
                persona.protocol,
                persona.nickname,
                persona.first_seen,
                persona.last_seen,
                persona.interaction_count,
                persona.intimacy,
                persona.relation,
                persona.traits,
                persona.preferences,
                persona.topics,
                persona.notes,
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

fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), rusqlite::Error> {
    let exists = {
        let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut rows = statement.query([])?;
        let mut found = false;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == column {
                found = true;
                break;
            }
        }
        found
    };

    if !exists {
        conn.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {definition}"
        ))?;
    }
    Ok(())
}

fn truncate_for_storage(value: &str) -> String {
    value.chars().take(16_384).collect()
}

/// 签名 URL 可能携带 rkey/token 等短期凭据，数据库只保存去除敏感查询参数的形式。
fn redact_url_for_storage(url: &str) -> String {
    let Some((base, query)) = url.split_once('?') else {
        return truncate_for_storage(url);
    };
    let safe_query = query
        .split('&')
        .filter(|part| {
            let key = part
                .split('=')
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase();
            !(key.contains("token")
                || key.contains("secret")
                || key.contains("rkey")
                || key.contains("signature")
                || key == "sig"
                || key == "auth"
                || key.ends_with("_key"))
        })
        .collect::<Vec<_>>();
    if safe_query.is_empty() {
        truncate_for_storage(base)
    } else {
        truncate_for_storage(&format!("{}?{}", base, safe_query.join("&")))
    }
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
        let id = database
            .begin_outbound_attempt(
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
            .expect("outbound attempt should insert");
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
    fn decision_trace_is_idempotent() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-decision-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = Database::open(path.to_str().expect("temporary path is not UTF-8"))
            .expect("database should open");
        database
            .insert_decision_trace(
                "event-1",
                "group-1",
                61.5,
                60.0,
                false,
                "reply",
                "score_reached",
                r#"{"question":true}"#,
                10,
            )
            .expect("trace should insert");
        database
            .insert_decision_trace(
                "event-1",
                "group-1",
                0.0,
                60.0,
                false,
                "skip",
                "duplicate",
                "{}",
                11,
            )
            .expect("duplicate trace should be ignored");

        let connection = database.conn.lock().expect("database lock should work");
        let row: (i64, String, f64) = connection
            .query_row(
                "SELECT COUNT(*), outcome, score FROM decision_traces WHERE event_key = 'event-1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("trace should exist");
        assert_eq!(row.0, 1);
        assert_eq!(row.1, "reply");
        assert_eq!(row.2, 61.5);

        drop(connection);
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }
}
