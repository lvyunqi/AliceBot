//! 记忆系统
//!
//! 三层记忆：短期（会话上下文） + 长期（重要事实） + 用户画像（识人）
//! 另含知识库提取和压缩反思。

mod candidates;
pub mod compact;
mod context;
mod knowledge;
pub mod long;
pub mod persona;
pub(crate) mod reflection;
pub(crate) mod retention;
pub(crate) mod search;
pub mod short;

pub(crate) use context::{ContextInput, assemble as assemble_prompt_context};
pub use persona::*;

use crate::pipeline::InMessage;

/// 观察用户（每条消息后更新画像）
pub async fn observe_user(msg: &InMessage) {
    persona::observe(msg).await;
    candidates::observe(msg).await;
    knowledge::observe(msg).await;
}

/// 推入短期上下文
pub async fn push_short_context(msg: &InMessage) {
    short::push(msg).await;
}

/// 获取用于 LLM 的短期上下文。
pub fn short_context(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    max_tokens: u32,
) -> Vec<short::ContextMessage> {
    short::get_context(protocol, session_type, session_id, max_tokens)
}

/// 在进程重启或插件重载后恢复有界短期上下文。
pub(crate) fn restore_short_context() -> Result<short::RestoreReport, String> {
    let database =
        crate::pipeline::try_db().ok_or_else(|| "database is not initialized".to_string())?;
    short::restore_from_database(
        &database,
        crate::pipeline::current_config().memories.short_size,
        chrono::Utc::now().timestamp_millis(),
    )
}

/// 推入已经成功发送的机器人回复。
pub async fn push_assistant_context(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    content: &str,
    timestamp: i64,
) {
    short::push_assistant(protocol, session_type, session_id, content, timestamp).await;
}

/// 动态插件实例停止时清理进程内易失记忆。
pub fn clear_runtime_state() {
    short::clear();
}

/// 一次请求者范围内的遗忘结果，不包含任何正文。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ForgetReport {
    pub long_memories_changed: usize,
    pub messages_redacted: usize,
}

/// 按协议和请求者主体遗忘自己的长期记忆及对应消息正文。
pub async fn forget_by_keyword(protocol: &str, subject_id: &str, keyword: &str) -> String {
    let keyword = keyword.trim();
    if keyword.is_empty() {
        return "我试着忘记这件事…嗯，好像本来也没记住什么 😅".to_string();
    }

    let protocol = protocol.trim();
    let subject_id = subject_id.trim();
    if protocol.is_empty() || subject_id.is_empty() {
        return "我还不能确认这条记忆属于谁，暂时不会删除".to_string();
    }
    let Some(database) = crate::pipeline::try_db() else {
        return "记忆数据库还没有准备好".to_string();
    };
    let now = chrono::Utc::now().timestamp_millis();
    match forget_by_keyword_in(&database, protocol, subject_id, keyword, now) {
        Ok(report) => {
            // 遗忘后清空进程内上下文，避免本轮旧正文继续进入模型请求。
            clear_runtime_state();
            format!(
                "已将 {} 条你的长期记忆标记为遗忘，并清理 {} 条消息正文",
                report.long_memories_changed, report.messages_redacted
            )
        }
        Err(_) => "记忆数据库暂时不可用".to_string(),
    }
}

fn forget_by_keyword_in(
    database: &crate::db::Database,
    protocol: &str,
    subject_id: &str,
    keyword: &str,
    now: i64,
) -> Result<ForgetReport, String> {
    let pattern = format!(
        "%{}%",
        keyword
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    );
    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "memory database lock failed".to_string())?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let long_memories_changed = transaction
        .execute(
            "UPDATE long_memory
             SET is_active = 0, status = 'forgotten', content = '[已遗忘]',
                 archived_at = ?1, updated_at = ?1
             WHERE protocol = ?2 AND subject_id = ?3
               AND scope IN ('user', 'user_session')
               AND status <> 'forgotten'
               AND content LIKE ?4 ESCAPE '\\'",
            rusqlite::params![now, protocol, subject_id, pattern],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "DELETE FROM memory_sources
             WHERE memory_id IN (
                 SELECT id FROM long_memory
                 WHERE protocol = ?1 AND subject_id = ?2
                   AND status = 'forgotten' AND archived_at = ?3
             )",
            rusqlite::params![protocol, subject_id, now],
        )
        .map_err(|error| error.to_string())?;
    let messages_redacted = transaction
        .execute(
            "UPDATE messages
             SET content = '', raw_json = NULL, media_url = NULL,
                 updated_at = MAX(COALESCE(updated_at, ?1), ?1)
             WHERE protocol = ?2 AND sender_id = ?3
               AND content LIKE ?4 ESCAPE '\\'",
            rusqlite::params![now, protocol, subject_id, pattern],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(ForgetReport {
        long_memories_changed,
        messages_redacted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_memory(
        database: &crate::db::Database,
        key: &str,
        protocol: &str,
        subject_id: &str,
        content: &str,
    ) {
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO long_memory
                 (normalized_key, protocol, session_type, scope, session_id, subject_id,
                  kind, content, importance, confidence, privacy, status, version,
                  is_active, created_at, updated_at)
                 VALUES (?1, ?2, 'private', 'user_session', 'private-1', ?3, 'fact',
                         ?4, 80, 85, 'normal', 'active', 1, 1, 1, 1)",
                rusqlite::params![key, protocol, subject_id, content],
            )
            .unwrap();
    }

    #[test]
    fn forget_is_subject_scoped_literal_and_preserves_shared_knowledge() {
        let database = crate::db::Database::open(":memory:").unwrap();
        insert_memory(&database, "target", "onebot11", "user-1", "100%_计划");
        insert_memory(&database, "other-user", "onebot11", "user-2", "100%_计划");
        insert_memory(
            &database,
            "other-protocol",
            "qq-official",
            "user-1",
            "100%_计划",
        );
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO knowledge
                 (normalized_key, subject, content, category, source, confidence, is_active,
                  access_count, created_at, updated_at, scope, status, version, protocol,
                  session_type, session_id)
                 VALUES ('shared', '群', '100%_计划', 'group_rule', 'message',
                         85, 1, 0, 1, 1, 'session', 'active', 1,
                         'onebot11', 'group', 'shared-group')",
                [],
            )
            .unwrap();
        let target_id = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT id FROM long_memory WHERE normalized_key = 'target'",
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
                 VALUES (?1, 'message', 'forget-message', 1, 1)",
                rusqlite::params![target_id],
            )
            .unwrap();
        let message = crate::pipeline::InMessage {
            event_key: "forget-message".to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "private".to_string(),
            session_id: "user-1".to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "用户".to_string(),
            message_id: "forget-message".to_string(),
            reply_to_id: String::new(),
            content: "我的 100%_计划".to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp: 1,
            safe_raw_json: r#"{"secret":"should disappear"}"#.to_string(),
        };
        database.insert_message(&message).unwrap();

        let report =
            forget_by_keyword_in(&database, "onebot11", "user-1", "100%_计划", 10).unwrap();
        assert_eq!(
            report,
            ForgetReport {
                long_memories_changed: 1,
                messages_redacted: 1,
            }
        );

        let connection = database.conn.lock().unwrap();
        let target: (String, i32, String) = connection
            .query_row(
                "SELECT status, is_active, content FROM long_memory WHERE normalized_key = 'target'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(target, ("forgotten".to_string(), 0, "[已遗忘]".to_string()));
        let source_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?1",
                rusqlite::params![target_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(source_count, 0);
        let other_user: String = connection
            .query_row(
                "SELECT status FROM long_memory WHERE normalized_key = 'other-user'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(other_user, "active");
        let other_protocol: String = connection
            .query_row(
                "SELECT status FROM long_memory WHERE normalized_key = 'other-protocol'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(other_protocol, "active");
        let shared_knowledge: (String, i32) = connection
            .query_row(
                "SELECT status, is_active FROM knowledge WHERE normalized_key = 'shared'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(shared_knowledge, ("active".to_string(), 1));
        let message_row: (String, Option<String>, Option<String>) = connection
            .query_row(
                "SELECT content, raw_json, media_url FROM messages WHERE event_key = 'forget-message'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(message_row, (String::new(), None, None));
    }
}
