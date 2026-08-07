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

/// Clear volatile memory when the dynamic plugin instance is stopped.
pub fn clear_runtime_state() {
    short::clear();
}

/// 按关键词遗忘
pub async fn forget_by_keyword(keyword: &str) -> String {
    let keyword = keyword.trim();
    if !keyword.is_empty() {
        let Some(database) = crate::pipeline::try_db() else {
            return "记忆数据库还没有准备好".to_string();
        };
        let pattern = format!(
            "%{}%",
            keyword
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_")
        );
        let Ok(connection) = database.conn.lock() else {
            return "记忆数据库暂时不可用".to_string();
        };
        let now = chrono::Utc::now().timestamp_millis();
        let long_changed = connection
            .execute(
                "UPDATE long_memory
                 SET is_active = 0, status = 'forgotten', archived_at = ?1, updated_at = ?1
                 WHERE status <> 'forgotten' AND content LIKE ?2 ESCAPE '\\'",
                rusqlite::params![now, pattern],
            )
            .unwrap_or(0);
        let knowledge_changed = connection
            .execute(
                "UPDATE knowledge
                 SET is_active = 0, status = 'forgotten', updated_at = ?1
                 WHERE status <> 'forgotten' AND content LIKE ?2 ESCAPE '\\'",
                rusqlite::params![now, pattern],
            )
            .unwrap_or(0);
        return format!(
            "已将 {} 条长期记忆和 {} 条知识标记为遗忘",
            long_changed, knowledge_changed
        );
    }
    "我试着忘记这件事…嗯，好像本来也没记住什么 😅".to_string()
}
