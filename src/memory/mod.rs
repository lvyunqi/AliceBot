//! 记忆系统
//!
//! 三层记忆：短期（会话上下文） + 长期（重要事实） + 用户画像（识人）
//! 另含知识库提取和压缩反思。

pub mod long;
pub mod persona;
pub mod short;

pub use persona::*;

use crate::pipeline::InMessage;

/// 观察用户（每条消息后更新画像）
pub async fn observe_user(msg: &InMessage) {
    // 更新用户画像
    persona::observe(msg).await;
}

/// 推入短期上下文
pub async fn push_short_context(msg: &InMessage) {
    short::push(msg).await;
}

/// 获取用于 LLM 的短期上下文。
pub fn short_context(session_id: &str, max_tokens: u32) -> Vec<short::ContextMessage> {
    short::get_context(session_id, max_tokens)
}

/// 推入已经成功发送的机器人回复。
pub async fn push_assistant_context(session_id: &str, content: &str, timestamp: i64) {
    short::push_assistant(session_id, content, timestamp).await;
}

/// 按关键词遗忘
pub async fn forget_by_keyword(keyword: &str) -> String {
    // TODO: 在 long_memory / knowledge 中查找并标记为非活跃
    let _ = keyword;
    "我试着忘记这件事…嗯，好像本来也没记住什么 😅".to_string()
}
