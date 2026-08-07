//! 短期记忆：按会话保存最近的用户与机器人消息。
//!
//! 这里使用进程内 LRU，消息原文仍然由 `messages` 表持久化。短期记忆只负责
//! 为下一次 LLM 请求提供低延迟上下文，不把数据库锁带进模型请求。
use std::collections::VecDeque;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use crate::pipeline::InMessage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMessage {
    pub event_key: String,
    pub role: String,
    pub content: String,
    pub speaker: String,
    pub timestamp: i64,
    pub is_key: bool,
}

struct SessionContext {
    session_id: String,
    messages: VecDeque<ContextMessage>,
}

static SHORT_CONTEXTS: Mutex<VecDeque<SessionContext>> = Mutex::new(VecDeque::new());

const MAX_SESSIONS: usize = 100;
const MIN_MAX_MESSAGES: usize = 5;
const MAX_MAX_MESSAGES: usize = 200;

/// 推入用户消息。
pub async fn push(msg: &InMessage) {
    push_message(
        &scoped_session_key(&msg.protocol, &msg.session_type, &msg.session_id),
        ContextMessage {
            event_key: msg.event_key.clone(),
            role: "user".to_string(),
            content: context_content(&msg.content, &msg.media, msg.has_media),
            speaker: speaker_label(msg),
            timestamp: msg.timestamp,
            is_key: false,
        },
    );
}

/// 推入已经成功发送的机器人消息。
pub async fn push_assistant(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    content: &str,
    timestamp: i64,
) {
    if content.trim().is_empty() {
        return;
    }
    push_message(
        &scoped_session_key(protocol, session_type, session_id),
        ContextMessage {
            event_key: String::new(),
            role: "assistant".to_string(),
            content: content.trim().to_string(),
            speaker: String::new(),
            timestamp,
            is_key: false,
        },
    );
}

fn push_message(session_id: &str, message: ContextMessage) {
    let limit = crate::pipeline::current_config()
        .memories
        .short_size
        .clamp(MIN_MAX_MESSAGES, MAX_MAX_MESSAGES);
    let Ok(mut contexts) = SHORT_CONTEXTS.lock() else {
        return;
    };

    if let Some(position) = contexts
        .iter()
        .position(|item| item.session_id == session_id)
    {
        let mut session = contexts
            .remove(position)
            .expect("position from VecDeque must remain valid");
        session.messages.push_back(message);
        while session.messages.len() > limit {
            session.messages.pop_front();
        }
        contexts.push_back(session);
    } else {
        let mut messages = VecDeque::new();
        messages.push_back(message);
        contexts.push_back(SessionContext {
            session_id: session_id.to_string(),
            messages,
        });
        while contexts.len() > MAX_SESSIONS {
            contexts.pop_front();
        }
    }
}

/// 获取按时间正序排列的上下文，并限制近似 token 数。
///
/// 近似算法按 Unicode 字符计数，ASCII 文本每 4 个字符约 1 token，中文等
/// 非 ASCII 字符按 1 个字符约 1 token 估算。它不依赖具体模型 tokenizer，
/// 但能稳定地防止上下文无限增长。
pub fn get_context(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    max_tokens: u32,
) -> Vec<ContextMessage> {
    let configured_limit = crate::pipeline::current_config()
        .memories
        .short_size
        .clamp(MIN_MAX_MESSAGES, MAX_MAX_MESSAGES);
    let max_messages = configured_limit.max(1);
    let max_tokens = max_tokens.max(1) as usize;
    let Ok(contexts) = SHORT_CONTEXTS.lock() else {
        return Vec::new();
    };
    let key = scoped_session_key(protocol, session_type, session_id);
    let Some(session) = contexts.iter().find(|item| item.session_id == key) else {
        return Vec::new();
    };

    let mut selected = VecDeque::new();
    let mut used_tokens = 0;
    for message in session.messages.iter().rev().take(max_messages) {
        let message_tokens = estimate_tokens(&message.content).max(1);
        if !selected.is_empty() && used_tokens + message_tokens > max_tokens {
            break;
        }
        selected.push_front(message.clone());
        used_tokens += message_tokens;
        if used_tokens >= max_tokens {
            break;
        }
    }
    selected.into_iter().collect()
}

/// Drop process-local context during plugin shutdown/reload.
pub fn clear() {
    if let Ok(mut contexts) = SHORT_CONTEXTS.lock() {
        contexts.clear();
    }
}

pub fn estimate_tokens(text: &str) -> usize {
    let ascii = text
        .chars()
        .filter(|character| character.is_ascii())
        .count();
    let non_ascii = text.chars().count().saturating_sub(ascii);
    ascii.div_ceil(4) + non_ascii
}

/// Build a collision-resistant in-process key for one protocol/session route.
///
/// The raw IDs are never sent to the model; the length prefixes only make the
/// internal key unambiguous even when an ID contains the separator character.
pub fn scoped_session_key(protocol: &str, session_type: &str, session_id: &str) -> String {
    format!(
        "{}:{}|{}:{}|{}:{}",
        protocol.len(),
        protocol,
        session_type.len(),
        session_type,
        session_id.len(),
        session_id
    )
}

/// Return a stable, privacy-preserving speaker label for prompt attribution.
/// A short digest keeps two users with the same nickname distinguishable
/// without exposing platform-specific IDs to the LLM.
pub fn speaker_label(msg: &InMessage) -> String {
    let nickname = msg
        .sender_name
        .chars()
        .filter(|character| !character.is_control())
        .take(24)
        .collect::<String>();
    let mut digest = Sha256::new();
    digest.update(msg.protocol.as_bytes());
    digest.update([0]);
    digest.update(msg.sender_id.as_bytes());
    let digest = digest.finalize();
    let suffix = digest[..3]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if nickname.trim().is_empty() {
        format!("成员#{suffix}")
    } else {
        format!("{}#{suffix}", nickname.trim())
    }
}

fn context_content(content: &str, media: &[crate::pipeline::MediaRef], has_media: bool) -> String {
    if !content.trim().is_empty() {
        return content.trim().to_string();
    }
    if has_media && !media.is_empty() {
        return format!("[收到{}个媒体附件]", media.len());
    }
    if has_media {
        return "[收到媒体附件]".to_string();
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(session_id: &str, content: &str, timestamp: i64) -> InMessage {
        InMessage {
            event_key: format!("test:{session_id}:{timestamp}"),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: session_id.to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "user".to_string(),
            message_id: timestamp.to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn context_is_chronological_and_contains_assistant_messages() {
        let session = format!("short-test-{}", std::process::id());
        push(&message(&session, "first", 1)).await;
        push_assistant("onebot11", "group", &session, "reply", 2).await;
        push(&message(&session, "second", 3)).await;

        let context = get_context("onebot11", "group", &session, 100);
        assert_eq!(
            context
                .iter()
                .map(|item| item.role.as_str())
                .collect::<Vec<_>>(),
            ["user", "assistant", "user"]
        );
        assert_eq!(context[0].content, "first");
        assert_eq!(context[1].content, "reply");
        assert_eq!(context[2].content, "second");
    }

    #[tokio::test]
    async fn context_budget_keeps_latest_message() {
        let session = format!("budget-test-{}", std::process::id());
        push(&message(&session, "old message", 1)).await;
        push(&message(&session, "latest", 2)).await;

        let context = get_context("onebot11", "group", &session, 2);
        assert_eq!(
            context.last().map(|item| item.content.as_str()),
            Some("latest")
        );
        assert!(context.len() <= 2);
    }

    #[tokio::test]
    async fn context_is_isolated_by_protocol_and_session_type() {
        let session = format!("route-test-{}", std::process::id());
        let mut onebot = message(&session, "onebot", 1);
        let mut official = message(&session, "official", 2);
        official.protocol = "qq-official".to_string();
        let mut private = message(&session, "private", 3);
        private.session_type = "private".to_string();

        push(&onebot).await;
        push(&official).await;
        push(&private).await;

        assert_eq!(
            get_context("onebot11", "group", &session, 100)[0].content,
            "onebot"
        );
        assert_eq!(
            get_context("qq-official", "group", &session, 100)[0].content,
            "official"
        );
        assert_eq!(
            get_context("onebot11", "private", &session, 100)[0].content,
            "private"
        );

        onebot.sender_name = "same name".to_string();
        official.sender_name = "same name".to_string();
        assert_ne!(speaker_label(&onebot), speaker_label(&official));
        assert!(!speaker_label(&onebot).contains(&onebot.sender_id));
    }
}
