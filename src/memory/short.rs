//! 短期记忆：按会话保存最近的用户与机器人消息。
//!
//! 这里使用进程内 LRU，消息原文仍然由 `messages` 表持久化。短期记忆只负责
//! 为下一次 LLM 请求提供低延迟上下文，不把数据库锁带进模型请求。
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use rusqlite::params;
use sha2::{Digest, Sha256};

use crate::db::Database;
use crate::pipeline::InMessage;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextMessage {
    pub event_key: String,
    /// Protocol-native message reference used to resolve QQ quote targets.
    pub message_ref_id: String,
    pub role: String,
    pub content: String,
    pub speaker: String,
    pub timestamp: i64,
    pub is_key: bool,
    /// In-memory media references from this turn. Signed URLs are never
    /// persisted; they are converted to inline vision data before sending.
    pub media: Vec<crate::pipeline::MediaRef>,
}

struct SessionContext {
    session_id: String,
    messages: VecDeque<ContextMessage>,
}

struct RecoveryRoute {
    protocol: String,
    session_type: String,
    session_id: String,
    last_activity: i64,
}

struct RestoredMessage {
    message: ContextMessage,
    source_kind: u8,
    source_id: i64,
}

struct RestoredSession {
    key: String,
    last_activity: i64,
    messages: VecDeque<ContextMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RestoreReport {
    pub sessions: usize,
    pub messages: usize,
    pub inbound_messages: usize,
    pub outbound_messages: usize,
}

static SHORT_CONTEXTS: Mutex<VecDeque<SessionContext>> = Mutex::new(VecDeque::new());

const MAX_SESSIONS: usize = 100;
const MIN_MAX_MESSAGES: usize = 5;
const MAX_MAX_MESSAGES: usize = 200;
const RESTORE_LOOKBACK_MILLIS: i64 = 7 * 24 * 60 * 60 * 1_000;
const RESTORE_ROUTE_SCAN_MULTIPLIER: usize = 16;
/// 推入用户消息。
pub async fn push(msg: &InMessage) {
    push_message(
        &scoped_session_key(&msg.protocol, &msg.session_type, &msg.session_id),
        ContextMessage {
            event_key: msg.event_key.clone(),
            message_ref_id: crate::pipeline::message_reference_id(msg),
            role: "user".to_string(),
            content: context_content(&msg.content, &msg.media, msg.has_media),
            speaker: speaker_label(msg),
            timestamp: msg.timestamp,
            is_key: false,
            media: msg.media.clone(),
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
            message_ref_id: String::new(),
            role: "assistant".to_string(),
            content: content.trim().to_string(),
            speaker: String::new(),
            timestamp,
            is_key: false,
            media: Vec::new(),
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

/// Rebuild the bounded in-process LRU from the persistent message journal.
///
/// Route discovery is bounded by time and row count. Each recovered route is
/// then queried independently with protocol/session-type/session-ID, time, and
/// message-count bounds, so reload cannot accidentally turn into a full-table
/// scan. Only completed or explicitly retained `record_only` inbound records
/// and host-accepted text replies participate in conversational history.
pub(crate) fn restore_from_database(
    database: &Database,
    configured_limit: usize,
    now: i64,
) -> Result<RestoreReport, String> {
    let per_session_limit = configured_limit.clamp(MIN_MAX_MESSAGES, MAX_MAX_MESSAGES);
    let cutoff = now.saturating_sub(RESTORE_LOOKBACK_MILLIS);
    let route_scan_limit = (MAX_SESSIONS * RESTORE_ROUTE_SCAN_MULTIPLIER) as i64;
    let mut routes = BTreeMap::new();
    let mut restored_sessions = Vec::new();

    {
        let connection = database
            .conn
            .lock()
            .map_err(|_| "database lock failed while restoring short context".to_string())?;

        {
            let mut statement = connection
                .prepare(
                    "SELECT protocol, session_type, session_id, created_at
                     FROM messages
                     WHERE direction = 'inbound'
                       AND processing_status IN ('processed', 'record_only')
                       AND created_at >= ?1
                     ORDER BY created_at DESC, id DESC
                     LIMIT ?2",
                )
                .map_err(|error| error.to_string())?;
            let rows = statement
                .query_map(params![cutoff, route_scan_limit], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })
                .map_err(|error| error.to_string())?;
            for row in rows {
                let (protocol, session_type, session_id, last_activity) =
                    row.map_err(|error| error.to_string())?;
                record_recovery_route(
                    &mut routes,
                    protocol,
                    session_type,
                    session_id,
                    last_activity,
                );
            }
        }

        {
            let mut statement = connection
                .prepare(
                    "SELECT protocol, session_type, session_id, updated_at
                     FROM outbound_messages
                     WHERE status = 'accepted'
                       AND TRIM(content) <> ''
                       AND updated_at >= ?1
                     ORDER BY updated_at DESC, id DESC
                     LIMIT ?2",
                )
                .map_err(|error| error.to_string())?;
            let rows = statement
                .query_map(params![cutoff, route_scan_limit], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                })
                .map_err(|error| error.to_string())?;
            for row in rows {
                let (protocol, session_type, session_id, last_activity) =
                    row.map_err(|error| error.to_string())?;
                record_recovery_route(
                    &mut routes,
                    protocol,
                    session_type,
                    session_id,
                    last_activity,
                );
            }
        }

        let mut selected_routes = routes.into_values().collect::<Vec<_>>();
        selected_routes.sort_by(|left, right| {
            right.last_activity.cmp(&left.last_activity).then_with(|| {
                scoped_session_key(&left.protocol, &left.session_type, &left.session_id).cmp(
                    &scoped_session_key(&right.protocol, &right.session_type, &right.session_id),
                )
            })
        });
        selected_routes.truncate(MAX_SESSIONS);

        let mut inbound_statement = connection
            .prepare(
                "SELECT id, COALESCE(event_key, ''),
                        COALESCE(message_ref_id, ''), sender_id,
                        COALESCE(sender_name, ''), content, has_media,
                        COALESCE(media_type, ''), COALESCE(media_url, ''),
                        COALESCE(media_requires_cache, 1), created_at
                 FROM messages
                 WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
                   AND direction = 'inbound'
                   AND processing_status IN ('processed', 'record_only')
                   AND created_at >= ?4
                 ORDER BY created_at DESC, id DESC
                 LIMIT ?5",
            )
            .map_err(|error| error.to_string())?;
        let mut outbound_statement = connection
            .prepare(
                "SELECT id, content, updated_at
                 FROM outbound_messages
                 WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
                   AND status = 'accepted'
                   AND TRIM(content) <> ''
                   AND updated_at >= ?4
                 ORDER BY updated_at DESC, id DESC
                 LIMIT ?5",
            )
            .map_err(|error| error.to_string())?;

        for route in selected_routes {
            let mut messages = Vec::new();
            let inbound_rows = inbound_statement
                .query_map(
                    params![
                        &route.protocol,
                        &route.session_type,
                        &route.session_id,
                        cutoff,
                        per_session_limit as i64,
                    ],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                            row.get::<_, String>(4)?,
                            row.get::<_, String>(5)?,
                            row.get::<_, i64>(6)? != 0,
                            row.get::<_, String>(7)?,
                            row.get::<_, String>(8)?,
                            row.get::<_, i64>(9)? != 0,
                            row.get::<_, i64>(10)?,
                        ))
                    },
                )
                .map_err(|error| error.to_string())?;
            for row in inbound_rows {
                let (
                    source_id,
                    event_key,
                    message_ref_id,
                    sender_id,
                    sender_name,
                    content,
                    has_media,
                    media_type,
                    media_url,
                    media_requires_cache,
                    timestamp,
                ) = row.map_err(|error| error.to_string())?;
                let content = context_content(&content, &[], has_media);
                if content.trim().is_empty() {
                    continue;
                }
                messages.push(RestoredMessage {
                    message: ContextMessage {
                        event_key: if event_key.is_empty() {
                            format!("journal:{source_id}")
                        } else {
                            event_key
                        },
                        message_ref_id,
                        role: "user".to_string(),
                        content,
                        speaker: speaker_label_for(&route.protocol, &sender_id, &sender_name),
                        timestamp,
                        is_key: false,
                        media: restored_media(media_type, media_url, media_requires_cache),
                    },
                    source_kind: 0,
                    source_id,
                });
            }

            let outbound_rows = outbound_statement
                .query_map(
                    params![
                        &route.protocol,
                        &route.session_type,
                        &route.session_id,
                        cutoff,
                        per_session_limit as i64,
                    ],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, i64>(2)?,
                        ))
                    },
                )
                .map_err(|error| error.to_string())?;
            for row in outbound_rows {
                let (source_id, content, timestamp) = row.map_err(|error| error.to_string())?;
                let content = content.trim();
                if content.is_empty() {
                    continue;
                }
                messages.push(RestoredMessage {
                    message: ContextMessage {
                        event_key: String::new(),
                        role: "assistant".to_string(),
                        content: content.to_string(),
                        speaker: String::new(),
                        message_ref_id: String::new(),
                        timestamp,
                        is_key: false,
                        media: Vec::new(),
                    },
                    source_kind: 1,
                    source_id,
                });
            }

            messages.sort_by(|left, right| {
                left.message
                    .timestamp
                    .cmp(&right.message.timestamp)
                    .then_with(|| left.source_kind.cmp(&right.source_kind))
                    .then_with(|| left.source_id.cmp(&right.source_id))
            });
            if messages.len() > per_session_limit {
                let first_retained = messages.len() - per_session_limit;
                messages.drain(..first_retained);
            }
            let Some(last_message) = messages.last() else {
                continue;
            };
            let key = scoped_session_key(&route.protocol, &route.session_type, &route.session_id);
            restored_sessions.push(RestoredSession {
                key,
                last_activity: last_message.message.timestamp,
                messages: messages
                    .into_iter()
                    .map(|item| item.message)
                    .collect::<VecDeque<_>>(),
            });
        }
    }

    restored_sessions.sort_by(|left, right| {
        left.last_activity
            .cmp(&right.last_activity)
            .then_with(|| left.key.cmp(&right.key))
    });
    let report = RestoreReport {
        sessions: restored_sessions.len(),
        messages: restored_sessions
            .iter()
            .map(|session| session.messages.len())
            .sum(),
        inbound_messages: restored_sessions
            .iter()
            .flat_map(|session| session.messages.iter())
            .filter(|message| message.role == "user")
            .count(),
        outbound_messages: restored_sessions
            .iter()
            .flat_map(|session| session.messages.iter())
            .filter(|message| message.role == "assistant")
            .count(),
    };

    let mut contexts = SHORT_CONTEXTS
        .lock()
        .map_err(|_| "short-context lock failed while restoring".to_string())?;
    contexts.clear();
    for session in restored_sessions {
        contexts.push_back(SessionContext {
            session_id: session.key,
            messages: session.messages,
        });
    }
    Ok(report)
}

fn record_recovery_route(
    routes: &mut BTreeMap<String, RecoveryRoute>,
    protocol: String,
    session_type: String,
    session_id: String,
    last_activity: i64,
) {
    let key = scoped_session_key(&protocol, &session_type, &session_id);
    match routes.get_mut(&key) {
        Some(existing) => existing.last_activity = existing.last_activity.max(last_activity),
        None => {
            routes.insert(
                key,
                RecoveryRoute {
                    protocol,
                    session_type,
                    session_id,
                    last_activity,
                },
            );
        }
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
    speaker_label_for(&msg.protocol, &msg.sender_id, &msg.sender_name)
}

pub(crate) fn speaker_label_for(protocol: &str, sender_id: &str, sender_name: &str) -> String {
    let nickname = sender_name
        .chars()
        .filter(|character| !character.is_control())
        .take(24)
        .collect::<String>();
    let mut digest = Sha256::new();
    digest.update(protocol.as_bytes());
    digest.update([0]);
    digest.update(sender_id.as_bytes());
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

fn restored_media(
    media_type: String,
    media_url: String,
    media_requires_cache: bool,
) -> Vec<crate::pipeline::MediaRef> {
    if media_url.is_empty() || media_url == "[invalid-media-url]" {
        return Vec::new();
    }
    vec![crate::pipeline::MediaRef {
        url: media_url,
        media_type: if media_type.is_empty() {
            "image".to_string()
        } else {
            media_type
        },
        requires_cache: media_requires_cache,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{OutboundAttempt, OutboundClaim};

    static TEST_CONTEXT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

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
        let _guard = TEST_CONTEXT_LOCK.lock().await;
        clear();
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
        clear();
    }

    #[tokio::test]
    async fn context_budget_keeps_latest_message() {
        let _guard = TEST_CONTEXT_LOCK.lock().await;
        clear();
        let session = format!("budget-test-{}", std::process::id());
        push(&message(&session, "old message", 1)).await;
        push(&message(&session, "latest", 2)).await;

        let context = get_context("onebot11", "group", &session, 2);
        assert_eq!(
            context.last().map(|item| item.content.as_str()),
            Some("latest")
        );
        assert!(context.len() <= 2);
        clear();
    }

    #[tokio::test]
    async fn context_is_isolated_by_protocol_and_session_type() {
        let _guard = TEST_CONTEXT_LOCK.lock().await;
        clear();
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
        clear();
    }

    #[test]
    fn restored_temporary_media_keeps_cache_lookup_identity() {
        let media = restored_media(
            "image/jpeg".to_string(),
            "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=abc".to_string(),
            true,
        );
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].media_type, "image/jpeg");
        assert!(media[0].url.contains("fileid=abc"));
        assert!(media[0].requires_cache);

        let public_media = restored_media(
            "image/png".to_string(),
            "https://example.test/image.png".to_string(),
            false,
        );
        assert_eq!(public_media[0].url, "https://example.test/image.png");
        assert!(!public_media[0].requires_cache);
    }

    #[allow(clippy::too_many_arguments)] // Test fixture mirrors the journal's route/state fields.
    fn persist_inbound(
        database: &Database,
        protocol: &str,
        session_type: &str,
        session_id: &str,
        event_key: &str,
        content: &str,
        has_media: bool,
        status: &str,
        timestamp: i64,
    ) {
        let message = InMessage {
            event_key: event_key.to_string(),
            protocol: protocol.to_string(),
            bot_account_id: String::new(),
            session_type: session_type.to_string(),
            session_id: session_id.to_string(),
            sender_id: "member-one".to_string(),
            sender_name: "恢复用户".to_string(),
            message_id: event_key.to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media,
            at_me: false,
            timestamp,
            safe_raw_json: "{}".to_string(),
        };
        database.insert_message(&message).unwrap();
        if status != "recorded" {
            database
                .set_message_processing_status(event_key, status, None, timestamp)
                .unwrap();
        }
    }

    #[allow(clippy::too_many_arguments)] // Test fixture mirrors the outbound audit fields.
    fn persist_outbound(
        database: &Database,
        protocol: &str,
        session_type: &str,
        session_id: &str,
        action_key: &str,
        content: &str,
        status: &str,
        timestamp: i64,
    ) {
        let id = match database
            .claim_outbound_attempt(
                &OutboundAttempt {
                    action_key: action_key.to_string(),
                    source_event_key: None,
                    protocol: protocol.to_string(),
                    bot_account_id: "bot-one".to_string(),
                    session_type: session_type.to_string(),
                    session_id: session_id.to_string(),
                    content: content.to_string(),
                    media_type: None,
                    media_url: None,
                },
                timestamp,
            )
            .unwrap()
        {
            OutboundClaim::Claimed(id) => id,
            other => panic!("unexpected outbound claim: {other:?}"),
        };
        database
            .finish_outbound_attempt(id, status, None, None, timestamp)
            .unwrap();
    }

    #[test]
    fn recovery_is_bounded_ordered_and_isolated_by_route() {
        let _guard = TEST_CONTEXT_LOCK.blocking_lock();
        clear();
        let database = Database::open(":memory:").unwrap();
        let now = RESTORE_LOOKBACK_MILLIS + 20_000;
        let cutoff = now - RESTORE_LOOKBACK_MILLIS;
        let session = "shared-session";

        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "expired",
            "expired record",
            false,
            "processed",
            cutoff - 1,
        );
        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "trimmed",
            "trimmed oldest",
            false,
            "processed",
            now - 600,
        );
        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "kept",
            "kept inbound",
            false,
            "processed",
            now - 500,
        );
        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "media",
            "",
            true,
            "record_only",
            now - 400,
        );
        persist_outbound(
            &database,
            "onebot11",
            "group",
            session,
            "accepted-first",
            "accepted first",
            "accepted",
            now - 300,
        );
        persist_outbound(
            &database,
            "onebot11",
            "group",
            session,
            "accepted-second",
            "accepted second",
            "accepted",
            now - 200,
        );
        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "latest",
            "latest inbound",
            false,
            "processed",
            now - 100,
        );
        persist_outbound(
            &database,
            "onebot11",
            "group",
            session,
            "rejected",
            "rejected outbound",
            "rejected",
            now - 50,
        );
        persist_inbound(
            &database,
            "onebot11",
            "group",
            session,
            "recorded",
            "not completed",
            false,
            "recorded",
            now - 25,
        );
        persist_inbound(
            &database,
            "qq-official",
            "group",
            session,
            "official",
            "official only",
            false,
            "processed",
            now - 90,
        );
        persist_inbound(
            &database,
            "onebot11",
            "private",
            session,
            "private",
            "private only",
            false,
            "processed",
            now - 80,
        );

        let report = restore_from_database(&database, 5, now).unwrap();
        assert_eq!(
            report,
            RestoreReport {
                sessions: 3,
                messages: 7,
                inbound_messages: 5,
                outbound_messages: 2,
            }
        );

        let group_context = get_context("onebot11", "group", session, 200);
        assert_eq!(
            group_context
                .iter()
                .map(|message| (message.role.as_str(), message.content.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("user", "kept inbound"),
                ("user", "[收到媒体附件]"),
                ("assistant", "accepted first"),
                ("assistant", "accepted second"),
                ("user", "latest inbound"),
            ]
        );
        assert!(!group_context[0].speaker.contains("member-one"));
        assert_eq!(
            get_context("onebot11", "group", session, 1)
                .last()
                .map(|message| message.content.as_str()),
            Some("latest inbound")
        );
        assert_eq!(
            get_context("qq-official", "group", session, 200)[0].content,
            "official only"
        );
        assert_eq!(
            get_context("onebot11", "private", session, 200)[0].content,
            "private only"
        );
        assert!(
            get_context("onebot11", "group", session, 200)
                .iter()
                .all(|message| !message.content.contains("trimmed")
                    && !message.content.contains("expired")
                    && !message.content.contains("not completed")
                    && !message.content.contains("rejected"))
        );
        clear();
    }
}
