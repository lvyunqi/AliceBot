//! 宿主发送封装与出站审计。
//!
//! 动态插件只把请求交给 QimenBot Host API，不在插件内实现协议上传。每次
//! 尝试都会先写入 pending，随后更新为 accepted 或具体失败状态。
use abi_stable_host_api::{BotApi, SendEnqueueStatus};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::db::{OutboundAttempt, OutboundClaim};

static NEXT_ACTION_ID: AtomicU64 = AtomicU64::new(1);

/// 保留旧的文本发送入口，适合没有入站事件关联的主动发送。
#[allow(dead_code)]
pub async fn send_text(account_id: &str, session_id: &str, session_type: &str, text: &str) -> bool {
    send_text_for_event(account_id, "unknown", session_id, session_type, text, None).await
}

/// 发送关联入站事件的文本消息。
pub async fn send_text_for_event(
    account_id: &str,
    protocol: &str,
    session_id: &str,
    session_type: &str,
    text: &str,
    source_event_key: Option<&str>,
) -> bool {
    let audit_id = match begin_attempt(
        "text",
        source_event_key,
        protocol,
        account_id,
        session_type,
        session_id,
        text,
        None,
        None,
    ) {
        Some(AttemptStart::Claimed(id)) => Some(id),
        Some(AttemptStart::AlreadyHandled) => return false,
        None => return false,
    };

    if account_id.trim().is_empty() {
        finish_attempt(audit_id, "invalid", None, Some("missing account_id"));
        log::warn!(
            "[AliceBot] 未配置发送 account_id，跳过发送 session_type={}, session_id={}",
            session_type,
            session_id
        );
        return false;
    }
    if text.trim().is_empty() {
        finish_attempt(audit_id, "invalid", None, Some("empty content"));
        return false;
    }

    let bot = BotApi::for_account(account_id);
    let status = match session_type {
        "private" | "dms" => bot.send_private_msg(session_id, text),
        "group" => bot.send_group_msg(session_id, text),
        "channel" => {
            let Some((guild_id, channel_id)) = session_id.split_once(':') else {
                finish_attempt(
                    audit_id,
                    "invalid",
                    None,
                    Some("channel session_id must be guild:channel"),
                );
                return false;
            };
            bot.send_guild_channel_msg(guild_id, channel_id, text)
        }
        other => {
            finish_attempt(audit_id, "invalid", None, Some("unsupported session type"));
            log::warn!("[AliceBot] 不支持的发送会话类型: {other}");
            return false;
        }
    };

    let host_status = format!("{status:?}");
    let accepted = matches!(status, SendEnqueueStatus::Accepted);
    finish_attempt(
        audit_id,
        if accepted { "accepted" } else { "rejected" },
        Some(&host_status),
        (!accepted).then_some("host rejected enqueue"),
    );
    log::trace!(
        "[AliceBot] 文本发送结果 status={}, session_type={}, session_id={}",
        host_status,
        session_type,
        session_id
    );
    accepted
}

/// 发送 URL 图片，具体协议转换交给宿主。
#[allow(dead_code)]
pub async fn send_image_url(
    account_id: &str,
    session_id: &str,
    session_type: &str,
    url: &str,
    caption: Option<&str>,
) -> bool {
    send_image_url_for_event(
        account_id,
        "unknown",
        session_id,
        session_type,
        url,
        caption,
        None,
    )
    .await
}

/// 发送关联入站事件的 URL 图片。
pub async fn send_image_url_for_event(
    account_id: &str,
    protocol: &str,
    session_id: &str,
    session_type: &str,
    url: &str,
    caption: Option<&str>,
    source_event_key: Option<&str>,
) -> bool {
    let content = caption.unwrap_or_default();
    let audit_id = match begin_attempt(
        "image",
        source_event_key,
        protocol,
        account_id,
        session_type,
        session_id,
        content,
        Some("image"),
        Some(url),
    ) {
        Some(AttemptStart::Claimed(id)) => Some(id),
        Some(AttemptStart::AlreadyHandled) => return false,
        None => return false,
    };

    if account_id.trim().is_empty() {
        finish_attempt(audit_id, "invalid", None, Some("missing account_id"));
        log::warn!("[AliceBot] 图片发送缺少 account_id，跳过发送");
        return false;
    }
    if !url.starts_with("https://") {
        finish_attempt(audit_id, "invalid", None, Some("image URL must use https"));
        log::warn!("[AliceBot] 图片 URL 无效，跳过发送");
        return false;
    }

    let mut segments = Vec::new();
    if let Some(caption) = caption.filter(|caption| !caption.is_empty()) {
        segments.push(serde_json::json!({
            "type": "text",
            "data": {"text": caption}
        }));
    }
    segments.push(serde_json::json!({
        "type": "image",
        "data": {"url": url}
    }));
    let segments_json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".to_string());

    let bot = BotApi::for_account(account_id);
    let status = match session_type {
        "private" | "dms" => bot.send_rich("private", session_id, "{}", &segments_json),
        "group" => bot.send_rich("group", session_id, "{}", &segments_json),
        "channel" => {
            let Some((guild_id, channel_id)) = session_id.split_once(':') else {
                finish_attempt(
                    audit_id,
                    "invalid",
                    None,
                    Some("channel session_id must be guild:channel"),
                );
                return false;
            };
            bot.send_rich(
                "channel",
                channel_id,
                &format!(r#"{{"guild_id":"{}"}}"#, escape_json(guild_id)),
                &segments_json,
            )
        }
        _ => {
            finish_attempt(audit_id, "invalid", None, Some("unsupported session type"));
            return false;
        }
    };

    let host_status = format!("{status:?}");
    let accepted = matches!(status, SendEnqueueStatus::Accepted);
    finish_attempt(
        audit_id,
        if accepted { "accepted" } else { "rejected" },
        Some(&host_status),
        (!accepted).then_some("host rejected enqueue"),
    );
    accepted
}

enum AttemptStart {
    Claimed(i64),
    AlreadyHandled,
}

fn begin_attempt(
    kind: &str,
    source_event_key: Option<&str>,
    protocol: &str,
    account_id: &str,
    session_type: &str,
    session_id: &str,
    content: &str,
    media_type: Option<&str>,
    media_url: Option<&str>,
) -> Option<AttemptStart> {
    let database = crate::pipeline::try_db()?;
    let action_key = source_event_key
        .filter(|key| !key.trim().is_empty())
        .map(|key| format!("reply:{key}:{kind}"))
        .unwrap_or_else(|| {
            let sequence = NEXT_ACTION_ID.fetch_add(1, Ordering::Relaxed);
            format!(
                "proactive:{kind}:{}:{sequence}",
                chrono::Utc::now().timestamp_millis()
            )
        });
    let attempt = OutboundAttempt {
        action_key,
        source_event_key: source_event_key
            .filter(|key| !key.trim().is_empty())
            .map(str::to_string),
        protocol: protocol.to_string(),
        bot_account_id: account_id.to_string(),
        session_type: session_type.to_string(),
        session_id: session_id.to_string(),
        content: content.to_string(),
        media_type: media_type.map(str::to_string),
        media_url: media_url.map(str::to_string),
    };
    match database.claim_outbound_attempt(&attempt, chrono::Utc::now().timestamp_millis()) {
        Ok(OutboundClaim::Claimed(id)) => Some(AttemptStart::Claimed(id)),
        Ok(OutboundClaim::AlreadyAccepted | OutboundClaim::InFlightOrUncertain) => {
            log::debug!(
                "[AliceBot] 跳过已认领或结果不确定的出站动作 action_key={}",
                attempt.action_key
            );
            Some(AttemptStart::AlreadyHandled)
        }
        Err(error) => {
            log::warn!("[AliceBot] 出站审计写入失败: {error}");
            None
        }
    }
}

fn finish_attempt(
    audit_id: Option<i64>,
    status: &str,
    host_status: Option<&str>,
    error: Option<&str>,
) {
    let Some(id) = audit_id else {
        return;
    };
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    if let Err(db_error) = database.finish_outbound_attempt(
        id,
        status,
        host_status,
        error,
        chrono::Utc::now().timestamp_millis(),
    ) {
        log::warn!("[AliceBot] 出站审计更新失败: {db_error}");
    }
}

fn escape_json(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Base64 发送保留给宿主能力明确的出站场景，当前入站 URL 链路不调用它。
#[allow(dead_code)]
pub async fn send_image_base64(
    _account_id: &str,
    _session_id: &str,
    _session_type: &str,
    _base64: &str,
    _caption: Option<&str>,
) {
    log::debug!("[AliceBot] Base64 图片发送尚未接入当前入站 URL 链路");
}
