//! Deterministic reply decision engine with persisted traces.
use crate::pipeline::InMessage;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static LAST_REPLY_AT: OnceLock<Mutex<HashMap<String, i64>>> = OnceLock::new();

#[derive(Debug, Default, Clone)]
pub struct ReplySignals {
    pub mentioned: bool,
    pub is_question: bool,
    pub is_reply_to_me: bool,
    pub topic_hit: bool,
    pub known_user: bool,
    pub intimacy: i32,
    pub recent_messages: i32,
    pub group_quiet: bool,
    pub emotional: bool,
    pub just_replied: bool,
    pub is_spam: bool,
    pub too_frequent: bool,
    pub out_of_topic: bool,
    pub has_media: bool,
}

#[derive(Debug, Clone)]
pub struct ReplyDecision {
    pub signals: ReplySignals,
    pub score: f32,
    pub threshold: f32,
    pub direct: bool,
    pub should_reply: bool,
    pub reason: String,
}

/// 判断是否回复，并将结果写入 decision_traces。
pub async fn should_reply(msg: &InMessage) -> bool {
    let decision = evaluate(msg);
    persist_trace(msg, &decision);
    log::trace!(
        "[AliceBot] decision: event_key={}, score={:.1}, threshold={:.1}, direct={}, reply={}, reason={}",
        msg.event_key,
        decision.score,
        decision.threshold,
        decision.direct,
        decision.should_reply,
        decision.reason
    );
    decision.should_reply
}

/// 纯函数式决策入口（数据库信号只读），适合离线回放和单元测试。
pub fn evaluate(msg: &InMessage) -> ReplyDecision {
    let config = crate::pipeline::current_config();
    if msg.content.trim().is_empty() && !msg.has_media {
        return ReplyDecision {
            signals: ReplySignals::default(),
            score: 0.0,
            threshold: 100.0,
            direct: false,
            should_reply: false,
            reason: "empty_message".to_string(),
        };
    }

    let signals = collect_signals(msg);
    let direct = signals.mentioned || signals.is_question || signals.is_reply_to_me;
    let threshold = if direct { 35.0 } else { 60.0 };
    let score = score(&signals, config.behavior.reply_bias);
    let (should_reply, reason) = if signals.is_spam {
        (false, "spam_detected")
    } else if signals.too_frequent {
        (false, "sender_burst")
    } else if !msg.at_me
        && is_in_cooldown(
            &msg.session_id,
            msg.timestamp,
            config.behavior.min_interval_sec,
        )
    {
        (false, "session_cooldown")
    } else if score >= threshold {
        (true, "score_reached")
    } else {
        (false, "below_threshold")
    };

    ReplyDecision {
        signals,
        score,
        threshold,
        direct,
        should_reply,
        reason: reason.to_string(),
    }
}

pub fn record_reply(session_id: &str, timestamp: i64) {
    if let Ok(mut state) = LAST_REPLY_AT
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        state.insert(session_id.to_string(), timestamp);
    }
}

fn persist_trace(msg: &InMessage, decision: &ReplyDecision) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let signals_json = json!({
        "mentioned": decision.signals.mentioned,
        "is_question": decision.signals.is_question,
        "is_reply_to_me": decision.signals.is_reply_to_me,
        "topic_hit": decision.signals.topic_hit,
        "known_user": decision.signals.known_user,
        "intimacy": decision.signals.intimacy,
        "recent_messages": decision.signals.recent_messages,
        "group_quiet": decision.signals.group_quiet,
        "emotional": decision.signals.emotional,
        "is_spam": decision.signals.is_spam,
        "too_frequent": decision.signals.too_frequent,
        "out_of_topic": decision.signals.out_of_topic,
        "has_media": decision.signals.has_media,
    })
    .to_string();
    if let Err(error) = database.insert_decision_trace(
        &msg.event_key,
        &msg.session_id,
        decision.score,
        decision.threshold,
        decision.direct,
        if decision.should_reply {
            "reply"
        } else {
            "skip"
        },
        &decision.reason,
        &signals_json,
        msg.timestamp,
    ) {
        log::debug!("[AliceBot] decision trace 写入失败: {error}");
    }
}

fn is_in_cooldown(session_id: &str, now: i64, interval_sec: u64) -> bool {
    let Some(last) = LAST_REPLY_AT
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()
        .and_then(|state| state.get(session_id).copied())
    else {
        return false;
    };
    now.saturating_sub(last) < (interval_sec as i64).saturating_mul(1_000)
}

fn collect_signals(msg: &InMessage) -> ReplySignals {
    let database = crate::pipeline::try_db();
    let (known_user, intimacy) = database
        .as_ref()
        .and_then(|database| {
            let connection = database.conn.lock().ok()?;
            connection
                .query_row(
                    "SELECT 1, intimacy FROM personas WHERE subject_id = ?1",
                    rusqlite::params![msg.sender_id],
                    |row| Ok((true, row.get::<_, i32>(1)?)),
                )
                .ok()
        })
        .unwrap_or((false, 0));

    let (recent_messages, sender_messages) = database
        .as_ref()
        .and_then(|database| {
            let connection = database.conn.lock().ok()?;
            let window_start = msg.timestamp.saturating_sub(60_000);
            connection
                .query_row(
                    "SELECT
                         (SELECT COUNT(*) FROM messages
                          WHERE session_id = ?1 AND direction = 'inbound' AND created_at >= ?2),
                         (SELECT COUNT(*) FROM messages
                          WHERE session_id = ?1 AND sender_id = ?3 AND direction = 'inbound'
                            AND created_at >= ?4)",
                    rusqlite::params![msg.session_id, window_start, msg.sender_id, window_start],
                    |row| Ok((row.get::<_, i32>(0)?, row.get::<_, i32>(1)?)),
                )
                .ok()
        })
        .unwrap_or((0, 0));

    let content = msg.content.trim();
    ReplySignals {
        mentioned: msg.at_me,
        is_question: looks_like_question(content),
        is_reply_to_me: msg.at_me && !msg.reply_to_id.is_empty(),
        topic_hit: false,
        known_user,
        intimacy,
        recent_messages,
        group_quiet: recent_messages <= 2,
        emotional: looks_emotional(content),
        just_replied: false,
        is_spam: looks_like_spam(content),
        too_frequent: sender_messages >= 4,
        out_of_topic: false,
        has_media: msg.has_media,
    }
}

fn looks_like_question(content: &str) -> bool {
    content.ends_with(['?', '？'])
        || [
            "吗",
            "么",
            "什么",
            "怎么",
            "为什么",
            "能不能",
            "有没有",
            "求助",
        ]
        .iter()
        .any(|word| content.contains(word))
}

fn looks_emotional(content: &str) -> bool {
    [
        "哈哈", "笑死", "呜呜", "难过", "生气", "急", "救命", "谢谢", "恭喜",
    ]
    .iter()
    .any(|word| content.contains(word))
        || content.contains('!')
        || content.contains('！')
}

fn looks_like_spam(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    let link_count = content.matches("http://").count() + content.matches("https://").count();
    link_count >= 3 || repeated_run(content) >= 8 || repeated_pattern(content)
}

fn repeated_run(content: &str) -> usize {
    let mut longest = 0;
    let mut current = 0;
    let mut previous = None;
    for character in content.chars() {
        if previous == Some(character) {
            current += 1;
        } else {
            current = 1;
            previous = Some(character);
        }
        longest = longest.max(current);
    }
    longest
}

fn repeated_pattern(content: &str) -> bool {
    let chars = content.chars().collect::<Vec<_>>();
    if chars.len() < 8 {
        return false;
    }
    (1..=3).any(|pattern_len| {
        chars
            .iter()
            .enumerate()
            .all(|(index, character)| *character == chars[index % pattern_len])
    })
}

/// 加权评分，输出范围 `0..100`。
pub fn score(signals: &ReplySignals, reply_bias: f32) -> f32 {
    let mut score = 50.0;
    if signals.mentioned {
        score += 35.0;
    }
    if signals.is_question {
        score += 28.0;
    }
    if signals.is_reply_to_me {
        score += 28.0;
    }
    if signals.topic_hit {
        score += 14.0;
    }
    if signals.known_user {
        score += 10.0;
    }
    score += (signals.intimacy as f32 / 100.0).clamp(0.0, 1.0) * 12.0;
    if signals.group_quiet {
        score += 12.0;
    } else if signals.recent_messages >= 8 {
        // 群消息越密集，越避免插入无关回复；直接提问仍由较低阈值保证优先。
        score -= ((signals.recent_messages - 7) as f32 * 2.0).min(20.0);
    }
    if signals.has_media {
        score += 4.0;
    }
    if signals.emotional {
        score += 10.0;
    }
    if signals.just_replied {
        score -= 24.0;
    }
    if signals.is_spam {
        score -= 55.0;
    }
    if signals.too_frequent {
        score -= 26.0;
    }
    if signals.out_of_topic {
        score -= 18.0;
    }
    (score + reply_bias.clamp(0.0, 1.0) * 20.0).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(content: &str) -> InMessage {
        InMessage {
            event_key: "test:event".to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "user".to_string(),
            message_id: "message-1".to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp: 1_000_000,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn direct_question_scores_above_plain_chat() {
        let plain = ReplySignals::default();
        let question = ReplySignals {
            is_question: true,
            mentioned: true,
            ..ReplySignals::default()
        };
        assert!(score(&question, 0.5) > score(&plain, 0.5));
    }

    #[test]
    fn repeated_text_is_spam() {
        assert!(looks_like_spam("哈哈哈哈哈哈哈哈哈哈"));
        assert!(!looks_like_spam("请问今天几点开会？"));
    }

    #[test]
    fn busy_group_is_penalized_but_direct_question_stays_strong() {
        let busy = ReplySignals {
            recent_messages: 15,
            group_quiet: false,
            ..ReplySignals::default()
        };
        let direct = ReplySignals {
            is_question: true,
            mentioned: true,
            recent_messages: 15,
            group_quiet: false,
            ..ReplySignals::default()
        };
        assert!(score(&busy, 0.5) < score(&ReplySignals::default(), 0.5));
        assert!(score(&direct, 0.5) >= 80.0);
    }

    #[test]
    fn empty_message_has_explainable_reason() {
        let result = evaluate(&message(""));
        assert!(!result.should_reply);
        assert_eq!(result.reason, "empty_message");
    }
}
