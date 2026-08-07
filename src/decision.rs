//! 带持久化决策 trace 的确定性回复决策引擎。
mod coalesce;
mod judge;
#[cfg(test)]
mod replay;
mod session;

use crate::pipeline::InMessage;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

pub(crate) use coalesce::CoalescedMessage;
pub(crate) use judge::ReplyStyleHint;
use judge::{ReplyJudgeStatus, ReplyJudgeTrace, apply_optional_reply_judge};

const POLICY_VERSION: &str = "reply-v3-gated";
const MIN_AUTONOMOUS_SCORE: f32 = 32.0;
const DIRECT_SCORE_THRESHOLD: f32 = 35.0;

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct ReplySignals {
    pub mentioned: bool,
    pub is_question: bool,
    pub is_reply_to_me: bool,
    pub topic_hit: bool,
    pub known_user: bool,
    pub intimacy: i32,
    pub recent_messages: i32,
    pub sender_messages: i32,
    pub activity_ewma: f32,
    pub burst_penalty: f32,
    pub recent_reply_penalty: f32,
    pub group_quiet: bool,
    pub emotional: bool,
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
    pub p_rule: f32,
    pub p_final: f32,
    pub random_value: f32,
    pub learned_reply_bias_offset: f32,
    /// 只保存分类状态与受限字段，不保存模型原始输出。
    llm_judge: Option<ReplyJudgeTrace>,
}

pub fn observe_message(msg: &InMessage) {
    let alpha = crate::pipeline::current_config().decision.activity_alpha;
    session::observe(msg, alpha);
}

pub async fn coalesce_message(msg: InMessage) -> Option<CoalescedMessage> {
    let window_ms = crate::pipeline::current_config()
        .decision
        .coalesce_window_ms;
    coalesce::coalesce(msg, window_ms).await
}

pub fn clear_runtime_state() {
    coalesce::clear();
}

/// 判断是否回复并返回分类器认可的有限风格提示，供同一轮生成消费。
pub async fn should_reply_with_style(
    msg: &InMessage,
    coalesced_count: usize,
) -> (bool, Option<ReplyStyleHint>) {
    let mut decision = evaluate(msg);
    apply_optional_reply_judge(msg, &mut decision).await;
    persist_trace(msg, &decision, coalesced_count);
    let style_hint = decision
        .llm_judge
        .as_ref()
        .filter(|judge| judge.status == ReplyJudgeStatus::Applied && decision.should_reply)
        .and_then(|judge| judge.style_hint.as_ref())
        .cloned();
    let judge_status = decision
        .llm_judge
        .as_ref()
        .map(|judge| judge.status.as_str())
        .unwrap_or("not_called");
    log::trace!(
        "[AliceBot] decision: event_key={}, score={:.1}, p_final={:.3}, random={:.3}, direct={}, judge={}, reply={}, reason={}",
        msg.event_key,
        decision.score,
        decision.p_final,
        decision.random_value,
        decision.direct,
        judge_status,
        decision.should_reply,
        decision.reason
    );
    (decision.should_reply, style_hint)
}

/// 纯函数式决策入口（数据库信号只读），适合离线回放和单元测试。
pub fn evaluate(msg: &InMessage) -> ReplyDecision {
    let config = crate::pipeline::current_config();
    let snapshot = session::load(msg);
    if msg.content.trim().is_empty() && !msg.has_media {
        return ReplyDecision {
            signals: ReplySignals::default(),
            score: 0.0,
            threshold: 100.0,
            direct: false,
            should_reply: false,
            reason: "empty_message".to_string(),
            p_rule: 0.0,
            p_final: 0.0,
            random_value: deterministic_random(&msg.event_key, POLICY_VERSION),
            learned_reply_bias_offset: 0.0,
            llm_judge: None,
        };
    }

    let signals = collect_signals(msg, &snapshot, &config);
    let direct = signals.mentioned || signals.is_question || signals.is_reply_to_me;
    let learned_reply_bias_offset = if direct {
        0.0
    } else {
        crate::memory::reflection::reply_bias_offset()
    };
    evaluate_with_signals_and_offset(msg, &config, &snapshot, signals, learned_reply_bias_offset)
}

#[cfg(test)]
fn evaluate_with_signals(
    msg: &InMessage,
    config: &crate::config::AppConfig,
    snapshot: &session::SessionSnapshot,
    signals: ReplySignals,
) -> ReplyDecision {
    evaluate_with_signals_and_offset(msg, config, snapshot, signals, 0.0)
}

fn evaluate_with_signals_and_offset(
    msg: &InMessage,
    config: &crate::config::AppConfig,
    snapshot: &session::SessionSnapshot,
    signals: ReplySignals,
    learned_reply_bias_offset: f32,
) -> ReplyDecision {
    let direct = signals.mentioned || signals.is_question || signals.is_reply_to_me;
    let threshold = if direct {
        DIRECT_SCORE_THRESHOLD
    } else {
        MIN_AUTONOMOUS_SCORE
    };
    let applied_offset = if direct {
        0.0
    } else {
        learned_reply_bias_offset
    };
    let score = score(
        &signals,
        reply_bias_for_decision(config.reply_bias(), direct, applied_offset),
    );
    let p_rule = sigmoid((score - 50.0) / 12.0);
    let prior_denominator = snapshot.reply_alpha + snapshot.reply_beta;
    let participation_prior = if prior_denominator > 0.0 {
        (snapshot.reply_alpha / prior_denominator).clamp(0.05, 0.95)
    } else {
        0.5
    };
    let prior_blend = (0.75 * p_rule + 0.25 * participation_prior).clamp(0.0, 1.0);
    let burst_gate = (1.0 - signals.burst_penalty.clamp(0.0, 1.0)).powi(2);
    let recent_reply_gate = (1.0 - signals.recent_reply_penalty.clamp(0.0, 1.0)).powi(2);
    let p_final = (prior_blend * burst_gate * recent_reply_gate).clamp(0.0, 1.0);
    let random_value = deterministic_random(&msg.event_key, POLICY_VERSION);
    let (should_reply, reason) = if signals.is_spam {
        (false, "spam_detected")
    } else if signals.too_frequent {
        (false, "sender_burst")
    } else if !msg.at_me
        && !signals.is_reply_to_me
        && is_in_cooldown(snapshot, msg.timestamp, config.min_interval_sec())
    {
        (false, "session_cooldown")
    } else if !config.decision.enabled && !direct {
        (false, "autonomous_disabled")
    } else if direct {
        (true, "direct_signal")
    } else if score < MIN_AUTONOMOUS_SCORE {
        (false, "below_minimum_score")
    } else if random_value <= p_final {
        (true, "sampled_reply")
    } else {
        (false, "sampled_skip")
    };

    ReplyDecision {
        signals,
        score,
        threshold,
        direct,
        should_reply,
        reason: reason.to_string(),
        p_rule,
        p_final,
        random_value,
        learned_reply_bias_offset: applied_offset,
        llm_judge: None,
    }
}

pub fn record_reply(msg: &InMessage, timestamp: i64) {
    session::record_outbound(msg, timestamp);
}

pub fn record_coalesced(batch: &CoalescedMessage) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let signals_json = json!({
        "coalesced_into": batch.message.event_key,
        "policy_version": POLICY_VERSION,
    })
    .to_string();
    for event_key in batch
        .source_event_keys
        .iter()
        .filter(|event_key| *event_key != &batch.message.event_key)
    {
        let trace = crate::db::DecisionTrace {
            event_key,
            session_id: &batch.message.session_id,
            policy_version: POLICY_VERSION,
            score: 0.0,
            threshold: 0.0,
            p_rule: 0.0,
            p_final: 0.0,
            random_value: deterministic_random(event_key, POLICY_VERSION),
            activity_ewma: 0.0,
            direct: false,
            outcome: "batch",
            reason: "coalesced_into_later_message",
            signals_json: &signals_json,
            coalesced_count: batch.source_event_keys.len(),
            created_at: batch.message.timestamp,
        };
        if let Err(error) = database.insert_decision_trace(&trace) {
            log::debug!("[AliceBot] coalesced decision trace write failed: {error}");
        }
    }
}

fn persist_trace(msg: &InMessage, decision: &ReplyDecision, coalesced_count: usize) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let llm_judge = decision.llm_judge.as_ref().map(|judge| {
        json!({
            "status": judge.status.as_str(),
            "reply": judge.reply,
            "confidence": judge.confidence,
            "style_hint": judge.style_hint.as_ref().map(ReplyStyleHint::as_str),
            "error_kind": judge.error_kind,
        })
    });
    let signals_json = json!({
        "mentioned": decision.signals.mentioned,
        "is_question": decision.signals.is_question,
        "is_reply_to_me": decision.signals.is_reply_to_me,
        "topic_hit": decision.signals.topic_hit,
        "known_user": decision.signals.known_user,
        "intimacy": decision.signals.intimacy,
        "recent_messages": decision.signals.recent_messages,
        "sender_messages": decision.signals.sender_messages,
        "activity_ewma": decision.signals.activity_ewma,
        "burst_penalty": decision.signals.burst_penalty,
        "recent_reply_penalty": decision.signals.recent_reply_penalty,
        "group_quiet": decision.signals.group_quiet,
        "emotional": decision.signals.emotional,
        "is_spam": decision.signals.is_spam,
        "too_frequent": decision.signals.too_frequent,
        "out_of_topic": decision.signals.out_of_topic,
        "has_media": decision.signals.has_media,
        "learned_reply_bias_offset": decision.learned_reply_bias_offset,
        "llm_judge": llm_judge,
    })
    .to_string();
    let trace = crate::db::DecisionTrace {
        event_key: &msg.event_key,
        session_id: &msg.session_id,
        policy_version: POLICY_VERSION,
        score: decision.score,
        threshold: decision.threshold,
        p_rule: decision.p_rule,
        p_final: decision.p_final,
        random_value: decision.random_value,
        activity_ewma: decision.signals.activity_ewma,
        direct: decision.direct,
        outcome: if decision.should_reply {
            "reply"
        } else {
            "skip"
        },
        reason: &decision.reason,
        signals_json: &signals_json,
        coalesced_count,
        created_at: msg.timestamp,
    };
    if let Err(error) = database.insert_decision_trace(&trace) {
        log::debug!("[AliceBot] decision trace 写入失败: {error}");
    }
}

fn is_in_cooldown(snapshot: &session::SessionSnapshot, now: i64, interval_sec: u64) -> bool {
    let Some(last) = snapshot.last_outbound_at else {
        return false;
    };
    now.saturating_sub(last) < (interval_sec as i64).saturating_mul(1_000)
}

fn collect_signals(
    msg: &InMessage,
    snapshot: &session::SessionSnapshot,
    config: &crate::config::AppConfig,
) -> ReplySignals {
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

    let quiet_threshold = config.decision.quiet_threshold.clamp(0.0, 1.0);
    let configured_burst = config.decision.burst_threshold.clamp(0.0, 1.0);
    let burst_threshold = if configured_burst <= quiet_threshold {
        (quiet_threshold + 0.1).min(1.0)
    } else {
        configured_burst
    };
    let burst_penalty = if snapshot.activity_ewma <= burst_threshold {
        0.0
    } else if burst_threshold >= 1.0 {
        1.0
    } else {
        ((snapshot.activity_ewma - burst_threshold) / (1.0 - burst_threshold)).clamp(0.0, 1.0)
    };
    let recent_reply_penalty = snapshot
        .last_outbound_at
        .map(|last| {
            let age = msg.timestamp.saturating_sub(last);
            let interval_ms = (config.min_interval_sec() as i64).saturating_mul(1_000);
            if age >= interval_ms || interval_ms == 0 {
                (snapshot.recent_outbound_count as f32 / 5.0).clamp(0.0, 0.6)
            } else {
                (1.0 - age as f32 / interval_ms as f32).clamp(0.0, 1.0)
            }
        })
        .unwrap_or(0.0);
    let content = msg.content.trim();
    ReplySignals {
        mentioned: msg.at_me,
        is_question: looks_like_question(content),
        is_reply_to_me: msg.at_me && !msg.reply_to_id.is_empty(),
        topic_hit: false,
        known_user,
        intimacy,
        recent_messages: snapshot.recent_messages,
        sender_messages: snapshot.sender_messages,
        activity_ewma: snapshot.activity_ewma,
        burst_penalty,
        recent_reply_penalty,
        group_quiet: snapshot.activity_ewma <= quiet_threshold,
        emotional: looks_emotional(content),
        is_spam: looks_like_spam(content),
        too_frequent: snapshot.sender_messages >= 5,
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
    }
    score -= signals.burst_penalty.clamp(0.0, 1.0) * 26.0;
    if signals.has_media {
        score += 4.0;
    }
    if signals.emotional {
        score += 10.0;
    }
    score -= signals.recent_reply_penalty.clamp(0.0, 1.0) * 24.0;
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

pub(crate) fn reply_bias_for_decision(
    base_reply_bias: f32,
    direct: bool,
    learned_reply_bias_offset: f32,
) -> f32 {
    if direct {
        base_reply_bias.clamp(0.0, 1.0)
    } else {
        (base_reply_bias + learned_reply_bias_offset.clamp(-0.15, 0.15)).clamp(0.0, 1.0)
    }
}

fn sigmoid(value: f32) -> f32 {
    (1.0 / (1.0 + (-value).exp())).clamp(0.0, 1.0)
}

fn deterministic_random(event_key: &str, policy_version: &str) -> f32 {
    let mut hasher = Sha256::new();
    hasher.update(event_key.as_bytes());
    hasher.update([0]);
    hasher.update(policy_version.as_bytes());
    let digest = hasher.finalize();
    let value = u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    );
    let mantissa = value >> 11;
    (mantissa as f64 / (1_u64 << 53) as f64) as f32
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
            activity_ewma: 0.95,
            burst_penalty: 0.67,
            group_quiet: false,
            ..ReplySignals::default()
        };
        let direct = ReplySignals {
            is_question: true,
            mentioned: true,
            recent_messages: 15,
            activity_ewma: 0.95,
            burst_penalty: 0.67,
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

    #[test]
    fn deterministic_random_is_stable_and_policy_scoped() {
        let first = deterministic_random("event-1", "policy-a");
        assert_eq!(first, deterministic_random("event-1", "policy-a"));
        assert_ne!(first, deterministic_random("event-1", "policy-b"));
        assert!((0.0..1.0).contains(&first));
    }

    #[test]
    fn sigmoid_probability_is_bounded_and_monotonic() {
        assert!(sigmoid(-2.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(2.0));
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn direct_requests_ignore_the_learned_reply_bias_offset() {
        assert_eq!(reply_bias_for_decision(0.5, true, 0.15), 0.5);
        assert_eq!(reply_bias_for_decision(0.5, false, 0.15), 0.65);
    }
}
