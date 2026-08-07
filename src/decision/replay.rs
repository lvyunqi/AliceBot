use serde::Deserialize;

use super::session::SessionSnapshot;
use super::{ReplyDecision, ReplySignals, evaluate_with_signals};
use crate::config::AppConfig;
use crate::pipeline::InMessage;

const FIXTURE: &str = include_str!("../../tests/fixtures/decision_replay_v1.json");
const REPLAY_TIMESTAMP: i64 = 1_000_000;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Label {
    Required,
    Optional,
    Avoid,
}

#[derive(Clone, Debug, Deserialize)]
struct ReplayCase {
    id: String,
    label: Label,
    content: String,
    #[serde(default)]
    signals: ReplySignals,
    #[serde(default)]
    last_outbound_age_ms: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Metrics {
    total: usize,
    replies: usize,
    required: usize,
    required_misses: usize,
    avoid: usize,
    avoid_replies: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ExpectedMetrics {
    replies: f32,
    required_misses: f32,
    avoid_replies: f32,
}

#[test]
fn current_policy_improves_fixed_replay_metrics_without_missing_direct_requests() {
    let cases = load_cases();
    assert_eq!(
        cases.len(),
        32,
        "fixture size changes require metric review"
    );

    let legacy = measure(&cases, |case| legacy_should_reply(case, 0.5));
    let current = measure(&cases, current_should_reply);
    let legacy_expected =
        expected_metrics(&cases, |case| f32::from(legacy_should_reply(case, 0.5)));
    let current_expected = expected_metrics(&cases, current_reply_probability);

    eprintln!(
        "legacy={legacy:?} current={current:?} legacy_expected={legacy_expected:?} current_expected={current_expected:?}"
    );
    assert_eq!(
        legacy,
        Metrics {
            total: 32,
            replies: 24,
            required: 10,
            required_misses: 0,
            avoid: 12,
            avoid_replies: 5,
        }
    );
    assert_eq!(
        current,
        Metrics {
            total: 32,
            replies: 23,
            required: 10,
            required_misses: 0,
            avoid: 12,
            avoid_replies: 4,
        }
    );
    assert_eq!(legacy_expected.required_misses, 0.0);
    assert_eq!(current_expected.required_misses, 0.0);
    assert!(current_expected.replies < legacy_expected.replies);
    assert!(current_expected.avoid_replies < legacy_expected.avoid_replies * 0.8);
}

#[test]
fn replay_results_are_identical_for_repeated_runs() {
    let cases = load_cases();
    let first = cases.iter().map(current_should_reply).collect::<Vec<_>>();
    let second = cases.iter().map(current_should_reply).collect::<Vec<_>>();
    assert_eq!(first, second);
}

fn load_cases() -> Vec<ReplayCase> {
    serde_json::from_str(FIXTURE).expect("decision replay fixture should be valid JSON")
}

fn measure(cases: &[ReplayCase], decide: impl Fn(&ReplayCase) -> bool) -> Metrics {
    let mut metrics = Metrics {
        total: cases.len(),
        ..Metrics::default()
    };
    for case in cases {
        let reply = decide(case);
        metrics.replies += usize::from(reply);
        match case.label {
            Label::Required => {
                metrics.required += 1;
                metrics.required_misses += usize::from(!reply);
            }
            Label::Avoid => {
                metrics.avoid += 1;
                metrics.avoid_replies += usize::from(reply);
            }
            Label::Optional => {}
        }
    }
    metrics
}

fn expected_metrics(
    cases: &[ReplayCase],
    probability: impl Fn(&ReplayCase) -> f32,
) -> ExpectedMetrics {
    let mut metrics = ExpectedMetrics::default();
    for case in cases {
        let probability = probability(case).clamp(0.0, 1.0);
        metrics.replies += probability;
        match case.label {
            Label::Required => metrics.required_misses += 1.0 - probability,
            Label::Avoid => metrics.avoid_replies += probability,
            Label::Optional => {}
        }
    }
    metrics
}

fn current_should_reply(case: &ReplayCase) -> bool {
    current_decision(case).should_reply
}

fn current_reply_probability(case: &ReplayCase) -> f32 {
    let decision = current_decision(case);
    match decision.reason.as_str() {
        "direct_signal" => 1.0,
        "sampled_reply" | "sampled_skip" => decision.p_final,
        _ => 0.0,
    }
}

fn current_decision(case: &ReplayCase) -> ReplyDecision {
    let config = AppConfig::default();
    let message = message(case);
    let snapshot = SessionSnapshot {
        activity_ewma: case.signals.activity_ewma,
        last_outbound_at: case
            .last_outbound_age_ms
            .map(|age| REPLAY_TIMESTAMP.saturating_sub(age)),
        recent_outbound_count: i32::from(case.signals.recent_reply_penalty > 0.0),
        recent_messages: case.signals.recent_messages,
        sender_messages: case.signals.sender_messages,
        ..SessionSnapshot::default()
    };
    evaluate_with_signals(&message, &config, &snapshot, case.signals.clone())
}

fn legacy_should_reply(case: &ReplayCase, reply_bias: f32) -> bool {
    let signals = &case.signals;
    if signals.is_spam || signals.too_frequent {
        return false;
    }
    if case.last_outbound_age_ms.is_some_and(|age| age < 15_000)
        && !signals.mentioned
        && !signals.is_reply_to_me
    {
        return false;
    }

    let direct = signals.mentioned || signals.is_question || signals.is_reply_to_me;
    legacy_score(signals, reply_bias) >= if direct { 35.0 } else { 60.0 }
}

fn legacy_score(signals: &ReplySignals, reply_bias: f32) -> f32 {
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
        score -= ((signals.recent_messages - 7) as f32 * 2.0).min(20.0);
    }
    if signals.has_media {
        score += 4.0;
    }
    if signals.emotional {
        score += 10.0;
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

fn message(case: &ReplayCase) -> InMessage {
    InMessage {
        event_key: format!("replay:{}", case.id),
        protocol: "onebot11".to_string(),
        bot_account_id: String::new(),
        session_type: "group".to_string(),
        session_id: "anonymous-group".to_string(),
        sender_id: "anonymous-user".to_string(),
        sender_name: String::new(),
        message_id: case.id.clone(),
        reply_to_id: if case.signals.is_reply_to_me {
            "anonymous-reply".to_string()
        } else {
            String::new()
        },
        content: case.content.clone(),
        media: Vec::new(),
        has_media: case.signals.has_media,
        at_me: case.signals.mentioned,
        timestamp: REPLAY_TIMESTAMP,
        safe_raw_json: "{}".to_string(),
    }
}
