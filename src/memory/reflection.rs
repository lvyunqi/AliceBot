//! Bounded, deterministic behavior calibration.
//!
//! This module never edits persona, safety rules, permissions, or user data.
//! It only learns a small additive bias for non-direct reply decisions from
//! delivery-confirmed outbound text and intentional skips.

use crate::config::{AppConfig, MAX_REFLECTION_LEARNING_RATE};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};

const CALIBRATION_ID: i64 = 1;
const MAX_REPLY_BIAS_OFFSET: f64 = 0.15;
const MAX_UNRELIABLE_REPLY_RATE: f64 = 0.20;
const MIN_SAFE_AUTONOMOUS_REPLY_RATE: f64 = 0.05;
const MAX_SAFE_AUTONOMOUS_REPLY_RATE: f64 = 0.45;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectionAction {
    Disabled,
    NotDue,
    NoNewDecisions,
    InsufficientSamples,
    UnreliableDelivery,
    Observed,
    Applied,
    RolledBack,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReflectionReport {
    pub action: ReflectionAction,
    pub cursor_start: i64,
    pub cursor_end: i64,
    pub observed_samples: i64,
    pub accepted_replies: i64,
    pub skipped: i64,
    pub unreliable_replies: i64,
}

impl ReflectionReport {
    fn empty(action: ReflectionAction) -> Self {
        Self {
            action,
            cursor_start: 0,
            cursor_end: 0,
            observed_samples: 0,
            accepted_replies: 0,
            skipped: 0,
            unreliable_replies: 0,
        }
    }

    fn from_stats(action: ReflectionAction, cursor_start: i64, stats: &ReflectionStats) -> Self {
        Self {
            action,
            cursor_start,
            cursor_end: stats.cursor_end,
            observed_samples: stats.observed_samples(),
            accepted_replies: stats.accepted_replies,
            skipped: stats.skipped,
            unreliable_replies: stats.unreliable_replies(),
        }
    }
}

#[derive(Debug, Default)]
struct ReflectionStats {
    cursor_end: i64,
    attempted_replies: i64,
    accepted_replies: i64,
    skipped: i64,
}

impl ReflectionStats {
    fn observed_samples(&self) -> i64 {
        self.accepted_replies.saturating_add(self.skipped)
    }

    fn unreliable_replies(&self) -> i64 {
        self.attempted_replies.saturating_sub(self.accepted_replies)
    }

    fn observed_reply_rate(&self) -> Option<f64> {
        let total = self.observed_samples();
        (total > 0).then(|| self.accepted_replies as f64 / total as f64)
    }

    fn unreliable_reply_rate(&self) -> f64 {
        if self.attempted_replies == 0 {
            0.0
        } else {
            self.unreliable_replies() as f64 / self.attempted_replies as f64
        }
    }
}

/// Run one low-frequency calibration pass after compaction. The cursor and all
/// audit records are committed together, so failed transactions are retried
/// from the same decision boundary.
pub async fn run_if_due(config: &AppConfig) -> Result<ReflectionReport, String> {
    let database =
        crate::pipeline::try_db().ok_or_else(|| "database is not initialized".to_string())?;
    run_if_due_at(&database, config, chrono::Utc::now().timestamp_millis())
}

pub(crate) fn reply_bias_offset() -> f32 {
    let Some(database) = crate::pipeline::try_db() else {
        return 0.0;
    };
    let Ok(connection) = database.conn.lock() else {
        return 0.0;
    };
    connection
        .query_row(
            "SELECT reply_bias_offset FROM behavior_calibration WHERE id = ?1",
            [CALIBRATION_ID],
            |row| row.get::<_, f64>(0),
        )
        .optional()
        .ok()
        .flatten()
        .unwrap_or(0.0)
        .clamp(-MAX_REPLY_BIAS_OFFSET, MAX_REPLY_BIAS_OFFSET) as f32
}

fn run_if_due_at(
    database: &crate::db::Database,
    config: &AppConfig,
    now: i64,
) -> Result<ReflectionReport, String> {
    if !config.memories.reflection_enabled {
        return Ok(ReflectionReport::empty(ReflectionAction::Disabled));
    }

    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "database lock failed".to_string())?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO behavior_calibration
             (id, reply_bias_offset, reflection_cursor, updated_at)
             VALUES (?1, 0, 0, 0)",
            [CALIBRATION_ID],
        )
        .map_err(|error| error.to_string())?;

    let (old_offset, cursor_start): (f64, i64) = transaction
        .query_row(
            "SELECT reply_bias_offset, reflection_cursor
             FROM behavior_calibration WHERE id = ?1",
            [CALIBRATION_ID],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|error| error.to_string())?;
    let last_run_at =
        get_meta_i64(&transaction, "reflection_last_run_at").map_err(|error| error.to_string())?;
    let interval_ms = config
        .memories
        .reflection_interval_hours
        .clamp(1, 168)
        .saturating_mul(3_600_000) as i64;
    if last_run_at.is_some_and(|last| now.saturating_sub(last) < interval_ms) {
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(ReflectionReport::empty(ReflectionAction::NotDue));
    }

    let cursor_end: i64 = transaction
        .query_row(
            "SELECT COALESCE(MAX(id), 0) FROM decision_traces WHERE id > ?1",
            [cursor_start],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    if cursor_end <= cursor_start {
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(ReflectionReport::empty(ReflectionAction::NoNewDecisions));
    }

    let stats =
        load_stats(&transaction, cursor_start, cursor_end).map_err(|error| error.to_string())?;
    let minimum_samples = config.memories.reflection_min_decisions.max(1) as i64;
    if stats.observed_samples() < minimum_samples {
        let report = ReflectionReport::from_stats(
            ReflectionAction::InsufficientSamples,
            cursor_start,
            &stats,
        );
        let _reflection_id = insert_reflection_log(
            &transaction,
            &stats,
            cursor_start,
            config,
            None,
            "insufficient_samples",
            "行为校准样本不足，保持当前回复倾向",
            now,
        )
        .map_err(|error| error.to_string())?;
        set_last_run_at(&transaction, now).map_err(|error| error.to_string())?;
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(report);
    }

    if stats.unreliable_reply_rate() > MAX_UNRELIABLE_REPLY_RATE {
        let report = ReflectionReport::from_stats(
            ReflectionAction::UnreliableDelivery,
            cursor_start,
            &stats,
        );
        let _reflection_id = insert_reflection_log(
            &transaction,
            &stats,
            cursor_start,
            config,
            stats.observed_reply_rate(),
            "unreliable_delivery",
            "行为校准检测到过多未确认的回复，保持当前回复倾向",
            now,
        )
        .map_err(|error| error.to_string())?;
        set_last_run_at(&transaction, now).map_err(|error| error.to_string())?;
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(report);
    }

    let observed_rate = stats
        .observed_reply_rate()
        .expect("minimum stable samples guarantees a reply rate");
    let safety_min = MIN_SAFE_AUTONOMOUS_REPLY_RATE;
    let safety_max = MAX_SAFE_AUTONOMOUS_REPLY_RATE;
    if observed_rate < safety_min || observed_rate > safety_max {
        let previous_change = transaction
            .query_row(
                "SELECT id, old_reply_bias_offset
                 FROM reflection_changes
                 WHERE status = 'applied'
                 ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?)),
            )
            .optional()
            .map_err(|error| error.to_string())?;
        let (action, action_key, summary) =
            if let Some((change_id, restored_offset)) = previous_change {
                let reflection_id = insert_reflection_log(
                    &transaction,
                    &stats,
                    cursor_start,
                    config,
                    Some(observed_rate),
                    "safety_rollback",
                    "自主回复率超出安全范围，已回滚上一笔行为校准",
                    now,
                )
                .map_err(|error| error.to_string())?;
                transaction
                    .execute(
                        "UPDATE reflection_changes
                     SET status = 'rolled_back', rolled_back_at = ?1
                     WHERE id = ?2 AND status = 'applied'",
                        params![now, change_id],
                    )
                    .map_err(|error| error.to_string())?;
                insert_reflection_change(
                    &transaction,
                    reflection_id,
                    cursor_start,
                    cursor_end,
                    old_offset,
                    restored_offset,
                    &stats,
                    observed_rate,
                    config.memories.reflection_target_autonomous_rate as f64,
                    "automatic_safety_rollback",
                    "rollback",
                    now,
                )
                .map_err(|error| error.to_string())?;
                update_calibration(&transaction, restored_offset, cursor_end, now)
                    .map_err(|error| error.to_string())?;
                (
                    ReflectionAction::RolledBack,
                    "safety_rollback",
                    "行为校准已自动回滚",
                )
            } else {
                let _reflection_id = insert_reflection_log(
                    &transaction,
                    &stats,
                    cursor_start,
                    config,
                    Some(observed_rate),
                    "safety_observation",
                    "自主回复率超出安全范围，但没有可回滚的校准记录",
                    now,
                )
                .map_err(|error| error.to_string())?;
                update_calibration(&transaction, old_offset, cursor_end, now)
                    .map_err(|error| error.to_string())?;
                (
                    ReflectionAction::Observed,
                    "safety_observation",
                    "行为校准仅记录安全观察",
                )
            };
        set_last_run_at(&transaction, now).map_err(|error| error.to_string())?;
        transaction.commit().map_err(|error| error.to_string())?;
        log::info!(
            "[AliceBot] {summary}: action={action_key}, rate={observed_rate:.3}, cursor={cursor_end}"
        );
        return Ok(ReflectionReport::from_stats(action, cursor_start, &stats));
    }

    let learning_rate = config
        .memories
        .reflection_learning_rate
        .clamp(0.0, MAX_REFLECTION_LEARNING_RATE) as f64;
    let target_rate =
        (config.memories.reflection_target_autonomous_rate as f64).clamp(safety_min, safety_max);
    let new_offset = (old_offset + learning_rate * (target_rate - observed_rate))
        .clamp(-MAX_REPLY_BIAS_OFFSET, MAX_REPLY_BIAS_OFFSET);
    let changed = (new_offset - old_offset).abs() > f64::EPSILON;
    let action = if changed {
        ReflectionAction::Applied
    } else {
        ReflectionAction::Observed
    };
    let action_key = if changed {
        "calibration_applied"
    } else {
        "calibration_observed"
    };
    let summary = if changed {
        "行为校准已小步更新自主回复倾向"
    } else {
        "行为校准观察结果已记录，回复倾向无需调整"
    };
    let reflection_id = insert_reflection_log(
        &transaction,
        &stats,
        cursor_start,
        config,
        Some(observed_rate),
        action_key,
        summary,
        now,
    )
    .map_err(|error| error.to_string())?;
    if changed {
        insert_reflection_change(
            &transaction,
            reflection_id,
            cursor_start,
            cursor_end,
            old_offset,
            new_offset,
            &stats,
            observed_rate,
            target_rate,
            "bounded_rate_feedback",
            "applied",
            now,
        )
        .map_err(|error| error.to_string())?;
    }
    update_calibration(&transaction, new_offset, cursor_end, now)
        .map_err(|error| error.to_string())?;
    set_last_run_at(&transaction, now).map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    log::info!(
        "[AliceBot] behavior calibration: action={action_key}, rate={observed_rate:.3}, offset={old_offset:.4}->{new_offset:.4}, cursor={cursor_end}"
    );
    Ok(ReflectionReport::from_stats(action, cursor_start, &stats))
}

fn load_stats(
    transaction: &Transaction<'_>,
    cursor_start: i64,
    cursor_end: i64,
) -> Result<ReflectionStats, rusqlite::Error> {
    transaction.query_row(
        "SELECT
             COALESCE(SUM(CASE WHEN trace.outcome = 'reply' THEN 1 ELSE 0 END), 0),
             COALESCE(SUM(CASE WHEN trace.outcome = 'skip' THEN 1 ELSE 0 END), 0),
             COALESCE(SUM(CASE
                 WHEN trace.outcome = 'reply' AND EXISTS (
                     SELECT 1 FROM outbound_messages AS outbound
                     WHERE outbound.source_event_key = trace.event_key
                       AND outbound.status = 'accepted'
                       AND COALESCE(outbound.media_type, '') = ''
                       AND TRIM(outbound.content) <> ''
                 ) THEN 1 ELSE 0 END), 0)
         FROM decision_traces AS trace
         WHERE trace.id > ?1 AND trace.id <= ?2
           AND trace.direct = 0
           AND trace.outcome IN ('reply', 'skip')",
        params![cursor_start, cursor_end],
        |row| {
            Ok(ReflectionStats {
                cursor_end,
                attempted_replies: row.get(0)?,
                skipped: row.get(1)?,
                accepted_replies: row.get(2)?,
            })
        },
    )
}

fn get_meta_i64(transaction: &Transaction<'_>, key: &str) -> Result<Option<i64>, rusqlite::Error> {
    transaction
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map(|value| value.and_then(|value| value.parse::<i64>().ok()))
}

fn set_last_run_at(transaction: &Transaction<'_>, now: i64) -> Result<(), rusqlite::Error> {
    transaction.execute(
        "INSERT INTO meta (key, value) VALUES ('reflection_last_run_at', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [now.to_string()],
    )?;
    Ok(())
}

fn update_calibration(
    transaction: &Transaction<'_>,
    reply_bias_offset: f64,
    cursor_end: i64,
    now: i64,
) -> Result<(), rusqlite::Error> {
    transaction.execute(
        "UPDATE behavior_calibration
         SET reply_bias_offset = ?1, reflection_cursor = ?2, updated_at = ?3
         WHERE id = ?4",
        params![
            reply_bias_offset.clamp(-MAX_REPLY_BIAS_OFFSET, MAX_REPLY_BIAS_OFFSET),
            cursor_end,
            now,
            CALIBRATION_ID,
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_reflection_change(
    transaction: &Transaction<'_>,
    reflection_id: i64,
    cursor_start: i64,
    cursor_end: i64,
    old_offset: f64,
    new_offset: f64,
    stats: &ReflectionStats,
    observed_rate: f64,
    target_rate: f64,
    reason: &str,
    status: &str,
    now: i64,
) -> Result<(), rusqlite::Error> {
    transaction.execute(
        "INSERT INTO reflection_changes
         (reflection_id, cursor_start, cursor_end, old_reply_bias_offset,
          new_reply_bias_offset, observed_reply_rate, target_reply_rate,
          accepted_reply_count, skip_count, unreliable_reply_count,
          reason, status, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            reflection_id,
            cursor_start,
            cursor_end,
            old_offset,
            new_offset,
            observed_rate,
            target_rate,
            stats.accepted_replies,
            stats.skipped,
            stats.unreliable_replies(),
            reason,
            status,
            now,
        ],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_reflection_log(
    transaction: &Transaction<'_>,
    stats: &ReflectionStats,
    cursor_start: i64,
    config: &AppConfig,
    observed_rate: Option<f64>,
    action: &str,
    summary: &str,
    now: i64,
) -> Result<i64, rusqlite::Error> {
    let insights = serde_json::json!({
        "action": action,
        "cursor_start": cursor_start,
        "cursor_end": stats.cursor_end,
        "attempted_replies": stats.attempted_replies,
        "accepted_replies": stats.accepted_replies,
        "skipped": stats.skipped,
        "unreliable_replies": stats.unreliable_replies(),
        "observed_samples": stats.observed_samples(),
        "observed_autonomous_rate": observed_rate,
        "target_autonomous_rate": config.memories.reflection_target_autonomous_rate,
        "learning_rate": config.memories.reflection_learning_rate,
    })
    .to_string();
    transaction.execute(
        "INSERT INTO reflection_log (triggered_by, summary, insights, created_at)
         VALUES ('scheduled_behavior_calibration', ?1, ?2, ?3)",
        params![summary, insights, now],
    )?;
    Ok(transaction.last_insert_rowid())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Database, DecisionTrace, OutboundAttempt, OutboundClaim};

    fn test_database() -> Database {
        Database::open(":memory:").expect("test database should open")
    }

    fn test_config(minimum_samples: u64) -> AppConfig {
        let mut config = AppConfig::default();
        config.memories.reflection_interval_hours = 1;
        config.memories.reflection_min_decisions = minimum_samples;
        config.memories.reflection_learning_rate = 0.05;
        config.memories.reflection_target_autonomous_rate = 0.20;
        config
    }

    fn insert_trace(database: &Database, event_key: &str, outcome: &str, direct: bool) {
        database
            .insert_decision_trace(&DecisionTrace {
                event_key,
                session_id: "reflection-group",
                policy_version: "reflection-test",
                score: 50.0,
                threshold: 32.0,
                p_rule: 0.5,
                p_final: 0.5,
                random_value: 0.5,
                activity_ewma: 0.0,
                direct,
                outcome,
                reason: "reflection-test",
                signals_json: "{}",
                coalesced_count: 1,
                created_at: 1,
            })
            .expect("decision trace should insert");
    }

    fn record_text_outbound(database: &Database, event_key: &str, status: &str) {
        let attempt = OutboundAttempt {
            action_key: format!("reflection:{event_key}:{status}"),
            source_event_key: Some(event_key.to_string()),
            protocol: "onebot11".to_string(),
            bot_account_id: "bot-1".to_string(),
            session_type: "group".to_string(),
            session_id: "reflection-group".to_string(),
            content: "reply".to_string(),
            media_type: None,
            media_url: None,
        };
        let id = match database
            .claim_outbound_attempt(&attempt, 1)
            .expect("outbound attempt should claim")
        {
            OutboundClaim::Claimed(id) => id,
            other => panic!("unexpected outbound claim: {other:?}"),
        };
        if status != "pending" {
            database
                .finish_outbound_attempt(id, status, Some(status), None, 2)
                .expect("outbound status should update");
        }
    }

    fn calibration(database: &Database) -> (f64, i64) {
        let connection = database.conn.lock().expect("database lock should work");
        connection
            .query_row(
                "SELECT reply_bias_offset, reflection_cursor
                 FROM behavior_calibration WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("calibration should exist")
    }

    #[test]
    fn accepted_text_and_skip_samples_apply_a_bounded_small_step() {
        let database = test_database();
        insert_trace(&database, "accepted", "reply", false);
        record_text_outbound(&database, "accepted", "accepted");
        for index in 0..9 {
            insert_trace(&database, &format!("skip-{index}"), "skip", false);
        }

        let report = run_if_due_at(&database, &test_config(10), 3_600_000)
            .expect("calibration should succeed");
        assert_eq!(report.action, ReflectionAction::Applied);
        assert_eq!(report.accepted_replies, 1);
        assert_eq!(report.skipped, 9);
        assert_eq!(report.unreliable_replies, 0);
        let (offset, cursor) = calibration(&database);
        assert!((offset - 0.005).abs() < 1e-6);
        assert_eq!(cursor, 10);

        let connection = database.conn.lock().expect("database lock should work");
        let change: (String, f64, f64) = connection
            .query_row(
                "SELECT status, old_reply_bias_offset, new_reply_bias_offset
                 FROM reflection_changes ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("applied change should be recorded");
        assert_eq!(change.0, "applied");
        assert_eq!(change.1, 0.0);
        assert!((change.2 - 0.005).abs() < 1e-6);
    }

    #[test]
    fn insufficient_stable_samples_leave_the_cursor_unchanged() {
        let database = test_database();
        insert_trace(&database, "accepted", "reply", false);
        record_text_outbound(&database, "accepted", "accepted");
        insert_trace(&database, "skip", "skip", false);

        let report = run_if_due_at(&database, &test_config(3), 3_600_000)
            .expect("observation should succeed");
        assert_eq!(report.action, ReflectionAction::InsufficientSamples);
        assert_eq!(calibration(&database).1, 0);
    }

    #[test]
    fn rejected_and_pending_replies_are_excluded_and_hold_calibration() {
        let database = test_database();
        insert_trace(&database, "accepted", "reply", false);
        record_text_outbound(&database, "accepted", "accepted");
        insert_trace(&database, "rejected", "reply", false);
        record_text_outbound(&database, "rejected", "rejected");
        insert_trace(&database, "pending", "reply", false);
        record_text_outbound(&database, "pending", "pending");
        for index in 0..2 {
            insert_trace(&database, &format!("skip-{index}"), "skip", false);
        }

        let report = run_if_due_at(&database, &test_config(3), 3_600_000)
            .expect("observation should succeed");
        assert_eq!(report.action, ReflectionAction::UnreliableDelivery);
        assert_eq!(report.accepted_replies, 1);
        assert_eq!(report.skipped, 2);
        assert_eq!(report.unreliable_replies, 2);
        assert_eq!(calibration(&database), (0.0, 0));
    }

    #[test]
    fn calibration_offset_is_clamped_to_the_hard_limit() {
        let database = test_database();
        database
            .conn
            .lock()
            .expect("database lock should work")
            .execute(
                "UPDATE behavior_calibration SET reply_bias_offset = 0.149 WHERE id = 1",
                [],
            )
            .expect("calibration seed should update");
        insert_trace(&database, "accepted", "reply", false);
        record_text_outbound(&database, "accepted", "accepted");
        for index in 0..19 {
            insert_trace(&database, &format!("skip-{index}"), "skip", false);
        }
        let mut config = test_config(20);
        config.memories.reflection_target_autonomous_rate = 0.45;

        let report =
            run_if_due_at(&database, &config, 3_600_000).expect("calibration should succeed");
        assert_eq!(report.action, ReflectionAction::Applied);
        assert_eq!(calibration(&database).0, MAX_REPLY_BIAS_OFFSET);
    }

    #[test]
    fn abnormal_observed_rate_rolls_back_the_latest_applied_change() {
        let database = test_database();
        {
            let connection = database.conn.lock().expect("database lock should work");
            connection
                .execute(
                    "INSERT INTO reflection_log (triggered_by, summary, insights, created_at)
                     VALUES ('test', 'seed', '{}', 1)",
                    [],
                )
                .expect("reflection log should seed");
            let reflection_id = connection.last_insert_rowid();
            connection
                .execute(
                    "INSERT INTO reflection_changes
                     (reflection_id, cursor_start, cursor_end, old_reply_bias_offset,
                      new_reply_bias_offset, accepted_reply_count, skip_count,
                      unreliable_reply_count, reason, status, created_at)
                     VALUES (?1, 0, 1, 0.04, 0.05, 1, 9, 0, 'seed', 'applied', 1)",
                    [reflection_id],
                )
                .expect("applied change should seed");
            connection
                .execute(
                    "UPDATE behavior_calibration
                     SET reply_bias_offset = 0.05 WHERE id = 1",
                    [],
                )
                .expect("calibration seed should update");
        }
        for index in 0..10 {
            insert_trace(&database, &format!("skip-{index}"), "skip", false);
        }

        let report =
            run_if_due_at(&database, &test_config(10), 3_600_000).expect("rollback should succeed");
        assert_eq!(report.action, ReflectionAction::RolledBack);
        assert_eq!(calibration(&database).0, 0.04);

        let connection = database.conn.lock().expect("database lock should work");
        let previous: (String, Option<i64>) = connection
            .query_row(
                "SELECT status, rolled_back_at FROM reflection_changes WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("seed change should exist");
        assert_eq!(previous, ("rolled_back".to_string(), Some(3_600_000)));
        let rollback_status: String = connection
            .query_row(
                "SELECT status FROM reflection_changes ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("rollback change should exist");
        assert_eq!(rollback_status, "rollback");
    }

    #[test]
    fn failed_transaction_does_not_advance_the_reflection_cursor() {
        let database = test_database();
        for index in 0..2 {
            insert_trace(&database, &format!("skip-{index}"), "skip", false);
        }
        database
            .conn
            .lock()
            .expect("database lock should work")
            .execute_batch(
                "CREATE TRIGGER force_reflection_failure
                 BEFORE INSERT ON reflection_log
                 WHEN NEW.triggered_by = 'scheduled_behavior_calibration'
                 BEGIN
                    SELECT RAISE(ABORT, 'forced reflection failure');
                 END;",
            )
            .expect("failure trigger should install");

        let error = run_if_due_at(&database, &test_config(2), 3_600_000)
            .expect_err("trigger should abort reflection");
        assert!(error.contains("forced reflection failure"));
        assert_eq!(calibration(&database).1, 0);
    }
}
