use rusqlite::{Connection, OptionalExtension, params};

use crate::pipeline::InMessage;

const ACTIVITY_HALF_LIFE_MS: f64 = 60_000.0;
const RECENT_MESSAGE_WINDOW_MS: i64 = 60_000;
const SENDER_BURST_WINDOW_MS: i64 = 10_000;
const RECENT_OUTBOUND_WINDOW_MS: i64 = 300_000;

#[derive(Debug, Clone)]
pub(super) struct SessionSnapshot {
    pub activity_ewma: f32,
    pub last_outbound_at: Option<i64>,
    pub recent_outbound_count: i32,
    pub reply_alpha: f32,
    pub reply_beta: f32,
    pub recent_messages: i32,
    pub sender_messages: i32,
}

impl Default for SessionSnapshot {
    fn default() -> Self {
        Self {
            activity_ewma: 0.0,
            last_outbound_at: None,
            recent_outbound_count: 0,
            reply_alpha: 1.0,
            reply_beta: 1.0,
            recent_messages: 0,
            sender_messages: 0,
        }
    }
}

pub(super) fn observe(msg: &InMessage, alpha: f32) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let Ok(connection) = database.conn.lock() else {
        return;
    };
    if let Err(error) = observe_with_connection(&connection, msg, alpha) {
        log::debug!("[AliceBot] session activity update failed: {error}");
    }
}

pub(super) fn load(msg: &InMessage) -> SessionSnapshot {
    let Some(database) = crate::pipeline::try_db() else {
        return SessionSnapshot::default();
    };
    let Ok(connection) = database.conn.lock() else {
        return SessionSnapshot::default();
    };
    match load_with_connection(&connection, msg) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            log::debug!("[AliceBot] session state read failed: {error}");
            SessionSnapshot::default()
        }
    }
}

pub(super) fn record_outbound(msg: &InMessage, sent_at: i64) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let Ok(connection) = database.conn.lock() else {
        return;
    };
    if let Err(error) = record_outbound_with_connection(&connection, msg, sent_at) {
        log::debug!("[AliceBot] session outbound state update failed: {error}");
    }
}

fn observe_with_connection(
    conn: &Connection,
    msg: &InMessage,
    alpha: f32,
) -> Result<(), rusqlite::Error> {
    let key = session_key(msg);
    let previous = conn
        .query_row(
            "SELECT last_message_at, activity_ewma, last_outbound_at,
                    recent_outbound_count
             FROM session_state WHERE session_key = ?1",
            params![key],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?,
                    row.get::<_, f32>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, i32>(3)?,
                ))
            },
        )
        .optional()?;

    let (last_message_at, previous_activity, last_outbound_at, outbound_count) =
        previous.unwrap_or((None, 0.0, None, 0));
    let activity_ewma = update_activity(previous_activity, last_message_at, msg.timestamp, alpha);
    let last_message_at = Some(last_message_at.unwrap_or(msg.timestamp).max(msg.timestamp));
    let recent_outbound_count = if last_outbound_at
        .is_some_and(|last| msg.timestamp.saturating_sub(last) <= RECENT_OUTBOUND_WINDOW_MS)
    {
        outbound_count.max(0)
    } else {
        0
    };

    conn.execute(
        "INSERT INTO session_state
         (session_key, protocol, session_type, session_id, last_message_at,
          last_outbound_at, recent_outbound_count, activity_ewma, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(session_key) DO UPDATE SET
            protocol = excluded.protocol,
            session_type = excluded.session_type,
            session_id = excluded.session_id,
            last_message_at = excluded.last_message_at,
            recent_outbound_count = excluded.recent_outbound_count,
            activity_ewma = excluded.activity_ewma,
            updated_at = excluded.updated_at",
        params![
            key,
            msg.protocol,
            msg.session_type,
            msg.session_id,
            last_message_at,
            last_outbound_at,
            recent_outbound_count,
            activity_ewma,
            last_message_at,
        ],
    )?;
    Ok(())
}

fn load_with_connection(
    conn: &Connection,
    msg: &InMessage,
) -> Result<SessionSnapshot, rusqlite::Error> {
    let key = session_key(msg);
    let mut snapshot = conn
        .query_row(
            "SELECT activity_ewma, last_outbound_at, recent_outbound_count,
                    reply_alpha, reply_beta
             FROM session_state WHERE session_key = ?1",
            params![key],
            |row| {
                Ok(SessionSnapshot {
                    activity_ewma: row.get(0)?,
                    last_outbound_at: row.get(1)?,
                    recent_outbound_count: row.get(2)?,
                    reply_alpha: row.get(3)?,
                    reply_beta: row.get(4)?,
                    ..SessionSnapshot::default()
                })
            },
        )
        .optional()?
        .unwrap_or_default();

    let recent_start = msg.timestamp.saturating_sub(RECENT_MESSAGE_WINDOW_MS);
    let sender_start = msg.timestamp.saturating_sub(SENDER_BURST_WINDOW_MS);
    (snapshot.recent_messages, snapshot.sender_messages) = conn.query_row(
        "SELECT
             (SELECT COUNT(*) FROM messages
              WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
                AND direction = 'inbound' AND created_at >= ?4 AND created_at <= ?5),
             (SELECT COUNT(*) FROM messages
              WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
                AND sender_id = ?6 AND direction = 'inbound'
                AND created_at >= ?7 AND created_at <= ?5)",
        params![
            msg.protocol,
            msg.session_type,
            msg.session_id,
            recent_start,
            msg.timestamp,
            msg.sender_id,
            sender_start,
        ],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok(snapshot)
}

fn record_outbound_with_connection(
    conn: &Connection,
    msg: &InMessage,
    sent_at: i64,
) -> Result<(), rusqlite::Error> {
    let key = session_key(msg);
    let previous = conn
        .query_row(
            "SELECT last_outbound_at, recent_outbound_count
             FROM session_state WHERE session_key = ?1",
            params![key],
            |row| Ok((row.get::<_, Option<i64>>(0)?, row.get::<_, i32>(1)?)),
        )
        .optional()?;
    let recent_count = match previous {
        Some((Some(last), count))
            if sent_at >= last && sent_at.saturating_sub(last) <= RECENT_OUTBOUND_WINDOW_MS =>
        {
            count.saturating_add(1).clamp(1, 100)
        }
        Some((Some(last), _)) if sent_at >= last => 1,
        Some((Some(_), count)) => count,
        _ => 1,
    };

    conn.execute(
        "INSERT INTO session_state
         (session_key, protocol, session_type, session_id, last_outbound_at,
          recent_outbound_count, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?5)
         ON CONFLICT(session_key) DO UPDATE SET
            last_outbound_at = MAX(COALESCE(session_state.last_outbound_at, 0), excluded.last_outbound_at),
            recent_outbound_count = ?6,
            updated_at = MAX(session_state.updated_at, excluded.updated_at)",
        params![
            key,
            msg.protocol,
            msg.session_type,
            msg.session_id,
            sent_at,
            recent_count,
        ],
    )?;
    Ok(())
}

fn update_activity(previous: f32, last_message_at: Option<i64>, now: i64, alpha: f32) -> f32 {
    let Some(last_message_at) = last_message_at else {
        return 0.0;
    };
    if now < last_message_at {
        return previous.clamp(0.0, 1.0);
    }

    let elapsed = now.saturating_sub(last_message_at) as f64;
    let decay = 0.5_f64.powf(elapsed / ACTIVITY_HALF_LIFE_MS);
    let decayed_previous = f64::from(previous.clamp(0.0, 1.0)) * decay;
    let instant = (1.0 - elapsed / ACTIVITY_HALF_LIFE_MS).clamp(0.0, 1.0);
    let alpha = f64::from(alpha.clamp(0.05, 1.0));
    (alpha * instant + (1.0 - alpha) * decayed_previous).clamp(0.0, 1.0) as f32
}

fn session_key(msg: &InMessage) -> String {
    format!(
        "{}:{}{}:{}{}:{}",
        msg.protocol.len(),
        msg.protocol,
        msg.session_type.len(),
        msg.session_type,
        msg.session_id.len(),
        msg.session_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn message(timestamp: i64) -> InMessage {
        InMessage {
            event_key: format!("session-test:{timestamp}"),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "activity-group".to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "user".to_string(),
            message_id: timestamp.to_string(),
            reply_to_id: String::new(),
            content: "hello".to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn rapid_activity_rises_and_quiet_time_decays() {
        let mut activity = 0.0;
        let mut last = Some(0);
        for now in [1_000, 2_000, 3_000, 4_000, 5_000, 6_000] {
            activity = update_activity(activity, last, now, 0.35);
            last = Some(now);
        }
        assert!(activity > 0.85);

        let decayed = update_activity(activity, last, 306_000, 0.35);
        assert!(decayed < 0.05);
    }

    #[test]
    fn session_activity_and_outbound_state_are_persisted() {
        let database = Database::open(":memory:").expect("database should open");
        let connection = database.conn.lock().unwrap();
        let first = message(1_000);
        let second = message(2_000);
        observe_with_connection(&connection, &first, 0.35).unwrap();
        observe_with_connection(&connection, &second, 0.35).unwrap();
        record_outbound_with_connection(&connection, &second, 2_500).unwrap();

        let snapshot = load_with_connection(&connection, &second).unwrap();
        assert!(snapshot.activity_ewma > 0.3);
        assert_eq!(snapshot.last_outbound_at, Some(2_500));
        assert_eq!(snapshot.recent_outbound_count, 1);
        assert_eq!(snapshot.reply_alpha, 1.0);
        assert_eq!(snapshot.reply_beta, 1.0);
    }

    #[test]
    fn cooldown_state_survives_database_reopen() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-session-state-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let msg = message(10_000);
        let database = Database::open(path.to_str().unwrap()).unwrap();
        {
            let connection = database.conn.lock().unwrap();
            observe_with_connection(&connection, &msg, 0.35).unwrap();
            record_outbound_with_connection(&connection, &msg, 10_500).unwrap();
        }
        drop(database);

        let reopened = Database::open(path.to_str().unwrap()).unwrap();
        let snapshot = load_with_connection(&reopened.conn.lock().unwrap(), &msg).unwrap();
        assert_eq!(snapshot.last_outbound_at, Some(10_500));
        assert_eq!(snapshot.recent_outbound_count, 1);
        drop(reopened);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    #[test]
    fn replay_window_does_not_count_future_messages() {
        let database = Database::open(":memory:").unwrap();
        let past = message(1_000);
        let current = message(2_000);
        let future = message(3_000);
        database.insert_message(&past).unwrap();
        database.insert_message(&current).unwrap();
        database.insert_message(&future).unwrap();

        let snapshot = load_with_connection(&database.conn.lock().unwrap(), &current).unwrap();
        assert_eq!(snapshot.recent_messages, 2);
        assert_eq!(snapshot.sender_messages, 2);
    }
}
