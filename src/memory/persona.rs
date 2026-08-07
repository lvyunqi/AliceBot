//! Transactional, protocol-scoped user profiles for cautious personalization.

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::pipeline::InMessage;

const INTIMACY_DECAY_HALF_LIFE_DAYS: f64 = 180.0;
const TOPIC_DECAY_HALF_LIFE_DAYS: f64 = 30.0;
const MILLIS_PER_DAY: f64 = 86_400_000.0;
const MAX_PROFILE_COUNTS: usize = 64;
const MAX_TOPICS_PER_PERSONA: usize = 128;
const MAX_NICKNAMES_PER_PERSONA: usize = 32;

#[derive(Debug, Clone)]
pub struct Persona {
    pub subject_id: String,
    pub protocol: String,
    pub nickname: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub interaction_count: i32,
    pub intimacy: i32,
    pub intimacy_ewma: f64,
    pub intimacy_updated_at: i64,
    pub relation: String,
    pub traits: String,
    pub preferences: String,
    pub topics: String,
    pub notes: String,
}

/// Observe one accepted journal event without replacing prior profile evidence.
pub async fn observe(msg: &InMessage) {
    let Some(database) = crate::pipeline::try_db() else {
        log::warn!("[AliceBot] persona observation skipped: database is not ready");
        return;
    };
    if let Err(error) = observe_from(&database, msg) {
        log::error!("[AliceBot] persona update failed: {error}");
    }
}

/// Return a bounded JSON summary scoped to one protocol identity domain.
pub fn summary(protocol: &str, subject_id: &str) -> Option<String> {
    let database = crate::pipeline::try_db()?;
    let connection = database.conn.lock().ok()?;
    let persona = load_from(&connection, protocol, subject_id)?;
    if persona.interaction_count == 0 {
        return None;
    }

    let aliases = load_aliases(&connection, protocol, subject_id)
        .unwrap_or_default()
        .into_iter()
        .map(|alias| prompt_safe_text(&alias))
        .collect::<Vec<_>>();
    let topics = load_top_topics(&connection, protocol, subject_id).unwrap_or_default();
    Some(
        json!({
            "nickname": prompt_safe_text(&persona.nickname),
            "nickname_history": aliases,
            "relation": persona.relation,
            "interactions": persona.interaction_count,
            "intimacy": persona.intimacy,
            "traits": top_count_entries(&persona.traits, 8),
            "preferences": top_count_entries(&persona.preferences, 8),
            "topics": topics,
        })
        .to_string(),
    )
}

fn observe_from(database: &crate::db::Database, msg: &InMessage) -> Result<bool, String> {
    if msg.protocol.trim().is_empty() || msg.sender_id.trim().is_empty() {
        return Err("persona identity is missing protocol or subject ID".to_string());
    }

    let observed_at = if msg.timestamp > 0 {
        msg.timestamp
    } else {
        chrono::Utc::now().timestamp_millis()
    };
    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "persona database lock failed".to_string())?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;

    transaction
        .execute(
            "INSERT OR IGNORE INTO personas
             (protocol, subject_id, nickname, first_seen, last_seen,
              interaction_count, intimacy, intimacy_ewma, intimacy_updated_at,
              relation, traits, preferences, topics, notes)
             VALUES (?1, ?2, '', ?3, ?3, 0, 0, 0, ?3,
                     'stranger', '{}', '{}', '{}', '')",
            params![msg.protocol, msg.sender_id, observed_at],
        )
        .map_err(|error| error.to_string())?;

    let event_key = observation_key(msg);
    let inserted = transaction
        .execute(
            "INSERT OR IGNORE INTO persona_observations
             (protocol, event_key, subject_id, session_id, observed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                msg.protocol,
                event_key,
                msg.sender_id,
                msg.session_id,
                observed_at
            ],
        )
        .map_err(|error| error.to_string())?;
    if inserted == 0 {
        transaction.commit().map_err(|error| error.to_string())?;
        return Ok(false);
    }

    let mut persona = load_from(&transaction, &msg.protocol, &msg.sender_id)
        .ok_or_else(|| "persona row disappeared during observation".to_string())?;
    let was_new = persona.interaction_count == 0;
    persona.first_seen = if persona.first_seen == 0 {
        observed_at
    } else {
        persona.first_seen.min(observed_at)
    };
    persona.last_seen = persona.last_seen.max(observed_at);
    persona.interaction_count = persona.interaction_count.saturating_add(1);
    persona.intimacy_ewma = update_intimacy(
        persona.intimacy_ewma,
        persona.intimacy_updated_at,
        observed_at,
        intimacy_signal(msg),
        intimacy_alpha(msg),
        was_new,
    );
    persona.intimacy = persona.intimacy_ewma.round().clamp(0.0, 100.0) as i32;
    persona.intimacy_updated_at = persona.intimacy_updated_at.max(observed_at);
    persona.relation = relation_for(persona.intimacy_ewma).to_string();
    persona.traits = increment_counts(&persona.traits, &trait_keys(msg), MAX_PROFILE_COUNTS);
    persona.preferences = increment_counts(
        &persona.preferences,
        &preference_keys(&msg.content),
        MAX_PROFILE_COUNTS,
    );
    let topics = topic_keys(&msg.content);
    persona.topics = increment_counts(&persona.topics, &topics, MAX_PROFILE_COUNTS);

    if let Some(current_nickname) = record_nickname(
        &transaction,
        &msg.protocol,
        &msg.sender_id,
        &msg.sender_name,
        &event_key,
        observed_at,
    )? {
        persona.nickname = current_nickname;
    }
    for topic in &topics {
        record_topic(
            &transaction,
            &msg.protocol,
            &msg.sender_id,
            topic,
            observed_at,
        )?;
    }

    transaction
        .execute(
            "UPDATE personas
             SET nickname = ?1, first_seen = ?2, last_seen = ?3,
                 interaction_count = ?4, intimacy = ?5, intimacy_ewma = ?6,
                 intimacy_updated_at = ?7, relation = ?8, traits = ?9,
                 preferences = ?10, topics = ?11, notes = ?12
             WHERE protocol = ?13 AND subject_id = ?14",
            params![
                persona.nickname,
                persona.first_seen,
                persona.last_seen,
                persona.interaction_count,
                persona.intimacy,
                persona.intimacy_ewma,
                persona.intimacy_updated_at,
                persona.relation,
                persona.traits,
                persona.preferences,
                persona.topics,
                persona.notes,
                msg.protocol,
                msg.sender_id,
            ],
        )
        .map_err(|error| error.to_string())?;
    prune_nicknames(&transaction, &msg.protocol, &msg.sender_id)?;
    prune_topics(&transaction, &msg.protocol, &msg.sender_id)?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(true)
}

fn load_from(connection: &Connection, protocol: &str, subject_id: &str) -> Option<Persona> {
    connection
        .query_row(
            "SELECT subject_id, protocol, nickname, first_seen, last_seen,
                    interaction_count, intimacy, intimacy_ewma, intimacy_updated_at,
                    relation, traits, preferences, topics, notes
             FROM personas WHERE protocol = ?1 AND subject_id = ?2",
            params![protocol, subject_id],
            |row| {
                Ok(Persona {
                    subject_id: row.get(0)?,
                    protocol: row.get(1)?,
                    nickname: row.get(2)?,
                    first_seen: row.get(3)?,
                    last_seen: row.get(4)?,
                    interaction_count: row.get(5)?,
                    intimacy: row.get(6)?,
                    intimacy_ewma: row.get(7)?,
                    intimacy_updated_at: row.get(8)?,
                    relation: row.get(9)?,
                    traits: row.get(10)?,
                    preferences: row.get(11)?,
                    topics: row.get(12)?,
                    notes: row.get(13)?,
                })
            },
        )
        .optional()
        .ok()
        .flatten()
}

fn record_nickname(
    transaction: &Transaction<'_>,
    protocol: &str,
    subject_id: &str,
    raw_nickname: &str,
    event_key: &str,
    observed_at: i64,
) -> Result<Option<String>, String> {
    let nickname = sanitize_nickname(raw_nickname);
    if nickname.is_empty() {
        return Ok(None);
    }

    transaction
        .execute(
            "INSERT INTO persona_nicknames
             (protocol, subject_id, nickname, first_seen, last_seen,
              seen_count, is_current, last_event_key)
             VALUES (?1, ?2, ?3, ?4, ?4, 1, 0, ?5)
             ON CONFLICT(protocol, subject_id, nickname) DO UPDATE SET
                first_seen = MIN(persona_nicknames.first_seen, excluded.first_seen),
                last_seen = MAX(persona_nicknames.last_seen, excluded.last_seen),
                seen_count = persona_nicknames.seen_count + 1,
                last_event_key = excluded.last_event_key",
            params![protocol, subject_id, nickname, observed_at, event_key],
        )
        .map_err(|error| error.to_string())?;

    let current_last_seen = transaction
        .query_row(
            "SELECT last_seen FROM persona_nicknames
             WHERE protocol = ?1 AND subject_id = ?2 AND is_current = 1
             ORDER BY last_seen DESC LIMIT 1",
            params![protocol, subject_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    if current_last_seen.is_some_and(|last_seen| observed_at < last_seen) {
        return Ok(None);
    }

    transaction
        .execute(
            "UPDATE persona_nicknames SET is_current = 0
             WHERE protocol = ?1 AND subject_id = ?2 AND is_current <> 0",
            params![protocol, subject_id],
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "UPDATE persona_nicknames SET is_current = 1
             WHERE protocol = ?1 AND subject_id = ?2 AND nickname = ?3",
            params![protocol, subject_id, nickname],
        )
        .map_err(|error| error.to_string())?;
    Ok(Some(nickname))
}

fn record_topic(
    transaction: &Transaction<'_>,
    protocol: &str,
    subject_id: &str,
    topic: &str,
    observed_at: i64,
) -> Result<(), String> {
    let existing = transaction
        .query_row(
            "SELECT score, last_seen FROM persona_topics
             WHERE protocol = ?1 AND subject_id = ?2 AND topic = ?3",
            params![protocol, subject_id, topic],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(|error| error.to_string())?;
    let score = existing
        .map(|(score, last_seen)| {
            let age_days = observed_at.saturating_sub(last_seen).max(0) as f64 / MILLIS_PER_DAY;
            (score * 0.5_f64.powf(age_days / TOPIC_DECAY_HALF_LIFE_DAYS) + 1.0).clamp(0.0, 100.0)
        })
        .unwrap_or(1.0);
    transaction
        .execute(
            "INSERT INTO persona_topics
             (protocol, subject_id, topic, count, score, first_seen, last_seen)
             VALUES (?1, ?2, ?3, 1, ?4, ?5, ?5)
             ON CONFLICT(protocol, subject_id, topic) DO UPDATE SET
                count = persona_topics.count + 1,
                score = excluded.score,
                first_seen = MIN(persona_topics.first_seen, excluded.first_seen),
                last_seen = MAX(persona_topics.last_seen, excluded.last_seen)",
            params![protocol, subject_id, topic, score, observed_at],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn prune_nicknames(
    transaction: &Transaction<'_>,
    protocol: &str,
    subject_id: &str,
) -> Result<(), String> {
    transaction
        .execute(
            "DELETE FROM persona_nicknames
             WHERE rowid IN (
                SELECT rowid FROM persona_nicknames
                WHERE protocol = ?1 AND subject_id = ?2 AND is_current = 0
                ORDER BY seen_count DESC, last_seen DESC, nickname ASC
                LIMIT -1 OFFSET ?3
             )",
            params![
                protocol,
                subject_id,
                MAX_NICKNAMES_PER_PERSONA.saturating_sub(1) as i64
            ],
        )
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn prune_topics(
    transaction: &Transaction<'_>,
    protocol: &str,
    subject_id: &str,
) -> Result<(), String> {
    transaction
        .execute(
            "DELETE FROM persona_topics
             WHERE rowid IN (
                SELECT rowid FROM persona_topics
                WHERE protocol = ?1 AND subject_id = ?2
                ORDER BY score DESC, count DESC, last_seen DESC, topic ASC
                LIMIT -1 OFFSET ?3
             )",
            params![protocol, subject_id, MAX_TOPICS_PER_PERSONA as i64],
        )
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn load_aliases(
    connection: &Connection,
    protocol: &str,
    subject_id: &str,
) -> Result<Vec<String>, rusqlite::Error> {
    let mut statement = connection.prepare(
        "SELECT nickname FROM persona_nicknames
         WHERE protocol = ?1 AND subject_id = ?2 AND is_current = 0
         ORDER BY seen_count DESC, last_seen DESC, nickname ASC LIMIT 3",
    )?;
    statement
        .query_map(params![protocol, subject_id], |row| row.get(0))?
        .collect()
}

fn load_top_topics(
    connection: &Connection,
    protocol: &str,
    subject_id: &str,
) -> Result<Vec<Value>, rusqlite::Error> {
    let mut statement = connection.prepare(
        "SELECT topic, count, score FROM persona_topics
         WHERE protocol = ?1 AND subject_id = ?2
         ORDER BY score DESC, count DESC, last_seen DESC, topic ASC LIMIT 8",
    )?;
    statement
        .query_map(params![protocol, subject_id], |row| {
            Ok(json!({
                "topic": row.get::<_, String>(0)?,
                "count": row.get::<_, i64>(1)?,
                "score": (row.get::<_, f64>(2)? * 100.0).round() / 100.0,
            }))
        })?
        .collect()
}

fn relation_for(intimacy: f64) -> &'static str {
    if intimacy >= 65.0 {
        "friend"
    } else if intimacy >= 25.0 {
        "familiar"
    } else {
        "stranger"
    }
}

fn intimacy_signal(msg: &InMessage) -> f64 {
    if msg.session_type == "private" {
        85.0
    } else if msg.at_me && !msg.reply_to_id.is_empty() {
        80.0
    } else if msg.at_me {
        70.0
    } else if !msg.reply_to_id.is_empty() {
        60.0
    } else if msg.content.trim().ends_with(['?', '？']) {
        30.0
    } else {
        20.0
    }
}

fn intimacy_alpha(msg: &InMessage) -> f64 {
    if msg.session_type == "private" || msg.at_me || !msg.reply_to_id.is_empty() {
        0.12
    } else {
        0.04
    }
}

fn update_intimacy(
    previous: f64,
    previous_at: i64,
    observed_at: i64,
    signal: f64,
    alpha: f64,
    first_observation: bool,
) -> f64 {
    if !first_observation && observed_at < previous_at {
        return previous.clamp(0.0, 100.0);
    }
    let age_days = observed_at.saturating_sub(previous_at).max(0) as f64 / MILLIS_PER_DAY;
    let decayed =
        previous.clamp(0.0, 100.0) * 0.5_f64.powf(age_days / INTIMACY_DECAY_HALF_LIFE_DAYS);
    let effective_alpha = if first_observation { 0.25 } else { alpha };
    (decayed + effective_alpha * (signal.clamp(0.0, 100.0) - decayed)).clamp(0.0, 100.0)
}

fn trait_keys(msg: &InMessage) -> Vec<String> {
    let mut keys = Vec::new();
    if msg.at_me {
        keys.push("direct_mentions".to_string());
    }
    if msg.has_media {
        keys.push("media_sender".to_string());
    }
    if msg.content.trim().ends_with(['?', '？']) {
        keys.push("questioning".to_string());
    }
    if msg.content.contains('!') || msg.content.contains('！') {
        keys.push("expressive".to_string());
    }
    if msg.content.chars().count() >= 80 {
        keys.push("detailed".to_string());
    }
    keys
}

fn preference_keys(content: &str) -> Vec<String> {
    if super::candidates::contains_sensitive_content(content)
        || content.contains("http://")
        || content.contains("https://")
    {
        return Vec::new();
    }
    [
        ("不喜欢", "dislikes"),
        ("喜欢", "likes"),
        ("爱", "likes"),
        ("讨厌", "dislikes"),
    ]
    .iter()
    .filter_map(|(marker, category)| {
        let start = content.find(marker)? + marker.len();
        let fragment = clean_fragment(&content[start..]);
        (!fragment.is_empty()).then(|| format!("{category}:{fragment}"))
    })
    .collect()
}

fn topic_keys(content: &str) -> Vec<String> {
    const KNOWN: &[&str] = &[
        "游戏",
        "学习",
        "工作",
        "代码",
        "猫",
        "狗",
        "音乐",
        "电影",
        "天气",
        "旅行",
        "吃饭",
        "考试",
        "机器人",
    ];
    const ASCII_TOPICS: &[&str] = &[
        "ai",
        "anthropic",
        "docker",
        "java",
        "javascript",
        "linux",
        "llm",
        "onebot",
        "openai",
        "python",
        "qimenbot",
        "qqbot",
        "rust",
        "sqlite",
        "typescript",
        "windows",
    ];

    let mut topics = KNOWN
        .iter()
        .filter(|topic| content.contains(**topic))
        .map(|topic| (*topic).to_string())
        .collect::<Vec<_>>();
    let mut ascii_word = String::new();
    let flush = |word: &mut String, topics: &mut Vec<String>| {
        if ASCII_TOPICS.contains(&word.as_str()) {
            topics.push(std::mem::take(word));
        } else {
            word.clear();
        }
    };
    for character in content.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            ascii_word.push(character.to_ascii_lowercase());
        } else {
            flush(&mut ascii_word, &mut topics);
        }
    }
    flush(&mut ascii_word, &mut topics);
    topics.sort();
    topics.dedup();
    topics.truncate(32);
    topics
}

fn clean_fragment(fragment: &str) -> String {
    fragment
        .trim_matches(|character: char| {
            character.is_whitespace() || "，。！？!?；;：:,、".contains(character)
        })
        .chars()
        .filter(|character| !character.is_control())
        .take(24)
        .collect()
}

fn sanitize_nickname(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(|character| !character.is_control())
        .take(64)
        .collect()
}

fn increment_counts(source: &str, keys: &[String], max_entries: usize) -> String {
    let mut object = serde_json::from_str::<Map<String, Value>>(source).unwrap_or_default();
    for key in keys {
        let count = object
            .get(key)
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .saturating_add(1);
        object.insert(key.clone(), Value::from(count));
    }
    if object.len() > max_entries {
        let mut entries = object.into_iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            right
                .1
                .as_i64()
                .unwrap_or(0)
                .cmp(&left.1.as_i64().unwrap_or(0))
                .then_with(|| left.0.cmp(&right.0))
        });
        entries.truncate(max_entries);
        object = entries.into_iter().collect();
    }
    serde_json::to_string(&object).unwrap_or_else(|_| "{}".to_string())
}

fn top_count_entries(source: &str, limit: usize) -> Value {
    let object = serde_json::from_str::<Map<String, Value>>(source).unwrap_or_default();
    let mut entries = object.into_iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .1
            .as_i64()
            .unwrap_or(0)
            .cmp(&left.1.as_i64().unwrap_or(0))
            .then_with(|| left.0.cmp(&right.0))
    });
    entries.truncate(limit);
    Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (prompt_safe_text(&key), value))
            .collect(),
    )
}

fn prompt_safe_text(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '<' => '＜',
            '>' => '＞',
            _ => character,
        })
        .collect()
}

fn observation_key(msg: &InMessage) -> String {
    if !msg.event_key.trim().is_empty() {
        return msg.event_key.clone();
    }
    let mut hasher = Sha256::new();
    for value in [
        msg.protocol.as_str(),
        msg.session_id.as_str(),
        msg.sender_id.as_str(),
        msg.message_id.as_str(),
    ] {
        hasher.update(value.as_bytes());
        hasher.update([0]);
    }
    hasher.update(msg.timestamp.to_le_bytes());
    let digest = hasher.finalize();
    format!(
        "persona:{}",
        digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(
        event_key: &str,
        protocol: &str,
        nickname: &str,
        content: &str,
        timestamp: i64,
    ) -> InMessage {
        InMessage {
            event_key: event_key.to_string(),
            protocol: protocol.to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            sender_id: "same-id".to_string(),
            sender_name: nickname.to_string(),
            message_id: event_key.to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: true,
            timestamp,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[test]
    fn profile_counters_preserve_existing_values_and_are_bounded() {
        let mut keys = vec!["questioning".to_string(), "expressive".to_string()];
        keys.extend((0..100).map(|index| format!("topic-{index}")));
        let updated = increment_counts(r#"{"questioning":2,"likes:cats":1}"#, &keys, 64);
        let value: Map<String, Value> = serde_json::from_str(&updated).expect("valid JSON");
        assert_eq!(value["questioning"], 3);
        assert!(value.len() <= 64);
    }

    #[test]
    fn preference_and_topic_extraction_is_bounded() {
        let preferences = preference_keys("我喜欢猫和音乐");
        assert_eq!(preferences, vec!["likes:猫和音乐".to_string()]);
        assert!(topic_keys("最近在玩游戏和写Rust代码").contains(&"游戏".to_string()));
        assert!(topic_keys("API token sk-secret-value").is_empty());
        assert!(preference_keys("我喜欢 api key sk-secret").is_empty());
        assert!(clean_fragment(" a very long fragment ").chars().count() <= 24);
        assert_eq!(
            prompt_safe_text("</speaker_profile>"),
            "＜/speaker_profile＞"
        );
    }

    #[test]
    fn intimacy_ewma_is_bounded_and_decays_after_inactivity() {
        let first = update_intimacy(0.0, 0, 1_000, 85.0, 0.12, true);
        let repeated = update_intimacy(first, 1_000, 2_000, 85.0, 0.12, false);
        let much_later = update_intimacy(
            80.0,
            1_000,
            1_000 + (180.0 * MILLIS_PER_DAY) as i64,
            20.0,
            0.04,
            false,
        );
        assert!(first > 0.0 && repeated > first);
        assert!((0.0..=100.0).contains(&much_later));
        assert!(much_later < 80.0);
        assert_eq!(update_intimacy(42.0, 2_000, 1_000, 85.0, 0.12, false), 42.0);
    }

    #[test]
    fn observations_are_idempotent_and_keep_protocol_scoped_history() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let first = message("event-1", "onebot11", "Alice", "聊游戏和Rust代码", 1_000);
        let second = message("event-2", "onebot11", "Alicia", "最近听音乐", 2_000);
        let official = message("event-1", "qq-official", "Official Alice", "聊电影", 3_000);

        assert!(observe_from(&database, &first).unwrap());
        assert!(!observe_from(&database, &first).unwrap());
        assert!(observe_from(&database, &second).unwrap());
        assert!(observe_from(&database, &official).unwrap());

        let connection = database.conn.lock().unwrap();
        let identities: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM personas WHERE subject_id = 'same-id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(identities, 2);
        let onebot: (String, i64, f64) = connection
            .query_row(
                "SELECT nickname, interaction_count, intimacy_ewma FROM personas
                 WHERE protocol = 'onebot11' AND subject_id = 'same-id'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(onebot.0, "Alicia");
        assert_eq!(onebot.1, 2);
        assert!(onebot.2 > 0.0);
        let nickname_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM persona_nicknames
                 WHERE protocol = 'onebot11' AND subject_id = 'same-id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(nickname_count, 2);
        let game_count: i64 = connection
            .query_row(
                "SELECT count FROM persona_topics
                 WHERE protocol = 'onebot11' AND subject_id = 'same-id' AND topic = '游戏'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(game_count, 1);
    }
}
