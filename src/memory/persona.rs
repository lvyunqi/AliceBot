//! Incremental user profiles for recognition and cautious personalization.
use serde_json::{Map, Value};

use crate::pipeline::InMessage;

#[derive(Debug, Clone)]
pub struct Persona {
    pub subject_id: String,
    pub protocol: String,
    pub nickname: String,
    pub first_seen: i64,
    pub last_seen: i64,
    pub interaction_count: i32,
    pub intimacy: i32,
    pub relation: String,
    pub traits: String,
    pub preferences: String,
    pub topics: String,
    pub notes: String,
}

/// Observe one message without replacing previously learned profile fields.
pub async fn observe(msg: &InMessage) {
    let Some(database) = crate::pipeline::try_db() else {
        log::warn!("[AliceBot] persona observation skipped: database is not ready");
        return;
    };
    let now = chrono::Utc::now().timestamp_millis();
    let existing = load(&database, &msg.sender_id);
    let mut persona = existing.unwrap_or_else(|| Persona {
        subject_id: msg.sender_id.clone(),
        protocol: msg.protocol.clone(),
        nickname: String::new(),
        first_seen: now,
        last_seen: now,
        interaction_count: 0,
        intimacy: 0,
        relation: "stranger".to_string(),
        traits: "{}".to_string(),
        preferences: "{}".to_string(),
        topics: "{}".to_string(),
        notes: String::new(),
    });

    persona.protocol = msg.protocol.clone();
    if persona.first_seen == 0 {
        persona.first_seen = now;
    }
    if !msg.sender_name.trim().is_empty() {
        persona.nickname = msg.sender_name.clone();
    }
    persona.last_seen = now;
    persona.interaction_count = persona.interaction_count.saturating_add(1);
    let intimacy_delta = if msg.at_me {
        2
    } else if msg.session_type == "private" {
        1
    } else {
        0
    };
    persona.intimacy = persona
        .intimacy
        .saturating_add(intimacy_delta)
        .clamp(0, 100);
    persona.relation = relation_for(persona.intimacy).to_string();
    persona.traits = increment_counts(&persona.traits, &trait_keys(msg));
    persona.preferences = increment_counts(&persona.preferences, &preference_keys(&msg.content));
    persona.topics = increment_counts(&persona.topics, &topic_keys(&msg.content));

    if let Err(error) = database.upsert_persona(&persona) {
        log::error!("[AliceBot] persona update failed: {error}");
    }
}

/// Return a bounded, non-authoritative profile summary for the current prompt.
pub fn summary(subject_id: &str) -> Option<String> {
    let database = crate::pipeline::try_db()?;
    let persona = load(&database, subject_id)?;
    if persona.interaction_count == 0 {
        return None;
    }
    Some(format!(
        "nickname={}, relation={}, interactions={}, intimacy={}, traits={}, preferences={}, topics={}",
        limit_text(&persona.nickname, 40),
        persona.relation,
        persona.interaction_count,
        persona.intimacy,
        limit_text(&persona.traits, 240),
        limit_text(&persona.preferences, 240),
        limit_text(&persona.topics, 240),
    ))
}

fn load(database: &crate::db::Database, subject_id: &str) -> Option<Persona> {
    let connection = database.conn.lock().ok()?;
    connection
        .query_row(
            "SELECT subject_id, protocol, nickname, first_seen, last_seen,
                    interaction_count, intimacy, relation, traits, preferences, topics, notes
             FROM personas WHERE subject_id = ?1",
            rusqlite::params![subject_id],
            |row| {
                Ok(Persona {
                    subject_id: row.get(0)?,
                    protocol: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    nickname: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    first_seen: row.get::<_, Option<i64>>(3)?.unwrap_or_default(),
                    last_seen: row.get::<_, Option<i64>>(4)?.unwrap_or_default(),
                    interaction_count: row.get::<_, Option<i32>>(5)?.unwrap_or_default(),
                    intimacy: row.get::<_, Option<i32>>(6)?.unwrap_or_default(),
                    relation: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
                    traits: row
                        .get::<_, Option<String>>(8)?
                        .unwrap_or_else(|| "{}".to_string()),
                    preferences: row
                        .get::<_, Option<String>>(9)?
                        .unwrap_or_else(|| "{}".to_string()),
                    topics: row
                        .get::<_, Option<String>>(10)?
                        .unwrap_or_else(|| "{}".to_string()),
                    notes: row.get::<_, Option<String>>(11)?.unwrap_or_default(),
                })
            },
        )
        .ok()
}

fn relation_for(intimacy: i32) -> &'static str {
    match intimacy {
        70..=100 => "friend",
        30..=69 => "familiar",
        _ => "stranger",
    }
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
    let known = [
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
    let mut topics = known
        .iter()
        .filter(|topic| content.contains(**topic))
        .map(|topic| (*topic).to_string())
        .collect::<Vec<_>>();
    let mut ascii_word = String::new();
    for character in content.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            ascii_word.push(character.to_ascii_lowercase());
        } else if ascii_word.len() >= 3 {
            topics.push(std::mem::take(&mut ascii_word));
        } else {
            ascii_word.clear();
        }
    }
    if ascii_word.len() >= 3 {
        topics.push(ascii_word);
    }
    topics.sort();
    topics.dedup();
    topics
}

fn clean_fragment(fragment: &str) -> String {
    fragment
        .trim_matches(|character: char| {
            character.is_whitespace() || "，。！？!?；;：:,、".contains(character)
        })
        .chars()
        .take(24)
        .collect()
}

fn increment_counts(source: &str, keys: &[String]) -> String {
    let mut object = serde_json::from_str::<Map<String, Value>>(source).unwrap_or_default();
    for key in keys {
        let count = object
            .get(key)
            .and_then(Value::as_i64)
            .unwrap_or(0)
            .saturating_add(1);
        object.insert(key.clone(), Value::from(count));
    }
    serde_json::to_string(&object).unwrap_or_else(|_| "{}".to_string())
}

fn limit_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_counters_preserve_existing_values() {
        let updated = increment_counts(
            r#"{"questioning":2,"likes:cats":1}"#,
            &["questioning".to_string(), "expressive".to_string()],
        );
        let value: Map<String, Value> = serde_json::from_str(&updated).expect("valid JSON");
        assert_eq!(value["questioning"], 3);
        assert_eq!(value["likes:cats"], 1);
        assert_eq!(value["expressive"], 1);
    }

    #[test]
    fn preference_and_topic_extraction_is_bounded() {
        let preferences = preference_keys("我喜欢猫和音乐");
        assert_eq!(preferences, vec!["likes:猫和音乐".to_string()]);
        assert!(topic_keys("最近在玩游戏和写Rust代码").contains(&"游戏".to_string()));
        assert!(clean_fragment(&" a very long fragment ").chars().count() <= 24);
    }
}
