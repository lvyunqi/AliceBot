//! 用户画像 — 识人
//!
//! 每条消息后更新。记录昵称、风格、亲密度、话题偏好等。

use crate::pipeline::{InMessage, db};

/// 用户画像
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

/// 观察用户（每条消息后调用）
pub async fn observe(msg: &InMessage) {
    let now = chrono::Utc::now().timestamp_millis();

    // 查询现有画像
    let existing = db()
        .conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT interaction_count, intimacy FROM personas WHERE subject_id = ?1",
            rusqlite::params![msg.sender_id],
            |row| Ok((row.get::<_, i32>(0)?, row.get::<_, i32>(1)?)),
        )
        .ok();

    let (count, intimacy) = existing.unwrap_or((0, 0));

    let intimacy_delta = if msg.at_me { 1 } else { 0 };
    let new_intimacy = (intimacy + intimacy_delta).clamp(0, 100);

    let persona = Persona {
        subject_id: msg.sender_id.clone(),
        protocol: msg.protocol.clone(),
        nickname: msg.sender_name.clone(),
        first_seen: now,
        last_seen: now,
        interaction_count: count + 1,
        intimacy: new_intimacy,
        relation: if new_intimacy > 50 {
            "friend".into()
        } else {
            "stranger".into()
        },
        traits: "{}".into(),
        preferences: "{}".into(),
        topics: "{}".into(),
        notes: String::new(),
    };

    if let Err(e) = db().upsert_persona(&persona) {
        log::error!("[AliceBot] 画像更新失败: {}", e);
    }
}
