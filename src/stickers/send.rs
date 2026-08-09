//! 表情包候选排序、投递反馈与保守的平台能力闸门。

use std::collections::HashSet;

const DAY_MILLIS: i64 = 86_400_000;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SendPolicy {
    pub(crate) max_chain: u32,
    pub(crate) daily_send_limit: u32,
    pub(crate) cooldown_sec: u64,
}

impl SendPolicy {
    fn legacy(max_chain: u32) -> Self {
        Self {
            max_chain,
            daily_send_limit: u32::MAX,
            cooldown_sec: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StickerCandidate {
    pub(crate) sticker_id: i64,
    pub(crate) url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UrlImageCapability {
    Supported,
    Unsupported,
}

/// 宿主已验证的 HTTPS URL 图片发送能力矩阵。
pub(crate) fn url_image_capability(protocol: &str) -> UrlImageCapability {
    match protocol {
        "onebot11" | "qq-official" => UrlImageCapability::Supported,
        _ => UrlImageCapability::Unsupported,
    }
}

/// 以来源事件为键的确定性概率闸门。
pub fn should_send(event_key: &str, probability: f32) -> bool {
    probability >= 1.0
        || (probability > 0.0
            && crate::stickers::collect::deterministic_sample(event_key)
                < probability.clamp(0.0, 1.0))
}

/// 返回与关键词匹配的最佳旧版 OneBot URL。
pub async fn send_sticker(session_id: &str, keyword: &str) -> Option<String> {
    choose_chain(session_id, keyword, 1)
        .await
        .into_iter()
        .next()
        .map(|(_, url)| url)
}

/// 保留原有仅按会话选择的 OneBot 兼容入口。
pub async fn choose_chain(session_id: &str, keyword: &str, max_chain: u32) -> Vec<(i64, String)> {
    choose_chain_for_route(
        "onebot11",
        "group",
        session_id,
        keyword,
        SendPolicy::legacy(max_chain),
    )
    .await
    .into_iter()
    .map(|candidate| (candidate.sticker_id, candidate.url))
    .collect()
}

/// 仅在路由的图片能力已验证时，选择长度受限的候选链。
pub(crate) async fn choose_chain_for_route(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    keyword: &str,
    policy: SendPolicy,
) -> Vec<StickerCandidate> {
    if url_image_capability(protocol) != UrlImageCapability::Supported {
        return Vec::new();
    }
    let Some(database) = crate::pipeline::try_db() else {
        return Vec::new();
    };
    choose_chain_in_database(
        &database,
        protocol,
        session_type,
        session_id,
        keyword,
        policy,
        chrono::Utc::now().timestamp_millis(),
    )
}

/// 记录宿主已接受的投递；拒绝和无效结果已保留在出站审计历史中。
pub(crate) fn record_accepted_delivery(sticker_id: i64, protocol: &str) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let Ok(connection) = database.conn.lock() else {
        return;
    };
    let now = chrono::Utc::now().timestamp_millis();
    let _ = connection.execute(
        "UPDATE stickers
         SET usage_count = COALESCE(usage_count, 0) + 1, last_used = ?1, updated_at = ?1
         WHERE id = ?2 AND protocol = ?3 AND url_requires_cache = 0",
        rusqlite::params![now, sticker_id, protocol],
    );
}

fn choose_chain_in_database(
    database: &crate::db::Database,
    protocol: &str,
    session_type: &str,
    session_id: &str,
    keyword: &str,
    policy: SendPolicy,
    now: i64,
) -> Vec<StickerCandidate> {
    let Ok(connection) = database.conn.lock() else {
        return Vec::new();
    };
    let capacity =
        route_send_capacity(&connection, protocol, session_type, session_id, policy, now)
            .unwrap_or(0);
    if capacity == 0 {
        return Vec::new();
    }
    let Some(first) =
        find_best_in_connection(&connection, protocol, session_type, session_id, keyword)
    else {
        return Vec::new();
    };

    let mut selected = vec![first.clone()];
    let mut seen = HashSet::from([first.sticker_id]);
    let mut frontier = vec![first.sticker_id];
    while selected.len() < capacity && !frontier.is_empty() {
        let mut next_frontier = Vec::new();
        for sticker_id in frontier {
            for related_id in crate::stickers::link::find_links_in_connection(
                &connection,
                sticker_id,
                now,
                crate::stickers::link::MAX_LINKS_PER_NODE,
            ) {
                if selected.len() >= capacity || !seen.insert(related_id) {
                    continue;
                }
                if let Some(candidate) = candidate_for_id(&connection, related_id, protocol) {
                    next_frontier.push(candidate.sticker_id);
                    selected.push(candidate);
                }
            }
        }
        frontier = next_frontier;
    }
    selected
}

fn route_send_capacity(
    connection: &rusqlite::Connection,
    protocol: &str,
    session_type: &str,
    session_id: &str,
    policy: SendPolicy,
    now: i64,
) -> Option<usize> {
    if policy.cooldown_sec > 0 {
        let last_sent: Option<i64> = connection
            .query_row(
                "SELECT MAX(created_at) FROM outbound_messages
                 WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
                   AND media_type = 'image' AND status = 'accepted'",
                rusqlite::params![protocol, session_type, session_id],
                |row| row.get(0),
            )
            .ok()?;
        let cooldown_millis = policy.cooldown_sec.saturating_mul(1_000) as i64;
        if last_sent.is_some_and(|last| now.saturating_sub(last) < cooldown_millis) {
            return Some(0);
        }
    }

    let day_start = now.div_euclid(DAY_MILLIS).saturating_mul(DAY_MILLIS);
    let sent_today: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM outbound_messages
             WHERE protocol = ?1 AND session_type = ?2 AND session_id = ?3
               AND media_type = 'image' AND status = 'accepted'
               AND created_at >= ?4 AND created_at <= ?5",
            rusqlite::params![protocol, session_type, session_id, day_start, now],
            |row| row.get(0),
        )
        .ok()?;
    let remaining = i64::from(policy.daily_send_limit.max(1)).saturating_sub(sent_today);
    if remaining <= 0 {
        return Some(0);
    }
    Some(
        policy
            .max_chain
            .clamp(1, 3)
            .min(remaining.min(i64::from(u32::MAX)) as u32) as usize,
    )
}

fn find_best_in_connection(
    connection: &rusqlite::Connection,
    protocol: &str,
    session_type: &str,
    session_id: &str,
    keyword: &str,
) -> Option<StickerCandidate> {
    let keyword = escape_like(keyword.trim());
    let pattern = if keyword.is_empty() {
        "%".to_string()
    } else {
        format!("%{keyword}%")
    };
    connection
        .query_row(
            "SELECT s.id, s.media_url
             FROM stickers AS s
             LEFT JOIN sticker_tags AS tag ON tag.sticker_id = s.id
             WHERE s.protocol = ?1 AND s.url_requires_cache = 0
               AND s.media_url LIKE 'https://%'
               AND (s.media_url LIKE ?4 ESCAPE '\\' OR tag.tag LIKE ?4 ESCAPE '\\')
             GROUP BY s.id
             ORDER BY
                (CASE WHEN s.media_url LIKE ?4 ESCAPE '\\' THEN 2 ELSE 0 END
                 + MAX(CASE WHEN tag.tag LIKE ?4 ESCAPE '\\' THEN 1 ELSE 0 END)) DESC,
                (s.source_session = ?3) DESC,
                COALESCE((
                    SELECT SUM(CASE feedback.status
                        WHEN 'accepted' THEN 2
                        WHEN 'rejected' THEN -2
                        WHEN 'invalid' THEN -3
                        ELSE 0
                    END)
                    FROM outbound_messages AS feedback
                    WHERE feedback.protocol = ?1
                      AND feedback.session_type = ?2
                      AND feedback.session_id = ?3
                      AND feedback.media_type = 'image'
                      AND feedback.media_url = s.media_url
                ), 0) DESC,
                COALESCE(s.last_used, 0) ASC,
                s.id ASC
             LIMIT 1",
            rusqlite::params![protocol, session_type, session_id, pattern],
            |row| {
                Ok(StickerCandidate {
                    sticker_id: row.get(0)?,
                    url: row.get(1)?,
                })
            },
        )
        .ok()
}

fn candidate_for_id(
    connection: &rusqlite::Connection,
    sticker_id: i64,
    protocol: &str,
) -> Option<StickerCandidate> {
    connection
        .query_row(
            "SELECT id, media_url FROM stickers
             WHERE id = ?1 AND protocol = ?2 AND url_requires_cache = 0
               AND media_url LIKE 'https://%'",
            rusqlite::params![sticker_id, protocol],
            |row| {
                Ok(StickerCandidate {
                    sticker_id: row.get(0)?,
                    url: row.get(1)?,
                })
            },
        )
        .ok()
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SendPolicy {
        SendPolicy {
            max_chain: 3,
            daily_send_limit: 30,
            cooldown_sec: 60,
        }
    }

    fn insert_sticker(
        database: &crate::db::Database,
        url: &str,
        tag: &str,
        source_session: &str,
    ) -> i64 {
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_session, usage_count, created_at, updated_at)
                 VALUES ('onebot11', ?1, ?1, 0, 'remote', ?2, 0, 1, 1)",
                rusqlite::params![url, source_session],
            )
            .unwrap();
        let id = connection.last_insert_rowid();
        connection
            .execute(
                "INSERT INTO sticker_tags (sticker_id, tag, weight) VALUES (?1, ?2, 1)",
                rusqlite::params![id, tag],
            )
            .unwrap();
        id
    }

    fn insert_feedback(database: &crate::db::Database, url: &str, status: &str, created_at: i64) {
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO outbound_messages
                 (action_key, protocol, session_type, session_id, content, media_type,
                  media_url, status, attempt_count, created_at, updated_at)
                 VALUES (?1, 'onebot11', 'group', 'group-1', '', 'image', ?2, ?3, 1, ?4, ?4)",
                rusqlite::params![
                    format!("feedback-{status}-{created_at}-{url}"),
                    url,
                    status,
                    created_at
                ],
            )
            .unwrap();
    }

    #[test]
    fn send_gate_is_deterministic() {
        assert!(should_send("event", 1.0));
        assert!(!should_send("event", 0.0));
        assert_eq!(should_send("event", 0.3), should_send("event", 0.3));
    }

    #[test]
    fn like_pattern_escapes_wildcards() {
        assert_eq!(escape_like("a_%"), "a\\_\\%");
    }

    #[test]
    fn capability_matrix_allows_verified_url_media_routes() {
        assert_eq!(
            url_image_capability("onebot11"),
            UrlImageCapability::Supported
        );
        assert_eq!(
            url_image_capability("qq-official"),
            UrlImageCapability::Supported
        );
        assert_eq!(
            url_image_capability("unknown"),
            UrlImageCapability::Unsupported
        );
    }

    #[test]
    fn ranking_prefers_route_tag_and_positive_delivery_feedback() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let generic = insert_sticker(
            &database,
            "https://example.test/generic.png",
            "开心",
            "other-group",
        );
        let preferred = insert_sticker(
            &database,
            "https://example.test/preferred.png",
            "开心",
            "group-1",
        );
        insert_feedback(
            &database,
            "https://example.test/preferred.png",
            "accepted",
            DAY_MILLIS * 10,
        );
        let candidates = choose_chain_in_database(
            &database,
            "onebot11",
            "group",
            "group-1",
            "开心",
            SendPolicy {
                cooldown_sec: 0,
                ..policy()
            },
            DAY_MILLIS * 11,
        );
        assert_eq!(
            candidates.first().map(|candidate| candidate.sticker_id),
            Some(preferred)
        );
        assert_ne!(
            candidates.first().map(|candidate| candidate.sticker_id),
            Some(generic)
        );
    }

    #[test]
    fn ranking_uses_positive_and_negative_delivery_feedback() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let rejected = insert_sticker(
            &database,
            "https://example.test/rejected.png",
            "happy",
            "group-1",
        );
        let accepted = insert_sticker(
            &database,
            "https://example.test/accepted.png",
            "happy",
            "group-1",
        );
        insert_feedback(
            &database,
            "https://example.test/rejected.png",
            "invalid",
            DAY_MILLIS * 10,
        );
        insert_feedback(
            &database,
            "https://example.test/accepted.png",
            "accepted",
            DAY_MILLIS * 10,
        );

        let candidates = choose_chain_in_database(
            &database,
            "onebot11",
            "group",
            "group-1",
            "happy",
            SendPolicy {
                cooldown_sec: 0,
                ..policy()
            },
            DAY_MILLIS * 11,
        );
        assert_eq!(
            candidates.first().map(|candidate| candidate.sticker_id),
            Some(accepted)
        );
        assert_ne!(
            candidates.first().map(|candidate| candidate.sticker_id),
            Some(rejected)
        );
    }

    #[test]
    fn cooldown_and_daily_limit_block_new_candidates() {
        let database = crate::db::Database::open(":memory:").unwrap();
        insert_sticker(
            &database,
            "https://example.test/happy.png",
            "开心",
            "group-1",
        );
        let now = DAY_MILLIS * 30 + 10_000;
        insert_feedback(
            &database,
            "https://example.test/previous.png",
            "accepted",
            now - 1_000,
        );
        assert!(
            choose_chain_in_database(
                &database,
                "onebot11",
                "group",
                "group-1",
                "开心",
                policy(),
                now,
            )
            .is_empty()
        );

        let daily_limited = SendPolicy {
            cooldown_sec: 0,
            daily_send_limit: 1,
            ..policy()
        };
        assert!(
            choose_chain_in_database(
                &database,
                "onebot11",
                "group",
                "group-1",
                "开心",
                daily_limited,
                now + 61_000,
            )
            .is_empty()
        );
    }

    #[test]
    fn weighted_links_cap_chain_length_and_stop_cycles() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let first = insert_sticker(
            &database,
            "https://example.test/first.png",
            "seed",
            "group-1",
        );
        let second = insert_sticker(
            &database,
            "https://example.test/second.png",
            "next",
            "group-1",
        );
        let third = insert_sticker(
            &database,
            "https://example.test/third.png",
            "next",
            "group-1",
        );
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO sticker_links
                 (sticker_a, sticker_b, edge_kind, co_count, weight, evidence_count, updated_at)
                 VALUES (?1, ?2, 'cooccur', 2, 2.0, 2, 1),
                        (?2, ?3, 'cooccur', 2, 2.0, 2, 1),
                        (?1, ?3, 'cooccur', 2, 2.0, 2, 1)",
                rusqlite::params![first, second, third],
            )
            .unwrap();
        drop(connection);

        let candidates = choose_chain_in_database(
            &database,
            "onebot11",
            "group",
            "group-1",
            "seed",
            SendPolicy {
                max_chain: 10,
                cooldown_sec: 0,
                ..policy()
            },
            DAY_MILLIS * 5,
        );
        let ids: Vec<_> = candidates
            .iter()
            .map(|candidate| candidate.sticker_id)
            .collect();
        let unique_count = ids.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(ids.first(), Some(&first));
        assert_eq!(ids.len(), 3);
        assert_eq!(unique_count, ids.len());
    }

    #[test]
    fn signed_url_reference_is_not_selected_and_cached_public_url_stays_sendable() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_session, usage_count, created_at, updated_at)
                 VALUES ('onebot11', 'https://example.test/signed.png', 'signed', 1,
                         'required', 'group-1', 100, 1, 1),
                        ('onebot11', 'https://example.test/public.png', 'public', 0,
                         'remote', 'group-1', 1, 1, 1),
                        ('onebot11', 'https://example.test/cached.png', 'cached', 0,
                         'cached', 'group-1', 1, 1, 1)",
                [],
            )
            .unwrap();
        drop(connection);

        let candidates = choose_chain_in_database(
            &database,
            "onebot11",
            "group",
            "group-1",
            "cached",
            SendPolicy {
                cooldown_sec: 0,
                ..policy()
            },
            DAY_MILLIS * 5,
        );
        assert_eq!(
            candidates,
            vec![StickerCandidate {
                sticker_id: 3,
                url: "https://example.test/cached.png".to_string(),
            }]
        );
        assert!(candidate_for_id(&database.conn.lock().unwrap(), 1, "onebot11").is_none());
    }
}
