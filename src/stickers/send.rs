//! Sticker candidate search and deterministic send policy.
use std::collections::HashSet;

/// Return the best URL matching a keyword.
pub async fn send_sticker(session_id: &str, keyword: &str) -> Option<String> {
    choose_chain(session_id, keyword, 1)
        .await
        .into_iter()
        .next()
        .map(|(_, url)| url)
}

/// Choose a bounded chain from the best candidate and its co-occurrence links.
pub async fn choose_chain(session_id: &str, keyword: &str, max_chain: u32) -> Vec<(i64, String)> {
    let Some((first_id, first_url)) = find_best(session_id, keyword).await else {
        return Vec::new();
    };
    let limit = max_chain.clamp(1, 10) as usize;
    let mut selected = vec![(first_id, first_url)];
    let mut seen = HashSet::from([first_id]);
    let mut frontier = vec![first_id];

    while selected.len() < limit && !frontier.is_empty() {
        let mut next_frontier = Vec::new();
        for sticker_id in frontier {
            for related_id in crate::stickers::link::find_links(sticker_id).await {
                if selected.len() >= limit || !seen.insert(related_id) {
                    continue;
                }
                if let Some(url) = url_for_id(related_id).await {
                    selected.push((related_id, url));
                    next_frontier.push(related_id);
                }
            }
        }
        frontier = next_frontier;
    }
    selected
}

/// Deterministic probability gate keyed by the source event.
pub fn should_send(event_key: &str, probability: f32) -> bool {
    probability >= 1.0
        || (probability > 0.0
            && crate::stickers::collect::deterministic_sample(event_key)
                < probability.clamp(0.0, 1.0))
}

async fn find_best(session_id: &str, keyword: &str) -> Option<(i64, String)> {
    let database = crate::pipeline::try_db()?;
    let connection = database.conn.lock().ok()?;
    let keyword = escape_like(keyword.trim());
    let pattern = if keyword.is_empty() {
        "%".to_string()
    } else {
        format!("%{keyword}%")
    };
    connection
        .query_row(
            "SELECT s.id, s.media_url
             FROM stickers s
             LEFT JOIN sticker_tags t ON t.sticker_id = s.id
             WHERE s.media_url LIKE ?1 ESCAPE '\\' OR t.tag LIKE ?1 ESCAPE '\\'
             GROUP BY s.id
             ORDER BY (s.source_session = ?2) DESC,
                      s.usage_count DESC,
                      COALESCE(s.last_used, s.created_at) DESC
             LIMIT 1",
            rusqlite::params![pattern, session_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .ok()
}

async fn url_for_id(sticker_id: i64) -> Option<String> {
    let database = crate::pipeline::try_db()?;
    let connection = database.conn.lock().ok()?;
    let url = connection
        .query_row(
            "SELECT media_url FROM stickers WHERE id = ?1",
            rusqlite::params![sticker_id],
            |row| row.get::<_, String>(0),
        )
        .ok()?;
    let now = chrono::Utc::now().timestamp_millis();
    let _ = connection.execute(
        "UPDATE stickers SET usage_count = usage_count + 1, last_used = ?1 WHERE id = ?2",
        rusqlite::params![now, sticker_id],
    );
    url.starts_with("https://").then_some(url)
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
}
