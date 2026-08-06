//! Sticker co-occurrence links.

/// Return related sticker IDs ordered by co-occurrence count.
pub async fn find_links(sticker_id: i64) -> Vec<i64> {
    let Some(database) = crate::pipeline::try_db() else {
        return Vec::new();
    };
    let Ok(connection) = database.conn.lock() else {
        return Vec::new();
    };
    let Ok(mut statement) = connection.prepare(
        "SELECT CASE WHEN sticker_a = ?1 THEN sticker_b ELSE sticker_a END
         FROM sticker_links
         WHERE sticker_a = ?1 OR sticker_b = ?1
         ORDER BY co_count DESC, updated_at DESC
         LIMIT 20",
    ) else {
        return Vec::new();
    };
    statement
        .query_map(rusqlite::params![sticker_id], |row| row.get::<_, i64>(0))
        .map(|rows| rows.filter_map(Result::ok).collect())
        .unwrap_or_default()
}

/// Record an undirected co-occurrence edge for stickers seen in one message.
pub async fn record_cooccurrence(sticker_ids: &[i64]) {
    if sticker_ids.len() < 2 {
        return;
    }
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let Ok(connection) = database.conn.lock() else {
        return;
    };
    let now = chrono::Utc::now().timestamp_millis();
    for (index, &left) in sticker_ids.iter().enumerate() {
        for &right in &sticker_ids[index + 1..] {
            if left == right {
                continue;
            }
            let (sticker_a, sticker_b) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            let _ = connection.execute(
                "INSERT INTO sticker_links (sticker_a, sticker_b, co_count, updated_at)
                 VALUES (?1, ?2, 1, ?3)
                 ON CONFLICT(sticker_a, sticker_b) DO UPDATE SET
                    co_count = co_count + 1, updated_at = excluded.updated_at",
                rusqlite::params![sticker_a, sticker_b, now],
            );
        }
    }
}
