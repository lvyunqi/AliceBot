//! 表情包加权共现图。

use rusqlite::{Connection, OptionalExtension, TransactionBehavior};
use std::cmp::Ordering;

const DAY_MILLIS: i64 = 86_400_000;
/// 共现边每经过一个半衰期，历史权重衰减为原来的一半。
pub(crate) const EDGE_HALF_LIFE_MILLIS: i64 = 14 * DAY_MILLIS;
pub(crate) const MAX_LINKS_PER_NODE: usize = 20;

#[derive(Debug, Clone)]
struct WeightedLink {
    sticker_id: i64,
    score: f64,
    evidence_count: i64,
    co_count: i64,
    updated_at: i64,
}

/// 计算指定时间点的边权，避免旧边因历史累计次数永久压过新边。
pub(crate) fn decayed_weight(weight: f64, updated_at: i64, now: i64) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let age = now.saturating_sub(updated_at).max(0) as f64;
    weight * 0.5_f64.powf(age / EDGE_HALF_LIFE_MILLIS as f64)
}

/// 返回按衰减权重排序的邻居；调用方负责在当前链中去重并检测环。
pub(crate) fn find_links_in_connection(
    connection: &Connection,
    sticker_id: i64,
    now: i64,
    limit: usize,
) -> Vec<i64> {
    let Ok(mut statement) = connection.prepare(
        "SELECT CASE WHEN sticker_a = ?1 THEN sticker_b ELSE sticker_a END,
                COALESCE(weight, CAST(co_count AS REAL)),
                COALESCE(evidence_count, co_count, 1),
                COALESCE(co_count, 0),
                COALESCE(updated_at, 0)
         FROM sticker_links
         WHERE (sticker_a = ?1 OR sticker_b = ?1)
           AND COALESCE(edge_kind, 'cooccur') = 'cooccur'",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = statement.query_map(rusqlite::params![sticker_id], |row| {
        let weight: f64 = row.get(1)?;
        let evidence_count: i64 = row.get(2)?;
        let co_count: i64 = row.get(3)?;
        let updated_at: i64 = row.get(4)?;
        Ok(WeightedLink {
            sticker_id: row.get(0)?,
            score: decayed_weight(weight, updated_at, now),
            evidence_count,
            co_count,
            updated_at,
        })
    }) else {
        return Vec::new();
    };

    let mut links: Vec<_> = rows.filter_map(Result::ok).collect();
    links.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.evidence_count.cmp(&left.evidence_count))
            .then_with(|| right.co_count.cmp(&left.co_count))
            .then_with(|| right.updated_at.cmp(&left.updated_at))
            .then_with(|| left.sticker_id.cmp(&right.sticker_id))
    });
    links
        .into_iter()
        .take(limit.min(MAX_LINKS_PER_NODE))
        .map(|link| link.sticker_id)
        .collect()
}

/// 返回按衰减共现权重排序的关联表情包 ID。
pub async fn find_links(sticker_id: i64) -> Vec<i64> {
    let Some(database) = crate::pipeline::try_db() else {
        return Vec::new();
    };
    let Ok(connection) = database.conn.lock() else {
        return Vec::new();
    };
    find_links_in_connection(
        &connection,
        sticker_id,
        chrono::Utc::now().timestamp_millis(),
        MAX_LINKS_PER_NODE,
    )
}

/// 将一个消息中的不同表情包记录为一次无向共现，重复事件只增长一次证据。
pub async fn record_cooccurrence(sticker_ids: &[i64]) {
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let Ok(mut connection) = database.conn.lock() else {
        return;
    };
    if let Err(error) = record_cooccurrence_in_connection(
        &mut connection,
        sticker_ids,
        chrono::Utc::now().timestamp_millis(),
    ) {
        log::warn!("[AliceBot] 表情包共现边更新失败: {error}");
    }
}

/// 在事务中更新共现边，供运行时和 SQLite 回归测试复用。
pub(crate) fn record_cooccurrence_in_connection(
    connection: &mut Connection,
    sticker_ids: &[i64],
    now: i64,
) -> Result<(), rusqlite::Error> {
    let mut unique_ids: Vec<i64> = sticker_ids
        .iter()
        .copied()
        .filter(|sticker_id| *sticker_id > 0)
        .collect();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    if unique_ids.len() < 2 {
        return Ok(());
    }

    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    for (index, &left) in unique_ids.iter().enumerate() {
        for &right in &unique_ids[index + 1..] {
            let (sticker_a, sticker_b) = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            let existing: Option<(f64, i64, i64, i64)> = transaction
                .query_row(
                    "SELECT COALESCE(weight, CAST(co_count AS REAL)),
                            COALESCE(evidence_count, co_count, 1),
                            COALESCE(co_count, 0), COALESCE(updated_at, 0)
                     FROM sticker_links
                     WHERE sticker_a = ?1 AND sticker_b = ?2",
                    rusqlite::params![sticker_a, sticker_b],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()?;
            let (old_weight, old_evidence, old_count, old_updated_at) =
                existing.unwrap_or((0.0, 0, 0, now));
            let next_weight = decayed_weight(old_weight, old_updated_at, now) + 1.0;
            let next_evidence = old_evidence.saturating_add(1).max(1);
            let next_count = old_count.saturating_add(1).max(1);
            transaction.execute(
                "INSERT INTO sticker_links
                    (sticker_a, sticker_b, edge_kind, co_count, weight, evidence_count, updated_at)
                 VALUES (?1, ?2, 'cooccur', ?3, ?4, ?5, ?6)
                 ON CONFLICT(sticker_a, sticker_b) DO UPDATE SET
                    co_count = excluded.co_count,
                    weight = excluded.weight,
                    evidence_count = excluded.evidence_count,
                    updated_at = excluded.updated_at",
                rusqlite::params![
                    sticker_a,
                    sticker_b,
                    next_count,
                    next_weight,
                    next_evidence,
                    now
                ],
            )?;
        }
    }
    transaction.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooccurrence_deduplicates_one_round_and_accumulates_weight() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let mut connection = database.conn.lock().unwrap();
        record_cooccurrence_in_connection(&mut connection, &[3, 1, 2, 1, 2], 1_000).unwrap();

        let first: (i64, i64, f64) = connection
            .query_row(
                "SELECT co_count, evidence_count, weight
                 FROM sticker_links WHERE sticker_a = 1 AND sticker_b = 2",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(first.0, 1);
        assert_eq!(first.1, 1);
        assert!((first.2 - 1.0).abs() < f64::EPSILON);

        record_cooccurrence_in_connection(&mut connection, &[2, 1], 2_000).unwrap();
        let second: (i64, i64, f64, i64) = connection
            .query_row(
                "SELECT co_count, evidence_count, weight, updated_at
                 FROM sticker_links WHERE sticker_a = 1 AND sticker_b = 2",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(second.0, 2);
        assert_eq!(second.1, 2);
        assert!(second.2 > 1.9);
        assert_eq!(second.3, 2_000);
    }

    #[test]
    fn newer_edges_outrank_old_edges_after_decay() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let connection = database.conn.lock().unwrap();
        let now = EDGE_HALF_LIFE_MILLIS * 4;
        connection
            .execute(
                "INSERT INTO sticker_links
                 (sticker_a, sticker_b, edge_kind, co_count, weight, evidence_count, updated_at)
                 VALUES (1, 2, 'cooccur', 10, 10.0, 10, 0),
                        (1, 3, 'cooccur', 1, 2.0, 1, ?1)",
                rusqlite::params![now],
            )
            .unwrap();
        assert_eq!(
            find_links_in_connection(&connection, 1, now, 20),
            vec![3, 2]
        );
    }
}
