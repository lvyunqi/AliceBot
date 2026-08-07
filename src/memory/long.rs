//! 长期记忆：持久化重要事实，并提供确定性的相关性检索。
use crate::pipeline::db;
use sha2::{Digest, Sha256};

/// 按重要性和最近访问时间获取长期记忆。
pub async fn retrieve_topk(session_id: &str, k: usize) -> Vec<String> {
    retrieve_relevant(session_id, None, "", k).await
}

/// 使用词法重叠、重要性、访问次数和新鲜度排序长期记忆。
///
/// 这是没有向量数据库时的可解释降级算法：英文按词匹配，中文按字符
/// 匹配，再用重要性和时间作为平局排序。以后接入 embedding 时可以保留
/// 这个路径作为离线或故障回退。
pub async fn retrieve_relevant(
    session_id: &str,
    subject_id: Option<&str>,
    query: &str,
    k: usize,
) -> Vec<String> {
    let database = db();
    retrieve_from(&database, session_id, subject_id, query, k)
}

fn retrieve_from(
    database: &crate::db::Database,
    session_id: &str,
    subject_id: Option<&str>,
    query: &str,
    k: usize,
) -> Vec<String> {
    if k == 0 {
        return Vec::new();
    }

    let now = chrono::Utc::now().timestamp_millis();
    let candidates = {
        let Ok(connection) = database.conn.lock() else {
            return Vec::new();
        };
        let candidate_limit = (k.saturating_mul(8)).clamp(k, 200) as i64;
        let mut statement = match connection.prepare(
            "SELECT id, content, importance, confidence, access_count, created_at
             FROM long_memory
             WHERE is_active = 1 AND status = 'active'
               AND (session_id = ?1 OR session_id IS NULL)
               AND (subject_id IS NULL OR subject_id = ?2)
             ORDER BY importance DESC, COALESCE(updated_at, created_at) DESC
             LIMIT ?3",
        ) {
            Ok(statement) => statement,
            Err(error) => {
                log::warn!("[AliceBot] 长期记忆检索失败: {error}");
                return Vec::new();
            }
        };
        let rows = match statement.query_map(
            rusqlite::params![session_id, subject_id, candidate_limit],
            |row| {
                Ok(MemoryCandidate {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    importance: row.get(2)?,
                    confidence: row.get(3)?,
                    access_count: row.get(4)?,
                    created_at: row.get(5)?,
                })
            },
        ) {
            Ok(rows) => rows,
            Err(error) => {
                log::warn!("[AliceBot] 长期记忆读取失败: {error}");
                return Vec::new();
            }
        };
        rows.filter_map(Result::ok).collect::<Vec<_>>()
    };

    let query_terms = terms(query);
    let mut ranked = candidates
        .into_iter()
        .map(|candidate| {
            let overlap = query_terms
                .iter()
                .filter(|term| candidate.content.contains(term.as_str()))
                .count();
            let age_days =
                (now.saturating_sub(candidate.created_at) as f64 / (86_400_000_f64)).max(0.0);
            let freshness = (20.0 - age_days).max(0.0);
            let score = overlap as i64 * 1_000
                + i64::from(candidate.importance) * 10
                + i64::from(candidate.confidence) * 5
                + i64::from(candidate.access_count.min(100))
                + freshness.round() as i64;
            (score, candidate)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| right.1.created_at.cmp(&left.1.created_at))
    });

    let selected = ranked.into_iter().take(k).collect::<Vec<_>>();
    if let Ok(connection) = database.conn.lock() {
        for (_, candidate) in &selected {
            if let Err(error) = connection.execute(
                "UPDATE long_memory SET access_count = access_count + 1, last_access = ?1
                 WHERE id = ?2",
                rusqlite::params![now, candidate.id],
            ) {
                log::debug!("[AliceBot] 更新长期记忆访问次数失败: {error}");
            }
        }
    }
    selected
        .into_iter()
        .map(|(_, candidate)| candidate.content)
        .collect()
}

/// 写入一条长期记忆。
pub async fn store(content: &str, session_id: Option<&str>, importance: i32) -> Result<(), String> {
    if content.trim().is_empty() {
        return Ok(());
    }
    let now = chrono::Utc::now().timestamp_millis();
    let database = db();
    let mut connection = database
        .conn
        .lock()
        .map_err(|_| "长期记忆数据库锁失败".to_string())?;
    let transaction = connection
        .transaction()
        .map_err(|error| error.to_string())?;
    let scope = if session_id.is_some() {
        "session"
    } else {
        "global"
    };
    let mut hasher = Sha256::new();
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(session_id.unwrap_or_default().as_bytes());
    hasher.update([0]);
    hasher.update(content.trim().as_bytes());
    let normalized_key = format!("manual:{}", hex_prefix(&hasher.finalize()));
    transaction
        .execute(
            "INSERT INTO long_memory
             (normalized_key, scope, session_id, content, kind, importance, confidence,
              privacy, status, version, is_active, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 'manual', ?5, 80, 'normal', 'active', 1, 1, ?6, ?6)
             ON CONFLICT(normalized_key, version) DO UPDATE SET
                importance = MAX(long_memory.importance, excluded.importance),
                updated_at = excluded.updated_at",
            rusqlite::params![
                normalized_key,
                scope,
                session_id,
                content.trim(),
                importance.clamp(0, 100),
                now
            ],
        )
        .map_err(|error| error.to_string())?;
    let memory_id: i64 = transaction
        .query_row(
            "SELECT id FROM long_memory WHERE normalized_key = ?1 AND version = 1",
            rusqlite::params![normalized_key],
            |row| row.get(0),
        )
        .map_err(|error| error.to_string())?;
    transaction
        .execute(
            "INSERT OR IGNORE INTO memory_sources
             (memory_id, source_type, source_id, evidence_weight, created_at)
             VALUES (?1, 'manual', ?2, 1, ?3)",
            rusqlite::params![memory_id, normalized_key, now],
        )
        .map_err(|error| error.to_string())?;
    transaction.commit().map_err(|error| error.to_string())?;
    Ok(())
}

struct MemoryCandidate {
    id: i64,
    content: String,
    importance: i32,
    confidence: i32,
    access_count: i32,
    created_at: i64,
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn terms(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut ascii_word = String::new();
    for character in text.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            ascii_word.push(character.to_ascii_lowercase());
            continue;
        }
        if !ascii_word.is_empty() {
            if ascii_word.len() >= 2 {
                result.push(std::mem::take(&mut ascii_word));
            } else {
                ascii_word.clear();
            }
        }
        if !character.is_ascii_whitespace() && !character.is_ascii_punctuation() {
            result.push(character.to_string());
        }
    }
    if !ascii_word.is_empty() && ascii_word.len() >= 2 {
        result.push(ascii_word);
    }
    result.sort();
    result.dedup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieval_excludes_candidates_forgotten_rows_and_other_subjects() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let connection = database.conn.lock().unwrap();
        connection
            .execute_batch(
                "INSERT INTO long_memory
                 (normalized_key, scope, session_id, subject_id, content, kind,
                  importance, confidence, privacy, status, version, is_active,
                  created_at, updated_at)
                 VALUES
                 ('candidate', 'user_session', 'group-1', 'user-1', '喜欢：候选',
                  'preference', 90, 80, 'normal', 'candidate', 1, 0, 10, 10),
                 ('active-own', 'user_session', 'group-1', 'user-1', '喜欢：咖啡',
                  'preference', 70, 90, 'normal', 'active', 1, 1, 11, 11),
                 ('active-other', 'user_session', 'group-1', 'user-2', '喜欢：茶',
                  'preference', 70, 90, 'normal', 'active', 1, 1, 12, 12),
                 ('forgotten', 'user_session', 'group-1', 'user-1', '喜欢：旧事',
                  'preference', 90, 90, 'normal', 'forgotten', 1, 0, 13, 13),
                 ('global', 'global', NULL, NULL, '群公告：周五开会',
                  'fact', 60, 80, 'normal', 'active', 1, 1, 14, 14);",
            )
            .unwrap();
        drop(connection);

        let memories = retrieve_from(&database, "group-1", Some("user-1"), "咖啡", 10);
        assert!(memories.contains(&"喜欢：咖啡".to_string()));
        assert!(memories.contains(&"群公告：周五开会".to_string()));
        assert!(!memories.contains(&"喜欢：候选".to_string()));
        assert!(!memories.contains(&"喜欢：茶".to_string()));
        assert!(!memories.contains(&"喜欢：旧事".to_string()));
    }
}
