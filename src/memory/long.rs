//! 长期记忆：持久化重要事实，并提供确定性的相关性检索。
use crate::pipeline::db;

/// 按重要性和最近访问时间获取长期记忆。
pub async fn retrieve_topk(session_id: &str, k: usize) -> Vec<String> {
    retrieve_relevant(session_id, "", k).await
}

/// 使用词法重叠、重要性、访问次数和新鲜度排序长期记忆。
///
/// 这是没有向量数据库时的可解释降级算法：英文按词匹配，中文按字符
/// 匹配，再用重要性和时间作为平局排序。以后接入 embedding 时可以保留
/// 这个路径作为离线或故障回退。
pub async fn retrieve_relevant(session_id: &str, query: &str, k: usize) -> Vec<String> {
    if k == 0 {
        return Vec::new();
    }

    let database = db();
    let now = chrono::Utc::now().timestamp_millis();
    let candidates = {
        let Ok(connection) = database.conn.lock() else {
            return Vec::new();
        };
        let candidate_limit = (k.saturating_mul(8)).clamp(k, 200) as i64;
        let mut statement = match connection.prepare(
            "SELECT id, content, importance, access_count, created_at
             FROM long_memory
             WHERE is_active = 1 AND (session_id = ?1 OR session_id IS NULL)
             ORDER BY importance DESC, COALESCE(updated_at, created_at) DESC
             LIMIT ?2",
        ) {
            Ok(statement) => statement,
            Err(error) => {
                log::warn!("[AliceBot] 长期记忆检索失败: {error}");
                return Vec::new();
            }
        };
        let rows =
            match statement.query_map(rusqlite::params![session_id, candidate_limit], |row| {
                Ok(MemoryCandidate {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    importance: row.get(2)?,
                    access_count: row.get(3)?,
                    created_at: row.get(4)?,
                })
            }) {
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
    let connection = database
        .conn
        .lock()
        .map_err(|_| "长期记忆数据库锁失败".to_string())?;
    connection
        .execute(
            "INSERT INTO long_memory (session_id, content, importance, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            rusqlite::params![session_id, content.trim(), importance.clamp(0, 100), now],
        )
        .map_err(|error| error.to_string())?;
    Ok(())
}

struct MemoryCandidate {
    id: i64,
    content: String,
    importance: i32,
    access_count: i32,
    created_at: i64,
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
