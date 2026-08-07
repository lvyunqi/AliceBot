//! Long-term memory persistence and deterministic hybrid retrieval.

use crate::memory::search::SearchBackend;
use crate::pipeline::db;
use rusqlite::{Connection, params_from_iter, types::Value};
use sha2::{Digest, Sha256};

const MMR_LAMBDA: f64 = 0.75;
const MEMORY_HALF_LIFE_DAYS: f64 = 180.0;
const ACCESS_HALF_LIFE_DAYS: f64 = 30.0;
const MILLIS_PER_DAY: f64 = 86_400_000.0;
const MAX_CANDIDATES: usize = 200;
const LEGACY_PROTOCOL: &str = "legacy";
const LEGACY_SESSION_TYPE: &str = "legacy";

#[derive(Clone, Copy)]
struct MemoryRoute<'a> {
    protocol: &'a str,
    session_type: &'a str,
    session_id: &'a str,
}

const STRUCTURED_FILTER: &str = r#"
    lm.is_active = 1
    AND lm.status = 'active'
    AND (
        (lm.scope = 'global' AND lm.session_id IS NULL AND lm.subject_id IS NULL
            AND (lm.protocol = '*' OR lm.protocol = ?1))
        OR (lm.scope = 'session' AND lm.protocol = ?1
            AND lm.session_type = ?2 AND lm.session_id = ?3
            AND lm.subject_id IS NULL)
        OR (lm.scope = 'user' AND lm.protocol = ?1
            AND (lm.session_type = '*' OR lm.session_type = ?2)
            AND lm.session_id IS NULL AND lm.subject_id = ?4)
        OR (lm.scope = 'user_session' AND lm.protocol = ?1
            AND lm.session_type = ?2 AND lm.session_id = ?3
            AND lm.subject_id = ?4)
    )
    AND (
        lm.privacy = 'normal'
        OR (lm.privacy = 'session_only' AND lm.session_id = ?3
            AND lm.protocol = ?1 AND lm.session_type = ?2)
    )
"#;

/// Retrieve high-value memories when there is no current query text.
pub async fn retrieve_topk(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    k: usize,
) -> Vec<String> {
    retrieve_relevant(protocol, session_type, session_id, None, "", k).await
}

/// Retrieve relevant memories through FTS5/BM25 with a lexical fallback and MMR diversity.
pub async fn retrieve_relevant(
    protocol: &str,
    session_type: &str,
    session_id: &str,
    subject_id: Option<&str>,
    query: &str,
    k: usize,
) -> Vec<String> {
    let database = db();
    retrieve_from(
        &database,
        protocol,
        session_type,
        session_id,
        subject_id,
        query,
        k,
    )
}

fn retrieve_from(
    database: &crate::db::Database,
    protocol: &str,
    session_type: &str,
    session_id: &str,
    subject_id: Option<&str>,
    query: &str,
    k: usize,
) -> Vec<String> {
    retrieve_from_at(
        database,
        MemoryRoute {
            protocol,
            session_type,
            session_id,
        },
        subject_id,
        query,
        k,
        chrono::Utc::now().timestamp_millis(),
    )
}

fn retrieve_from_at(
    database: &crate::db::Database,
    route: MemoryRoute<'_>,
    subject_id: Option<&str>,
    query: &str,
    k: usize,
    now: i64,
) -> Vec<String> {
    if k == 0 {
        return Vec::new();
    }

    let query_terms = terms(query);
    let candidate_limit = k.saturating_mul(12).clamp(k, MAX_CANDIDATES) as i64;
    let loaded = {
        let Ok(connection) = database.conn.lock() else {
            return Vec::new();
        };
        load_candidates(
            &connection,
            database.memory_search,
            route,
            subject_id,
            query,
            &query_terms,
            candidate_limit,
        )
    };
    let (candidates, source) = match loaded {
        Ok(result) => result,
        Err(error) => {
            log::warn!("[AliceBot] long-memory retrieval failed: {error}");
            return Vec::new();
        }
    };

    let ranked = rank_candidates(candidates, source, &query_terms, now);
    let selected = select_mmr(ranked, k);
    if let Ok(connection) = database.conn.lock() {
        for candidate in &selected {
            if let Err(error) = connection.execute(
                "UPDATE long_memory
                 SET access_count = access_count + 1, last_access = ?1
                 WHERE id = ?2",
                rusqlite::params![now, candidate.id],
            ) {
                log::debug!("[AliceBot] failed to update memory access statistics: {error}");
            }
        }
    }

    selected
        .into_iter()
        .map(|candidate| candidate.content)
        .collect()
}

fn load_candidates(
    connection: &Connection,
    backend: SearchBackend,
    route: MemoryRoute<'_>,
    subject_id: Option<&str>,
    query: &str,
    query_terms: &[String],
    limit: i64,
) -> Result<(Vec<MemoryCandidate>, RetrievalSource), rusqlite::Error> {
    if backend == SearchBackend::Fts5
        && let Some(match_query) = crate::memory::search::match_query(query)
    {
        match load_fts_candidates(connection, route, subject_id, &match_query, limit) {
            Ok(candidates) if !candidates.is_empty() => {
                return Ok((candidates, RetrievalSource::Fts5));
            }
            Ok(_) => {}
            Err(error) => {
                log::debug!("[AliceBot] FTS5 query failed; using lexical fallback: {error}");
            }
        }
    }

    let mut candidates =
        load_lexical_candidates(connection, route, subject_id, query_terms, limit)?;
    if candidates.is_empty() && !query_terms.is_empty() {
        candidates = load_lexical_candidates(connection, route, subject_id, &[], limit)?;
    }
    Ok((candidates, RetrievalSource::Lexical))
}

fn load_fts_candidates(
    connection: &Connection,
    route: MemoryRoute<'_>,
    subject_id: Option<&str>,
    match_query: &str,
    limit: i64,
) -> Result<Vec<MemoryCandidate>, rusqlite::Error> {
    let sql = format!(
        "SELECT lm.id, lm.content, lm.scope, lm.importance, lm.confidence,
                lm.access_count, lm.last_access, lm.created_at, lm.updated_at,
                COALESCE((
                    SELECT SUM(ms.evidence_weight)
                    FROM memory_sources AS ms
                    WHERE ms.memory_id = lm.id
                ), 0),
                bm25(memory_fts)
         FROM memory_fts
         JOIN long_memory AS lm ON lm.id = memory_fts.rowid
         WHERE memory_fts MATCH ?5 AND {STRUCTURED_FILTER}
         ORDER BY bm25(memory_fts) ASC, lm.id ASC
         LIMIT ?6"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(
        rusqlite::params![
            route.protocol,
            route.session_type,
            route.session_id,
            subject_id,
            match_query,
            limit
        ],
        |row| {
            Ok(MemoryCandidate {
                id: row.get(0)?,
                content: row.get(1)?,
                scope: row.get(2)?,
                importance: row.get(3)?,
                confidence: row.get(4)?,
                access_count: row.get(5)?,
                last_access: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
                source_weight: row.get(9)?,
                bm25: row.get(10)?,
                tokens: Vec::new(),
                relevance: 0.0,
                score: 0.0,
            })
        },
    )?;
    rows.collect()
}

fn load_lexical_candidates(
    connection: &Connection,
    route: MemoryRoute<'_>,
    subject_id: Option<&str>,
    query_terms: &[String],
    limit: i64,
) -> Result<Vec<MemoryCandidate>, rusqlite::Error> {
    let mut sql = format!(
        "SELECT lm.id, lm.content, lm.scope, lm.importance, lm.confidence,
                lm.access_count, lm.last_access, lm.created_at, lm.updated_at,
                COALESCE((
                    SELECT SUM(ms.evidence_weight)
                    FROM memory_sources AS ms
                    WHERE ms.memory_id = lm.id
                ), 0)
         FROM long_memory AS lm
         WHERE {STRUCTURED_FILTER}"
    );
    let mut values = vec![
        Value::Text(route.protocol.to_string()),
        Value::Text(route.session_type.to_string()),
        Value::Text(route.session_id.to_string()),
        subject_id
            .map(|subject| Value::Text(subject.to_string()))
            .unwrap_or(Value::Null),
    ];

    if !query_terms.is_empty() {
        let predicates = query_terms
            .iter()
            .enumerate()
            .map(|(index, _)| format!("LOWER(lm.content) LIKE ?{} ESCAPE '\\'", index + 5))
            .collect::<Vec<_>>();
        sql.push_str(" AND (");
        sql.push_str(&predicates.join(" OR "));
        sql.push(')');
        values.extend(
            query_terms
                .iter()
                .map(|term| Value::Text(format!("%{}%", escape_like(term)))),
        );
    }

    let limit_index = values.len() + 1;
    sql.push_str(&format!(
        " ORDER BY lm.importance DESC, lm.confidence DESC,
                   COALESCE(lm.updated_at, lm.created_at) DESC, lm.id ASC
          LIMIT ?{limit_index}"
    ));
    values.push(Value::Integer(limit));

    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(MemoryCandidate {
            id: row.get(0)?,
            content: row.get(1)?,
            scope: row.get(2)?,
            importance: row.get(3)?,
            confidence: row.get(4)?,
            access_count: row.get(5)?,
            last_access: row.get(6)?,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
            source_weight: row.get(9)?,
            bm25: None,
            tokens: Vec::new(),
            relevance: 0.0,
            score: 0.0,
        })
    })?;
    rows.collect()
}

fn rank_candidates(
    mut candidates: Vec<MemoryCandidate>,
    source: RetrievalSource,
    query_terms: &[String],
    now: i64,
) -> Vec<MemoryCandidate> {
    if source == RetrievalSource::Fts5 {
        candidates.sort_by(|left, right| {
            left.bm25
                .unwrap_or(f64::MAX)
                .total_cmp(&right.bm25.unwrap_or(f64::MAX))
                .then_with(|| left.id.cmp(&right.id))
        });
    }

    let bm25_best = candidates
        .iter()
        .filter_map(|candidate| candidate.bm25)
        .reduce(f64::min);
    let bm25_worst = candidates
        .iter()
        .filter_map(|candidate| candidate.bm25)
        .reduce(f64::max);

    for (rank, candidate) in candidates.iter_mut().enumerate() {
        candidate.tokens = terms(&candidate.content);
        let lexical = lexical_relevance(query_terms, &candidate.content, &candidate.tokens);
        candidate.relevance = if source == RetrievalSource::Fts5 {
            let reciprocal_rank = 1.0 / (1.0 + rank as f64 * 0.15);
            let bm25_quality = match (candidate.bm25, bm25_best, bm25_worst) {
                (Some(value), Some(best), Some(worst)) if worst - best > f64::EPSILON => {
                    ((worst - value) / (worst - best)).clamp(0.0, 1.0)
                }
                (Some(_), _, _) => 1.0,
                _ => 0.0,
            };
            (0.65 * reciprocal_rank + 0.15 * bm25_quality + 0.20 * lexical).clamp(0.0, 1.0)
        } else {
            lexical
        };
        candidate.score = retrieval_score(candidate, now);
    }

    candidates.sort_by(compare_base_rank);
    candidates
}

fn retrieval_score(candidate: &MemoryCandidate, now: i64) -> f64 {
    let updated_at = candidate.updated_at.unwrap_or(candidate.created_at);
    let memory_age_days = age_days(now, updated_at);
    let decayed_importance = (f64::from(candidate.importance.clamp(0, 100)) / 100.0)
        * 0.5_f64.powf(memory_age_days / MEMORY_HALF_LIFE_DAYS);
    let confidence = f64::from(candidate.confidence.clamp(0, 100)) / 100.0;
    let scope_match = match candidate.scope.as_str() {
        "user_session" => 1.0,
        "user" => 0.9,
        "session" => 0.8,
        "global" => 0.65,
        _ => 0.0,
    };
    let access_frequency =
        (1.0 + f64::from(candidate.access_count.clamp(0, 100))).ln() / 101.0_f64.ln();
    let access_anchor = candidate.last_access.unwrap_or(updated_at);
    let access_freshness = 0.5_f64.powf(age_days(now, access_anchor) / ACCESS_HALF_LIFE_DAYS);
    let recent_access = 0.5 * access_frequency + 0.5 * access_freshness;
    let source_quality = (candidate.source_weight.max(0) as f64 / 3.0).clamp(0.0, 1.0);

    0.35 * candidate.relevance
        + 0.25 * decayed_importance
        + 0.15 * confidence
        + 0.10 * scope_match
        + 0.10 * recent_access
        + 0.05 * source_quality
}

fn select_mmr(mut remaining: Vec<MemoryCandidate>, k: usize) -> Vec<MemoryCandidate> {
    let mut selected = Vec::with_capacity(k.min(remaining.len()));
    while selected.len() < k && !remaining.is_empty() {
        let mut best_index = 0;
        let mut best_mmr = mmr_score(&remaining[0], &selected);
        for index in 1..remaining.len() {
            let candidate_mmr = mmr_score(&remaining[index], &selected);
            if candidate_mmr
                .total_cmp(&best_mmr)
                .then_with(|| compare_base_rank(&remaining[best_index], &remaining[index]))
                .is_gt()
            {
                best_index = index;
                best_mmr = candidate_mmr;
            }
        }
        selected.push(remaining.remove(best_index));
    }
    selected
}

fn mmr_score(candidate: &MemoryCandidate, selected: &[MemoryCandidate]) -> f64 {
    let redundancy = selected
        .iter()
        .map(|existing| term_similarity(&candidate.tokens, &existing.tokens))
        .reduce(f64::max)
        .unwrap_or(0.0);
    MMR_LAMBDA * candidate.score - (1.0 - MMR_LAMBDA) * redundancy
}

fn compare_base_rank(left: &MemoryCandidate, right: &MemoryCandidate) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| {
            right
                .updated_at
                .unwrap_or(right.created_at)
                .cmp(&left.updated_at.unwrap_or(left.created_at))
        })
        .then_with(|| left.id.cmp(&right.id))
}

fn lexical_relevance(query_terms: &[String], content: &str, content_terms: &[String]) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let lowered = content.to_lowercase();
    let matched = query_terms
        .iter()
        .filter(|term| lowered.contains(term.as_str()))
        .count();
    let coverage = matched as f64 / query_terms.len() as f64;
    (0.8 * coverage + 0.2 * term_similarity(query_terms, content_terms)).clamp(0.0, 1.0)
}

fn term_similarity(left: &[String], right: &[String]) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.iter().filter(|term| right.contains(term)).count();
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn age_days(now: i64, timestamp: i64) -> f64 {
    now.saturating_sub(timestamp).max(0) as f64 / MILLIS_PER_DAY
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Store a manually supplied memory without a route.
///
/// The compatibility entry point uses a legacy route and is intentionally not
/// returned to normal protocol queries. New callers should use
/// [`store_for_route`] so persistent memory has an explicit isolation boundary.
pub async fn store(content: &str, session_id: Option<&str>, importance: i32) -> Result<(), String> {
    store_for_route(
        content,
        LEGACY_PROTOCOL,
        LEGACY_SESSION_TYPE,
        session_id,
        importance,
    )
    .await
}

/// Store a manually supplied long-term memory with an explicit route.
pub async fn store_for_route(
    content: &str,
    protocol: &str,
    session_type: &str,
    session_id: Option<&str>,
    importance: i32,
) -> Result<(), String> {
    let database = db();
    store_in(
        &database,
        content,
        protocol,
        session_type,
        session_id,
        importance,
    )
}

fn store_in(
    database: &crate::db::Database,
    content: &str,
    protocol: &str,
    session_type: &str,
    session_id: Option<&str>,
    importance: i32,
) -> Result<(), String> {
    if content.trim().is_empty() {
        return Ok(());
    }
    let protocol = if protocol.trim().is_empty() {
        LEGACY_PROTOCOL
    } else {
        protocol.trim()
    };
    let session_type = if session_id.is_some() {
        if session_type.trim().is_empty() {
            LEGACY_SESSION_TYPE
        } else {
            session_type.trim()
        }
    } else {
        "*"
    };
    let now = chrono::Utc::now().timestamp_millis();
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
    hasher.update(protocol.as_bytes());
    hasher.update([0]);
    hasher.update(session_type.as_bytes());
    hasher.update([0]);
    hasher.update(scope.as_bytes());
    hasher.update([0]);
    hasher.update(session_id.unwrap_or_default().as_bytes());
    hasher.update([0]);
    hasher.update(content.trim().as_bytes());
    let normalized_key = format!("manual:{}", hex_prefix(&hasher.finalize()));
    transaction
        .execute(
            "INSERT INTO long_memory
             (normalized_key, protocol, session_type, scope, session_id, content, kind, importance, confidence,
              privacy, status, version, is_active, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'manual', ?7, 80, 'normal', 'active', 1, 1, ?8, ?8)
             ON CONFLICT DO UPDATE SET
                importance = MAX(long_memory.importance, excluded.importance),
                updated_at = excluded.updated_at",
            rusqlite::params![
                normalized_key,
                protocol,
                session_type,
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
            "SELECT id FROM long_memory
             WHERE normalized_key = ?1 AND protocol = ?2 AND session_type = ?3
               AND COALESCE(session_id, '') = COALESCE(?4, '') AND version = 1",
            rusqlite::params![normalized_key, protocol, session_type, session_id],
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetrievalSource {
    Fts5,
    Lexical,
}

#[derive(Debug, Clone)]
struct MemoryCandidate {
    id: i64,
    content: String,
    scope: String,
    importance: i32,
    confidence: i32,
    access_count: i32,
    last_access: Option<i64>,
    created_at: i64,
    updated_at: Option<i64>,
    source_weight: i64,
    bm25: Option<f64>,
    tokens: Vec<String>,
    relevance: f64,
    score: f64,
}

fn hex_prefix(bytes: &[u8]) -> String {
    bytes[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn terms(text: &str) -> Vec<String> {
    const MAX_TERMS: usize = 32;

    let mut result = Vec::new();
    let mut ascii_word = String::new();
    let mut unicode_run = Vec::new();

    let flush_ascii = |word: &mut String, result: &mut Vec<String>| {
        if word.len() >= 2 {
            result.push(std::mem::take(word));
        } else {
            word.clear();
        }
    };
    let flush_unicode = |run: &mut Vec<char>, result: &mut Vec<String>| {
        result.extend(run.iter().map(char::to_string));
        result.extend(
            run.windows(2)
                .map(|window| window.iter().collect::<String>()),
        );
        run.clear();
    };

    for character in text.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            flush_unicode(&mut unicode_run, &mut result);
            ascii_word.push(character.to_ascii_lowercase());
        } else if character.is_alphanumeric() {
            flush_ascii(&mut ascii_word, &mut result);
            unicode_run.push(character);
        } else {
            flush_ascii(&mut ascii_word, &mut result);
            flush_unicode(&mut unicode_run, &mut result);
        }
    }
    flush_ascii(&mut ascii_word, &mut result);
    flush_unicode(&mut unicode_run, &mut result);
    result.sort();
    result.dedup();
    result.truncate(MAX_TERMS);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestMemory<'a> {
        key: &'a str,
        scope: &'a str,
        session_id: Option<&'a str>,
        subject_id: Option<&'a str>,
        content: &'a str,
        privacy: &'a str,
        status: &'a str,
        active: bool,
        importance: i32,
        created_at: i64,
    }

    fn insert_memory(database: &crate::db::Database, memory: TestMemory<'_>) {
        insert_memory_for_route(database, memory, "onebot11", "group");
    }

    fn insert_memory_for_route(
        database: &crate::db::Database,
        memory: TestMemory<'_>,
        protocol: &str,
        session_type: &str,
    ) {
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO long_memory
                 (normalized_key, protocol, session_type, scope, session_id, subject_id,
                  content, kind, importance, confidence, privacy, status, version,
                  is_active, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'fact', ?8, 85, ?9, ?10,
                         1, ?11, ?12, ?12)",
                rusqlite::params![
                    memory.key,
                    protocol,
                    session_type,
                    memory.scope,
                    memory.session_id,
                    memory.subject_id,
                    memory.content,
                    memory.importance,
                    memory.privacy,
                    memory.status,
                    i32::from(memory.active),
                    memory.created_at,
                ],
            )
            .unwrap();
    }

    #[test]
    fn retrieval_enforces_state_scope_subject_and_privacy_filters() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let now = chrono::Utc::now().timestamp_millis();
        let rows = [
            TestMemory {
                key: "candidate",
                scope: "user_session",
                session_id: Some("group-1"),
                subject_id: Some("user-1"),
                content: "候选记忆",
                privacy: "normal",
                status: "candidate",
                active: false,
                importance: 90,
                created_at: now,
            },
            TestMemory {
                key: "active-own",
                scope: "user_session",
                session_id: Some("group-1"),
                subject_id: Some("user-1"),
                content: "喜欢咖啡",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 70,
                created_at: now,
            },
            TestMemory {
                key: "active-other",
                scope: "user_session",
                session_id: Some("group-1"),
                subject_id: Some("user-2"),
                content: "喜欢茶",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 70,
                created_at: now,
            },
            TestMemory {
                key: "forgotten",
                scope: "user_session",
                session_id: Some("group-1"),
                subject_id: Some("user-1"),
                content: "已经遗忘",
                privacy: "normal",
                status: "forgotten",
                active: false,
                importance: 90,
                created_at: now,
            },
            TestMemory {
                key: "global",
                scope: "global",
                session_id: None,
                subject_id: None,
                content: "群公告周五开会",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 60,
                created_at: now,
            },
            TestMemory {
                key: "sensitive",
                scope: "user_session",
                session_id: Some("group-1"),
                subject_id: Some("user-1"),
                content: "敏感记忆",
                privacy: "sensitive",
                status: "active",
                active: true,
                importance: 100,
                created_at: now,
            },
            TestMemory {
                key: "session-only",
                scope: "session",
                session_id: Some("group-1"),
                subject_id: None,
                content: "仅本会话可见",
                privacy: "session_only",
                status: "active",
                active: true,
                importance: 65,
                created_at: now,
            },
            TestMemory {
                key: "other-session-only",
                scope: "session",
                session_id: Some("group-2"),
                subject_id: None,
                content: "其他会话私有",
                privacy: "session_only",
                status: "active",
                active: true,
                importance: 100,
                created_at: now,
            },
        ];
        for row in rows {
            insert_memory(&database, row);
        }

        let memories = retrieve_from(
            &database,
            "onebot11",
            "group",
            "group-1",
            Some("user-1"),
            "",
            20,
        );
        assert!(memories.contains(&"喜欢咖啡".to_string()));
        assert!(memories.contains(&"群公告周五开会".to_string()));
        assert!(memories.contains(&"仅本会话可见".to_string()));
        assert!(!memories.contains(&"候选记忆".to_string()));
        assert!(!memories.contains(&"喜欢茶".to_string()));
        assert!(!memories.contains(&"已经遗忘".to_string()));
        assert!(!memories.contains(&"敏感记忆".to_string()));
        assert!(!memories.contains(&"其他会话私有".to_string()));
    }

    #[test]
    fn retrieval_isolates_identical_ids_by_protocol_and_session_type() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let now = chrono::Utc::now().timestamp_millis();
        let rows = [
            ("onebot", "onebot11", "group", "OneBot 群记忆"),
            ("official", "qq-official", "group", "官方 QQ 群记忆"),
            ("private", "onebot11", "private", "OneBot 私聊记忆"),
        ];
        for (key, protocol, session_type, content) in rows {
            insert_memory_for_route(
                &database,
                TestMemory {
                    key,
                    scope: "user_session",
                    session_id: Some("same-session-id"),
                    subject_id: Some("same-user-id"),
                    content,
                    privacy: "normal",
                    status: "active",
                    active: true,
                    importance: 70,
                    created_at: now,
                },
                protocol,
                session_type,
            );
        }

        let onebot_group = retrieve_from(
            &database,
            "onebot11",
            "group",
            "same-session-id",
            Some("same-user-id"),
            "",
            10,
        );
        let official_group = retrieve_from(
            &database,
            "qq-official",
            "group",
            "same-session-id",
            Some("same-user-id"),
            "",
            10,
        );
        let onebot_private = retrieve_from(
            &database,
            "onebot11",
            "private",
            "same-session-id",
            Some("same-user-id"),
            "",
            10,
        );

        assert_eq!(onebot_group, vec!["OneBot 群记忆"]);
        assert_eq!(official_group, vec!["官方 QQ 群记忆"]);
        assert_eq!(onebot_private, vec!["OneBot 私聊记忆"]);
    }

    #[test]
    fn manual_store_keys_include_the_explicit_route() {
        let database = crate::db::Database::open(":memory:").unwrap();
        store_in(
            &database,
            "同一条手工记忆",
            "onebot11",
            "group",
            Some("same-session-id"),
            70,
        )
        .unwrap();
        store_in(
            &database,
            "同一条手工记忆",
            "onebot11",
            "group",
            Some("same-session-id"),
            80,
        )
        .unwrap();
        store_in(
            &database,
            "同一条手工记忆",
            "qq-official",
            "group",
            Some("same-session-id"),
            70,
        )
        .unwrap();

        let connection = database.conn.lock().unwrap();
        let (count, keys, max_importance): (i64, i64, i32) = connection
            .query_row(
                "SELECT COUNT(*), COUNT(DISTINCT normalized_key), MAX(importance)
                 FROM long_memory",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!((count, keys, max_importance), (2, 2, 80));
    }

    #[test]
    fn bm25_retrieval_prefers_matching_memory_when_fts5_is_available() {
        let database = crate::db::Database::open(":memory:").unwrap();
        if database.memory_search != SearchBackend::Fts5 {
            return;
        }
        let now = chrono::Utc::now().timestamp_millis();
        insert_memory(
            &database,
            TestMemory {
                key: "rust",
                scope: "global",
                session_id: None,
                subject_id: None,
                content: "Rust ownership borrowing and memory safety",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 55,
                created_at: now,
            },
        );
        insert_memory(
            &database,
            TestMemory {
                key: "garden",
                scope: "global",
                session_id: None,
                subject_id: None,
                content: "Tomato gardening and soil moisture",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 100,
                created_at: now,
            },
        );

        let memories = retrieve_from(
            &database,
            "onebot11",
            "group",
            "group-1",
            None,
            "ownership borrowing",
            1,
        );
        assert_eq!(memories, vec!["Rust ownership borrowing and memory safety"]);
    }

    #[test]
    fn forced_lexical_backend_retrieves_chinese_substrings() {
        let mut database = crate::db::Database::open(":memory:").unwrap();
        database.memory_search = SearchBackend::Lexical;
        let now = chrono::Utc::now().timestamp_millis();
        insert_memory(
            &database,
            TestMemory {
                key: "coffee",
                scope: "session",
                session_id: Some("group-1"),
                subject_id: None,
                content: "大家约好周末去喝咖啡",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 50,
                created_at: now,
            },
        );
        insert_memory(
            &database,
            TestMemory {
                key: "unrelated",
                scope: "session",
                session_id: Some("group-1"),
                subject_id: None,
                content: "天气预报说明天有雨",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 100,
                created_at: now,
            },
        );

        let memories = retrieve_from(&database, "onebot11", "group", "group-1", None, "咖啡", 1);
        assert_eq!(memories, vec!["大家约好周末去喝咖啡"]);
    }

    #[test]
    fn fts_query_error_falls_back_to_lexical_retrieval() {
        let database = crate::db::Database::open(":memory:").unwrap();
        if database.memory_search != SearchBackend::Fts5 {
            return;
        }
        let now = chrono::Utc::now().timestamp_millis();
        insert_memory(
            &database,
            TestMemory {
                key: "fallback",
                scope: "global",
                session_id: None,
                subject_id: None,
                content: "fallback retrieval remains available",
                privacy: "normal",
                status: "active",
                active: true,
                importance: 50,
                created_at: now,
            },
        );
        database
            .conn
            .lock()
            .unwrap()
            .execute_batch(
                "DROP TRIGGER memory_fts_ai;
                 DROP TRIGGER memory_fts_ad;
                 DROP TRIGGER memory_fts_au;
                 DROP TABLE memory_fts;",
            )
            .unwrap();

        let memories = retrieve_from(
            &database,
            "onebot11",
            "group",
            "group-1",
            None,
            "fallback retrieval",
            1,
        );
        assert_eq!(memories, vec!["fallback retrieval remains available"]);
    }

    #[test]
    fn mmr_prefers_a_diverse_second_memory() {
        let mut candidates = vec![
            ranked_test_candidate(1, "rust ownership borrowing safety", 0.95),
            ranked_test_candidate(2, "rust ownership borrowing rules", 0.93),
            ranked_test_candidate(3, "rust async concurrency runtime", 0.90),
        ];
        candidates.sort_by(compare_base_rank);

        let selected = select_mmr(candidates, 2);
        assert_eq!(
            selected.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn equal_scores_use_stable_id_tie_breaking() {
        let mut candidates = vec![
            ranked_test_candidate(2, "second distinct memory", 0.8),
            ranked_test_candidate(1, "first distinct memory", 0.8),
        ];
        candidates.sort_by(compare_base_rank);

        let selected = select_mmr(candidates, 2);
        assert_eq!(
            selected.iter().map(|item| item.id).collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    fn ranked_test_candidate(id: i64, content: &str, score: f64) -> MemoryCandidate {
        MemoryCandidate {
            id,
            content: content.to_string(),
            scope: "global".to_string(),
            importance: 50,
            confidence: 80,
            access_count: 0,
            last_access: None,
            created_at: 10,
            updated_at: Some(10),
            source_weight: 0,
            bm25: None,
            tokens: terms(content),
            relevance: 0.0,
            score,
        }
    }
}
