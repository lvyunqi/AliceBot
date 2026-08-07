//! Recoverable deterministic conversation compaction.
use crate::config::AppConfig;
use sha2::{Digest, Sha256};

/// Compress new inbound journal rows into per-session long-memory summaries.
/// Raw messages remain available for audit and replay; the cursor only controls
/// which rows are summarized into the retrieval layer.
pub async fn run_if_due(config: &AppConfig) -> Result<usize, String> {
    let database =
        crate::pipeline::try_db().ok_or_else(|| "database is not initialized".to_string())?;
    let cursor = database
        .get_meta("compaction_cursor")
        .map_err(|error| error.to_string())?
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0);

    let (end_cursor, message_count) = {
        let connection = database
            .conn
            .lock()
            .map_err(|_| "database lock failed".to_string())?;
        connection
            .query_row(
                "SELECT COALESCE(MAX(id), 0), COUNT(*) FROM messages
                 WHERE id > ?1 AND direction = 'inbound'",
                rusqlite::params![cursor],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .map_err(|error| error.to_string())?
    };

    if message_count == 0 || message_count < config.memories.compress_min_messages as i64 {
        return Ok(0);
    }

    let run_key = format!("messages:{cursor}:{end_cursor}");
    let started_at = chrono::Utc::now().timestamp_millis();
    if !database
        .begin_compaction_run(&run_key, cursor, end_cursor, started_at)
        .map_err(|error| error.to_string())?
    {
        return Ok(0);
    }

    let batches = match load_batches(&database, cursor, end_cursor) {
        Ok(batches) => batches,
        Err(error) => {
            finish_failed(&database, &run_key, &error);
            return Err(error);
        }
    };

    let processed_count = batches.iter().map(|batch| batch.count).sum::<i64>();
    let finished_at = chrono::Utc::now().timestamp_millis();
    let transaction_result = (|| -> Result<(), String> {
        let mut connection = database
            .conn
            .lock()
            .map_err(|_| "database lock failed".to_string())?;
        let transaction = connection
            .transaction()
            .map_err(|error| error.to_string())?;
        for batch in &batches {
            let samples = if batch.samples.is_empty() {
                "(无文本消息)".to_string()
            } else {
                batch.samples.join(" | ")
            };
            let summary = format!(
                "会话摘要：本批包含 {} 条消息，时间范围 {} 到 {}。代表性片段：{}",
                batch.count, batch.first_at, batch.last_at, samples
            );
            let normalized_key = summary_key(&run_key, &batch.session_id);
            transaction
                .execute(
                    "INSERT INTO long_memory
                     (normalized_key, scope, session_id, content, kind, importance,
                      confidence, privacy, status, version, is_active, created_at, updated_at)
                     VALUES (?1, 'session', ?2, ?3, 'conversation_summary', ?4,
                             80, 'normal', 'active', 1, 1, ?5, ?5)",
                    rusqlite::params![
                        normalized_key,
                        batch.session_id,
                        summary,
                        i32::try_from(config.memories.importance_threshold)
                            .unwrap_or(30)
                            .clamp(40, 100),
                        finished_at
                    ],
                )
                .map_err(|error| error.to_string())?;
            let memory_id = transaction.last_insert_rowid();
            transaction
                .execute(
                    "INSERT INTO memory_sources
                     (memory_id, source_type, source_id, evidence_weight, created_at)
                     VALUES (?1, 'reflection', ?2, 1, ?3)",
                    rusqlite::params![memory_id, run_key, finished_at],
                )
                .map_err(|error| error.to_string())?;
        }
        let insights = serde_json::json!({
            "cursor_start": cursor,
            "cursor_end": end_cursor,
            "sessions": batches.len(),
            "messages": processed_count,
        })
        .to_string();
        transaction
            .execute(
                "INSERT INTO reflection_log (triggered_by, summary, insights, created_at)
                 VALUES ('scheduled_compaction', ?1, ?2, ?3)",
                rusqlite::params![
                    format!(
                        "压缩 {} 条消息，生成 {} 个会话摘要",
                        processed_count,
                        batches.len()
                    ),
                    insights,
                    finished_at
                ],
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('compaction_cursor', ?1)",
                rusqlite::params![end_cursor.to_string()],
            )
            .map_err(|error| error.to_string())?;
        transaction
            .execute(
                "UPDATE compaction_runs
                 SET status = 'completed', processed_count = ?1, finished_at = ?2
                 WHERE run_key = ?3",
                rusqlite::params![processed_count, finished_at, run_key],
            )
            .map_err(|error| error.to_string())?;
        transaction.commit().map_err(|error| error.to_string())
    })();

    if let Err(error) = transaction_result {
        finish_failed(&database, &run_key, &error);
        return Err(error);
    }
    log::info!(
        "[AliceBot] compaction completed: messages={}, sessions={}, cursor={}",
        processed_count,
        batches.len(),
        end_cursor
    );
    Ok(processed_count as usize)
}

fn summary_key(run_key: &str, session_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(run_key.as_bytes());
    hasher.update([0]);
    hasher.update(session_id.as_bytes());
    let digest = hasher.finalize();
    let suffix = digest[..12]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("summary:{suffix}")
}

fn finish_failed(database: &crate::db::Database, run_key: &str, error: &str) {
    let _ = database.finish_compaction_run(
        run_key,
        "failed",
        0,
        Some(error),
        chrono::Utc::now().timestamp_millis(),
    );
}

fn load_batches(
    database: &crate::db::Database,
    cursor: i64,
    end_cursor: i64,
) -> Result<Vec<SessionBatch>, String> {
    let connection = database
        .conn
        .lock()
        .map_err(|_| "database lock failed".to_string())?;
    let mut statement = connection
        .prepare(
            "SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)
             FROM messages
             WHERE id > ?1 AND id <= ?2 AND direction = 'inbound'
             GROUP BY session_id
             ORDER BY COUNT(*) DESC",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map(rusqlite::params![cursor, end_cursor], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(|error| error.to_string())?;
    let groups = rows
        .filter_map(Result::ok)
        .collect::<Vec<(String, i64, i64, i64)>>();
    drop(statement);

    let mut batches = Vec::new();
    for (session_id, count, first_at, last_at) in groups {
        let mut sample_statement = connection
            .prepare(
                "SELECT content FROM messages
                 WHERE session_id = ?1 AND id > ?2 AND id <= ?3
                   AND direction = 'inbound' AND TRIM(content) <> ''
                 ORDER BY id DESC LIMIT 6",
            )
            .map_err(|error| error.to_string())?;
        let sample_rows = sample_statement
            .query_map(rusqlite::params![session_id, cursor, end_cursor], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|error| error.to_string())?;
        let samples = sample_rows
            .filter_map(Result::ok)
            .map(|sample| sample.chars().take(120).collect::<String>())
            .collect::<Vec<_>>();
        batches.push(SessionBatch {
            session_id,
            count,
            first_at,
            last_at,
            samples,
        });
    }
    Ok(batches)
}

struct SessionBatch {
    session_id: String,
    count: i64,
    first_at: i64,
    last_at: i64,
    samples: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_config_has_a_nonzero_default() {
        let config = AppConfig::default();
        assert!(config.memories.compress_interval_hours > 0);
        assert!(config.memories.compress_min_messages > 0);
    }

    #[tokio::test]
    async fn compaction_writes_summary_and_advances_cursor() {
        let path = std::env::temp_dir().join(format!(
            "alicebot-compaction-test-{}-{}.db",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let database = crate::db::Database::open(path.to_str().expect("path should be UTF-8"))
            .expect("database should open");
        database
            .insert_message(&crate::pipeline::InMessage {
                event_key: "compact:event-1".to_string(),
                protocol: "onebot11".to_string(),
                bot_account_id: String::new(),
                session_type: "group".to_string(),
                session_id: "group-compact".to_string(),
                sender_id: "user-1".to_string(),
                sender_name: "user".to_string(),
                message_id: "message-1".to_string(),
                reply_to_id: String::new(),
                content: "remember this topic".to_string(),
                media: Vec::new(),
                has_media: false,
                at_me: false,
                timestamp: 1_000,
                safe_raw_json: "{}".to_string(),
            })
            .expect("message should insert");
        crate::pipeline::set_db(database);

        let mut config = AppConfig::default();
        config.memories.compress_min_messages = 1;
        assert_eq!(
            run_if_due(&config).await.expect("compaction should work"),
            1
        );
        let database = crate::pipeline::try_db().expect("database should remain installed");
        let connection = database.conn.lock().expect("database lock should work");
        let summary_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM long_memory WHERE kind = 'conversation_summary'",
                [],
                |row| row.get(0),
            )
            .expect("summary should exist");
        assert_eq!(summary_count, 1);
        let source_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM memory_sources WHERE source_type = 'reflection'",
                [],
                |row| row.get(0),
            )
            .expect("summary source should exist");
        assert_eq!(source_count, 1);
        drop(connection);
        assert_eq!(
            database.get_meta("compaction_cursor").unwrap().as_deref(),
            Some("1")
        );
        crate::pipeline::clear_db();
        drop(database);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }
}
