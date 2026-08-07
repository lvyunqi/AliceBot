//! Optional SQLite FTS5 cache for long-memory retrieval.

use rusqlite::{Connection, OptionalExtension, TransactionBehavior};

const FTS_TABLE_SQL: &str = r#"
CREATE VIRTUAL TABLE memory_fts USING fts5(
    content,
    content = 'long_memory',
    content_rowid = 'id',
    tokenize = 'trigram'
);
"#;

const FTS_INSERT_TRIGGER_SQL: &str = r#"
CREATE TRIGGER memory_fts_ai AFTER INSERT ON long_memory BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
"#;

const FTS_DELETE_TRIGGER_SQL: &str = r#"
CREATE TRIGGER memory_fts_ad AFTER DELETE ON long_memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
"#;

const FTS_UPDATE_TRIGGER_SQL: &str = r#"
CREATE TRIGGER memory_fts_au AFTER UPDATE OF content ON long_memory BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
"#;

const DROP_CACHE_SQL: &str = r#"
DROP TRIGGER IF EXISTS memory_fts_ai;
DROP TRIGGER IF EXISTS memory_fts_ad;
DROP TRIGGER IF EXISTS memory_fts_au;
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchBackend {
    Fts5,
    Lexical,
}

pub(crate) fn initialize(conn: &mut Connection) -> Result<SearchBackend, rusqlite::Error> {
    if !fts5_available(conn)? {
        return Ok(SearchBackend::Lexical);
    }
    if !cache_is_healthy(conn)? {
        rebuild_cache(conn)?;
    }
    Ok(SearchBackend::Fts5)
}

pub(crate) fn match_query(query: &str) -> Option<String> {
    const MAX_TERMS: usize = 16;
    const MAX_TERM_CHARS: usize = 64;

    let mut terms = Vec::new();
    let mut current = String::new();
    let flush = |current: &mut String, terms: &mut Vec<String>| {
        if current.chars().count() >= 3 {
            let term = current
                .to_lowercase()
                .chars()
                .take(MAX_TERM_CHARS)
                .collect::<String>();
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        current.clear();
    };

    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character);
        } else {
            flush(&mut current, &mut terms);
            if terms.len() >= MAX_TERMS {
                break;
            }
        }
    }
    if terms.len() < MAX_TERMS {
        flush(&mut current, &mut terms);
    }
    terms.truncate(MAX_TERMS);

    if terms.is_empty() {
        None
    } else {
        Some(
            terms
                .into_iter()
                .map(|term| format!(r#""{}""#, term.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(" OR "),
        )
    }
}

fn fts5_available(conn: &Connection) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT sqlite_compileoption_used('ENABLE_FTS5')",
        [],
        |row| row.get::<_, i64>(0),
    )
    .map(|enabled| enabled != 0)
}

fn cache_is_healthy(conn: &Connection) -> Result<bool, rusqlite::Error> {
    let expected = [
        ("table", "memory_fts", FTS_TABLE_SQL),
        ("trigger", "memory_fts_ai", FTS_INSERT_TRIGGER_SQL),
        ("trigger", "memory_fts_ad", FTS_DELETE_TRIGGER_SQL),
        ("trigger", "memory_fts_au", FTS_UPDATE_TRIGGER_SQL),
    ];

    for (kind, name, expected_sql) in expected {
        let Some(actual_sql) = schema_sql(conn, kind, name)? else {
            return Ok(false);
        };
        if normalize_sql(&actual_sql) != normalize_sql(expected_sql) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn rebuild_cache(conn: &mut Connection) -> Result<(), rusqlite::Error> {
    let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(DROP_CACHE_SQL)?;

    let existing_kind = transaction
        .query_row(
            "SELECT type FROM sqlite_schema WHERE name = 'memory_fts' LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    match existing_kind.as_deref() {
        Some("view") => transaction.execute_batch("DROP VIEW memory_fts;")?,
        Some("table") => transaction.execute_batch("DROP TABLE memory_fts;")?,
        Some(_) | None => {}
    }

    transaction.execute_batch(FTS_TABLE_SQL)?;
    transaction.execute_batch(FTS_INSERT_TRIGGER_SQL)?;
    transaction.execute_batch(FTS_DELETE_TRIGGER_SQL)?;
    transaction.execute_batch(FTS_UPDATE_TRIGGER_SQL)?;
    transaction.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')", [])?;
    transaction.commit()
}

fn schema_sql(
    conn: &Connection,
    kind: &str,
    name: &str,
) -> Result<Option<String>, rusqlite::Error> {
    Ok(conn
        .query_row(
            "SELECT sql FROM sqlite_schema WHERE type = ?1 AND name = ?2",
            [kind, name],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten())
}

fn normalize_sql(sql: &str) -> String {
    sql.chars()
        .filter(|character| !character.is_whitespace() && *character != ';')
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_agrees_with_sqlite_fts5_capability_and_cache_table() {
        let database = crate::db::Database::open(":memory:").unwrap();
        let connection = database.conn.lock().unwrap();
        let available = fts5_available(&connection).unwrap();
        let table_exists = schema_sql(&connection, "table", "memory_fts")
            .unwrap()
            .is_some();

        assert_eq!(database.memory_search == SearchBackend::Fts5, available);
        assert_eq!(table_exists, available);
        if available {
            assert!(cache_is_healthy(&connection).unwrap());
        }
    }

    #[test]
    fn missing_trigger_repairs_and_rebuilds_external_content_cache() {
        let database = crate::db::Database::open(":memory:").unwrap();
        if database.memory_search != SearchBackend::Fts5 {
            return;
        }

        let mut connection = database.conn.lock().unwrap();
        connection
            .execute_batch("DROP TRIGGER memory_fts_ai;")
            .unwrap();
        connection
            .execute(
                "INSERT INTO long_memory
                 (normalized_key, scope, session_id, content, status, is_active, created_at)
                 VALUES ('repair-test', 'session', 'group-1',
                         'repairable searchable memory', 'active', 1, 10)",
                [],
            )
            .unwrap();

        assert_eq!(initialize(&mut connection).unwrap(), SearchBackend::Fts5);
        assert!(cache_is_healthy(&connection).unwrap());
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM memory_fts WHERE memory_fts MATCH 'repairable'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn match_query_quotes_user_terms_instead_of_accepting_fts_operators() {
        assert_eq!(
            match_query("rust\" OR * 数据库"),
            Some(r#""rust" OR "数据库""#.to_string())
        );
        assert_eq!(match_query("hi"), None);
    }
}
