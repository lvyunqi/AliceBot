use rusqlite::{Connection, DatabaseName, OptionalExtension, Transaction, TransactionBehavior};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(crate) const LATEST_SCHEMA_VERSION: i64 = 16;
pub(crate) const MEMORY_SEARCH_CACHE_CONTRACT: &str = "fts5-external-content-v1";

#[derive(Debug, thiserror::Error)]
pub(crate) enum DatabaseError {
    #[error("SQLite operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("invalid database schema version: {0:?}")]
    InvalidSchemaVersion(String),
    #[error(
        "database schema version {found} is newer than supported version {supported}; upgrade AliceBot"
    )]
    SchemaTooNew { found: i64, supported: i64 },
    #[error("database schema validation failed: {0}")]
    SchemaValidation(String),
    #[error("no migration is registered for schema version {0}")]
    MissingMigration(i64),
}

pub(crate) struct MigrationReport {
    pub from: i64,
    pub to: i64,
    pub backup_path: Option<PathBuf>,
}

pub(crate) fn has_existing_database(path: &Path) -> bool {
    path != Path::new(":memory:")
        && std::fs::metadata(path)
            .map(|metadata| metadata.is_file() && metadata.len() > 0)
            .unwrap_or(false)
}

pub(crate) fn prepare_database(
    path: &Path,
    conn: &mut Connection,
    had_existing_database: bool,
) -> Result<MigrationReport, DatabaseError> {
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "foreign_keys", true)?;

    let from = read_schema_version(conn)?;
    if from > LATEST_SCHEMA_VERSION {
        return Err(DatabaseError::SchemaTooNew {
            found: from,
            supported: LATEST_SCHEMA_VERSION,
        });
    }

    let backup_path = if had_existing_database && from < LATEST_SCHEMA_VERSION {
        Some(create_backup(conn, path, from, LATEST_SCHEMA_VERSION)?)
    } else {
        None
    };

    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    migrate_to(conn, LATEST_SCHEMA_VERSION)?;
    validate_latest_schema(conn)?;

    Ok(MigrationReport {
        from,
        to: LATEST_SCHEMA_VERSION,
        backup_path,
    })
}

fn read_schema_version(conn: &Connection) -> Result<i64, DatabaseError> {
    if !object_exists(conn, "table", "meta")? {
        return Ok(0);
    }

    let value = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();

    let Some(value) = value else {
        return Ok(0);
    };
    let version = value
        .trim()
        .parse::<i64>()
        .map_err(|_| DatabaseError::InvalidSchemaVersion(value.clone()))?;
    if version < 0 {
        return Err(DatabaseError::InvalidSchemaVersion(value));
    }
    Ok(version)
}

fn migrate_to(conn: &mut Connection, target: i64) -> Result<(), DatabaseError> {
    let mut current = read_schema_version(conn)?;
    while current < target {
        let next = current + 1;
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        apply_migration(&transaction, next)?;
        write_schema_version(&transaction, next)?;
        transaction.commit()?;
        current = next;
    }
    Ok(())
}

fn apply_migration(transaction: &Transaction<'_>, version: i64) -> Result<(), DatabaseError> {
    match version {
        1 => migration_1_baseline(transaction)?,
        2 => migration_2_journal_and_audit(transaction)?,
        3 => migration_3_compaction(transaction)?,
        4 => migration_4_llm_audit(transaction)?,
        // Earlier builds wrote versions 3 and 4 without a real migration chain.
        // Re-run every idempotent step once before adopting the formal system.
        5 => {
            migration_1_baseline(transaction)?;
            migration_2_journal_and_audit(transaction)?;
            migration_3_compaction(transaction)?;
            migration_4_llm_audit(transaction)?;
        }
        6 => migration_6_session_state(transaction)?,
        7 => migration_7_structured_memory(transaction)?,
        8 => migration_8_memory_retrieval(transaction)?,
        9 => migration_9_persona_and_knowledge_evidence(transaction)?,
        10 => migration_10_delivery_idempotency(transaction)?,
        11 => migration_11_memory_route_isolation(transaction)?,
        12 => migration_12_short_context_recovery_indexes(transaction)?,
        13 => migration_13_behavior_calibration(transaction)?,
        14 => migration_14_media_url_privacy(transaction)?,
        15 => migration_15_sticker_cache(transaction)?,
        16 => migration_16_sticker_graph(transaction)?,
        _ => return Err(DatabaseError::MissingMigration(version)),
    }
    Ok(())
}

fn write_schema_version(conn: &Connection, version: i64) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [version.to_string()],
    )?;
    Ok(())
}

fn migration_1_baseline(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS messages (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol       TEXT NOT NULL,
            direction      TEXT NOT NULL,
            session_type   TEXT NOT NULL,
            session_id     TEXT NOT NULL,
            sender_id      TEXT NOT NULL,
            sender_name    TEXT,
            message_id     TEXT,
            content        TEXT NOT NULL,
            raw_json       TEXT,
            has_media      INTEGER DEFAULT 0,
            media_type     TEXT,
            reply_to_id    TEXT,
            at_me          INTEGER DEFAULT 0,
            sentiment      INTEGER,
            created_at     INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_msg_session_time
            ON messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_msg_sender_time
            ON messages(sender_id, created_at);

        CREATE TABLE IF NOT EXISTS long_memory (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT,
            subject_id    TEXT,
            content       TEXT NOT NULL,
            kind          TEXT DEFAULT 'fact',
            importance    INTEGER DEFAULT 50,
            is_active     INTEGER DEFAULT 1,
            access_count  INTEGER DEFAULT 0,
            last_access   INTEGER,
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_mem_active_imp
            ON long_memory(is_active, importance);

        CREATE TABLE IF NOT EXISTS personas (
            subject_id        TEXT PRIMARY KEY,
            protocol          TEXT,
            nickname          TEXT,
            first_seen        INTEGER,
            last_seen         INTEGER,
            interaction_count INTEGER DEFAULT 0,
            intimacy          INTEGER DEFAULT 0,
            relation          TEXT,
            traits            TEXT,
            preferences       TEXT,
            topics            TEXT,
            notes             TEXT
        );

        CREATE TABLE IF NOT EXISTS knowledge (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            subject       TEXT,
            content       TEXT NOT NULL,
            category      TEXT,
            source        TEXT,
            confidence    INTEGER DEFAULT 60,
            is_active     INTEGER DEFAULT 1,
            access_count  INTEGER DEFAULT 0,
            created_at    INTEGER NOT NULL,
            updated_at    INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_know_subject
            ON knowledge(subject, is_active);

        CREATE TABLE IF NOT EXISTS stickers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            protocol       TEXT,
            kind           TEXT DEFAULT 'image',
            media_url      TEXT NOT NULL,
            file_hash      TEXT,
            source_user    TEXT,
            source_session TEXT,
            usage_count    INTEGER DEFAULT 0,
            last_used      INTEGER,
            created_at     INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sticker_tags (
            sticker_id INTEGER NOT NULL,
            tag        TEXT NOT NULL,
            weight     INTEGER DEFAULT 1,
            PRIMARY KEY (sticker_id, tag)
        );

        CREATE TABLE IF NOT EXISTS sticker_links (
            sticker_a INTEGER NOT NULL,
            sticker_b INTEGER NOT NULL,
            co_count  INTEGER DEFAULT 1,
            updated_at INTEGER,
            PRIMARY KEY (sticker_a, sticker_b)
        );

        CREATE TABLE IF NOT EXISTS reflection_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            triggered_by TEXT,
            summary      TEXT,
            insights     TEXT,
            created_at   INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        "#,
    )?;
    Ok(())
}

fn migration_2_journal_and_audit(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(conn, "messages", "event_key", "TEXT")?;
    ensure_column(conn, "messages", "action_key", "TEXT")?;
    ensure_column(conn, "messages", "bot_account_id", "TEXT")?;
    ensure_column(conn, "messages", "media_url", "TEXT")?;
    ensure_column(conn, "messages", "updated_at", "INTEGER")?;
    conn.execute(
        "UPDATE messages SET updated_at = created_at WHERE updated_at IS NULL",
        [],
    )?;
    conn.execute_batch(
        r#"
        CREATE UNIQUE INDEX IF NOT EXISTS ux_msg_event_direction
            ON messages(event_key, direction) WHERE event_key IS NOT NULL;

        CREATE TABLE IF NOT EXISTS outbound_messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            action_key       TEXT NOT NULL,
            source_event_key TEXT,
            protocol         TEXT NOT NULL,
            bot_account_id   TEXT,
            session_type     TEXT NOT NULL,
            session_id       TEXT NOT NULL,
            content          TEXT NOT NULL DEFAULT '',
            media_type       TEXT,
            media_url        TEXT,
            status           TEXT NOT NULL DEFAULT 'pending',
            host_status      TEXT,
            error            TEXT,
            attempt_count    INTEGER NOT NULL DEFAULT 1,
            created_at       INTEGER NOT NULL,
            updated_at       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_outbound_session_time
            ON outbound_messages(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_outbound_status_time
            ON outbound_messages(status, created_at);

        CREATE TABLE IF NOT EXISTS decision_traces (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key    TEXT NOT NULL UNIQUE,
            session_id   TEXT NOT NULL,
            score        REAL NOT NULL,
            threshold    REAL NOT NULL,
            direct       INTEGER NOT NULL,
            outcome      TEXT NOT NULL,
            reason       TEXT NOT NULL,
            signals_json TEXT NOT NULL,
            created_at   INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_decision_session_time
            ON decision_traces(session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_decision_outcome_time
            ON decision_traces(outcome, created_at);
        "#,
    )?;
    Ok(())
}

fn migration_3_compaction(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS compaction_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_key         TEXT NOT NULL UNIQUE,
            cursor_start    INTEGER NOT NULL,
            cursor_end      INTEGER NOT NULL,
            status          TEXT NOT NULL,
            processed_count INTEGER NOT NULL DEFAULT 0,
            error           TEXT,
            started_at      INTEGER NOT NULL,
            finished_at     INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_compaction_status_time
            ON compaction_runs(status, started_at);
        "#,
    )?;
    Ok(())
}

fn migration_4_llm_audit(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS llm_calls (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            task         TEXT NOT NULL,
            provider_id  TEXT NOT NULL,
            protocol     TEXT NOT NULL,
            model        TEXT,
            attempt      INTEGER NOT NULL,
            status       TEXT NOT NULL DEFAULT 'started',
            error_kind   TEXT,
            input_chars  INTEGER NOT NULL DEFAULT 0,
            output_chars INTEGER NOT NULL DEFAULT 0,
            latency_ms   INTEGER,
            created_at   INTEGER NOT NULL,
            updated_at   INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_llm_calls_task_time
            ON llm_calls(task, created_at);
        CREATE INDEX IF NOT EXISTS idx_llm_calls_status_time
            ON llm_calls(status, created_at);
        "#,
    )?;
    Ok(())
}

fn migration_6_session_state(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS session_state (
            session_key           TEXT PRIMARY KEY,
            protocol              TEXT NOT NULL,
            session_type          TEXT NOT NULL,
            session_id            TEXT NOT NULL,
            last_message_at       INTEGER,
            last_outbound_at      INTEGER,
            recent_outbound_count INTEGER NOT NULL DEFAULT 0,
            activity_ewma         REAL NOT NULL DEFAULT 0,
            short_summary         TEXT,
            short_cursor          INTEGER NOT NULL DEFAULT 0,
            reply_alpha           REAL NOT NULL DEFAULT 1,
            reply_beta            REAL NOT NULL DEFAULT 1,
            updated_at            INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_session_state_route
            ON session_state(protocol, session_type, session_id);
        "#,
    )?;
    ensure_column(conn, "decision_traces", "policy_version", "TEXT")?;
    ensure_column(conn, "decision_traces", "p_rule", "REAL")?;
    ensure_column(conn, "decision_traces", "p_final", "REAL")?;
    ensure_column(conn, "decision_traces", "random_value", "REAL")?;
    ensure_column(conn, "decision_traces", "activity_ewma", "REAL")?;
    ensure_column(
        conn,
        "decision_traces",
        "coalesced_count",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    Ok(())
}

fn migration_7_structured_memory(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(
        conn,
        "long_memory",
        "normalized_key",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "long_memory",
        "scope",
        "TEXT NOT NULL DEFAULT 'session'",
    )?;
    ensure_column(
        conn,
        "long_memory",
        "confidence",
        "INTEGER NOT NULL DEFAULT 50",
    )?;
    ensure_column(
        conn,
        "long_memory",
        "privacy",
        "TEXT NOT NULL DEFAULT 'normal'",
    )?;
    ensure_column(
        conn,
        "long_memory",
        "status",
        "TEXT NOT NULL DEFAULT 'candidate'",
    )?;
    ensure_column(conn, "long_memory", "version", "INTEGER NOT NULL DEFAULT 1")?;
    ensure_column(conn, "long_memory", "superseded_by", "INTEGER")?;
    ensure_column(conn, "long_memory", "archived_at", "INTEGER")?;

    ensure_column(
        conn,
        "knowledge",
        "normalized_key",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    ensure_column(
        conn,
        "knowledge",
        "scope",
        "TEXT NOT NULL DEFAULT 'session'",
    )?;
    ensure_column(
        conn,
        "knowledge",
        "status",
        "TEXT NOT NULL DEFAULT 'candidate'",
    )?;
    ensure_column(conn, "knowledge", "version", "INTEGER NOT NULL DEFAULT 1")?;
    ensure_column(conn, "knowledge", "superseded_by", "INTEGER")?;

    conn.execute_batch(
        r#"
        UPDATE long_memory
        SET normalized_key = 'legacy:memory:' || id
        WHERE normalized_key = '';
        UPDATE long_memory
        SET status = CASE WHEN is_active = 1 THEN 'active' ELSE 'forgotten' END;

        UPDATE knowledge
        SET normalized_key = 'legacy:knowledge:' || id
        WHERE normalized_key = '';
        UPDATE knowledge
        SET status = CASE WHEN is_active = 1 THEN 'active' ELSE 'forgotten' END;

        CREATE TABLE IF NOT EXISTS memory_sources (
            memory_id       INTEGER NOT NULL,
            source_type     TEXT NOT NULL,
            source_id       TEXT NOT NULL,
            evidence_weight INTEGER NOT NULL DEFAULT 1,
            created_at      INTEGER NOT NULL,
            PRIMARY KEY (memory_id, source_type, source_id),
            FOREIGN KEY (memory_id) REFERENCES long_memory(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_memory_key_version
            ON long_memory(normalized_key, version);
        CREATE INDEX IF NOT EXISTS idx_memory_retrieve_v7
            ON long_memory(status, scope, subject_id, importance, confidence);
        CREATE INDEX IF NOT EXISTS idx_memory_sources_source
            ON memory_sources(source_type, source_id);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_knowledge_key_version
            ON knowledge(normalized_key, version);
        CREATE INDEX IF NOT EXISTS idx_knowledge_retrieve_v7
            ON knowledge(status, scope, category, confidence);
        "#,
    )?;
    Ok(())
}

fn migration_8_memory_retrieval(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        UPDATE long_memory
        SET scope = CASE
            WHEN session_id IS NOT NULL AND subject_id IS NOT NULL THEN 'user_session'
            WHEN session_id IS NOT NULL THEN 'session'
            WHEN subject_id IS NOT NULL THEN 'user'
            ELSE 'global'
        END
        WHERE normalized_key LIKE 'legacy:memory:%';

        CREATE INDEX IF NOT EXISTS idx_memory_retrieve_v8
            ON long_memory(
                status,
                is_active,
                privacy,
                scope,
                session_id,
                subject_id,
                importance DESC,
                updated_at DESC
            );

        INSERT INTO meta (key, value)
        VALUES ('memory_search_cache_contract', 'fts5-external-content-v1')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value;
        "#,
    )?;
    Ok(())
}

fn migration_9_persona_and_knowledge_evidence(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(conn, "knowledge", "protocol", "TEXT")?;
    ensure_column(conn, "knowledge", "session_type", "TEXT")?;
    ensure_column(conn, "knowledge", "session_id", "TEXT")?;
    conn.execute_batch(
        r#"
        CREATE TABLE personas_v9 (
            protocol             TEXT NOT NULL,
            subject_id           TEXT NOT NULL,
            nickname             TEXT NOT NULL DEFAULT '',
            first_seen           INTEGER NOT NULL DEFAULT 0,
            last_seen            INTEGER NOT NULL DEFAULT 0,
            interaction_count    INTEGER NOT NULL DEFAULT 0,
            intimacy             INTEGER NOT NULL DEFAULT 0,
            intimacy_ewma        REAL NOT NULL DEFAULT 0,
            intimacy_updated_at  INTEGER NOT NULL DEFAULT 0,
            relation             TEXT NOT NULL DEFAULT 'stranger',
            traits               TEXT NOT NULL DEFAULT '{}',
            preferences          TEXT NOT NULL DEFAULT '{}',
            topics               TEXT NOT NULL DEFAULT '{}',
            notes                TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (protocol, subject_id)
        );

        INSERT INTO personas_v9
            (protocol, subject_id, nickname, first_seen, last_seen,
             interaction_count, intimacy, intimacy_ewma, intimacy_updated_at,
             relation, traits, preferences, topics, notes)
        SELECT COALESCE(NULLIF(protocol, ''), 'unknown'),
               subject_id,
               COALESCE(nickname, ''),
               COALESCE(first_seen, 0),
               COALESCE(last_seen, 0),
               COALESCE(interaction_count, 0),
               COALESCE(intimacy, 0),
               COALESCE(intimacy, 0),
               COALESCE(last_seen, 0),
               COALESCE(NULLIF(relation, ''), 'stranger'),
               COALESCE(NULLIF(traits, ''), '{}'),
               COALESCE(NULLIF(preferences, ''), '{}'),
               COALESCE(NULLIF(topics, ''), '{}'),
               COALESCE(notes, '')
        FROM personas
        WHERE subject_id IS NOT NULL AND TRIM(subject_id) <> '';

        DROP TABLE personas;
        ALTER TABLE personas_v9 RENAME TO personas;

        CREATE INDEX idx_persona_subject_protocol
            ON personas(subject_id, protocol);

        CREATE TABLE persona_observations (
            protocol     TEXT NOT NULL,
            event_key    TEXT NOT NULL,
            subject_id   TEXT NOT NULL,
            session_id   TEXT NOT NULL,
            observed_at  INTEGER NOT NULL,
            PRIMARY KEY (protocol, event_key),
            FOREIGN KEY (protocol, subject_id)
                REFERENCES personas(protocol, subject_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_persona_observation_subject
            ON persona_observations(protocol, subject_id, observed_at);

        CREATE TABLE persona_nicknames (
            protocol        TEXT NOT NULL,
            subject_id      TEXT NOT NULL,
            nickname        TEXT NOT NULL,
            first_seen      INTEGER NOT NULL,
            last_seen       INTEGER NOT NULL,
            seen_count      INTEGER NOT NULL DEFAULT 1,
            is_current      INTEGER NOT NULL DEFAULT 0,
            last_event_key  TEXT NOT NULL,
            PRIMARY KEY (protocol, subject_id, nickname),
            FOREIGN KEY (protocol, subject_id)
                REFERENCES personas(protocol, subject_id) ON DELETE CASCADE
        );
        INSERT INTO persona_nicknames
            (protocol, subject_id, nickname, first_seen, last_seen,
             seen_count, is_current, last_event_key)
        SELECT protocol, subject_id, nickname, first_seen, last_seen,
               MAX(interaction_count, 1), 1, 'migration:v9'
        FROM personas
        WHERE nickname <> '';
        CREATE INDEX idx_persona_nickname_current
            ON persona_nicknames(protocol, subject_id, is_current, last_seen);

        CREATE TABLE persona_topics (
            protocol    TEXT NOT NULL,
            subject_id  TEXT NOT NULL,
            topic       TEXT NOT NULL,
            count       INTEGER NOT NULL DEFAULT 1,
            score       REAL NOT NULL DEFAULT 1,
            first_seen  INTEGER NOT NULL,
            last_seen   INTEGER NOT NULL,
            PRIMARY KEY (protocol, subject_id, topic),
            FOREIGN KEY (protocol, subject_id)
                REFERENCES personas(protocol, subject_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_persona_topics_rank
            ON persona_topics(protocol, subject_id, score DESC, count DESC, last_seen DESC);

        CREATE TABLE knowledge_sources (
            knowledge_id       INTEGER NOT NULL,
            source_type        TEXT NOT NULL,
            source_id          TEXT NOT NULL,
            source_subject_id  TEXT NOT NULL,
            evidence_weight    INTEGER NOT NULL DEFAULT 1,
            created_at         INTEGER NOT NULL,
            PRIMARY KEY (knowledge_id, source_type, source_id),
            FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_knowledge_sources_source
            ON knowledge_sources(source_type, source_id);
        CREATE INDEX idx_knowledge_sources_subject
            ON knowledge_sources(knowledge_id, source_subject_id);
        CREATE INDEX idx_knowledge_scope_v9
            ON knowledge(
                status, is_active, scope, protocol,
                session_type, session_id, category, confidence
            );

        INSERT OR IGNORE INTO knowledge_sources
            (knowledge_id, source_type, source_id, source_subject_id,
             evidence_weight, created_at)
        SELECT id, 'legacy', source, 'legacy', 1, COALESCE(created_at, 0)
        FROM knowledge
        WHERE source IS NOT NULL AND TRIM(source) <> '';
        "#,
    )?;
    Ok(())
}

fn migration_10_delivery_idempotency(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(
        conn,
        "messages",
        "processing_status",
        "TEXT NOT NULL DEFAULT 'legacy'",
    )?;
    ensure_column(conn, "messages", "processing_error", "TEXT")?;
    ensure_column(conn, "messages", "processed_at", "INTEGER")?;
    conn.execute_batch(
        r#"
        UPDATE messages
        SET processing_status = 'legacy'
        WHERE processing_status IS NULL OR processing_status = '';

        UPDATE outbound_messages
        SET action_key = action_key || ':legacy-duplicate:' || id
        WHERE id NOT IN (
            SELECT MIN(id) FROM outbound_messages GROUP BY action_key
        );

        CREATE INDEX idx_messages_processing_v10
            ON messages(processing_status, created_at);
        CREATE UNIQUE INDEX ux_outbound_action_key
            ON outbound_messages(action_key);
        "#,
    )?;
    Ok(())
}

fn migration_11_memory_route_isolation(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(conn, "long_memory", "protocol", "TEXT NOT NULL DEFAULT ''")?;
    ensure_column(
        conn,
        "long_memory",
        "session_type",
        "TEXT NOT NULL DEFAULT ''",
    )?;
    conn.execute_batch(
        r#"
        UPDATE long_memory
        SET protocol = CASE
            WHEN scope = 'global' THEN '*'
            ELSE COALESCE(
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT m.protocol) = 1 THEN MAX(m.protocol)
                    END
                    FROM memory_sources AS ms
                    JOIN messages AS m ON m.event_key = ms.source_id
                    WHERE ms.memory_id = long_memory.id
                      AND ms.source_type = 'message'
                      AND m.direction = 'inbound'
                ),
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT m.protocol) = 1 THEN MAX(m.protocol)
                    END
                    FROM messages AS m
                    WHERE (long_memory.session_id IS NULL
                           OR m.session_id = long_memory.session_id)
                      AND (long_memory.subject_id IS NULL
                           OR m.sender_id = long_memory.subject_id)
                      AND m.direction = 'inbound'
                ),
                'legacy'
            )
        END
        WHERE TRIM(COALESCE(protocol, '')) = '';

        UPDATE long_memory
        SET session_type = CASE
            WHEN scope IN ('global', 'user') THEN '*'
            ELSE COALESCE(
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT m.session_type) = 1
                        THEN MAX(m.session_type)
                    END
                    FROM memory_sources AS ms
                    JOIN messages AS m ON m.event_key = ms.source_id
                    WHERE ms.memory_id = long_memory.id
                      AND ms.source_type = 'message'
                      AND m.direction = 'inbound'
                ),
                (
                    SELECT CASE
                        WHEN COUNT(DISTINCT m.session_type) = 1
                        THEN MAX(m.session_type)
                    END
                    FROM messages AS m
                    WHERE (long_memory.session_id IS NULL
                           OR m.session_id = long_memory.session_id)
                      AND (long_memory.subject_id IS NULL
                           OR m.sender_id = long_memory.subject_id)
                      AND m.direction = 'inbound'
                ),
                'legacy'
            )
        END
        WHERE TRIM(COALESCE(session_type, '')) = '';

        UPDATE long_memory
        SET session_type = 'legacy'
        WHERE protocol = 'legacy'
          AND scope IN ('session', 'user_session');

        DROP INDEX IF EXISTS ux_memory_key_version;
        CREATE UNIQUE INDEX ux_memory_route_key_version
            ON long_memory(
                normalized_key,
                protocol,
                session_type,
                COALESCE(session_id, ''),
                version
            );

        CREATE INDEX IF NOT EXISTS idx_memory_retrieve_v11
            ON long_memory(
                status,
                is_active,
                privacy,
                protocol,
                session_type,
                scope,
                session_id,
                subject_id,
                importance DESC,
                updated_at DESC
            );
        "#,
    )?;
    Ok(())
}

fn migration_12_short_context_recovery_indexes(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE INDEX IF NOT EXISTS idx_messages_recovery_recent_v12
            ON messages(direction, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_recovery_route_v12
            ON messages(
                protocol,
                session_type,
                session_id,
                direction,
                created_at DESC,
                id DESC
            );
        CREATE INDEX IF NOT EXISTS idx_outbound_recovery_recent_v12
            ON outbound_messages(status, updated_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_outbound_recovery_route_v12
            ON outbound_messages(
                protocol,
                session_type,
                session_id,
                status,
                updated_at DESC,
                id DESC
            );
        "#,
    )?;
    Ok(())
}

fn migration_13_behavior_calibration(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS behavior_calibration (
            id                INTEGER PRIMARY KEY CHECK (id = 1),
            reply_bias_offset REAL NOT NULL DEFAULT 0,
            reflection_cursor INTEGER NOT NULL DEFAULT 0,
            updated_at        INTEGER NOT NULL DEFAULT 0
        );
        INSERT OR IGNORE INTO behavior_calibration
            (id, reply_bias_offset, reflection_cursor, updated_at)
            VALUES (1, 0, 0, 0);

        CREATE TABLE IF NOT EXISTS reflection_changes (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            reflection_id         INTEGER NOT NULL REFERENCES reflection_log(id) ON DELETE CASCADE,
            cursor_start          INTEGER NOT NULL,
            cursor_end            INTEGER NOT NULL,
            old_reply_bias_offset REAL NOT NULL,
            new_reply_bias_offset REAL NOT NULL,
            observed_reply_rate   REAL,
            target_reply_rate     REAL,
            accepted_reply_count  INTEGER NOT NULL DEFAULT 0,
            skip_count            INTEGER NOT NULL DEFAULT 0,
            unreliable_reply_count INTEGER NOT NULL DEFAULT 0,
            reason                TEXT NOT NULL,
            status                TEXT NOT NULL,
            created_at            INTEGER NOT NULL,
            rolled_back_at        INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_decision_reflection_v13
            ON decision_traces(direct, outcome, id);
        CREATE INDEX IF NOT EXISTS idx_reflection_changes_status_time_v13
            ON reflection_changes(status, created_at DESC, id DESC);
        "#,
    )?;
    Ok(())
}

fn migration_14_media_url_privacy(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(conn, "stickers", "url_hash", "TEXT")?;
    ensure_column(
        conn,
        "stickers",
        "url_requires_cache",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "stickers",
        "cache_status",
        "TEXT NOT NULL DEFAULT 'url_only'",
    )?;
    ensure_column(conn, "stickers", "updated_at", "INTEGER NOT NULL DEFAULT 0")?;

    sanitize_media_url_column(conn, "messages")?;
    sanitize_media_url_column(conn, "outbound_messages")?;

    let stickers = {
        let mut statement = conn.prepare(
            "SELECT id, media_url, file_hash, created_at
             FROM stickers ORDER BY id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    for (id, raw_url, legacy_url_hash, created_at) in stickers {
        match crate::media::sanitize_remote_media_url(&raw_url, true) {
            Some(media) => {
                let status = if media.requires_cache {
                    "required"
                } else {
                    "remote"
                };
                conn.execute(
                    "UPDATE stickers
                     SET media_url = ?1, url_hash = ?2, file_hash = NULL,
                         url_requires_cache = ?3, cache_status = ?4,
                         updated_at = CASE WHEN updated_at = 0 THEN ?5 ELSE updated_at END
                     WHERE id = ?6",
                    rusqlite::params![
                        media.storage_url,
                        media.identity_hash,
                        i32::from(media.requires_cache),
                        status,
                        created_at,
                        id,
                    ],
                )?;
            }
            None => {
                conn.execute(
                    "UPDATE stickers
                     SET media_url = '[invalid-media-url]', url_hash = ?1, file_hash = NULL,
                         url_requires_cache = 1, cache_status = 'invalid',
                         updated_at = CASE WHEN updated_at = 0 THEN ?2 ELSE updated_at END
                     WHERE id = ?3",
                    rusqlite::params![legacy_url_hash, created_at, id],
                )?;
            }
        }
    }

    conn.execute_batch(
        r#"
        CREATE INDEX IF NOT EXISTS idx_stickers_url_hash_v14
            ON stickers(url_hash);
        CREATE INDEX IF NOT EXISTS idx_stickers_delivery_v14
            ON stickers(url_requires_cache, cache_status, last_used DESC, id);
        "#,
    )?;
    Ok(())
}

fn migration_15_sticker_cache(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(conn, "stickers", "cache_path", "TEXT")?;
    ensure_column(conn, "stickers", "cache_size", "INTEGER")?;
    ensure_column(conn, "stickers", "cache_error", "TEXT")?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS sticker_sources (
            sticker_id   INTEGER NOT NULL,
            url_hash     TEXT NOT NULL,
            protocol     TEXT NOT NULL DEFAULT '',
            source_user  TEXT NOT NULL DEFAULT '',
            source_session TEXT NOT NULL DEFAULT '',
            first_seen   INTEGER NOT NULL,
            last_seen    INTEGER NOT NULL,
            seen_count   INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (sticker_id, url_hash),
            FOREIGN KEY (sticker_id) REFERENCES stickers(id) ON DELETE CASCADE
        );

        INSERT OR IGNORE INTO sticker_sources
            (sticker_id, url_hash, protocol, source_user, source_session,
             first_seen, last_seen, seen_count)
        SELECT id,
               COALESCE(NULLIF(url_hash, ''), 'legacy-url:' || id),
               COALESCE(protocol, ''),
               COALESCE(source_user, ''),
               COALESCE(source_session, ''),
               created_at,
               COALESCE(last_used, created_at),
               MAX(COALESCE(usage_count, 1), 1)
        FROM stickers;

        CREATE INDEX IF NOT EXISTS idx_sticker_sources_url_hash_v15
            ON sticker_sources(url_hash, sticker_id);
        CREATE INDEX IF NOT EXISTS idx_stickers_file_hash_v15
            ON stickers(file_hash);
        CREATE INDEX IF NOT EXISTS idx_stickers_cache_status_v15
            ON stickers(cache_status, cache_size, updated_at DESC, id);
        "#,
    )?;
    Ok(())
}

fn migration_16_sticker_graph(conn: &Connection) -> Result<(), rusqlite::Error> {
    ensure_column(
        conn,
        "sticker_links",
        "edge_kind",
        "TEXT NOT NULL DEFAULT 'cooccur'",
    )?;
    ensure_column(conn, "sticker_links", "weight", "REAL NOT NULL DEFAULT 1")?;
    ensure_column(
        conn,
        "sticker_links",
        "evidence_count",
        "INTEGER NOT NULL DEFAULT 1",
    )?;
    conn.execute_batch(
        r#"
        UPDATE sticker_links
        SET edge_kind = COALESCE(NULLIF(edge_kind, ''), 'cooccur'),
            weight = MAX(COALESCE(co_count, 1), 0),
            evidence_count = MAX(COALESCE(co_count, 1), 1);

        CREATE INDEX IF NOT EXISTS idx_sticker_links_a_weight_v16
            ON sticker_links(sticker_a, weight DESC, updated_at DESC, sticker_b ASC);
        CREATE INDEX IF NOT EXISTS idx_sticker_links_b_weight_v16
            ON sticker_links(sticker_b, weight DESC, updated_at DESC, sticker_a ASC);
        "#,
    )?;
    Ok(())
}

fn sanitize_media_url_column(conn: &Connection, table: &str) -> Result<(), rusqlite::Error> {
    let rows = {
        let mut statement = conn.prepare(&format!(
            "SELECT id, media_url FROM {table} WHERE media_url IS NOT NULL"
        ))?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    let update = format!("UPDATE {table} SET media_url = ?1 WHERE id = ?2");
    for (id, raw_url) in rows {
        conn.execute(
            &update,
            rusqlite::params![crate::media::redact_url_for_storage(&raw_url), id],
        )?;
    }
    Ok(())
}

fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), rusqlite::Error> {
    if !column_exists(conn, table, column)? {
        conn.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {definition}"
        ))?;
    }
    Ok(())
}

fn validate_latest_schema(conn: &Connection) -> Result<(), DatabaseError> {
    let version = read_schema_version(conn)?;
    if version != LATEST_SCHEMA_VERSION {
        return Err(DatabaseError::SchemaValidation(format!(
            "expected version {LATEST_SCHEMA_VERSION}, found {version}"
        )));
    }

    let required = [
        (
            "messages",
            &[
                "event_key",
                "bot_account_id",
                "content",
                "media_url",
                "processing_status",
                "processing_error",
                "processed_at",
                "updated_at",
            ][..],
        ),
        (
            "outbound_messages",
            &["action_key", "status", "host_status", "updated_at"][..],
        ),
        (
            "decision_traces",
            &[
                "event_key",
                "signals_json",
                "outcome",
                "policy_version",
                "p_rule",
                "p_final",
                "random_value",
                "activity_ewma",
                "coalesced_count",
            ][..],
        ),
        (
            "llm_calls",
            &["provider_id", "status", "input_chars", "updated_at"][..],
        ),
        (
            "compaction_runs",
            &["run_key", "cursor_start", "status"][..],
        ),
        (
            "session_state",
            &[
                "session_key",
                "last_message_at",
                "last_outbound_at",
                "activity_ewma",
                "reply_alpha",
                "reply_beta",
            ][..],
        ),
        (
            "long_memory",
            &[
                "content",
                "protocol",
                "session_type",
                "importance",
                "confidence",
                "normalized_key",
                "scope",
                "privacy",
                "status",
                "version",
                "superseded_by",
                "is_active",
            ][..],
        ),
        (
            "memory_sources",
            &["memory_id", "source_type", "source_id", "evidence_weight"][..],
        ),
        (
            "personas",
            &[
                "protocol",
                "subject_id",
                "nickname",
                "intimacy",
                "intimacy_ewma",
                "intimacy_updated_at",
            ][..],
        ),
        (
            "persona_observations",
            &["event_key", "protocol", "subject_id", "observed_at"][..],
        ),
        (
            "persona_nicknames",
            &[
                "protocol",
                "subject_id",
                "nickname",
                "seen_count",
                "is_current",
                "last_event_key",
            ][..],
        ),
        (
            "persona_topics",
            &["protocol", "subject_id", "topic", "count", "score"][..],
        ),
        (
            "knowledge",
            &[
                "content",
                "confidence",
                "normalized_key",
                "scope",
                "protocol",
                "session_type",
                "session_id",
                "status",
                "version",
                "superseded_by",
                "is_active",
            ][..],
        ),
        (
            "knowledge_sources",
            &[
                "knowledge_id",
                "source_type",
                "source_id",
                "source_subject_id",
                "evidence_weight",
            ][..],
        ),
        (
            "stickers",
            &[
                "media_url",
                "file_hash",
                "url_hash",
                "url_requires_cache",
                "cache_status",
                "cache_path",
                "cache_size",
                "cache_error",
                "source_session",
                "updated_at",
            ][..],
        ),
        ("sticker_tags", &["sticker_id", "tag", "weight"][..]),
        (
            "sticker_links",
            &[
                "sticker_a",
                "sticker_b",
                "co_count",
                "edge_kind",
                "weight",
                "evidence_count",
            ][..],
        ),
        (
            "sticker_sources",
            &[
                "sticker_id",
                "url_hash",
                "protocol",
                "source_user",
                "source_session",
                "first_seen",
                "last_seen",
                "seen_count",
            ][..],
        ),
        (
            "reflection_log",
            &["triggered_by", "insights", "created_at"][..],
        ),
        (
            "behavior_calibration",
            &["reply_bias_offset", "reflection_cursor", "updated_at"][..],
        ),
        (
            "reflection_changes",
            &[
                "reflection_id",
                "old_reply_bias_offset",
                "new_reply_bias_offset",
                "status",
                "rolled_back_at",
            ][..],
        ),
        ("meta", &["key", "value"][..]),
    ];

    for (table, columns) in required {
        if !object_exists(conn, "table", table)? {
            return Err(DatabaseError::SchemaValidation(format!(
                "required table {table} is missing"
            )));
        }
        for column in columns {
            if !column_exists(conn, table, column)? {
                return Err(DatabaseError::SchemaValidation(format!(
                    "required column {table}.{column} is missing"
                )));
            }
        }
    }

    let required_indexes = [
        "ux_msg_event_direction",
        "idx_messages_processing_v10",
        "idx_outbound_status_time",
        "ux_outbound_action_key",
        "idx_decision_outcome_time",
        "idx_llm_calls_status_time",
        "idx_compaction_status_time",
        "ux_session_state_route",
        "ux_memory_route_key_version",
        "idx_memory_sources_source",
        "idx_memory_retrieve_v8",
        "idx_memory_retrieve_v11",
        "idx_messages_recovery_recent_v12",
        "idx_messages_recovery_route_v12",
        "idx_outbound_recovery_recent_v12",
        "idx_outbound_recovery_route_v12",
        "idx_decision_reflection_v13",
        "idx_reflection_changes_status_time_v13",
        "idx_stickers_url_hash_v14",
        "idx_stickers_delivery_v14",
        "idx_sticker_sources_url_hash_v15",
        "idx_stickers_file_hash_v15",
        "idx_stickers_cache_status_v15",
        "ux_knowledge_key_version",
        "idx_persona_subject_protocol",
        "idx_persona_observation_subject",
        "idx_persona_nickname_current",
        "idx_persona_topics_rank",
        "idx_knowledge_sources_source",
        "idx_knowledge_sources_subject",
        "idx_knowledge_scope_v9",
    ];
    for index in required_indexes {
        if !object_exists(conn, "index", index)? {
            return Err(DatabaseError::SchemaValidation(format!(
                "required index {index} is missing"
            )));
        }
    }

    let search_contract = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'memory_search_cache_contract'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if search_contract.as_deref() != Some(MEMORY_SEARCH_CACHE_CONTRACT) {
        return Err(DatabaseError::SchemaValidation(
            "memory search cache contract is missing or unsupported".to_string(),
        ));
    }
    Ok(())
}

fn object_exists(conn: &Connection, kind: &str, name: &str) -> Result<bool, rusqlite::Error> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM sqlite_schema WHERE type = ?1 AND name = ?2",
            [kind, name],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool, rusqlite::Error> {
    let mut statement = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        if row.get::<_, String>(1)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn create_backup(
    conn: &Connection,
    database_path: &Path,
    from: i64,
    to: i64,
) -> Result<PathBuf, DatabaseError> {
    let backup_path = next_backup_path(database_path, from, to);
    if let Err(error) = conn.backup(DatabaseName::Main, &backup_path, None) {
        let _ = std::fs::remove_file(&backup_path);
        return Err(error.into());
    }
    Ok(backup_path)
}

fn next_backup_path(database_path: &Path, from: i64, to: i64) -> PathBuf {
    let parent = database_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = database_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("alicebot.db");
    let base = format!("{file_name}.pre-v{from}-to-v{to}.bak");
    let mut suffix = 0_u64;
    loop {
        let name = if suffix == 0 {
            base.clone()
        } else {
            format!("{base}.{suffix}")
        };
        let candidate = parent.join(name);
        if !candidate.exists() {
            return candidate;
        }
        suffix = suffix.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDatabase {
        directory: PathBuf,
        path: PathBuf,
    }

    impl TempDatabase {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should follow Unix epoch")
                .as_nanos();
            let directory = std::env::temp_dir().join(format!(
                "alicebot-migration-{label}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir_all(&directory).expect("temporary directory should be created");
            let path = directory.join("alicebot.db");
            Self { directory, path }
        }

        fn as_str(&self) -> &str {
            self.path
                .to_str()
                .expect("temporary database path should be UTF-8")
        }

        fn backups(&self) -> Vec<PathBuf> {
            let mut paths = std::fs::read_dir(&self.directory)
                .expect("temporary directory should be readable")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.contains(".pre-v") && name.contains(".bak"))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            paths.sort();
            paths
        }
    }

    impl Drop for TempDatabase {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }

    #[test]
    fn new_database_reaches_latest_version_without_backup() {
        let temporary = TempDatabase::new("new");
        let database = Database::open(temporary.as_str()).expect("new database should open");
        assert_eq!(
            database.get_meta("schema_version").unwrap(),
            Some(LATEST_SCHEMA_VERSION.to_string())
        );
        drop(database);
        assert!(temporary.backups().is_empty());

        let reopened = Database::open(temporary.as_str()).expect("database should reopen");
        drop(reopened);
        assert!(temporary.backups().is_empty());
    }

    #[test]
    fn unversioned_legacy_database_is_backed_up_and_backfilled() {
        let temporary = TempDatabase::new("legacy");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 1).unwrap();
        connection
            .execute("DELETE FROM meta WHERE key = 'schema_version'", [])
            .unwrap();
        connection
            .execute(
                "INSERT INTO messages
                 (protocol, direction, session_type, session_id, sender_id, content, created_at)
                 VALUES ('onebot11', 'inbound', 'group', 'group-1', 'user-1', 'legacy', 42)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("legacy database should migrate");
        let connection = database.conn.lock().unwrap();
        let row: (String, i64) = connection
            .query_row(
                "SELECT content, updated_at FROM messages WHERE content = 'legacy'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(row, ("legacy".to_string(), 42));
        assert!(object_exists(&connection, "table", "llm_calls").unwrap());
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 0);
        assert!(!column_exists(&backup, "messages", "event_key").unwrap());
    }

    #[test]
    fn version_four_database_is_reconciled_once_and_preserves_data() {
        let temporary = TempDatabase::new("version-four");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 4).unwrap();
        connection
            .execute(
                "INSERT INTO personas (subject_id, nickname, interaction_count)
                 VALUES ('user-1', 'before-upgrade', 7)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version four should migrate");
        assert_eq!(
            database.get_meta("schema_version").unwrap(),
            Some(LATEST_SCHEMA_VERSION.to_string())
        );
        let interaction_count: i64 = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT interaction_count FROM personas WHERE subject_id = 'user-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(interaction_count, 7);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 4);

        drop(backup);
        let reopened = Database::open(temporary.as_str()).expect("latest database should reopen");
        drop(reopened);
        assert_eq!(temporary.backups().len(), 1);
    }

    #[test]
    fn version_five_database_gains_session_state_and_probability_trace_columns() {
        let temporary = TempDatabase::new("version-five");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 5).unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version five should migrate");
        let connection = database.conn.lock().unwrap();
        assert!(object_exists(&connection, "table", "session_state").unwrap());
        assert!(column_exists(&connection, "decision_traces", "p_final").unwrap());
        assert!(column_exists(&connection, "decision_traces", "random_value").unwrap());
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 5);
        assert!(!object_exists(&backup, "table", "session_state").unwrap());
    }

    #[test]
    fn version_six_database_backfills_structured_memory_state() {
        let temporary = TempDatabase::new("version-six");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 6).unwrap();
        connection
            .execute(
                "INSERT INTO long_memory
                 (session_id, content, importance, is_active, created_at, updated_at)
                 VALUES ('group-1', 'active legacy memory', 60, 1, 10, 10),
                        ('group-1', 'forgotten legacy memory', 60, 0, 11, 11)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version six should migrate");
        let connection = database.conn.lock().unwrap();
        let rows = connection
            .prepare("SELECT normalized_key, status FROM long_memory ORDER BY id")
            .unwrap()
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows[0].1, "active");
        assert_eq!(rows[1].1, "forgotten");
        assert_ne!(rows[0].0, rows[1].0);
        assert!(object_exists(&connection, "table", "memory_sources").unwrap());
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 6);
        assert!(!object_exists(&backup, "table", "memory_sources").unwrap());
    }

    #[test]
    fn version_seven_database_gains_memory_retrieval_contract_and_index() {
        let temporary = TempDatabase::new("version-seven");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 7).unwrap();
        connection
            .execute(
                "INSERT INTO long_memory
                 (normalized_key, scope, session_id, content, kind, importance,
                  confidence, privacy, status, version, is_active, created_at, updated_at)
                 VALUES ('legacy:memory:search', 'session', 'group-1',
                         'searchable legacy memory', 'fact', 60, 80, 'normal',
                         'active', 1, 1, 10, 10)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version seven should migrate");
        assert_eq!(
            database.get_meta("schema_version").unwrap(),
            Some(LATEST_SCHEMA_VERSION.to_string())
        );
        assert_eq!(
            database
                .get_meta("memory_search_cache_contract")
                .unwrap()
                .as_deref(),
            Some(MEMORY_SEARCH_CACHE_CONTRACT)
        );
        let connection = database.conn.lock().unwrap();
        assert!(object_exists(&connection, "index", "idx_memory_retrieve_v8").unwrap());
        let scope: String = connection
            .query_row(
                "SELECT scope FROM long_memory WHERE normalized_key = 'legacy:memory:search'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(scope, "session");
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 7);
        assert!(!object_exists(&backup, "index", "idx_memory_retrieve_v8").unwrap());
        assert!(!object_exists(&backup, "table", "memory_fts").unwrap());
    }

    #[test]
    fn version_eight_database_gains_protocol_scoped_personas_and_knowledge_sources() {
        let temporary = TempDatabase::new("version-eight");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 8).unwrap();
        connection
            .execute(
                "INSERT INTO personas
                 (subject_id, protocol, nickname, first_seen, last_seen,
                  interaction_count, intimacy, relation, traits, preferences, topics, notes)
                 VALUES ('user-1', 'onebot11', 'before-upgrade', 10, 20,
                         7, 35, 'familiar', '{}', '{}', '{}', '')",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO knowledge
                 (normalized_key, subject, content, category, scope, source,
                  confidence, status, version, is_active, created_at, updated_at)
                 VALUES ('legacy:knowledge:test', 'schedule', 'Friday meeting',
                         'group_fact', 'session', 'legacy-message', 70,
                         'candidate', 1, 0, 10, 10)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version eight should migrate");
        let connection = database.conn.lock().unwrap();
        let persona: (String, String, f64) = connection
            .query_row(
                "SELECT protocol, nickname, intimacy_ewma
                 FROM personas WHERE protocol = 'onebot11' AND subject_id = 'user-1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            persona,
            ("onebot11".to_string(), "before-upgrade".to_string(), 35.0)
        );
        assert!(object_exists(&connection, "table", "persona_nicknames").unwrap());
        assert!(object_exists(&connection, "table", "persona_topics").unwrap());
        assert!(object_exists(&connection, "table", "knowledge_sources").unwrap());
        let nickname_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM persona_nicknames WHERE is_current = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(nickname_count, 1);
        assert!(column_exists(&connection, "knowledge", "protocol").unwrap());
        assert!(column_exists(&connection, "knowledge", "session_id").unwrap());
        let source_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM knowledge_sources", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(source_count, 1);
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 8);
        assert!(!object_exists(&backup, "table", "persona_nicknames").unwrap());
        assert!(!object_exists(&backup, "table", "knowledge_sources").unwrap());
    }

    #[test]
    fn version_nine_database_gains_processing_state_and_unique_action_keys() {
        let temporary = TempDatabase::new("version-nine");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 9).unwrap();
        connection
            .execute(
                "INSERT INTO messages
                 (event_key, protocol, direction, session_type, session_id,
                  sender_id, content, created_at, updated_at)
                 VALUES ('legacy-event', 'onebot11', 'inbound', 'group',
                         'group-1', 'user-1', 'legacy', 10, 10)",
                [],
            )
            .unwrap();
        connection
            .execute_batch(
                "INSERT INTO outbound_messages
                 (action_key, protocol, session_type, session_id, content,
                  status, attempt_count, created_at, updated_at)
                 VALUES ('duplicate-action', 'onebot11', 'group', 'group-1',
                         'first', 'accepted', 1, 10, 10),
                        ('duplicate-action', 'onebot11', 'group', 'group-1',
                         'second', 'rejected', 1, 11, 11);",
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version nine should migrate");
        let connection = database.conn.lock().unwrap();
        assert!(column_exists(&connection, "messages", "processing_status").unwrap());
        assert!(object_exists(&connection, "index", "ux_outbound_action_key").unwrap());
        let processing_status: String = connection
            .query_row(
                "SELECT processing_status FROM messages WHERE event_key = 'legacy-event'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(processing_status, "legacy");
        let distinct_actions: i64 = connection
            .query_row(
                "SELECT COUNT(DISTINCT action_key) FROM outbound_messages",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(distinct_actions, 2);
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 9);
        assert!(!column_exists(&backup, "messages", "processing_status").unwrap());
        assert!(!object_exists(&backup, "index", "ux_outbound_action_key").unwrap());
    }

    #[test]
    fn version_ten_database_gains_protocol_scoped_long_memory() {
        let temporary = TempDatabase::new("version-ten");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 10).unwrap();
        connection
            .execute_batch(
                "INSERT INTO messages
                 (event_key, protocol, direction, session_type, session_id,
                  sender_id, content, created_at, updated_at)
                 VALUES ('unique-onebot', 'onebot11', 'inbound', 'group',
                         'unique-session', 'user-1', 'unique route', 10, 10),
                        ('shared-onebot', 'onebot11', 'inbound', 'group',
                         'shared-session', 'user-1', 'onebot route', 11, 11),
                        ('shared-official', 'qq-official', 'inbound', 'group',
                         'shared-session', 'user-2', 'official route', 12, 12);

                 INSERT INTO long_memory
                 (normalized_key, scope, session_id, content, kind, importance,
                  confidence, privacy, status, version, is_active, created_at, updated_at)
                 VALUES ('legacy-unique', 'session', 'unique-session',
                         'unique memory', 'fact', 60, 80, 'normal', 'active', 1, 1, 10, 10),
                        ('legacy-ambiguous', 'session', 'shared-session',
                         'ambiguous memory', 'fact', 60, 80, 'normal', 'active', 1, 1, 11, 11),
                        ('legacy-global', 'global', NULL,
                         'global memory', 'fact', 60, 80, 'normal', 'active', 1, 1, 12, 12);",
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version ten should migrate");
        let connection = database.conn.lock().unwrap();
        assert!(column_exists(&connection, "long_memory", "protocol").unwrap());
        assert!(column_exists(&connection, "long_memory", "session_type").unwrap());
        assert!(object_exists(&connection, "index", "idx_memory_retrieve_v11").unwrap());
        assert!(object_exists(&connection, "index", "ux_memory_route_key_version").unwrap());
        assert!(!object_exists(&connection, "index", "ux_memory_key_version").unwrap());
        let routes = connection
            .prepare(
                "SELECT normalized_key, protocol, session_type
                 FROM long_memory ORDER BY normalized_key",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            routes,
            vec![
                (
                    "legacy-ambiguous".to_string(),
                    "legacy".to_string(),
                    "legacy".to_string(),
                ),
                (
                    "legacy-global".to_string(),
                    "*".to_string(),
                    "*".to_string(),
                ),
                (
                    "legacy-unique".to_string(),
                    "onebot11".to_string(),
                    "group".to_string(),
                ),
            ]
        );
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 10);
        assert!(!column_exists(&backup, "long_memory", "protocol").unwrap());
        assert!(!object_exists(&backup, "index", "idx_memory_retrieve_v11").unwrap());
        assert!(object_exists(&backup, "index", "ux_memory_key_version").unwrap());
    }

    #[test]
    fn version_eleven_database_gains_short_context_recovery_indexes() {
        let temporary = TempDatabase::new("version-eleven");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 11).unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version eleven should migrate");
        let connection = database.conn.lock().unwrap();
        for index in [
            "idx_messages_recovery_recent_v12",
            "idx_messages_recovery_route_v12",
            "idx_outbound_recovery_recent_v12",
            "idx_outbound_recovery_route_v12",
        ] {
            assert!(object_exists(&connection, "index", index).unwrap());
        }
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 11);
        assert!(!object_exists(&backup, "index", "idx_messages_recovery_recent_v12").unwrap());
    }

    #[test]
    fn version_twelve_database_gains_behavior_calibration_tables() {
        let temporary = TempDatabase::new("version-twelve");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 12).unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version twelve should migrate");
        let connection = database.conn.lock().unwrap();
        assert!(object_exists(&connection, "table", "behavior_calibration").unwrap());
        assert!(object_exists(&connection, "table", "reflection_changes").unwrap());
        assert!(object_exists(&connection, "index", "idx_decision_reflection_v13").unwrap());
        assert!(
            object_exists(
                &connection,
                "index",
                "idx_reflection_changes_status_time_v13"
            )
            .unwrap()
        );
        let calibration: (f64, i64) = connection
            .query_row(
                "SELECT reply_bias_offset, reflection_cursor
                 FROM behavior_calibration WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(calibration, (0.0, 0));
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 12);
        assert!(!object_exists(&backup, "table", "behavior_calibration").unwrap());
        assert!(!object_exists(&backup, "index", "idx_decision_reflection_v13").unwrap());
    }

    #[test]
    fn version_thirteen_database_redacts_media_urls_and_marks_cache_requirements() {
        let temporary = TempDatabase::new("version-thirteen");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 13).unwrap();
        let signed_url = "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=abc&rkey=secret&spec=0#fragment";
        connection
            .execute(
                "INSERT INTO messages
                 (event_key, protocol, direction, session_type, session_id, sender_id,
                  content, media_url, created_at, updated_at)
                 VALUES ('media-event', 'qq-official', 'inbound', 'group', 'group-1',
                         'user-1', '', ?1, 1, 1)",
                [signed_url],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO outbound_messages
                 (action_key, protocol, session_type, session_id, media_url, created_at, updated_at)
                 VALUES ('media-action', 'qq-official', 'group', 'group-1', ?1, 1, 1)",
                [signed_url],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, file_hash, source_session, created_at)
                 VALUES ('qq-official', ?1, 'legacy-url-hash', 'group-1', 1)",
                [signed_url],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version thirteen should migrate");
        let connection = database.conn.lock().unwrap();
        let sticker: (String, String, Option<String>, i64, String) = connection
            .query_row(
                "SELECT media_url, url_hash, file_hash, url_requires_cache, cache_status
                 FROM stickers LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert!(!sticker.0.contains("rkey"));
        assert!(!sticker.0.contains("fragment"));
        assert!(sticker.0.contains("fileid=abc"));
        assert_eq!(sticker.1.len(), 64);
        assert_eq!(sticker.2, None);
        assert_eq!((sticker.3, sticker.4.as_str()), (1, "required"));
        for table in ["messages", "outbound_messages"] {
            let url: String = connection
                .query_row(
                    &format!("SELECT media_url FROM {table} LIMIT 1"),
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert!(!url.contains("rkey"));
            assert!(!url.contains("secret"));
        }
        assert!(object_exists(&connection, "index", "idx_stickers_url_hash_v14").unwrap());
        assert!(object_exists(&connection, "index", "idx_stickers_delivery_v14").unwrap());
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 13);
        assert!(!column_exists(&backup, "stickers", "url_hash").unwrap());
        let raw_url: String = backup
            .query_row("SELECT media_url FROM stickers LIMIT 1", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert!(raw_url.contains("rkey=secret"));
    }

    #[test]
    fn version_fourteen_database_gains_sticker_cache_metadata_and_sources() {
        let temporary = TempDatabase::new("version-fourteen");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 14).unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_user, source_session, usage_count, last_used, created_at, updated_at)
                 VALUES ('onebot11', 'https://example.test/image.png', 'stable-url-hash', 0,
                         'remote', 'user-1', 'session-1', 3, 2, 1, 2)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version fourteen should migrate");
        let connection = database.conn.lock().unwrap();
        for column in ["cache_path", "cache_size", "cache_error"] {
            assert!(column_exists(&connection, "stickers", column).unwrap());
        }
        assert!(object_exists(&connection, "table", "sticker_sources").unwrap());
        for index in [
            "idx_sticker_sources_url_hash_v15",
            "idx_stickers_file_hash_v15",
            "idx_stickers_cache_status_v15",
        ] {
            assert!(object_exists(&connection, "index", index).unwrap());
        }
        let source: (String, String, String, i64, i64, i64) = connection
            .query_row(
                "SELECT url_hash, protocol, source_user, first_seen, last_seen, seen_count
                 FROM sticker_sources",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            source,
            (
                "stable-url-hash".to_string(),
                "onebot11".to_string(),
                "user-1".to_string(),
                1,
                2,
                3,
            )
        );
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 14);
        assert!(!object_exists(&backup, "table", "sticker_sources").unwrap());
        assert!(!column_exists(&backup, "stickers", "cache_path").unwrap());
    }

    #[test]
    fn version_fifteen_database_gains_weighted_sticker_links() {
        let temporary = TempDatabase::new("version-fifteen");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 15).unwrap();
        connection
            .execute(
                "INSERT INTO sticker_links (sticker_a, sticker_b, co_count, updated_at)
                 VALUES (1, 2, 4, 100)",
                [],
            )
            .unwrap();
        drop(connection);

        let database = Database::open(temporary.as_str()).expect("version fifteen should migrate");
        let connection = database.conn.lock().unwrap();
        for column in ["edge_kind", "weight", "evidence_count"] {
            assert!(column_exists(&connection, "sticker_links", column).unwrap());
        }
        let edge: (String, f64, i64) = connection
            .query_row(
                "SELECT edge_kind, weight, evidence_count
                 FROM sticker_links WHERE sticker_a = 1 AND sticker_b = 2",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(edge.0, "cooccur");
        assert_eq!(edge.1, 4.0);
        assert_eq!(edge.2, 4);
        drop(connection);
        drop(database);

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 15);
        assert!(!column_exists(&backup, "sticker_links", "weight").unwrap());
    }

    #[test]
    fn failed_migration_rolls_back_schema_and_version_but_keeps_backup() {
        let temporary = TempDatabase::new("rollback");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 1).unwrap();
        connection
            .execute_batch("CREATE VIEW outbound_messages AS SELECT 1 AS incompatible")
            .unwrap();
        drop(connection);

        let error = match Database::open(temporary.as_str()) {
            Ok(_) => panic!("migration should fail"),
            Err(error) => error,
        };
        assert!(matches!(error, DatabaseError::Sqlite(_)));

        let connection = Connection::open(&temporary.path).unwrap();
        assert_eq!(read_schema_version(&connection).unwrap(), 1);
        assert!(!column_exists(&connection, "messages", "event_key").unwrap());
        assert!(!object_exists(&connection, "index", "ux_msg_event_direction").unwrap());
        assert!(object_exists(&connection, "view", "outbound_messages").unwrap());

        let backups = temporary.backups();
        assert_eq!(backups.len(), 1);
        let backup = Connection::open(&backups[0]).unwrap();
        assert_eq!(read_schema_version(&backup).unwrap(), 1);
        assert!(!column_exists(&backup, "messages", "event_key").unwrap());
    }

    #[test]
    fn newer_schema_is_rejected_without_overwriting_or_backup() {
        let temporary = TempDatabase::new("newer");
        let mut connection = Connection::open(&temporary.path).unwrap();
        migrate_to(&mut connection, 1).unwrap();
        write_schema_version(&connection, LATEST_SCHEMA_VERSION + 1).unwrap();
        drop(connection);

        let error = match Database::open(temporary.as_str()) {
            Ok(_) => panic!("newer schema should fail"),
            Err(error) => error,
        };
        match error {
            DatabaseError::SchemaTooNew { found, supported } => {
                assert_eq!(found, LATEST_SCHEMA_VERSION + 1);
                assert_eq!(supported, LATEST_SCHEMA_VERSION);
            }
            other => panic!("unexpected error: {other}"),
        }
        assert!(temporary.backups().is_empty());

        let connection = Connection::open(&temporary.path).unwrap();
        assert_eq!(
            read_schema_version(&connection).unwrap(),
            LATEST_SCHEMA_VERSION + 1
        );
    }
}
