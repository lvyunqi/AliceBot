//! Bounded asynchronous media caching for sticker assets.

use crate::db::Database;
use crate::media::sanitize_remote_media_url;
use base64::Engine;
use reqwest::redirect::Policy;
use rusqlite::OptionalExtension;
use sha2::{Digest, Sha256};
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex as AsyncMutex;

const MIB: u64 = 1_048_576;

/// Per-download and total-disk bounds captured when a task is queued.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CachePolicy {
    pub(crate) max_file_bytes: u64,
    pub(crate) max_total_bytes: u64,
    pub(crate) timeout_sec: u64,
}

impl CachePolicy {
    pub(crate) fn from_config(config: &crate::config::StickerConfig) -> Self {
        Self {
            max_file_bytes: u64::from(config.cache_max_file_mib).saturating_mul(MIB),
            max_total_bytes: u64::from(config.cache_max_total_mib).saturating_mul(MIB),
            timeout_sec: config.cache_timeout_sec,
        }
    }
}

/// An owned cache job. The source URL lives only in memory while the worker downloads it.
pub(crate) struct CacheTask {
    database: Arc<Database>,
    sticker_id: i64,
    source_url: String,
    cache_root: PathBuf,
    policy: CachePolicy,
}

impl CacheTask {
    pub(crate) fn new(
        database: Arc<Database>,
        sticker_id: i64,
        source_url: String,
        cache_root: PathBuf,
        policy: CachePolicy,
    ) -> Self {
        Self {
            database,
            sticker_id,
            source_url,
            cache_root,
            policy,
        }
    }

    /// Put a queued or interrupted task back into a retryable URL-only state.
    pub(crate) fn reset_for_retry(&self, reason: &'static str) {
        let Ok(connection) = self.database.conn.lock() else {
            return;
        };
        let _ = connection.execute(
            "UPDATE stickers
             SET cache_status = CASE WHEN url_requires_cache = 1 THEN 'required' ELSE 'remote' END,
                 cache_error = ?1, updated_at = MAX(COALESCE(updated_at, ?2), ?2)
             WHERE id = ?3 AND cache_status IN ('queued', 'caching')",
            rusqlite::params![
                reason,
                chrono::Utc::now().timestamp_millis(),
                self.sticker_id
            ],
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheError {
    InvalidSource,
    HttpStatus,
    Network,
    Timeout,
    UnsupportedContentType,
    TooLarge,
    Filesystem,
    Database,
}

impl CacheError {
    fn status(self) -> &'static str {
        match self {
            Self::InvalidSource => "invalid_source",
            Self::HttpStatus => "http_status",
            Self::Network => "network_error",
            Self::Timeout => "timeout",
            Self::UnsupportedContentType => "unsupported_content_type",
            Self::TooLarge => "too_large",
            Self::Filesystem => "filesystem_error",
            Self::Database => "database_error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheOutcome {
    Cached,
    Deduplicated,
    QuotaExceeded,
}

/// Process one queued job. All failure details are reduced to a stable category.
pub(crate) async fn process(task: &CacheTask) {
    if !claim_caching(task) {
        return;
    }
    let downloaded = match download_to_temp(task).await {
        Ok(downloaded) => downloaded,
        Err(error) => {
            mark_failed(task, error.status());
            return;
        }
    };
    if let Err(error) = finalize_download(task, downloaded).await {
        mark_failed(task, error.status());
    }
}

fn claim_caching(task: &CacheTask) -> bool {
    let Ok(connection) = task.database.conn.lock() else {
        return false;
    };
    connection
        .execute(
            "UPDATE stickers
             SET cache_status = 'caching', cache_error = NULL,
                 updated_at = MAX(COALESCE(updated_at, ?1), ?1)
             WHERE id = ?2 AND cache_status = 'queued'",
            rusqlite::params![chrono::Utc::now().timestamp_millis(), task.sticker_id],
        )
        .map(|changed| changed == 1)
        .unwrap_or(false)
}

fn mark_failed(task: &CacheTask, status: &'static str) {
    let Ok(connection) = task.database.conn.lock() else {
        return;
    };
    let _ = connection.execute(
        "UPDATE stickers
         SET cache_status = 'failed', cache_path = NULL, cache_size = NULL,
             file_hash = NULL, cache_error = ?1,
             updated_at = MAX(COALESCE(updated_at, ?2), ?2)
         WHERE id = ?3 AND cache_status IN ('queued', 'caching')",
        rusqlite::params![
            status,
            chrono::Utc::now().timestamp_millis(),
            task.sticker_id
        ],
    );
}

/// Mark a URL as queued only once. A full/closed runtime queue calls `reset_for_retry`.
pub(crate) fn queue_if_needed(
    database: &Arc<Database>,
    sticker_id: i64,
    cache_root: PathBuf,
    source_url: &str,
    policy: CachePolicy,
) -> Option<CacheTask> {
    let Ok(connection) = database.conn.lock() else {
        return None;
    };
    let changed = connection
        .execute(
            "UPDATE stickers
             SET cache_status = 'queued', cache_error = NULL,
                 updated_at = MAX(COALESCE(updated_at, ?1), ?1)
             WHERE id = ?2 AND cache_status IN ('remote', 'required', 'failed', 'quota_exceeded')",
            rusqlite::params![chrono::Utc::now().timestamp_millis(), sticker_id],
        )
        .ok()?;
    (changed == 1).then(|| {
        CacheTask::new(
            database.clone(),
            sticker_id,
            source_url.to_owned(),
            cache_root,
            policy,
        )
    })
}

/// Read a cached image for a redacted historical URL. The URL is only used to
/// resolve the content hash; the returned bytes never leave the plugin except
/// as the bounded multimodal request that asked for them.
pub(crate) async fn cached_image_data_for_url(raw_url: &str) -> Option<(String, String)> {
    let database = crate::pipeline::try_db()?;
    let root = crate::pipeline::sticker_cache_root()?;
    let path = cached_image_path_for_url(&database, &root, raw_url)?;
    let metadata = tokio::fs::metadata(&path).await.ok()?;
    if metadata.len() > crate::media::MAX_VISION_IMAGE_BYTES as u64 {
        return None;
    }
    let bytes = tokio::fs::read(path).await.ok()?;
    let media_type = crate::media::image_content_type_from_bytes(&bytes)?;
    Some((
        media_type.to_string(),
        base64::engine::general_purpose::STANDARD.encode(bytes),
    ))
}

fn cached_image_path_for_url(
    database: &Database,
    cache_root: &std::path::Path,
    raw_url: &str,
) -> Option<std::path::PathBuf> {
    let sanitized = sanitize_remote_media_url(raw_url, true)?;
    let cache_path = {
        let connection = database.conn.lock().ok()?;
        connection
            .query_row(
                "SELECT sticker.cache_path
                 FROM stickers AS sticker
                 WHERE sticker.cache_status = 'cached'
                   AND sticker.cache_path IS NOT NULL
                   AND (sticker.url_hash = ?1 OR EXISTS (
                       SELECT 1 FROM sticker_sources AS source
                       WHERE source.sticker_id = sticker.id AND source.url_hash = ?1
                   ))
                 ORDER BY sticker.id ASC
                 LIMIT 1",
                rusqlite::params![sanitized.identity_hash],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .ok()
            .flatten()
    }?;
    let relative = std::path::PathBuf::from(&cache_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return None;
    }
    Some(cache_root.join(relative))
}

async fn download_to_temp(task: &CacheTask) -> Result<DownloadedFile, CacheError> {
    tokio::time::timeout(
        Duration::from_secs(task.policy.timeout_sec.max(1)),
        download_to_temp_inner(task),
    )
    .await
    .map_err(|_| CacheError::Timeout)?
}

async fn download_to_temp_inner(task: &CacheTask) -> Result<DownloadedFile, CacheError> {
    if sanitize_remote_media_url(&task.source_url, true).is_none() {
        return Err(CacheError::InvalidSource);
    }

    let client = http_client_for(&task.source_url).await?;
    let response = client
        .get(&task.source_url)
        .send()
        .await
        .map_err(|_| CacheError::Network)?;
    if !response.status().is_success() {
        return Err(CacheError::HttpStatus);
    }
    if let Some(length) = response.content_length()
        && length > task.policy.max_file_bytes
    {
        return Err(CacheError::TooLarge);
    }
    if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
        let content_type = content_type
            .to_str()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !content_type.is_empty()
            && !content_type.starts_with("image/")
            && content_type != "application/octet-stream"
        {
            return Err(CacheError::UnsupportedContentType);
        }
    }

    let temporary_directory = task.cache_root.join(".tmp");
    fs::create_dir_all(&temporary_directory)
        .await
        .map_err(|_| CacheError::Filesystem)?;
    let nonce = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary_path = temporary_directory.join(format!(
        "cache-{}-{}-{}.part",
        task.sticker_id,
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        nonce
    ));
    let mut temporary = TempFile::new(temporary_path);
    let mut file = fs::File::create(&temporary.path)
        .await
        .map_err(|_| CacheError::Filesystem)?;
    let mut response = response;
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    loop {
        let chunk = response.chunk().await.map_err(|_| CacheError::Network)?;
        let Some(chunk) = chunk else {
            break;
        };
        size = size.saturating_add(chunk.len() as u64);
        if size > task.policy.max_file_bytes {
            return Err(CacheError::TooLarge);
        }
        file.write_all(&chunk)
            .await
            .map_err(|_| CacheError::Filesystem)?;
        digest.update(&chunk);
    }
    file.flush().await.map_err(|_| CacheError::Filesystem)?;
    file.sync_all().await.map_err(|_| CacheError::Filesystem)?;
    drop(file);
    temporary.armed = true;

    Ok(DownloadedFile {
        temporary,
        content_hash: digest_hex(digest),
        size,
    })
}

static FINALIZE_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
static TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

async fn http_client_for(source_url: &str) -> Result<reqwest::Client, CacheError> {
    let url = reqwest::Url::parse(source_url).map_err(|_| CacheError::InvalidSource)?;
    let host = url.host_str().ok_or(CacheError::InvalidSource)?.to_owned();
    let port = url
        .port_or_known_default()
        .ok_or(CacheError::InvalidSource)?;
    let addresses = tokio::net::lookup_host((host.as_str(), port))
        .await
        .map_err(|_| CacheError::InvalidSource)?
        .filter(|address| is_public_media_address(address.ip()))
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(CacheError::InvalidSource);
    }

    // Pin the checked result into this request's resolver to prevent a later DNS lookup
    // from changing a public hostname into a loopback or private destination.
    reqwest::Client::builder()
        .redirect(Policy::none())
        .resolve_to_addrs(&host, &addresses)
        .build()
        .map_err(|_| CacheError::Network)
}

fn is_public_media_address(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            let [first, second, third, _] = address.octets();
            !address.is_private()
                && !address.is_loopback()
                && !address.is_link_local()
                && !address.is_unspecified()
                && !address.is_multicast()
                && !address.is_broadcast()
                && !address.is_documentation()
                && first != 0
                && !(first == 100 && (64..=127).contains(&second))
                && !(first == 192 && second == 0)
                && !(first == 192 && second == 88 && third == 99)
                && !(first == 198 && matches!(second, 18 | 19))
                && first < 240
        }
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return is_public_media_address(IpAddr::V4(mapped));
            }
            let segments = address.segments();
            let special_range = address.is_unspecified()
                || address.is_loopback()
                || address.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
                || (segments[0] == 0x2001 && segments[1] == 0x0002)
                || (segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0x0001);
            !special_range
        }
    }
}

fn finalize_lock() -> &'static AsyncMutex<()> {
    FINALIZE_LOCK.get_or_init(|| AsyncMutex::new(()))
}

struct DownloadedFile {
    temporary: TempFile,
    content_hash: String,
    size: u64,
}

struct TempFile {
    path: PathBuf,
    armed: bool,
}

impl TempFile {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn digest_hex(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

async fn finalize_download(
    task: &CacheTask,
    mut downloaded: DownloadedFile,
) -> Result<CacheOutcome, CacheError> {
    let _guard = finalize_lock().lock().await;
    let duplicate_id = {
        let connection = task
            .database
            .conn
            .lock()
            .map_err(|_| CacheError::Database)?;
        connection
            .query_row(
                "SELECT id FROM stickers
                 WHERE file_hash = ?1 AND cache_status = 'cached' AND id <> ?2
                 ORDER BY id LIMIT 1",
                rusqlite::params![downloaded.content_hash, task.sticker_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(|_| CacheError::Database)?
    };
    if let Some(canonical_id) = duplicate_id {
        remove_temporary(&mut downloaded.temporary).await?;
        merge_duplicate_sticker(&task.database, task.sticker_id, canonical_id)?;
        return Ok(CacheOutcome::Deduplicated);
    }

    let used_bytes = {
        let connection = task
            .database
            .conn
            .lock()
            .map_err(|_| CacheError::Database)?;
        connection
            .query_row(
                "SELECT COALESCE(SUM(cache_size), 0) FROM stickers
                 WHERE cache_status = 'cached'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|_| CacheError::Database)?
            .max(0) as u64
    };
    if used_bytes.saturating_add(downloaded.size) > task.policy.max_total_bytes {
        remove_temporary(&mut downloaded.temporary).await?;
        let connection = task
            .database
            .conn
            .lock()
            .map_err(|_| CacheError::Database)?;
        connection
            .execute(
                "UPDATE stickers
                 SET cache_status = 'quota_exceeded', cache_path = NULL, cache_size = NULL,
                     file_hash = NULL, cache_error = 'quota_exceeded',
                     updated_at = MAX(COALESCE(updated_at, ?1), ?1)
                 WHERE id = ?2 AND cache_status = 'caching'",
                rusqlite::params![chrono::Utc::now().timestamp_millis(), task.sticker_id],
            )
            .map_err(|_| CacheError::Database)?;
        return Ok(CacheOutcome::QuotaExceeded);
    }

    let relative_path = cache_relative_path(&downloaded.content_hash);
    let destination = task.cache_root.join(&relative_path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|_| CacheError::Filesystem)?;
    }
    match fs::metadata(&destination).await {
        Ok(_) => remove_temporary(&mut downloaded.temporary).await?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::rename(&downloaded.temporary.path, &destination)
                .await
                .map_err(|_| CacheError::Filesystem)?;
            downloaded.temporary.disarm();
        }
        Err(_) => return Err(CacheError::Filesystem),
    }

    let connection = task
        .database
        .conn
        .lock()
        .map_err(|_| CacheError::Database)?;
    let changed = connection
        .execute(
            "UPDATE stickers
             SET file_hash = ?1, cache_path = ?2, cache_size = ?3,
                 cache_status = 'cached', cache_error = NULL,
                 updated_at = MAX(COALESCE(updated_at, ?4), ?4)
             WHERE id = ?5 AND cache_status = 'caching'",
            rusqlite::params![
                downloaded.content_hash,
                relative_path.to_string_lossy().replace('\\', "/"),
                downloaded.size as i64,
                chrono::Utc::now().timestamp_millis(),
                task.sticker_id
            ],
        )
        .map_err(|_| CacheError::Database)?;
    if changed != 1 {
        return Err(CacheError::Database);
    }
    Ok(CacheOutcome::Cached)
}

async fn remove_temporary(temporary: &mut TempFile) -> Result<(), CacheError> {
    match fs::remove_file(&temporary.path).await {
        Ok(()) => {
            temporary.disarm();
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            temporary.disarm();
            Ok(())
        }
        Err(_) => Err(CacheError::Filesystem),
    }
}

fn cache_relative_path(content_hash: &str) -> PathBuf {
    PathBuf::from("sha256")
        .join(&content_hash[..2])
        .join(format!("{content_hash}.bin"))
}

fn merge_duplicate_sticker(
    database: &Database,
    duplicate_id: i64,
    canonical_id: i64,
) -> Result<(), CacheError> {
    if duplicate_id == canonical_id {
        return Ok(());
    }
    let mut connection = database.conn.lock().map_err(|_| CacheError::Database)?;
    let transaction = connection
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|_| CacheError::Database)?;
    let now = chrono::Utc::now().timestamp_millis();

    transaction
        .execute(
            "UPDATE stickers
             SET usage_count = COALESCE(usage_count, 0) +
                    (SELECT COALESCE(usage_count, 0) FROM stickers WHERE id = ?2),
                 last_used = MAX(COALESCE(last_used, 0),
                    COALESCE((SELECT last_used FROM stickers WHERE id = ?2), 0)),
                 updated_at = MAX(COALESCE(updated_at, ?3), ?3)
             WHERE id = ?1",
            rusqlite::params![canonical_id, duplicate_id, now],
        )
        .map_err(|_| CacheError::Database)?;

    let tags = {
        let mut statement = transaction
            .prepare("SELECT tag, COALESCE(weight, 1) FROM sticker_tags WHERE sticker_id = ?1")
            .map_err(|_| CacheError::Database)?;
        let rows = statement
            .query_map(rusqlite::params![duplicate_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|_| CacheError::Database)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheError::Database)?
    };
    for (tag, weight) in tags {
        transaction
            .execute(
                "INSERT INTO sticker_tags (sticker_id, tag, weight) VALUES (?1, ?2, ?3)
                 ON CONFLICT(sticker_id, tag) DO UPDATE SET
                    weight = COALESCE(sticker_tags.weight, 0) + excluded.weight",
                rusqlite::params![canonical_id, tag, weight],
            )
            .map_err(|_| CacheError::Database)?;
    }

    let sources = {
        let mut statement = transaction
            .prepare(
                "SELECT url_hash, protocol, source_user, source_session,
                        first_seen, last_seen, seen_count
                 FROM sticker_sources WHERE sticker_id = ?1",
            )
            .map_err(|_| CacheError::Database)?;
        let rows = statement
            .query_map(rusqlite::params![duplicate_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            })
            .map_err(|_| CacheError::Database)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheError::Database)?
    };
    for (url_hash, protocol, source_user, source_session, first_seen, last_seen, seen_count) in
        sources
    {
        transaction
            .execute(
                "INSERT INTO sticker_sources
                     (sticker_id, url_hash, protocol, source_user, source_session,
                      first_seen, last_seen, seen_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(sticker_id, url_hash) DO UPDATE SET
                    first_seen = MIN(sticker_sources.first_seen, excluded.first_seen),
                    last_seen = MAX(sticker_sources.last_seen, excluded.last_seen),
                    seen_count = sticker_sources.seen_count + excluded.seen_count",
                rusqlite::params![
                    canonical_id,
                    url_hash,
                    protocol,
                    source_user,
                    source_session,
                    first_seen,
                    last_seen,
                    seen_count.max(1)
                ],
            )
            .map_err(|_| CacheError::Database)?;
    }

    let links = {
        let mut statement = transaction
            .prepare(
                "SELECT sticker_a, sticker_b, COALESCE(co_count, 1), COALESCE(updated_at, 0)
                 FROM sticker_links WHERE sticker_a = ?1 OR sticker_b = ?1",
            )
            .map_err(|_| CacheError::Database)?;
        let rows = statement
            .query_map(rusqlite::params![duplicate_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })
            .map_err(|_| CacheError::Database)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|_| CacheError::Database)?
    };
    for (sticker_a, sticker_b, co_count, updated_at) in links {
        let other_id = if sticker_a == duplicate_id {
            sticker_b
        } else {
            sticker_a
        };
        transaction
            .execute(
                "DELETE FROM sticker_links WHERE sticker_a = ?1 AND sticker_b = ?2",
                rusqlite::params![sticker_a, sticker_b],
            )
            .map_err(|_| CacheError::Database)?;
        if other_id == canonical_id {
            continue;
        }
        let (new_a, new_b) = if canonical_id < other_id {
            (canonical_id, other_id)
        } else {
            (other_id, canonical_id)
        };
        transaction
            .execute(
                "INSERT INTO sticker_links (sticker_a, sticker_b, co_count, updated_at)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(sticker_a, sticker_b) DO UPDATE SET
                    co_count = sticker_links.co_count + excluded.co_count,
                    updated_at = MAX(sticker_links.updated_at, excluded.updated_at)",
                rusqlite::params![new_a, new_b, co_count, updated_at],
            )
            .map_err(|_| CacheError::Database)?;
    }

    transaction
        .execute(
            "DELETE FROM sticker_tags WHERE sticker_id = ?1",
            rusqlite::params![duplicate_id],
        )
        .map_err(|_| CacheError::Database)?;
    transaction
        .execute(
            "DELETE FROM sticker_sources WHERE sticker_id = ?1",
            rusqlite::params![duplicate_id],
        )
        .map_err(|_| CacheError::Database)?;
    transaction
        .execute(
            "DELETE FROM stickers WHERE id = ?1",
            rusqlite::params![duplicate_id],
        )
        .map_err(|_| CacheError::Database)?;
    transaction.commit().map_err(|_| CacheError::Database)
}

#[cfg(test)]
async fn cache_bytes_for_test(task: &CacheTask, bytes: &[u8]) -> Result<CacheOutcome, CacheError> {
    if bytes.len() as u64 > task.policy.max_file_bytes {
        return Err(CacheError::TooLarge);
    }
    let temporary_directory = task.cache_root.join(".tmp");
    fs::create_dir_all(&temporary_directory)
        .await
        .map_err(|_| CacheError::Filesystem)?;
    let nonce = TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut temporary =
        TempFile::new(temporary_directory.join(format!("test-{}-{nonce}.part", task.sticker_id)));
    let mut file = fs::File::create(&temporary.path)
        .await
        .map_err(|_| CacheError::Filesystem)?;
    file.write_all(bytes)
        .await
        .map_err(|_| CacheError::Filesystem)?;
    file.flush().await.map_err(|_| CacheError::Filesystem)?;
    file.sync_all().await.map_err(|_| CacheError::Filesystem)?;
    drop(file);
    temporary.armed = true;
    let mut digest = Sha256::new();
    digest.update(bytes);
    finalize_download(
        task,
        DownloadedFile {
            temporary,
            content_hash: digest_hex(digest),
            size: bytes.len() as u64,
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "alicebot-sticker-cache-{label}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn policy(max_file_bytes: u64, max_total_bytes: u64) -> CachePolicy {
        CachePolicy {
            max_file_bytes,
            max_total_bytes,
            timeout_sec: 5,
        }
    }

    #[test]
    fn media_download_rejects_private_and_special_network_ranges() {
        for address in [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "172.16.0.1",
            "192.168.1.1",
            "198.18.0.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ] {
            let address = address.parse::<IpAddr>().unwrap();
            assert!(
                !is_public_media_address(address),
                "{address} must be rejected"
            );
        }
        for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            let address = address.parse::<IpAddr>().unwrap();
            assert!(
                is_public_media_address(address),
                "{address} should be allowed"
            );
        }
    }

    fn add_sticker(database: &Database, url_hash: &str, source_url: &str) -> i64 {
        let connection = database.conn.lock().unwrap();
        connection
            .execute(
                "INSERT INTO stickers
                 (protocol, media_url, url_hash, url_requires_cache, cache_status,
                  source_user, source_session, usage_count, created_at, updated_at)
                 VALUES ('onebot11', ?1, ?2, 0, 'caching', 'user', 'session', 1, 1, 1)",
                params![source_url, url_hash],
            )
            .unwrap();
        let id = connection.last_insert_rowid();
        connection
            .execute(
                "INSERT INTO sticker_sources
                 (sticker_id, url_hash, protocol, source_user, source_session,
                  first_seen, last_seen, seen_count)
                 VALUES (?1, ?2, 'onebot11', 'user', 'session', 1, 1, 1)",
                params![id, url_hash],
            )
            .unwrap();
        id
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cache_writes_content_hash_and_atomic_relative_path() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("atomic");
        let id = add_sticker(&database, "url-1", "https://example.test/image.png");
        let task = CacheTask::new(
            database.clone(),
            id,
            "https://example.test/image.png".to_string(),
            root.clone(),
            policy(1024, 4096),
        );

        let outcome = cache_bytes_for_test(&task, b"image-bytes").await.unwrap();
        assert_eq!(outcome, CacheOutcome::Cached);
        let (status, file_hash, cache_path, cache_size): (String, String, String, i64) = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT cache_status, file_hash, cache_path, cache_size FROM stickers WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        let expected_hash = {
            let mut digest = Sha256::new();
            digest.update(b"image-bytes");
            digest_hex(digest)
        };
        assert_eq!(status, "cached");
        assert_eq!(file_hash, expected_hash);
        assert_eq!(cache_size, 11);
        let cached_path = root.join(&cache_path);
        assert_eq!(std::fs::read(&cached_path).unwrap(), b"image-bytes");
        let temporary_files = std::fs::read_dir(root.join(".tmp"))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(temporary_files.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn redacted_historical_url_resolves_the_surviving_cache_file() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("history-lookup");
        let raw_url =
            "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=abc&rkey=temporary&spec=0";
        let sanitized = sanitize_remote_media_url(raw_url, true).unwrap();
        let id = add_sticker(&database, &sanitized.identity_hash, &sanitized.storage_url);
        let task = CacheTask::new(
            database.clone(),
            id,
            raw_url.to_string(),
            root.clone(),
            policy(1024, 4096),
        );
        let bytes = b"\x89PNG\r\n\x1a\nrestored-image";
        cache_bytes_for_test(&task, bytes).await.unwrap();

        let cached = cached_image_path_for_url(&database, &root, &sanitized.storage_url)
            .expect("redacted URL should resolve cached content");
        assert_eq!(std::fs::read(cached).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn identical_content_merges_sources_and_does_not_duplicate_assets() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("dedup");
        let first = add_sticker(&database, "url-1", "https://example.test/one.png");
        let second = add_sticker(&database, "url-2", "https://example.test/two.png");
        let first_task = CacheTask::new(
            database.clone(),
            first,
            "https://example.test/one.png".to_string(),
            root.clone(),
            policy(1024, 4096),
        );
        let second_task = CacheTask::new(
            database.clone(),
            second,
            "https://example.test/two.png".to_string(),
            root.clone(),
            policy(1024, 4096),
        );
        assert_eq!(
            cache_bytes_for_test(&first_task, b"same-image")
                .await
                .unwrap(),
            CacheOutcome::Cached
        );
        assert_eq!(
            cache_bytes_for_test(&second_task, b"same-image")
                .await
                .unwrap(),
            CacheOutcome::Deduplicated
        );

        let connection = database.conn.lock().unwrap();
        let sticker_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM stickers", [], |row| row.get(0))
            .unwrap();
        let source_count: i64 = connection
            .query_row("SELECT COUNT(*) FROM sticker_sources", [], |row| row.get(0))
            .unwrap();
        assert_eq!(sticker_count, 1);
        assert_eq!(source_count, 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn oversized_content_leaves_no_cache_file_or_hash() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("oversized");
        let id = add_sticker(&database, "url-1", "https://example.test/image.png");
        let task = CacheTask::new(
            database.clone(),
            id,
            "https://example.test/image.png".to_string(),
            root.clone(),
            policy(3, 4096),
        );
        assert_eq!(
            cache_bytes_for_test(&task, b"too-large").await,
            Err(CacheError::TooLarge)
        );
        mark_failed(&task, CacheError::TooLarge.status());
        let row: (String, Option<String>, Option<String>) = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT cache_status, file_hash, cache_path FROM stickers WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, ("failed".to_string(), None, None));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_source_worker_failure_leaves_a_retryable_url_only_record() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("invalid-source");
        let id = add_sticker(&database, "url-1", "http://example.test/image.png");
        database
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE stickers SET cache_status = 'queued' WHERE id = ?1",
                params![id],
            )
            .unwrap();
        let task = CacheTask::new(
            database.clone(),
            id,
            "http://example.test/image.png".to_string(),
            root.clone(),
            policy(1024, 4096),
        );

        process(&task).await;

        let row: (String, Option<String>, Option<String>, Option<String>) = database
            .conn
            .lock()
            .unwrap()
            .query_row(
                "SELECT cache_status, file_hash, cache_path, cache_error
                 FROM stickers WHERE id = ?1",
                params![id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            (
                "failed".to_string(),
                None,
                None,
                Some("invalid_source".to_string()),
            )
        );
        assert!(!root.join(".tmp").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn total_cache_quota_preserves_existing_asset_and_leaves_new_url_only() {
        let database = Arc::new(Database::open(":memory:").unwrap());
        let root = temp_root("quota");
        let first = add_sticker(&database, "url-1", "https://example.test/one.png");
        let second = add_sticker(&database, "url-2", "https://example.test/two.png");
        let first_task = CacheTask::new(
            database.clone(),
            first,
            "https://example.test/one.png".to_string(),
            root.clone(),
            policy(16, 4),
        );
        let second_task = CacheTask::new(
            database.clone(),
            second,
            "https://example.test/two.png".to_string(),
            root.clone(),
            policy(16, 4),
        );

        assert_eq!(
            cache_bytes_for_test(&first_task, b"abc").await.unwrap(),
            CacheOutcome::Cached
        );
        assert_eq!(
            cache_bytes_for_test(&second_task, b"de").await.unwrap(),
            CacheOutcome::QuotaExceeded
        );

        let connection = database.conn.lock().unwrap();
        let cached_count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM stickers WHERE cache_status = 'cached'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let rejected: (String, Option<String>, Option<String>) = connection
            .query_row(
                "SELECT cache_status, file_hash, cache_path FROM stickers WHERE id = ?1",
                params![second],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(cached_count, 1);
        assert_eq!(rejected, ("quota_exceeded".to_string(), None, None));
        drop(connection);
        let temporary_files = std::fs::read_dir(root.join(".tmp"))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(temporary_files.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
