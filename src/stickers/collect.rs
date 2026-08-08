//! Incoming URL media collection.

use rusqlite::{OptionalExtension, TransactionBehavior};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

use crate::stickers::cache::{CachePolicy, CacheTask};

const COLLECTION_SCORE_THRESHOLD: f32 = 0.45;
const DEFAULT_DAILY_COLLECT_LIMIT: u32 = 100;
const DAY_MILLIS: i64 = 86_400_000;

pub(crate) struct CollectedSticker {
    pub(crate) sticker_id: i64,
    pub(crate) cache_task: Option<CacheTask>,
}

pub(crate) struct CollectionPolicy {
    pub(crate) probability: f32,
    pub(crate) daily_collect_limit: u32,
    pub(crate) cache: Option<(CachePolicy, PathBuf)>,
}

/// Compatibility entry point for callers without policy metadata.
pub async fn maybe_collect(image_url: &str, context: &str) -> bool {
    maybe_collect_with_metadata(
        image_url,
        context,
        "unknown",
        "",
        "",
        CollectionPolicy {
            probability: 1.0,
            daily_collect_limit: DEFAULT_DAILY_COLLECT_LIMIT,
            cache: None,
        },
    )
    .await
    .is_some()
}

/// Apply deterministic sampling, scoring and a daily bound before writing a sticker.
pub(crate) async fn maybe_collect_with_metadata(
    image_url: &str,
    context: &str,
    protocol: &str,
    source_user: &str,
    source_session: &str,
    policy: CollectionPolicy,
) -> Option<CollectedSticker> {
    let media = crate::media::sanitize_remote_media_url(image_url, true)?;
    if is_sensitive_context(context) {
        return None;
    }
    if policy.probability <= 0.0 {
        return None;
    }
    if policy.probability < 1.0
        && deterministic_sample(&media.identity_hash) >= policy.probability.clamp(0.0, 1.0)
    {
        return None;
    }

    let database = crate::pipeline::try_db()?;
    let now = chrono::Utc::now().timestamp_millis();
    let requires_cache = i32::from(media.requires_cache);
    let cache_status = if media.requires_cache {
        "required"
    } else {
        "remote"
    };
    let sticker_id = {
        let mut connection = database.conn.lock().ok()?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .ok()?;
        let existing = transaction
            .query_row(
                "SELECT s.id FROM stickers AS s
                 LEFT JOIN sticker_sources AS source ON source.sticker_id = s.id
                 WHERE s.url_hash = ?1 OR source.url_hash = ?1
                 ORDER BY s.id LIMIT 1",
                rusqlite::params![media.identity_hash],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .ok()?;

        if existing.is_none() {
            let score = collection_score(context, false, true);
            if !daily_collection_has_capacity(&transaction, now, policy.daily_collect_limit).ok()?
                || score < COLLECTION_SCORE_THRESHOLD
            {
                transaction.commit().ok()?;
                return None;
            }
        }

        let sticker_id = if let Some(id) = existing {
            transaction
                .execute(
                    "UPDATE stickers
                     SET usage_count = usage_count + 1, last_used = ?1, updated_at = ?1,
                         media_url = CASE
                             WHEN url_requires_cache = 1 AND ?2 = 0 THEN ?3
                             ELSE media_url
                         END,
                         url_requires_cache = MIN(url_requires_cache, ?2),
                         cache_status = CASE
                             WHEN ?2 = 0 AND cache_status NOT IN ('cached', 'caching', 'queued')
                                 THEN 'remote'
                             ELSE cache_status
                         END
                     WHERE id = ?4",
                    rusqlite::params![now, requires_cache, media.storage_url, id],
                )
                .ok()?;
            id
        } else {
            transaction
                .execute(
                    "INSERT INTO stickers
                     (protocol, kind, media_url, url_hash, url_requires_cache, cache_status,
                      source_user, source_session, usage_count, last_used, created_at, updated_at)
                     VALUES (?1, 'image', ?2, ?3, ?4, ?5, ?6, ?7, 1, ?8, ?8, ?8)",
                    rusqlite::params![
                        protocol,
                        media.storage_url,
                        media.identity_hash,
                        requires_cache,
                        cache_status,
                        source_user,
                        source_session,
                        now
                    ],
                )
                .ok()?;
            transaction.last_insert_rowid()
        };

        transaction
            .execute(
                "INSERT INTO sticker_sources
                     (sticker_id, url_hash, protocol, source_user, source_session,
                      first_seen, last_seen, seen_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, 1)
                 ON CONFLICT(sticker_id, url_hash) DO UPDATE SET
                    last_seen = MAX(sticker_sources.last_seen, excluded.last_seen),
                    seen_count = sticker_sources.seen_count + 1",
                rusqlite::params![
                    sticker_id,
                    media.identity_hash,
                    protocol,
                    source_user,
                    source_session,
                    now
                ],
            )
            .ok()?;

        for tag in tags_from_context(context) {
            transaction
                .execute(
                    "INSERT INTO sticker_tags (sticker_id, tag, weight) VALUES (?1, ?2, 1)
                     ON CONFLICT(sticker_id, tag) DO UPDATE SET weight = weight + 1",
                    rusqlite::params![sticker_id, tag],
                )
                .ok()?;
        }
        transaction.commit().ok()?;
        sticker_id
    };

    let cache_task = policy.cache.and_then(|(cache_policy, cache_root)| {
        crate::stickers::cache::queue_if_needed(
            &database,
            sticker_id,
            cache_root,
            image_url,
            cache_policy,
        )
    });
    Some(CollectedSticker {
        sticker_id,
        cache_task,
    })
}

fn daily_collection_has_capacity(
    transaction: &rusqlite::Transaction<'_>,
    now: i64,
    daily_collect_limit: u32,
) -> rusqlite::Result<bool> {
    let day_start = now.div_euclid(DAY_MILLIS).saturating_mul(DAY_MILLIS);
    let collected_today: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM stickers
         WHERE created_at >= ?1 AND created_at <= ?2",
        rusqlite::params![day_start, now],
        |row| row.get(0),
    )?;
    Ok(collected_today < i64::from(daily_collect_limit.max(1)))
}

pub fn deterministic_sample(value: &str) -> f32 {
    let digest = Sha256::digest(value.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes) as f64 as f32 / u64::MAX as f32
}

/// Score a new candidate using the bounded rule-based collection signals.
pub(crate) fn collection_score(context: &str, is_duplicate: bool, platform_quality: bool) -> f32 {
    let labels = label_candidates(context);
    let emotional_salience = if labels.iter().any(|candidate| {
        matches!(
            candidate.tag.as_str(),
            "开心" | "难过" | "生气" | "哈哈" | "无语" | "震惊"
        )
    }) {
        1.0
    } else {
        0.0
    };
    let topic_relevance = if labels.iter().any(|candidate| {
        matches!(
            candidate.tag.as_str(),
            "猫" | "狗" | "游戏" | "加班" | "下雨" | "群公告" | "问题"
        )
    }) {
        1.0
    } else if !context.trim().is_empty() {
        0.4
    } else {
        0.0
    };
    let repeat_signal = if context.contains("哈哈") || context.contains("!!") {
        0.8
    } else {
        0.0
    };
    let sensitivity = if is_sensitive_context(context) {
        1.0
    } else {
        0.0
    };
    let novelty = if is_duplicate { 0.0 } else { 1.0 };
    let duplicate_score = if is_duplicate { 1.0 } else { 0.0 };
    let platform_quality = if platform_quality { 1.0 } else { 0.0 };
    let score: f32 = 0.30 * novelty
        + 0.20 * emotional_salience
        + 0.20 * topic_relevance
        + 0.15 * repeat_signal
        + 0.10 * 0.5
        + 0.05 * platform_quality
        - 0.35 * sensitivity
        - 0.25 * duplicate_score;
    score.clamp(0.0, 1.0)
}

fn is_sensitive_context(context: &str) -> bool {
    let normalized = context.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "private key",
        "http://",
        "https://",
        "身份证",
        "银行卡",
        "手机号",
        "密码",
        "隐私",
        "私密",
        "病历",
        "医疗",
        "未成年",
        "儿童",
        "色情",
        "裸照",
        "钓鱼",
        "恶意链接",
        "木马",
        "勒索",
        "自杀",
    ];
    MARKERS.iter().any(|marker| {
        if marker.is_ascii() {
            normalized.contains(marker)
        } else {
            context.contains(marker)
        }
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LabelCandidate {
    tag: String,
}

fn label_candidates(context: &str) -> Vec<LabelCandidate> {
    const KNOWN_LABELS: &[&str] = &[
        "开心",
        "难过",
        "生气",
        "哈哈",
        "无语",
        "震惊",
        "可爱",
        "猫",
        "狗",
        "谢谢",
        "恭喜",
        "问号",
        "游戏",
        "加班",
        "下雨",
        "群公告",
        "问题",
    ];
    let mut labels = vec![LabelCandidate {
        tag: "image".to_string(),
    }];
    for &tag in KNOWN_LABELS {
        if context.contains(tag) {
            labels.push(LabelCandidate {
                tag: tag.to_string(),
            });
        }
    }
    labels.sort_by(|left, right| left.tag.cmp(&right.tag));
    labels.dedup_by(|left, right| left.tag == right.tag);
    labels.truncate(8);
    labels
}

fn tags_from_context(context: &str) -> Vec<String> {
    label_candidates(context)
        .into_iter()
        .map(|candidate| candidate.tag)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_is_deterministic_and_bounded() {
        let first = deterministic_sample("https://example.test/a.png");
        assert_eq!(first, deterministic_sample("https://example.test/a.png"));
        assert!((0.0..=1.0).contains(&first));
    }

    #[test]
    fn invalid_media_is_not_collected() {
        assert!(tags_from_context("今天哈哈开心").contains(&"哈哈".to_string()));
        assert!(
            crate::media::sanitize_remote_media_url("http://example.test/a.png", true).is_none()
        );
    }

    #[test]
    fn credential_rotation_keeps_collection_sampling_stable() {
        let first = crate::media::sanitize_remote_media_url(
            "https://example.test/a.png?fileid=1&rkey=old",
            true,
        )
        .unwrap();
        let second = crate::media::sanitize_remote_media_url(
            "https://example.test/a.png?rkey=new&fileid=1",
            true,
        )
        .unwrap();
        assert_eq!(first.identity_hash, second.identity_hash);
        assert_eq!(
            deterministic_sample(&first.identity_hash),
            deterministic_sample(&second.identity_hash)
        );
    }

    #[test]
    fn sensitive_context_is_rejected_before_persistence() {
        assert!(is_sensitive_context("please paste the password"));
        assert!(is_sensitive_context("这是未成年人的私密照片"));
        assert!(is_sensitive_context(
            "https://malicious.example.test/payload"
        ));
        assert!(!is_sensitive_context("今天哈哈开心"));
    }

    #[test]
    fn collection_score_rewards_signal_and_penalizes_duplicates() {
        let new_score = collection_score("猫猫太可爱了哈哈", false, true);
        let duplicate_score = collection_score("猫猫太可爱了哈哈", true, true);
        let unsupported_score = collection_score("猫猫太可爱了哈哈", false, false);
        let low_signal_score = collection_score("", false, true);
        assert!(new_score > duplicate_score);
        assert!(new_score > unsupported_score);
        assert!(low_signal_score < COLLECTION_SCORE_THRESHOLD);
        assert!(new_score >= COLLECTION_SCORE_THRESHOLD);
        assert!((0.0..=1.0).contains(&new_score));
    }

    #[test]
    fn label_candidates_are_deduplicated_and_bounded() {
        let labels = label_candidates("开心难过生气哈哈无语震惊可爱猫狗谢谢恭喜问号游戏加班下雨");
        assert_eq!(
            labels.first().map(|label| label.tag.as_str()),
            Some("image")
        );
        assert!(labels.len() <= 8);
        let cat_labels = label_candidates("猫猫");
        assert_eq!(
            cat_labels.iter().filter(|label| label.tag == "猫").count(),
            1
        );
    }

    #[test]
    fn daily_collection_limit_uses_the_current_utc_day_only() {
        let mut connection = rusqlite::Connection::open_in_memory().unwrap();
        connection
            .execute_batch("CREATE TABLE stickers (created_at INTEGER NOT NULL)")
            .unwrap();
        let now = DAY_MILLIS * 20 + 10;
        connection
            .execute("INSERT INTO stickers (created_at) VALUES (?1)", [now])
            .unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        assert!(!daily_collection_has_capacity(&transaction, now, 1).unwrap());
        transaction.commit().unwrap();

        let next_day = now + DAY_MILLIS;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        assert!(daily_collection_has_capacity(&transaction, next_day, 1).unwrap());
    }
}
