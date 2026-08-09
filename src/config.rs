//! 配置解析
//!
//! 从 QimenBot 宿主传入的 JSON 配置解析为类型安全的 Rust 结构。
//! API 0.6 的 Schema 校验由宿主完成，这里只做 Rust 层反序列化。

use serde::Deserialize;
use std::collections::HashSet;
use std::fmt;

pub(crate) const MAX_REFLECTION_LEARNING_RATE: f32 = 0.05;
pub(crate) const MIN_REFLECTION_TARGET_AUTONOMOUS_RATE: f32 = 0.05;
pub(crate) const MAX_REFLECTION_TARGET_AUTONOMOUS_RATE: f32 = 0.45;

/// 应用配置（完整）
#[derive(Clone, Default, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub persona: PersonaConfig,

    #[serde(default)]
    pub llm: LlmConfig,

    #[serde(default)]
    pub behavior: BehaviorConfig,

    #[serde(default)]
    pub decision: DecisionConfig,

    #[serde(default)]
    pub memories: MemoryConfig,

    #[serde(default)]
    pub privacy: PrivacyConfig,

    #[serde(default)]
    pub observability: ObservabilityConfig,

    #[serde(default)]
    pub stickers: StickerConfig,

    #[serde(default)]
    pub send: SendConfig,
}

impl fmt::Debug for AppConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AppConfig")
            .field("persona", &"[REDACTED]")
            .field("llm", &self.llm)
            .field("behavior", &self.behavior)
            .field("decision", &self.decision)
            .field("memories", &self.memories)
            .field("privacy", &self.privacy)
            .field("observability", &self.observability)
            .field("stickers", &self.stickers)
            .field("send", &"[REDACTED]")
            .finish()
    }
}

impl AppConfig {
    pub fn reply_bias(&self) -> f32 {
        self.decision
            .reply_bias
            .unwrap_or(self.behavior.reply_bias)
            .clamp(0.0, 1.0)
    }

    pub fn min_interval_sec(&self) -> u64 {
        self.decision
            .min_interval_sec
            .unwrap_or(self.behavior.min_interval_sec)
            .clamp(1, 300)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SendConfig {
    /// 宿主 bots.account_id 对应的稳定账号选择器，不是 secret。
    #[serde(default)]
    pub account_id: String,
}

/// 人物立绘
#[derive(Debug, Clone, Deserialize)]
pub struct PersonaConfig {
    #[serde(default = "default_name")]
    pub name: String,

    #[serde(default = "default_gender")]
    pub gender: String,

    #[serde(default)]
    pub age: u8,

    #[serde(default)]
    pub personality: String,

    #[serde(default)]
    pub background: String,

    #[serde(default)]
    pub speaking_style: String,
}

fn default_name() -> String {
    "Alice".into()
}
fn default_gender() -> String {
    "女".into()
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            name: default_name(),
            gender: default_gender(),
            age: 22,
            personality: "俏皮、有点毒舌、好奇心强".into(),
            background: "一个喜欢猫和咖啡的二次元少女".into(),
            speaking_style: "简短口语化，直接回答，不堆表情或语气词".into(),
        }
    }
}

/// LLM 提供商配置
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_llm_timeout")]
    pub request_timeout_ms: u64,

    #[serde(default = "default_retry_limit")]
    pub retry_limit: u32,

    /// Enable bounded native tool-calling for conversational replies.
    #[serde(default = "default_true")]
    pub agent_enabled: bool,

    /// Maximum tool rounds per reply; kept deliberately small to bound cost.
    #[serde(default = "default_agent_max_steps")]
    pub agent_max_steps: u32,

    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            request_timeout_ms: default_llm_timeout(),
            retry_limit: default_retry_limit(),
            agent_enabled: true,
            agent_max_steps: default_agent_max_steps(),
            providers: Vec::new(),
        }
    }
}

fn default_llm_timeout() -> u64 {
    30_000
}

fn default_retry_limit() -> u32 {
    1
}

fn default_agent_max_steps() -> u32 {
    3
}

#[derive(Clone, Deserialize)]
pub struct ProviderConfig {
    pub id: String,

    #[serde(default = "default_protocol")]
    pub protocol: String,

    #[serde(default = "default_base_url")]
    pub base_url: String,

    #[serde(default)]
    pub api_key: String,

    #[serde(default = "default_model")]
    pub model: String,

    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub priority: u32,

    /// Whether this provider/model accepts image content blocks.
    #[serde(default)]
    pub supports_vision: bool,

    /// Optional model override used when the current turn contains images.
    #[serde(default)]
    pub vision_model: Option<String>,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("id", &self.id)
            .field("protocol", &self.protocol)
            .field("base_url", &"[REDACTED]")
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("enabled", &self.enabled)
            .field("priority", &self.priority)
            .field("supports_vision", &self.supports_vision)
            .field("vision_model", &self.vision_model)
            .finish()
    }
}

fn default_protocol() -> String {
    "openai".into()
}
fn default_base_url() -> String {
    "https://api.openai.com/v1".into()
}
fn default_model() -> String {
    "gpt-4o-mini".into()
}
fn default_true() -> bool {
    true
}

/// 拟人行为参数
#[derive(Debug, Clone, Deserialize)]
pub struct BehaviorConfig {
    #[serde(default = "default_reply_bias")]
    pub reply_bias: f32,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    #[serde(default = "default_max_context")]
    pub max_context_tokens: u32,

    #[serde(default = "default_min_interval")]
    pub min_interval_sec: u64,

    #[serde(default = "default_allow_typos")]
    pub allow_typos: bool,

    #[serde(default = "default_emoji_usage")]
    pub emoji_usage: f32,
}

fn default_reply_bias() -> f32 {
    0.5
}
fn default_temperature() -> f32 {
    1.0
}
fn default_max_tokens() -> u32 {
    512
}
fn default_max_context() -> u32 {
    4000
}
fn default_min_interval() -> u64 {
    15
}
fn default_allow_typos() -> bool {
    false
}
fn default_emoji_usage() -> f32 {
    0.1
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            reply_bias: default_reply_bias(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            max_context_tokens: default_max_context(),
            min_interval_sec: default_min_interval(),
            allow_typos: default_allow_typos(),
            emoji_usage: default_emoji_usage(),
        }
    }
}

/// 自主回复策略。概率和冷却字段可在一个配置版本内回退到旧的 `behavior` 位置。
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// 仅在规则评分模糊区调用可选 LLM 分类器，默认关闭以避免额外模型成本。
    #[serde(default)]
    pub reply_judge_enabled: bool,

    #[serde(default)]
    pub reply_bias: Option<f32>,

    #[serde(default)]
    pub min_interval_sec: Option<u64>,

    #[serde(default = "default_coalesce_window_ms")]
    pub coalesce_window_ms: u64,

    #[serde(default = "default_activity_alpha")]
    pub activity_alpha: f32,

    #[serde(default = "default_quiet_threshold")]
    pub quiet_threshold: f32,

    #[serde(default = "default_burst_threshold")]
    pub burst_threshold: f32,
}

fn default_coalesce_window_ms() -> u64 {
    900
}

fn default_activity_alpha() -> f32 {
    0.35
}

fn default_quiet_threshold() -> f32 {
    0.25
}

fn default_burst_threshold() -> f32 {
    0.85
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reply_judge_enabled: false,
            reply_bias: None,
            min_interval_sec: None,
            coalesce_window_ms: default_coalesce_window_ms(),
            activity_alpha: default_activity_alpha(),
            quiet_threshold: default_quiet_threshold(),
            burst_threshold: default_burst_threshold(),
        }
    }
}

/// 记忆策略
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_short_size")]
    pub short_size: usize,

    #[serde(default = "default_long_topk")]
    pub long_topk: usize,

    #[serde(default = "default_compress_interval")]
    pub compress_interval_hours: u64,

    #[serde(default = "default_compress_messages")]
    pub compress_min_messages: u64,

    #[serde(default = "default_importance_threshold")]
    pub importance_threshold: u32,

    #[serde(default = "default_true")]
    pub knowledge_enabled: bool,

    #[serde(default = "default_true")]
    pub reflection_enabled: bool,

    #[serde(default = "default_reflection_interval")]
    pub reflection_interval_hours: u64,

    #[serde(default = "default_reflection_min_decisions")]
    pub reflection_min_decisions: u64,

    #[serde(default = "default_reflection_learning_rate")]
    pub reflection_learning_rate: f32,

    #[serde(default = "default_reflection_target_autonomous_rate")]
    pub reflection_target_autonomous_rate: f32,
}

fn default_short_size() -> usize {
    30
}
fn default_long_topk() -> usize {
    10
}
fn default_compress_interval() -> u64 {
    6
}
fn default_compress_messages() -> u64 {
    200
}
fn default_importance_threshold() -> u32 {
    30
}
fn default_reflection_interval() -> u64 {
    24
}
fn default_reflection_min_decisions() -> u64 {
    100
}
fn default_reflection_learning_rate() -> f32 {
    0.02
}
fn default_reflection_target_autonomous_rate() -> f32 {
    0.20
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_size: default_short_size(),
            long_topk: default_long_topk(),
            compress_interval_hours: default_compress_interval(),
            compress_min_messages: default_compress_messages(),
            importance_threshold: default_importance_threshold(),
            knowledge_enabled: true,
            reflection_enabled: true,
            reflection_interval_hours: default_reflection_interval(),
            reflection_min_decisions: default_reflection_min_decisions(),
            reflection_learning_rate: default_reflection_learning_rate(),
            reflection_target_autonomous_rate: default_reflection_target_autonomous_rate(),
        }
    }
}

/// 原始事件和消息 journal 的隐私保留边界。
#[derive(Debug, Clone, Deserialize)]
pub struct PrivacyConfig {
    /// 是否在 journal 中保存已经脱敏的原始事件 JSON。
    #[serde(default = "default_true")]
    pub store_raw_events: bool,

    /// 原始事件 JSON 的最长保留天数，正文 journal 可继续保留。
    #[serde(default = "default_raw_event_retention_days")]
    pub raw_event_retention_days: u32,

    /// 未被派生数据引用的已完成入站事件最长保留天数。
    #[serde(default = "default_message_retention_days")]
    pub message_retention_days: u32,
}

fn default_raw_event_retention_days() -> u32 {
    30
}

fn default_message_retention_days() -> u32 {
    180
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            store_raw_events: true,
            raw_event_retention_days: default_raw_event_retention_days(),
            message_retention_days: default_message_retention_days(),
        }
    }
}

/// 脱敏审计和临时协议诊断的开关。
#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    /// AliceBot 可选诊断的最低日志级别，不修改宿主全局日志过滤器。
    #[serde(default = "default_observability_level")]
    pub level: String,

    /// 是否持久化可回放的脱敏决策轨迹。
    #[serde(default = "default_true")]
    pub decision_trace: bool,

    /// 是否持久化不含提示词和响应正文的 LLM 调用指标。
    #[serde(default = "default_true")]
    pub llm_metrics: bool,

    /// 是否输出已脱敏的原始协议事件，仅用于短时排错。
    #[serde(default)]
    pub raw_protocol_debug: bool,
}

fn default_observability_level() -> String {
    "info".to_string()
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            level: default_observability_level(),
            decision_trace: true,
            llm_metrics: true,
            raw_protocol_debug: false,
        }
    }
}

impl ObservabilityConfig {
    /// 原始协议日志只能在显式调试级别下输出，避免普通信息日志泄露用户事件。
    pub(crate) fn raw_protocol_debug_enabled(&self) -> bool {
        self.raw_protocol_debug && matches!(self.level.as_str(), "debug" | "trace")
    }
}

/// 表情包策略
#[derive(Debug, Clone, Deserialize)]
pub struct StickerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_true")]
    pub auto_collect: bool,

    /// 每 UTC 日最多新增收藏的媒体数量。
    #[serde(default = "default_daily_collect_limit")]
    pub daily_collect_limit: u32,

    /// 是否把已收藏的远程媒体异步缓存到插件数据目录。
    #[serde(default = "default_true")]
    pub cache_media: bool,

    /// 单个缓存媒体的最大体积（MiB）。
    #[serde(default = "default_cache_max_file_mib")]
    pub cache_max_file_mib: u32,

    /// 全部缓存媒体的最大体积（MiB）。
    #[serde(default = "default_cache_max_total_mib")]
    pub cache_max_total_mib: u32,

    /// 单次媒体下载的超时秒数。
    #[serde(default = "default_cache_timeout_sec")]
    pub cache_timeout_sec: u64,

    #[serde(default = "default_collect_prob")]
    pub collect_probability: f32,

    #[serde(default = "default_send_prob")]
    pub send_probability: f32,

    /// 每 UTC 日最多接受的表情包 URL 发送数。
    #[serde(default = "default_daily_send_limit")]
    pub daily_send_limit: u32,

    /// 同一路由两次自动表情包发送的最小间隔。
    #[serde(default = "default_sticker_cooldown_sec")]
    pub sticker_cooldown_sec: u64,

    #[serde(default = "default_true")]
    pub link_enabled: bool,

    #[serde(default = "default_max_chain")]
    pub max_chain: u32,
}

fn default_collect_prob() -> f32 {
    1.0
}
fn default_daily_collect_limit() -> u32 {
    100
}
fn default_cache_max_file_mib() -> u32 {
    8
}
fn default_cache_max_total_mib() -> u32 {
    256
}
fn default_cache_timeout_sec() -> u64 {
    15
}
fn default_send_prob() -> f32 {
    0.4
}
fn default_daily_send_limit() -> u32 {
    30
}
fn default_sticker_cooldown_sec() -> u64 {
    300
}
fn default_max_chain() -> u32 {
    2
}

impl Default for StickerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_collect: true,
            daily_collect_limit: default_daily_collect_limit(),
            cache_media: true,
            cache_max_file_mib: default_cache_max_file_mib(),
            cache_max_total_mib: default_cache_max_total_mib(),
            cache_timeout_sec: default_cache_timeout_sec(),
            collect_probability: default_collect_prob(),
            send_probability: default_send_prob(),
            daily_send_limit: default_daily_send_limit(),
            sticker_cooldown_sec: default_sticker_cooldown_sec(),
            link_enabled: true,
            max_chain: default_max_chain(),
        }
    }
}

/// 解析宿主传入的 JSON 配置
pub fn parse_config(json: &str) -> Result<AppConfig, String> {
    if json.is_empty() {
        return Ok(AppConfig::default());
    }
    serde_json::from_str::<AppConfig>(json).map_err(|e| format!("JSON 解析失败: {}", e))
}

pub fn parse_and_validate_config(json: &str) -> Result<AppConfig, String> {
    let config = parse_config(json)?;
    validate(&config)?;
    Ok(config)
}

fn validate(config: &AppConfig) -> Result<(), String> {
    let quiet = config.decision.quiet_threshold;
    let burst = config.decision.burst_threshold;
    if !(0.0..=1.0).contains(&quiet) || !(0.0..=1.0).contains(&burst) {
        return Err("decision quiet/burst thresholds must be between 0 and 1".to_string());
    }
    if quiet >= burst {
        return Err("decision.quiet_threshold must be lower than burst_threshold".to_string());
    }
    if !(0.05..=1.0).contains(&config.decision.activity_alpha) {
        return Err("decision.activity_alpha must be between 0.05 and 1".to_string());
    }
    if config.decision.coalesce_window_ms > 3_000 {
        return Err("decision.coalesce_window_ms cannot exceed 3000".to_string());
    }

    let memories = &config.memories;
    if !(1..=168).contains(&memories.reflection_interval_hours) {
        return Err("memories.reflection_interval_hours must be between 1 and 168".to_string());
    }
    if !(10..=10_000).contains(&memories.reflection_min_decisions) {
        return Err("memories.reflection_min_decisions must be between 10 and 10000".to_string());
    }
    if !(0.0..=MAX_REFLECTION_LEARNING_RATE).contains(&memories.reflection_learning_rate)
        || memories.reflection_learning_rate == 0.0
    {
        return Err(format!(
            "memories.reflection_learning_rate must be greater than 0 and no more than {MAX_REFLECTION_LEARNING_RATE}"
        ));
    }
    if !(MIN_REFLECTION_TARGET_AUTONOMOUS_RATE..=MAX_REFLECTION_TARGET_AUTONOMOUS_RATE)
        .contains(&memories.reflection_target_autonomous_rate)
    {
        return Err(format!(
            "memories.reflection_target_autonomous_rate must be between {MIN_REFLECTION_TARGET_AUTONOMOUS_RATE} and {MAX_REFLECTION_TARGET_AUTONOMOUS_RATE}"
        ));
    }

    let privacy = &config.privacy;
    if !(1..=3_650).contains(&privacy.raw_event_retention_days) {
        return Err("privacy.raw_event_retention_days must be between 1 and 3650".to_string());
    }
    if !(1..=3_650).contains(&privacy.message_retention_days) {
        return Err("privacy.message_retention_days must be between 1 and 3650".to_string());
    }
    if privacy.raw_event_retention_days > privacy.message_retention_days {
        return Err(
            "privacy.raw_event_retention_days cannot exceed message_retention_days".to_string(),
        );
    }

    let observability = &config.observability;
    if !matches!(
        observability.level.as_str(),
        "error" | "warn" | "info" | "debug" | "trace"
    ) {
        return Err("observability.level must be error, warn, info, debug, or trace".to_string());
    }
    if observability.raw_protocol_debug && !observability.raw_protocol_debug_enabled() {
        return Err(
            "observability.raw_protocol_debug requires observability.level debug or trace"
                .to_string(),
        );
    }

    let stickers = &config.stickers;
    if !(1..=1_000).contains(&stickers.daily_collect_limit) {
        return Err("stickers.daily_collect_limit must be between 1 and 1000".to_string());
    }
    if !(1..=32).contains(&stickers.cache_max_file_mib) {
        return Err("stickers.cache_max_file_mib must be between 1 and 32".to_string());
    }
    if !(16..=1_024).contains(&stickers.cache_max_total_mib) {
        return Err("stickers.cache_max_total_mib must be between 16 and 1024".to_string());
    }
    if stickers.cache_max_total_mib < stickers.cache_max_file_mib {
        return Err(
            "stickers.cache_max_total_mib cannot be smaller than cache_max_file_mib".to_string(),
        );
    }
    if !(1..=60).contains(&stickers.cache_timeout_sec) {
        return Err("stickers.cache_timeout_sec must be between 1 and 60".to_string());
    }
    if !(1..=1_000).contains(&stickers.daily_send_limit) {
        return Err("stickers.daily_send_limit must be between 1 and 1000".to_string());
    }
    if stickers.sticker_cooldown_sec > 86_400 {
        return Err("stickers.sticker_cooldown_sec cannot exceed 86400".to_string());
    }
    if !(1..=3).contains(&stickers.max_chain) {
        return Err("stickers.max_chain must be between 1 and 3".to_string());
    }
    if !(1..=5).contains(&config.llm.agent_max_steps) {
        return Err("llm.agent_max_steps must be between 1 and 5".to_string());
    }

    let mut provider_ids = HashSet::new();
    for provider in &config.llm.providers {
        let id = provider.id.trim();
        if id.is_empty() {
            return Err("LLM provider id cannot be empty".to_string());
        }
        if !provider_ids.insert(id) {
            return Err(format!("duplicate LLM provider id: {id}"));
        }
        if !matches!(provider.protocol.as_str(), "openai" | "anthropic") {
            return Err(format!(
                "unsupported protocol for provider {id}: {}",
                provider.protocol
            ));
        }
        if provider
            .vision_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(format!("vision_model for provider {id} cannot be empty"));
        }
        let url = reqwest::Url::parse(&provider.base_url)
            .map_err(|_| format!("invalid base_url for provider {id}"))?;
        if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
            return Err(format!(
                "provider {id} base_url must be an absolute HTTP(S) URL"
            ));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(format!(
                "provider {id} base_url must not contain credentials"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_behavior_values_remain_effective() {
        let config =
            parse_and_validate_config(r#"{"behavior":{"reply_bias":0.2,"min_interval_sec":45}}"#)
                .expect("legacy configuration should parse and validate");
        assert_eq!(config.reply_bias(), 0.2);
        assert_eq!(config.min_interval_sec(), 45);
    }

    #[test]
    fn decision_values_override_legacy_locations() {
        let config = parse_and_validate_config(
            r#"{
                "behavior":{"reply_bias":0.2,"min_interval_sec":45},
                "decision":{"reply_bias":0.8,"min_interval_sec":9}
            }"#,
        )
        .expect("new configuration should parse and validate");
        assert_eq!(config.reply_bias(), 0.8);
        assert_eq!(config.min_interval_sec(), 9);
    }

    #[test]
    fn business_validation_rejects_threshold_inversion() {
        let error = parse_and_validate_config(
            r#"{"decision":{"quiet_threshold":0.9,"burst_threshold":0.4}}"#,
        )
        .expect_err("inverted thresholds should fail");
        assert!(error.contains("quiet_threshold"));
    }

    #[test]
    fn business_validation_rejects_unsafe_reflection_parameters() {
        let learning_rate =
            parse_and_validate_config(r#"{"memories":{"reflection_learning_rate":0.06}}"#)
                .expect_err("learning rate above the hard limit should fail");
        assert!(learning_rate.contains("reflection_learning_rate"));

        let target =
            parse_and_validate_config(r#"{"memories":{"reflection_target_autonomous_rate":0.5}}"#)
                .expect_err("target above the safety range should fail");
        assert!(target.contains("reflection_target_autonomous_rate"));
    }

    #[test]
    fn business_validation_rejects_duplicate_provider_ids() {
        let error = parse_and_validate_config(
            r#"{
                "llm":{"providers":[
                    {"id":"same","base_url":"https://example.com","protocol":"openai"},
                    {"id":"same","base_url":"https://example.org","protocol":"anthropic"}
                ]}
            }"#,
        )
        .expect_err("duplicate provider IDs should fail");
        assert!(error.contains("duplicate"));
        assert!(!error.contains("api_key"));
    }

    #[test]
    fn business_validation_rejects_non_http_provider_url_without_echoing_it() {
        let error = parse_and_validate_config(
            r#"{
                "llm":{"providers":[
                    {"id":"primary","base_url":"ftp://user:secret@example.com","protocol":"openai"}
                ]}
            }"#,
        )
        .expect_err("non-HTTP provider URL should fail");
        assert!(error.contains("HTTP(S)"));
        assert!(!error.contains("secret"));
    }

    #[test]
    fn configuration_debug_redacts_secrets_and_prompt_fields() {
        let config = parse_and_validate_config(
            r#"{
                "persona":{"background":"persona-secret"},
                "send":{"account_id":"account-secret"},
                "llm":{"providers":[{
                    "id":"primary",
                    "base_url":"https://example.com/v1?access_token=url-secret",
                    "api_key":"api-key-secret",
                    "protocol":"openai"
                }]}
            }"#,
        )
        .expect("configuration should be valid");

        let debug = format!("{config:?}");
        assert!(!debug.contains("persona-secret"));
        assert!(!debug.contains("account-secret"));
        assert!(!debug.contains("url-secret"));
        assert!(!debug.contains("api-key-secret"));
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn schema_exposes_decision_section_and_hides_legacy_fields() {
        let schema: serde_json::Value = serde_json::from_str(include_str!("../config.schema.json"))
            .expect("configuration schema should be valid JSON");
        assert!(schema["properties"]["decision"].is_object());
        assert!(schema["properties"]["decision"]["properties"]["reply_judge_enabled"].is_object());
        assert!(schema["properties"]["memories"]["properties"]["reflection_enabled"].is_object());
        assert!(schema["properties"]["privacy"]["properties"]["store_raw_events"].is_object());
        assert!(schema["properties"]["observability"]["properties"]["decision_trace"].is_object());
        assert!(schema["properties"]["stickers"]["properties"]["daily_collect_limit"].is_object());
        assert!(schema["properties"]["llm"]["properties"]["providers"]["items"]["properties"]["supports_vision"].is_object());
        assert!(schema["properties"]["llm"]["properties"]["providers"]["items"]["properties"]["vision_model"].is_object());
        let provider_required =
            schema["properties"]["llm"]["properties"]["providers"]["items"]["required"]
                .as_array()
                .expect("provider required fields should be an array");
        assert!(
            !provider_required
                .iter()
                .any(|field| field.as_str() == Some("api_key"))
        );
        assert!(
            schema["properties"]["privacy"]["properties"]["raw_event_retention_days"].is_object()
        );
        assert!(schema["properties"]["behavior"]["properties"]["reply_bias"].is_null());
        assert!(schema["properties"]["behavior"]["properties"]["min_interval_sec"].is_null());
    }

    #[test]
    fn reply_judge_is_explicitly_opt_in() {
        assert!(!AppConfig::default().decision.reply_judge_enabled);

        let config = parse_and_validate_config(r#"{"decision":{"reply_judge_enabled":true}}"#)
            .expect("reply judge configuration should be valid");
        assert!(config.decision.reply_judge_enabled);
    }

    #[test]
    fn concise_chat_defaults_match_the_configuration_schema() {
        let defaults = AppConfig::default();
        assert_eq!(
            defaults.persona.speaking_style,
            "简短口语化，直接回答，不堆表情或语气词"
        );
        assert!(!defaults.behavior.allow_typos);
        assert_eq!(defaults.behavior.emoji_usage, 0.1);

        let parsed = parse_and_validate_config("{}")
            .expect("an empty configuration should use concise chat defaults");
        assert!(!parsed.behavior.allow_typos);
        assert_eq!(parsed.behavior.emoji_usage, 0.1);

        let schema: serde_json::Value = serde_json::from_str(include_str!("../config.schema.json"))
            .expect("configuration schema should be valid JSON");
        assert_eq!(
            schema["properties"]["persona"]["properties"]["speaking_style"]["default"],
            "简短口语化，直接回答，不堆表情或语气词"
        );
        assert_eq!(
            schema["properties"]["behavior"]["properties"]["allow_typos"]["default"],
            false
        );
        assert_eq!(
            schema["properties"]["behavior"]["properties"]["emoji_usage"]["default"],
            0.1
        );
    }

    #[test]
    fn provider_without_api_key_represents_a_cleared_secret() {
        let config = parse_and_validate_config(
            r#"{
                "llm":{"providers":[{
                    "id":"cleared",
                    "protocol":"openai",
                    "base_url":"https://example.test/v1",
                    "model":"test-model"
                }]}
            }"#,
        )
        .expect("a provider without a secret should remain a valid disabled route");
        assert_eq!(config.llm.providers[0].api_key, "");
    }

    #[test]
    fn vision_provider_configuration_is_optional_and_validated() {
        let config = parse_and_validate_config(
            r#"{"llm":{"providers":[{"id":"vision","protocol":"openai","base_url":"https://example.test/v1","model":"text","supports_vision":true,"vision_model":"vision"}]}}"#,
        )
        .expect("vision provider should be valid");
        assert!(config.llm.providers[0].supports_vision);
        assert_eq!(
            config.llm.providers[0].vision_model.as_deref(),
            Some("vision")
        );

        let error = parse_and_validate_config(
            r#"{"llm":{"providers":[{"id":"vision","protocol":"openai","base_url":"https://example.test/v1","model":"text","vision_model":" "}]}}"#,
        )
        .expect_err("blank vision model should fail");
        assert!(error.contains("vision_model"));
    }

    #[test]
    fn privacy_defaults_keep_raw_events_bounded_and_validated() {
        let defaults = AppConfig::default().privacy;
        assert!(defaults.store_raw_events);
        assert_eq!(defaults.raw_event_retention_days, 30);
        assert_eq!(defaults.message_retention_days, 180);

        let config = parse_and_validate_config(
            r#"{"privacy":{"store_raw_events":false,"raw_event_retention_days":7,"message_retention_days":14}}"#,
        )
        .expect("privacy configuration should be valid");
        assert!(!config.privacy.store_raw_events);
        assert_eq!(config.privacy.raw_event_retention_days, 7);
        assert_eq!(config.privacy.message_retention_days, 14);
    }

    #[test]
    fn privacy_validation_rejects_unbounded_or_inverted_retention() {
        let too_long = parse_and_validate_config(r#"{"privacy":{"message_retention_days":3651}}"#)
            .expect_err("retention beyond the hard limit should fail");
        assert!(too_long.contains("message_retention_days"));

        let inverted = parse_and_validate_config(
            r#"{"privacy":{"raw_event_retention_days":31,"message_retention_days":30}}"#,
        )
        .expect_err("raw retention beyond message retention should fail");
        assert!(inverted.contains("cannot exceed"));
    }

    #[test]
    fn observability_defaults_are_safe_and_raw_debug_is_explicit() {
        let defaults = AppConfig::default().observability;
        assert_eq!(defaults.level, "info");
        assert!(defaults.decision_trace);
        assert!(defaults.llm_metrics);
        assert!(!defaults.raw_protocol_debug_enabled());

        let enabled = parse_and_validate_config(
            r#"{"observability":{"level":"debug","raw_protocol_debug":true,"decision_trace":false,"llm_metrics":false}}"#,
        )
        .expect("debug diagnostics should be explicit and valid");
        assert!(enabled.observability.raw_protocol_debug_enabled());
        assert!(!enabled.observability.decision_trace);
        assert!(!enabled.observability.llm_metrics);

        let invalid_level = parse_and_validate_config(r#"{"observability":{"level":"verbose"}}"#)
            .expect_err("unknown observability level should fail");
        assert!(invalid_level.contains("observability.level"));

        let unsafe_debug = parse_and_validate_config(
            r#"{"observability":{"level":"info","raw_protocol_debug":true}}"#,
        )
        .expect_err("raw protocol output must require debug level");
        assert!(unsafe_debug.contains("raw_protocol_debug"));
    }

    #[test]
    fn sticker_cache_limits_default_safely_and_reject_invalid_ranges() {
        let defaults = AppConfig::default().stickers;
        assert!(defaults.cache_media);
        assert_eq!(defaults.collect_probability, 1.0);
        assert_eq!(defaults.daily_collect_limit, 100);
        assert_eq!(defaults.cache_max_file_mib, 8);
        assert_eq!(defaults.cache_max_total_mib, 256);
        assert_eq!(defaults.cache_timeout_sec, 15);
        assert_eq!(defaults.daily_send_limit, 30);
        assert_eq!(defaults.sticker_cooldown_sec, 300);
        assert_eq!(defaults.max_chain, 2);

        let too_large = parse_and_validate_config(r#"{"stickers":{"cache_max_file_mib":33}}"#)
            .expect_err("oversized cache file limit should fail");
        assert!(too_large.contains("cache_max_file_mib"));

        let inverted = parse_and_validate_config(
            r#"{"stickers":{"cache_max_file_mib":8,"cache_max_total_mib":4}}"#,
        )
        .expect_err("cache total below a single file should fail");
        assert!(inverted.contains("cache_max_total_mib"));

        let invalid_daily_limit =
            parse_and_validate_config(r#"{"stickers":{"daily_collect_limit":0}}"#)
                .expect_err("zero daily collection limit should fail");
        assert!(invalid_daily_limit.contains("daily_collect_limit"));

        let invalid_send_limit =
            parse_and_validate_config(r#"{"stickers":{"daily_send_limit":0}}"#)
                .expect_err("zero daily send limit should fail");
        assert!(invalid_send_limit.contains("daily_send_limit"));

        let invalid_cooldown =
            parse_and_validate_config(r#"{"stickers":{"sticker_cooldown_sec":86401}}"#)
                .expect_err("unbounded sticker cooldown should fail");
        assert!(invalid_cooldown.contains("sticker_cooldown_sec"));

        let invalid_chain = parse_and_validate_config(r#"{"stickers":{"max_chain":4}}"#)
            .expect_err("sticker chains longer than three should fail");
        assert!(invalid_chain.contains("max_chain"));
    }
}
