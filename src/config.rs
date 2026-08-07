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
            speaking_style: "口语化，爱用表情词".into(),
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

    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            request_timeout_ms: default_llm_timeout(),
            retry_limit: default_retry_limit(),
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

    #[serde(default)]
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
fn default_emoji_usage() -> f32 {
    0.6
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            reply_bias: default_reply_bias(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            max_context_tokens: default_max_context(),
            min_interval_sec: default_min_interval(),
            allow_typos: true,
            emoji_usage: default_emoji_usage(),
        }
    }
}

/// Autonomous reply policy. Optional probability/cooldown fields fall back to
/// their legacy `behavior` locations for one configuration version.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

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

/// 表情包策略
#[derive(Debug, Clone, Deserialize)]
pub struct StickerConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default = "default_true")]
    pub auto_collect: bool,

    #[serde(default = "default_collect_prob")]
    pub collect_probability: f32,

    #[serde(default = "default_send_prob")]
    pub send_probability: f32,

    #[serde(default = "default_true")]
    pub link_enabled: bool,

    #[serde(default = "default_max_chain")]
    pub max_chain: u32,
}

fn default_collect_prob() -> f32 {
    0.3
}
fn default_send_prob() -> f32 {
    0.4
}
fn default_max_chain() -> u32 {
    3
}

impl Default for StickerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_collect: true,
            collect_probability: default_collect_prob(),
            send_probability: default_send_prob(),
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
        assert!(schema["properties"]["memories"]["properties"]["reflection_enabled"].is_object());
        assert!(schema["properties"]["behavior"]["properties"]["reply_bias"].is_null());
        assert!(schema["properties"]["behavior"]["properties"]["min_interval_sec"].is_null());
    }
}
