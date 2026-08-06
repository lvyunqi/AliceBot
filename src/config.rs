//! 配置解析
//!
//! 从 QimenBot 宿主传入的 JSON 配置解析为类型安全的 Rust 结构。
//! API 0.6 的 Schema 校验由宿主完成，这里只做 Rust 层反序列化。

use serde::Deserialize;

/// 应用配置（完整）
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub persona: PersonaConfig,

    #[serde(default)]
    pub llm: LlmConfig,

    #[serde(default)]
    pub behavior: BehaviorConfig,

    #[serde(default)]
    pub memories: MemoryConfig,

    #[serde(default)]
    pub stickers: StickerConfig,

    #[serde(default)]
    pub send: SendConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            persona: PersonaConfig::default(),
            llm: LlmConfig::default(),
            behavior: BehaviorConfig::default(),
            memories: MemoryConfig::default(),
            stickers: StickerConfig::default(),
            send: SendConfig::default(),
        }
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

#[derive(Debug, Clone, Deserialize)]
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

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            short_size: default_short_size(),
            long_topk: default_long_topk(),
            compress_interval_hours: default_compress_interval(),
            compress_min_messages: default_compress_messages(),
            importance_threshold: default_importance_threshold(),
            knowledge_enabled: true,
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
