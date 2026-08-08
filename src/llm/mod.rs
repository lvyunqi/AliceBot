//! LLM protocol abstraction, provider fallback, and redacted call auditing.
pub mod anthropic;
pub mod openai;

#[cfg(test)]
pub(crate) mod test_support;

use async_trait::async_trait;
use std::fmt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// HTTPS image URLs that should be sent as multimodal content blocks.
    /// URLs are never included in Debug output or persisted in LLM audits.
    pub image_urls: Vec<String>,
}

impl fmt::Debug for ChatMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatMessage")
            .field("role", &self.role)
            .field("content_chars", &self.content.chars().count())
            .field("image_count", &self.image_urls.len())
            .finish()
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            image_urls: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            image_urls: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            image_urls: Vec::new(),
        }
    }

    pub fn with_image_urls(mut self, image_urls: impl IntoIterator<Item = String>) -> Self {
        self.image_urls = image_urls
            .into_iter()
            .filter(|url| url.starts_with("https://"))
            .take(4)
            .collect();
        self
    }

    pub fn has_images(&self) -> bool {
        !self.image_urls.is_empty()
    }

    pub fn without_images(mut self) -> Self {
        self.image_urls.clear();
        self
    }
}

#[derive(Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
}

impl fmt::Debug for ChatRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatRequest")
            .field("model", &self.model)
            .field(
                "system_chars",
                &self
                    .system
                    .as_ref()
                    .map(|text| text.chars().count())
                    .unwrap_or(0),
            )
            .field("message_count", &self.messages.len())
            .field(
                "message_chars",
                &self
                    .messages
                    .iter()
                    .map(|message| message.content.chars().count())
                    .sum::<usize>(),
            )
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[derive(Clone)]
pub struct ChatResponse {
    pub text: String,
}

impl fmt::Debug for ChatResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatResponse")
            .field("text_chars", &self.text.chars().count())
            .finish()
    }
}

#[derive(Clone)]
pub struct LlmError {
    pub kind: ErrorKind,
    pub message: String,
}

impl fmt::Debug for LlmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LlmError")
            .field("kind", &self.kind)
            .field("message", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    Timeout,
    RateLimited,
    Auth,
    InvalidRequest,
    Server,
    Parse,
    NoProvider,
    Unknown,
}

/// 将 transport 错误压缩为稳定分类，避免 URL 查询参数进入错误文本或日志。
pub(crate) fn transport_error(error: &reqwest::Error) -> LlmError {
    if error.is_timeout() {
        LlmError {
            kind: ErrorKind::Timeout,
            message: "provider request timed out".to_string(),
        }
    } else {
        LlmError {
            kind: ErrorKind::Unknown,
            message: "provider transport request failed".to_string(),
        }
    }
}

/// HTTP 错误只暴露状态码和分类，不读取或保留上游响应正文。
pub(crate) fn http_status_error(kind: ErrorKind, status: u16) -> LlmError {
    LlmError {
        kind,
        message: format!("provider returned HTTP {status}"),
    }
}

/// JSON 解析失败使用固定文本，避免解析器携带响应片段或请求 URL。
pub(crate) fn response_parse_error(provider: &str) -> LlmError {
    LlmError {
        kind: ErrorKind::Parse,
        message: format!("{provider} response was not valid JSON"),
    }
}

#[async_trait]
pub trait Llm: Send + Sync {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
}

struct ProviderClient {
    id: String,
    protocol: String,
    model: String,
    supports_vision: bool,
    vision_model: Option<String>,
    client: Box<dyn Llm>,
}

/// Providers are tried by priority; retryable errors move through the configured fallback chain.
pub struct LlmClient {
    providers: Vec<ProviderClient>,
    retry_limit: u32,
}

impl LlmClient {
    pub fn from_config(config: &crate::config::LlmConfig) -> Self {
        let timeout = Duration::from_millis(config.request_timeout_ms.max(100));
        let mut providers = if config.enabled {
            config
                .providers
                .iter()
                .filter(|provider| provider.enabled && !provider.api_key.trim().is_empty())
                .filter_map(|provider| {
                    Some((
                        provider.priority,
                        provider.id.clone(),
                        ProviderClient {
                            id: provider.id.clone(),
                            protocol: provider.protocol.clone(),
                            model: provider.model.clone(),
                            supports_vision: provider.supports_vision,
                            vision_model: provider
                                .vision_model
                                .clone()
                                .filter(|model| !model.trim().is_empty()),
                            client: create_client(
                                &provider.protocol,
                                &provider.base_url,
                                &provider.api_key,
                                timeout,
                            )?,
                        },
                    ))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        providers.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

        Self {
            providers: providers.into_iter().map(|(_, _, client)| client).collect(),
            retry_limit: config.retry_limit.min(3),
        }
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }

    pub async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_with_task("chat", request).await
    }

    pub async fn chat_with_task(
        &self,
        task: &str,
        request: &ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        let mut last_error = None;
        let input_chars = input_chars(request);

        for provider in &self.providers {
            let mut attempt = 0;
            loop {
                let mut provider_request = request.clone();
                if provider_request.model.trim().is_empty() {
                    provider_request.model = if provider_request_has_images(&provider_request)
                        && provider.supports_vision
                    {
                        provider
                            .vision_model
                            .clone()
                            .unwrap_or_else(|| provider.model.clone())
                    } else {
                        provider.model.clone()
                    };
                }
                if provider_request_has_images(&provider_request) && !provider.supports_vision {
                    provider_request.messages = provider_request
                        .messages
                        .into_iter()
                        .map(ChatMessage::without_images)
                        .collect();
                }
                let audit_id = begin_audit(
                    task,
                    provider,
                    &provider_request.model,
                    attempt + 1,
                    input_chars,
                );
                let started = Instant::now();

                match provider.client.chat(&provider_request).await {
                    Ok(response) if !response.text.trim().is_empty() => {
                        finish_audit(
                            audit_id,
                            "success",
                            None,
                            response.text.chars().count(),
                            started,
                        );
                        return Ok(response);
                    }
                    Ok(_) => {
                        finish_audit(audit_id, "empty", Some(&ErrorKind::Parse), 0, started);
                        last_error = Some(LlmError {
                            kind: ErrorKind::Parse,
                            message: format!("provider {} returned empty text", provider.id),
                        });
                    }
                    Err(error) => {
                        finish_audit(audit_id, "error", Some(&error.kind), 0, started);
                        let retryable = matches!(
                            &error.kind,
                            ErrorKind::Timeout | ErrorKind::RateLimited | ErrorKind::Server
                        );
                        last_error = Some(error.clone());
                        if retryable && attempt < self.retry_limit {
                            attempt += 1;
                            let delay = 100_u64.saturating_mul(2_u64.saturating_pow(attempt));
                            tokio::time::sleep(Duration::from_millis(delay.min(2_000))).await;
                            continue;
                        }
                    }
                }
                break;
            }
        }

        Err(last_error.unwrap_or(LlmError {
            kind: ErrorKind::NoProvider,
            message: "no usable LLM provider".to_string(),
        }))
    }
}

fn input_chars(request: &ChatRequest) -> usize {
    request
        .system
        .as_ref()
        .map(|text| text.chars().count())
        .unwrap_or(0)
        + request
            .messages
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>()
}

fn provider_request_has_images(request: &ChatRequest) -> bool {
    request.messages.iter().any(ChatMessage::has_images)
}

fn begin_audit(
    task: &str,
    provider: &ProviderClient,
    model: &str,
    attempt: u32,
    input_chars: usize,
) -> Option<i64> {
    if !crate::pipeline::current_config().observability.llm_metrics {
        return None;
    }
    let database = crate::pipeline::try_db()?;
    match database.begin_llm_call(
        task,
        &provider.id,
        &provider.protocol,
        model,
        attempt,
        input_chars,
        chrono::Utc::now().timestamp_millis(),
    ) {
        Ok(id) => Some(id),
        Err(error) => {
            log::debug!("[AliceBot] LLM audit insert failed: {error}");
            None
        }
    }
}

fn finish_audit(
    audit_id: Option<i64>,
    status: &str,
    error_kind: Option<&ErrorKind>,
    output_chars: usize,
    started: Instant,
) {
    let Some(id) = audit_id else {
        return;
    };
    let Some(database) = crate::pipeline::try_db() else {
        return;
    };
    let error_kind = error_kind.map(|kind| format!("{kind:?}"));
    if let Err(error) = database.finish_llm_call(
        id,
        status,
        error_kind.as_deref(),
        output_chars,
        started.elapsed().as_millis() as u64,
        chrono::Utc::now().timestamp_millis(),
    ) {
        log::debug!("[AliceBot] LLM audit update failed: {error}");
    }
}

pub fn create_client(
    protocol: &str,
    base_url: &str,
    api_key: &str,
    timeout: Duration,
) -> Option<Box<dyn Llm>> {
    match protocol {
        "openai" => Some(Box::new(openai::OpenAiClient::new(
            base_url, api_key, timeout,
        ))),
        "anthropic" => Some(Box::new(anthropic::AnthropicClient::new(
            base_url, api_key, timeout,
        ))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn empty_provider_set_is_explicit() {
        let config = crate::config::LlmConfig::default();
        let client = LlmClient::from_config(&config);
        assert_eq!(client.provider_count(), 0);
    }

    #[test]
    fn audit_size_counts_characters_without_exposing_content() {
        let request = ChatRequest {
            model: "model".to_string(),
            system: Some("system".to_string()),
            messages: vec![ChatMessage::user("你好")],
            temperature: 1.0,
            max_tokens: 10,
        };
        assert_eq!(input_chars(&request), 8);
    }

    #[test]
    fn debug_output_exposes_only_llm_metrics_and_error_kind() {
        let request = ChatRequest {
            model: "model".to_string(),
            system: Some("system-prompt-secret".to_string()),
            messages: vec![ChatMessage::user("user-prompt-secret")],
            temperature: 1.0,
            max_tokens: 10,
        };
        let response = ChatResponse {
            text: "model-response-secret".to_string(),
        };
        let error = LlmError {
            kind: ErrorKind::InvalidRequest,
            message: "raw-http-body-secret".to_string(),
        };

        let debug = format!("{request:?} {response:?} {error:?}");
        assert!(!debug.contains("system-prompt-secret"));
        assert!(!debug.contains("user-prompt-secret"));
        assert!(!debug.contains("model-response-secret"));
        assert!(!debug.contains("raw-http-body-secret"));
        assert!(debug.contains("InvalidRequest"));
    }

    struct RetryMock {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Llm for RetryMock {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
                Err(LlmError {
                    kind: ErrorKind::Server,
                    message: "temporary".to_string(),
                })
            } else {
                Ok(ChatResponse {
                    text: "ok".to_string(),
                })
            }
        }
    }

    #[tokio::test]
    async fn retryable_provider_error_is_retried_before_success() {
        let mock = RetryMock {
            calls: AtomicUsize::new(0),
        };
        let client = LlmClient {
            providers: vec![ProviderClient {
                id: "mock".to_string(),
                protocol: "test".to_string(),
                model: "model".to_string(),
                supports_vision: false,
                vision_model: None,
                client: Box::new(mock),
            }],
            retry_limit: 1,
        };
        let request = ChatRequest {
            model: String::new(),
            system: None,
            messages: vec![ChatMessage::user("hello")],
            temperature: 1.0,
            max_tokens: 10,
        };
        assert_eq!(client.chat(&request).await.unwrap().text, "ok");
    }

    struct CaptureMock {
        request: std::sync::Arc<std::sync::Mutex<Option<ChatRequest>>>,
    }

    #[async_trait]
    impl Llm for CaptureMock {
        async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            *self.request.lock().expect("capture lock should work") = Some(req.clone());
            Ok(ChatResponse {
                text: "ok".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn vision_provider_uses_override_model_and_keeps_images() {
        let capture = std::sync::Arc::new(std::sync::Mutex::new(None));
        let client = LlmClient {
            providers: vec![ProviderClient {
                id: "vision".to_string(),
                protocol: "test".to_string(),
                model: "text-model".to_string(),
                supports_vision: true,
                vision_model: Some("vision-model".to_string()),
                client: Box::new(CaptureMock {
                    request: capture.clone(),
                }),
            }],
            retry_limit: 0,
        };
        let request = ChatRequest {
            model: String::new(),
            system: None,
            messages: vec![
                ChatMessage::user("describe")
                    .with_image_urls(vec!["https://example.test/image.png".to_string()]),
            ],
            temperature: 0.0,
            max_tokens: 16,
        };

        client
            .chat(&request)
            .await
            .expect("vision request should work");
        let captured = capture
            .lock()
            .expect("capture lock should work")
            .clone()
            .expect("provider should receive a request");
        assert_eq!(captured.model, "vision-model");
        assert_eq!(captured.messages[0].image_urls.len(), 1);
    }

    #[tokio::test]
    async fn text_provider_receives_a_safe_text_only_fallback() {
        let capture = std::sync::Arc::new(std::sync::Mutex::new(None));
        let client = LlmClient {
            providers: vec![ProviderClient {
                id: "text".to_string(),
                protocol: "test".to_string(),
                model: "text-model".to_string(),
                supports_vision: false,
                vision_model: None,
                client: Box::new(CaptureMock {
                    request: capture.clone(),
                }),
            }],
            retry_limit: 0,
        };
        let request = ChatRequest {
            model: String::new(),
            system: None,
            messages: vec![
                ChatMessage::user("describe")
                    .with_image_urls(vec!["https://example.test/image.png".to_string()]),
            ],
            temperature: 0.0,
            max_tokens: 16,
        };

        client
            .chat(&request)
            .await
            .expect("fallback request should work");
        let captured = capture
            .lock()
            .expect("capture lock should work")
            .clone()
            .expect("provider should receive a request");
        assert_eq!(captured.model, "text-model");
        assert!(captured.messages[0].image_urls.is_empty());
    }
}
