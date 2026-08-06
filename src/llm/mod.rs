//! LLM protocol abstraction, provider fallback, and redacted call auditing.
pub mod anthropic;
pub mod openai;

#[cfg(test)]
pub(crate) mod test_support;

use async_trait::async_trait;
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

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct LlmError {
    pub kind: ErrorKind,
    pub message: String,
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

pub(crate) fn truncate_error(message: String) -> String {
    message.chars().take(1_024).collect()
}

#[async_trait]
pub trait Llm: Send + Sync {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;
}

struct ProviderClient {
    id: String,
    protocol: String,
    model: String,
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
                    provider_request.model = provider.model.clone();
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

fn begin_audit(
    task: &str,
    provider: &ProviderClient,
    model: &str,
    attempt: u32,
    input_chars: usize,
) -> Option<i64> {
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
}
