//! LLM 协议抽象与主备客户端。

pub mod anthropic;
pub mod openai;

#[cfg(test)]
pub(crate) mod test_support;

use async_trait::async_trait;
use std::time::Duration;

/// 聊天消息角色。
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
    model: String,
    client: Box<dyn Llm>,
}

/// 按 priority 排序的 provider 集合，失败后才切换下一个 provider。
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
        let mut last_error = None;

        for provider in &self.providers {
            let mut attempt = 0;
            loop {
                let mut provider_request = request.clone();
                if provider_request.model.trim().is_empty() {
                    provider_request.model = provider.model.clone();
                }

                match provider.client.chat(&provider_request).await {
                    Ok(response) if !response.text.trim().is_empty() => return Ok(response),
                    Ok(_) => {
                        last_error = Some(LlmError {
                            kind: ErrorKind::Parse,
                            message: format!("provider {} 返回空文本", provider.id),
                        });
                    }
                    Err(error) => {
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
            message: "没有可用的 LLM provider".to_string(),
        }))
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

    #[test]
    fn empty_provider_set_is_explicit() {
        let config = crate::config::LlmConfig::default();
        let client = LlmClient::from_config(&config);
        assert_eq!(client.provider_count(), 0);
    }
}
