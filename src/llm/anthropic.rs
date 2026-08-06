//! Anthropic Messages API client.
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

use super::*;

pub struct AnthropicClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl AnthropicClient {
    pub fn new(base_url: &str, api_key: &str, timeout: Duration) -> Self {
        Self {
            client: Client::builder()
                .timeout(timeout)
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl Llm for AnthropicClient {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let url = format!("{}/messages", self.base_url);
        let mut messages = Vec::new();
        for message in &req.messages {
            match message.role {
                Role::System => {}
                Role::User => messages.push(serde_json::json!({
                    "role": "user",
                    "content": message.content,
                })),
                Role::Assistant => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": message.content,
                })),
            }
        }

        if messages.is_empty()
            || messages
                .first()
                .and_then(|message| message["role"].as_str())
                != Some("user")
        {
            return Err(LlmError {
                kind: ErrorKind::InvalidRequest,
                message: "Anthropic messages must start with user".to_string(),
            });
        }

        let mut body = serde_json::json!({
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
            "temperature": req.temperature,
        });
        let system_text = req
            .system
            .as_deref()
            .or_else(|| {
                req.messages.iter().find_map(|message| match &message.role {
                    Role::System => Some(message.content.as_str()),
                    _ => None,
                })
            })
            .unwrap_or("");
        if !system_text.is_empty() {
            body["system"] = serde_json::json!(system_text);
        }

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|error| LlmError {
                kind: if error.is_timeout() {
                    ErrorKind::Timeout
                } else {
                    ErrorKind::Unknown
                },
                message: error.to_string(),
            })?;

        let status = response.status();
        if !status.is_success() {
            let error_text = truncate_error(response.text().await.unwrap_or_default());
            return Err(LlmError {
                kind: status_error_kind(status.as_u16()),
                message: error_text,
            });
        }

        let data: serde_json::Value = response.json().await.map_err(|error| LlmError {
            kind: ErrorKind::Parse,
            message: error.to_string(),
        })?;
        let text = data["content"]
            .as_array()
            .and_then(|items| items.first())
            .and_then(|item| item["text"].as_str())
            .ok_or_else(|| LlmError {
                kind: ErrorKind::Parse,
                message: "unable to parse Anthropic response".to_string(),
            })?
            .to_string();
        Ok(ChatResponse { text })
    }
}

fn status_error_kind(status: u16) -> ErrorKind {
    match status {
        401 => ErrorKind::Auth,
        429 => ErrorKind::RateLimited,
        400..=499 => ErrorKind::InvalidRequest,
        500..=599 => ErrorKind::Server,
        _ => ErrorKind::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::test_support::spawn_json_server;

    #[tokio::test]
    async fn sends_anthropic_contract_and_parses_response() {
        let (base_url, server) = spawn_json_server(
            200,
            r#"{"content":[{"type":"text","text":"hello from mock"}]}"#,
        );
        let client = AnthropicClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "claude-mock".to_string(),
            system: Some("be concise".to_string()),
            messages: vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")],
            temperature: 0.6,
            max_tokens: 64,
        };

        let response = client
            .chat(&request)
            .await
            .expect("mock response should parse");
        let raw_request = server.join().expect("mock server should finish");
        let (headers, body) = raw_request
            .split_once("\r\n\r\n")
            .expect("request should contain headers");
        let body: serde_json::Value = serde_json::from_str(body).expect("request should be JSON");
        assert_eq!(response.text, "hello from mock");
        assert!(headers.to_ascii_lowercase().contains("x-api-key: test-key"));
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("anthropic-version: 2023-06-01")
        );
        assert_eq!(body["model"], "claude-mock");
        assert_eq!(body["system"], "be concise");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][1]["role"], "assistant");
    }

    #[tokio::test]
    async fn rejects_history_starting_with_assistant() {
        let client =
            AnthropicClient::new("http://127.0.0.1:1", "test-key", Duration::from_millis(100));
        let request = ChatRequest {
            model: "claude-mock".to_string(),
            system: None,
            messages: vec![ChatMessage::assistant("orphan")],
            temperature: 1.0,
            max_tokens: 32,
        };

        let error = client
            .chat(&request)
            .await
            .expect_err("invalid history should fail");
        assert_eq!(error.kind, ErrorKind::InvalidRequest);
    }
}
