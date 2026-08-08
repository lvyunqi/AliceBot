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
                    "content": content_value(message),
                })),
                Role::Assistant => messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": content_value(message),
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
            .map_err(|error| transport_error(&error))?;

        let status = response.status();
        if !status.is_success() {
            return Err(http_status_error(
                status_error_kind(status.as_u16()),
                status.as_u16(),
            ));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|_| response_parse_error("Anthropic"))?;
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

fn content_value(message: &ChatMessage) -> serde_json::Value {
    if message.image_urls.is_empty() {
        return serde_json::json!(message.content);
    }

    let mut blocks = Vec::with_capacity(message.image_urls.len() + 1);
    if !message.content.trim().is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }
    blocks.extend(message.image_urls.iter().map(|url| {
        serde_json::json!({
            "type": "image",
            "source": {"type": "url", "url": url},
        })
    }));
    serde_json::Value::Array(blocks)
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

    #[tokio::test]
    async fn serializes_anthropic_vision_content_blocks() {
        let (base_url, server) =
            spawn_json_server(200, r#"{"content":[{"type":"text","text":"cat"}]}"#);
        let client = AnthropicClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "claude-vision".to_string(),
            system: None,
            messages: vec![
                ChatMessage::user("describe this")
                    .with_image_urls(vec!["https://example.test/cat.png".to_string()]),
            ],
            temperature: 0.2,
            max_tokens: 32,
        };

        client
            .chat(&request)
            .await
            .expect("mock response should parse");
        let raw_request = server.join().expect("mock server should finish");
        let (_, body) = raw_request
            .split_once("\r\n\r\n")
            .expect("request should contain headers");
        let body: serde_json::Value = serde_json::from_str(body).expect("request should be JSON");
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["messages"][0]["content"][1]["type"], "image");
        assert_eq!(body["messages"][0]["content"][1]["source"]["type"], "url");
    }

    #[tokio::test]
    async fn http_error_does_not_expose_body_prompt_or_api_key() {
        let (base_url, server) = spawn_json_server(
            500,
            r#"{"error":"raw-body-secret","echo":"prompt-secret api-key-secret"}"#,
        );
        let client = AnthropicClient::new(&base_url, "api-key-secret", Duration::from_secs(5));
        let request = ChatRequest {
            model: "claude-mock".to_string(),
            system: Some("prompt-secret".to_string()),
            messages: vec![ChatMessage::user("hello")],
            temperature: 0.6,
            max_tokens: 64,
        };

        let error = client
            .chat(&request)
            .await
            .expect_err("HTTP 500 should fail");
        let _ = server.join().expect("mock server should finish");
        assert_eq!(error.kind, ErrorKind::Server);
        assert_eq!(error.message, "provider returned HTTP 500");
        assert!(!format!("{error:?}").contains("raw-body-secret"));
        assert!(!error.message.contains("prompt-secret"));
        assert!(!error.message.contains("api-key-secret"));
    }
}
