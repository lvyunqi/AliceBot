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
                Role::User => messages.push(message_value(message, "user")),
                Role::Assistant => messages.push(message_value(message, "assistant")),
                Role::Tool => messages.push(message_value(message, "user")),
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
        if !req.tools.is_empty() {
            body["tools"] = serde_json::Value::Array(
                req.tools
                    .iter()
                    .map(|tool| {
                        serde_json::json!({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.parameters,
                        })
                    })
                    .collect(),
            );
        }
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
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        if let Some(items) = data["content"].as_array() {
            for item in items {
                match item["type"].as_str() {
                    Some("text") => {
                        if let Some(text) = item["text"].as_str() {
                            text_parts.push(text.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let Some(name) = item["name"].as_str().map(str::trim) else {
                            continue;
                        };
                        if name.is_empty() {
                            continue;
                        }
                        let arguments = item
                            .get("input")
                            .and_then(|input| serde_json::to_string(input).ok())
                            .unwrap_or_else(|| "{}".to_string());
                        tool_calls.push(ToolCall {
                            id: item["id"]
                                .as_str()
                                .filter(|value| !value.trim().is_empty())
                                .unwrap_or("call-unknown")
                                .chars()
                                .take(128)
                                .collect(),
                            name: name.chars().take(80).collect(),
                            arguments: arguments.chars().take(16_384).collect(),
                        });
                    }
                    _ => {}
                }
            }
        }
        let text = text_parts.join("\n");
        if text.trim().is_empty() && tool_calls.is_empty() {
            return Err(LlmError {
                kind: ErrorKind::Parse,
                message: "unable to parse Anthropic response".to_string(),
            });
        }
        Ok(ChatResponse { text, tool_calls })
    }
}

fn content_value(message: &ChatMessage) -> serde_json::Value {
    if message.image_urls.is_empty() && message.image_data.is_empty() {
        return serde_json::json!(message.content);
    }

    let mut blocks = Vec::with_capacity(message.image_urls.len() + message.image_data.len() + 1);
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
    blocks.extend(message.image_data.iter().map(|image| {
        serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.base64,
            },
        })
    }));
    serde_json::Value::Array(blocks)
}

fn message_value(message: &ChatMessage, role: &str) -> serde_json::Value {
    if message.role == Role::Tool {
        return serde_json::json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id.as_deref().unwrap_or_default(),
                "content": message.content,
            }],
        });
    }
    if !message.tool_calls.is_empty() {
        let mut blocks = Vec::new();
        if !message.content.trim().is_empty() {
            blocks.push(serde_json::json!({"type": "text", "text": message.content}));
        }
        blocks.extend(message.tool_calls.iter().map(|call| {
            let input = serde_json::from_str::<serde_json::Value>(&call.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            serde_json::json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": input,
            })
        }));
        return serde_json::json!({"role": role, "content": blocks});
    }
    serde_json::json!({
        "role": role,
        "content": content_value(message),
    })
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
            tools: Vec::new(),
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
            tools: Vec::new(),
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
            tools: Vec::new(),
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
    async fn serializes_tools_and_parses_tool_use_blocks() {
        let (base_url, server) = spawn_json_server(
            200,
            r#"{"content":[{"type":"tool_use","id":"tool-1","name":"search_memory","input":{"query":"咖啡"}}]}"#,
        );
        let client = AnthropicClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "claude-mock".to_string(),
            system: None,
            messages: vec![ChatMessage::user("你记得我的偏好吗？")],
            temperature: 0.0,
            max_tokens: 64,
            tools: vec![ChatTool {
                name: "search_memory".to_string(),
                description: "read-only".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        };

        let response = client
            .chat(&request)
            .await
            .expect("tool response should parse");
        let raw_request = server.join().expect("mock server should finish");
        let (_, body) = raw_request
            .split_once("\r\n\r\n")
            .expect("request should contain headers");
        let body: serde_json::Value = serde_json::from_str(body).expect("request should be JSON");
        assert_eq!(body["tools"][0]["name"], "search_memory");
        assert_eq!(response.tool_calls[0].id, "tool-1");
        assert_eq!(response.tool_calls[0].name, "search_memory");
        assert_eq!(response.tool_calls[0].arguments, r#"{"query":"咖啡"}"#);
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
            tools: Vec::new(),
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
