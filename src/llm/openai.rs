//! OpenAI-compatible chat completions client.
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

use super::*;

pub struct OpenAiClient {
    client: Client,
    base_url: String,
    api_key: String,
}

impl OpenAiClient {
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
impl Llm for OpenAiClient {
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut messages: Vec<serde_json::Value> = req.messages.iter().map(message_value).collect();

        if let Some(system) = req.system.as_deref()
            && !messages.iter().any(|message| message["role"] == "system")
        {
            messages.insert(
                0,
                serde_json::json!({
                    "role": "system",
                    "content": system,
                }),
            );
        }

        let mut body = serde_json::json!({
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": false,
        });
        if !req.tools.is_empty() {
            body["tools"] = tools_value(&req.tools);
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
            .map_err(|_| response_parse_error("OpenAI"))?;
        let message = data["choices"][0].get("message").ok_or_else(|| LlmError {
            kind: ErrorKind::Parse,
            message: "unable to parse OpenAI response".to_string(),
        })?;
        let text = message["content"].as_str().unwrap_or_default().to_string();
        let tool_calls = message["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|call| {
                        let function = call.get("function")?;
                        let name = function.get("name")?.as_str()?.trim();
                        if name.is_empty() {
                            return None;
                        }
                        let id = call
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .filter(|value| !value.trim().is_empty())
                            .unwrap_or("call-unknown");
                        let arguments = function
                            .get("arguments")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("{}");
                        Some(ToolCall {
                            id: id.chars().take(128).collect(),
                            name: name.chars().take(80).collect(),
                            arguments: arguments.chars().take(16_384).collect(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if text.trim().is_empty() && tool_calls.is_empty() {
            return Err(LlmError {
                kind: ErrorKind::Parse,
                message: "unable to parse OpenAI response".to_string(),
            });
        }
        Ok(ChatResponse { text, tool_calls })
    }
}

fn content_value(message: &ChatMessage) -> serde_json::Value {
    if message.image_urls.is_empty() && message.image_data.is_empty() {
        return serde_json::json!(message.content);
    }

    // Put visual blocks first. This is accepted by the OpenAI contract and
    // avoids gateways that inspect only the first block for vision routing.
    let mut parts = Vec::with_capacity(message.image_urls.len() + message.image_data.len() + 1);
    parts.extend(message.image_urls.iter().map(|url| {
        serde_json::json!({
            "type": "image_url",
            "image_url": {"url": url},
        })
    }));
    parts.extend(message.image_data.iter().map(|image| {
        serde_json::json!({
            "type": "image_url",
            "image_url": {"url": format!("data:{};base64,{}", image.media_type, image.base64)},
        })
    }));
    if !message.content.trim().is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }
    serde_json::Value::Array(parts)
}

fn message_value(message: &ChatMessage) -> serde_json::Value {
    let mut value = serde_json::json!({
        "role": message.role.as_str(),
        "content": content_value(message),
    });
    if !message.tool_calls.is_empty() {
        value["tool_calls"] = serde_json::Value::Array(
            message
                .tool_calls
                .iter()
                .map(|call| {
                    serde_json::json!({
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    })
                })
                .collect(),
        );
        if message.content.trim().is_empty() {
            value["content"] = serde_json::Value::Null;
        }
    }
    if message.role == Role::Tool {
        value["tool_call_id"] =
            serde_json::json!(message.tool_call_id.as_deref().unwrap_or_default());
    }
    value
}

fn tools_value(tools: &[ChatTool]) -> serde_json::Value {
    if tools.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::Value::Array(
        tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect(),
    )
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
    async fn sends_openai_contract_and_parses_response() {
        let (base_url, server) = spawn_json_server(
            200,
            r#"{"choices":[{"message":{"content":"hello from mock"}}]}"#,
        );
        let client = OpenAiClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "mock-model".to_string(),
            system: Some("be concise".to_string()),
            messages: vec![ChatMessage::user("hello")],
            temperature: 0.4,
            max_tokens: 64,
            tools: Vec::new(),
        };

        let response = client
            .chat(&request)
            .await
            .expect("mock response should parse");
        let raw_request = server.join().expect("mock server should finish");
        let (_, body) = raw_request
            .split_once("\r\n\r\n")
            .expect("request should contain headers");
        let body: serde_json::Value = serde_json::from_str(body).expect("request should be JSON");
        assert_eq!(response.text, "hello from mock");
        assert_eq!(body["model"], "mock-model");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be concise");
        assert_eq!(body["messages"][1]["content"], "hello");
        assert_eq!(body["stream"], false);
        assert!(body.get("tools").is_none());
    }

    #[tokio::test]
    async fn serializes_tools_and_parses_tool_calls() {
        let (base_url, server) = spawn_json_server(
            200,
            r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"search_history","arguments":"{\"query\":\"图片\"}"}}]}}]}"#,
        );
        let client = OpenAiClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "mock-model".to_string(),
            system: None,
            messages: vec![ChatMessage::user("刚才谁发了图？")],
            temperature: 0.0,
            max_tokens: 64,
            tools: vec![ChatTool {
                name: "search_history".to_string(),
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
        assert_eq!(body["tools"][0]["function"]["name"], "search_history");
        assert_eq!(response.tool_calls[0].id, "call-1");
        assert_eq!(response.tool_calls[0].name, "search_history");
        assert_eq!(response.text, "");
    }

    #[tokio::test]
    async fn http_error_does_not_expose_body_prompt_or_api_key() {
        let (base_url, server) = spawn_json_server(
            400,
            r#"{"error":"raw-body-secret","echo":"prompt-secret api-key-secret"}"#,
        );
        let client = OpenAiClient::new(&base_url, "api-key-secret", Duration::from_secs(5));
        let request = ChatRequest {
            model: "mock-model".to_string(),
            system: None,
            messages: vec![ChatMessage::user("prompt-secret")],
            temperature: 0.4,
            max_tokens: 64,
            tools: Vec::new(),
        };

        let error = client
            .chat(&request)
            .await
            .expect_err("HTTP 400 should fail");
        let _ = server.join().expect("mock server should finish");
        assert_eq!(error.kind, ErrorKind::InvalidRequest);
        assert_eq!(error.message, "provider returned HTTP 400");
        assert!(!format!("{error:?}").contains("raw-body-secret"));
        assert!(!error.message.contains("prompt-secret"));
        assert!(!error.message.contains("api-key-secret"));
    }

    #[tokio::test]
    async fn serializes_openai_vision_content_blocks() {
        let (base_url, server) =
            spawn_json_server(200, r#"{"choices":[{"message":{"content":"cat"}}]}"#);
        let client = OpenAiClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "vision-model".to_string(),
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
        assert_eq!(body["messages"][0]["content"][0]["type"], "image_url");
        assert_eq!(body["messages"][0]["content"][1]["type"], "text");
        assert_eq!(
            body["messages"][0]["content"][0]["image_url"]["url"],
            "https://example.test/cat.png"
        );
    }

    #[tokio::test]
    async fn serializes_openai_inline_vision_data_uri() {
        let (base_url, server) =
            spawn_json_server(200, r#"{"choices":[{"message":{"content":"cat"}}]}"#);
        let client = OpenAiClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "vision-model".to_string(),
            system: None,
            messages: vec![
                ChatMessage::user("describe this").with_image_data(vec![ImageData {
                    media_type: "image/jpeg".to_string(),
                    base64: "/9j/".to_string(),
                }]),
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
        assert_eq!(
            body["messages"][0]["content"][0]["image_url"]["url"],
            "data:image/jpeg;base64,/9j/"
        );
    }

    #[tokio::test]
    async fn parse_error_does_not_expose_success_body() {
        let (base_url, server) = spawn_json_server(200, "raw-success-body-secret");
        let client = OpenAiClient::new(&base_url, "test-key", Duration::from_secs(5));
        let request = ChatRequest {
            model: "mock-model".to_string(),
            system: None,
            messages: vec![ChatMessage::user("hello")],
            temperature: 0.4,
            max_tokens: 64,
            tools: Vec::new(),
        };

        let error = client
            .chat(&request)
            .await
            .expect_err("invalid JSON should fail");
        let _ = server.join().expect("mock server should finish");
        assert_eq!(error.kind, ErrorKind::Parse);
        assert_eq!(error.message, "OpenAI response was not valid JSON");
        assert!(!format!("{error:?}").contains("raw-success-body-secret"));
    }
}
