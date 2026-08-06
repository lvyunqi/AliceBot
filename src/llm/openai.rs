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
        let mut messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .map(|message| {
                serde_json::json!({
                    "role": message.role.as_str(),
                    "content": message.content,
                })
            })
            .collect();

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

        let body = serde_json::json!({
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": false,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
        let text = data["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| LlmError {
                kind: ErrorKind::Parse,
                message: "unable to parse OpenAI response".to_string(),
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
    }
}
