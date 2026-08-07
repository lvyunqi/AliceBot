use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::pipeline::InMessage;

const MAX_COALESCE_WINDOW_MS: u64 = 3_000;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BatchKey {
    protocol: String,
    session_type: String,
    session_id: String,
    sender_id: String,
}

struct PendingBatch {
    generation: u64,
    deadline: Instant,
    messages: Vec<InMessage>,
}

#[derive(Default)]
struct Coalescer {
    next_generation: u64,
    pending: HashMap<BatchKey, PendingBatch>,
}

static COALESCER: LazyLock<Mutex<Coalescer>> = LazyLock::new(|| Mutex::new(Coalescer::default()));

pub(crate) struct CoalescedMessage {
    pub message: InMessage,
    pub source_event_keys: Vec<String>,
}

pub(super) async fn coalesce(
    message: InMessage,
    configured_window_ms: u64,
) -> Option<CoalescedMessage> {
    let window_ms = configured_window_ms.min(MAX_COALESCE_WINDOW_MS);
    let key = batch_key(&message);
    if window_ms == 0 || message.session_type != "group" || is_immediate(&message) {
        return Some(flush_immediate(key, message));
    }

    let window = Duration::from_millis(window_ms);
    let now = Instant::now();
    let (generation, wait) = {
        let mut state = COALESCER.lock().ok()?;
        state.next_generation = state.next_generation.wrapping_add(1).max(1);
        let generation = state.next_generation;
        let max_batch_ms = window_ms.saturating_mul(3).min(MAX_COALESCE_WINDOW_MS);
        let deadline = now + Duration::from_millis(max_batch_ms.max(window_ms));
        let pending = state.pending.entry(key.clone()).or_insert(PendingBatch {
            generation,
            deadline,
            messages: Vec::new(),
        });
        pending.generation = generation;
        pending.messages.push(message);
        let due = std::cmp::min(now + window, pending.deadline);
        (generation, due.saturating_duration_since(now))
    };

    tokio::time::sleep(wait).await;

    let messages = {
        let mut state = COALESCER.lock().ok()?;
        let pending = state.pending.get(&key)?;
        if pending.generation != generation && Instant::now() < pending.deadline {
            return None;
        }
        state.pending.remove(&key)?.messages
    };
    Some(merge(messages))
}

pub(super) fn clear() {
    if let Ok(mut state) = COALESCER.lock() {
        state.pending.clear();
    }
}

fn flush_immediate(key: BatchKey, message: InMessage) -> CoalescedMessage {
    let mut prior = COALESCER
        .lock()
        .ok()
        .and_then(|mut state| state.pending.remove(&key))
        .map(|pending| pending.messages)
        .unwrap_or_default();
    let mut source_event_keys = prior
        .drain(..)
        .map(|item| item.event_key)
        .collect::<Vec<_>>();
    source_event_keys.push(message.event_key.clone());
    CoalescedMessage {
        message,
        source_event_keys,
    }
}

fn merge(mut messages: Vec<InMessage>) -> CoalescedMessage {
    messages.sort_by(|left, right| {
        left.timestamp
            .cmp(&right.timestamp)
            .then_with(|| left.event_key.cmp(&right.event_key))
    });
    let source_event_keys = messages
        .iter()
        .map(|message| message.event_key.clone())
        .collect::<Vec<_>>();
    let mut representative = messages
        .last()
        .cloned()
        .expect("a pending coalescing batch always contains a message");

    representative.content = messages
        .iter()
        .filter_map(|message| {
            let content = message.content.trim();
            (!content.is_empty()).then_some(content)
        })
        .collect::<Vec<_>>()
        .join("\n");
    representative.media = messages
        .iter()
        .flat_map(|message| message.media.iter().cloned())
        .collect();
    representative.has_media = !representative.media.is_empty();
    representative.at_me = messages.iter().any(|message| message.at_me);
    if let Some(reply_to_id) = messages
        .iter()
        .rev()
        .map(|message| message.reply_to_id.as_str())
        .find(|reply_to_id| !reply_to_id.is_empty())
    {
        representative.reply_to_id = reply_to_id.to_string();
    }
    if let Some(account_id) = messages
        .iter()
        .rev()
        .map(|message| message.bot_account_id.as_str())
        .find(|account_id| !account_id.is_empty())
    {
        representative.bot_account_id = account_id.to_string();
    }

    CoalescedMessage {
        message: representative,
        source_event_keys,
    }
}

fn batch_key(message: &InMessage) -> BatchKey {
    BatchKey {
        protocol: message.protocol.clone(),
        session_type: message.session_type.clone(),
        session_id: message.session_id.clone(),
        sender_id: message.sender_id.clone(),
    }
}

fn is_immediate(message: &InMessage) -> bool {
    message.at_me
        || !message.reply_to_id.is_empty()
        || ["救命", "紧急", "急急", "help", "HELP"]
            .iter()
            .any(|word| message.content.contains(word))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(session_id: &str, content: &str, timestamp: i64) -> InMessage {
        InMessage {
            event_key: format!("coalesce:{session_id}:{timestamp}"),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: session_id.to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "user".to_string(),
            message_id: timestamp.to_string(),
            reply_to_id: String::new(),
            content: content.to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp,
            safe_raw_json: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn consecutive_messages_are_merged_once() {
        let session = format!("merge-{}", std::process::id());
        let first = tokio::spawn(coalesce(message(&session, "first", 1), 30));
        tokio::time::sleep(Duration::from_millis(5)).await;
        let second = coalesce(message(&session, "second", 2), 30)
            .await
            .expect("latest message should own the batch");

        assert!(first.await.unwrap().is_none());
        assert_eq!(second.message.content, "first\nsecond");
        assert_eq!(second.source_event_keys.len(), 2);
    }

    #[tokio::test]
    async fn mention_flushes_pending_messages_without_waiting() {
        let session = format!("direct-{}", std::process::id());
        let first = tokio::spawn(coalesce(message(&session, "first", 1), 40));
        tokio::time::sleep(Duration::from_millis(5)).await;
        let mut direct = message(&session, "@bot help", 2);
        direct.at_me = true;

        let batch = coalesce(direct, 40)
            .await
            .expect("direct message should be immediate");
        assert_eq!(batch.message.content, "@bot help");
        assert_eq!(batch.source_event_keys.len(), 2);
        assert!(first.await.unwrap().is_none());
    }
}
