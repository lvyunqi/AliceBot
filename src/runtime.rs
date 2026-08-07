//! 自建 tokio runtime 管理。
//!
//! 动态插件回调是同步 FFI，不能把宿主请求引用带入异步任务。runtime 只接收
//! 已经拥有所有字段的任务，并在 shutdown 时停止和等待插件自己的后台任务。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::{JoinHandle, JoinSet};

use crate::config::AppConfig;
use crate::pipeline::{DirectAskTask, InMessage};

const MESSAGE_QUEUE_CAPACITY: usize = 1024;
const MESSAGE_CONCURRENCY: usize = 32;
const DIRECT_ASK_QUEUE_CAPACITY: usize = 32;
const DIRECT_ASK_CONCURRENCY: usize = 4;

struct RuntimeState {
    runtime: Option<Runtime>,
    shutdown: Arc<AtomicBool>,
    stop_notify: Arc<Notify>,
    background_tasks: Vec<JoinHandle<()>>,
    message_sender: Option<mpsc::Sender<InMessage>>,
    direct_ask_sender: Option<mpsc::Sender<DirectAskTask>>,
}

/// 可重复启动和关闭的插件 runtime。
pub struct PluginRuntime {
    state: Mutex<RuntimeState>,
}

impl PluginRuntime {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(RuntimeState {
                runtime: None,
                shutdown: Arc::new(AtomicBool::new(false)),
                stop_notify: Arc::new(Notify::new()),
                background_tasks: Vec::new(),
                message_sender: None,
                direct_ask_sender: None,
            }),
        }
    }

    /// 为一次 init 创建新的 runtime。reload 前旧 runtime 已由 shutdown 释放。
    pub fn start(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "runtime 状态锁已损坏".to_string())?;

        if state.runtime.is_none() {
            let runtime = Runtime::new().map_err(|e| format!("创建 tokio runtime 失败: {e}"))?;
            state.shutdown = Arc::new(AtomicBool::new(false));
            state.stop_notify = Arc::new(Notify::new());
            let (sender, receiver) = mpsc::channel(MESSAGE_QUEUE_CAPACITY);
            let message_worker = runtime.handle().spawn(message_dispatcher(
                receiver,
                state.shutdown.clone(),
                state.stop_notify.clone(),
            ));
            let (direct_ask_sender, direct_ask_receiver) = mpsc::channel(DIRECT_ASK_QUEUE_CAPACITY);
            let direct_ask_worker = runtime.handle().spawn(direct_ask_dispatcher(
                direct_ask_receiver,
                state.shutdown.clone(),
                state.stop_notify.clone(),
            ));
            state.runtime = Some(runtime);
            state.background_tasks = vec![message_worker, direct_ask_worker];
            state.message_sender = Some(sender);
            state.direct_ask_sender = Some(direct_ask_sender);
        } else {
            state.shutdown.store(false, Ordering::Release);
        }

        Ok(())
    }

    fn handle(&self) -> Option<Handle> {
        let state = self.state.lock().ok()?;
        state
            .runtime
            .as_ref()
            .map(|runtime| runtime.handle().clone())
    }

    /// 在 runtime 上执行一个需要同步返回结果的短任务。
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        self.handle()
            .expect("plugin runtime 尚未初始化")
            .block_on(future)
    }

    /// 将已拥有所有字段的入站消息送入有界处理队列。
    pub fn submit_message(&self, message: InMessage) -> MessageSubmitResult {
        let Ok(state) = self.state.lock() else {
            return MessageSubmitResult::Unavailable;
        };
        let Some(sender) = state.message_sender.as_ref() else {
            return MessageSubmitResult::Unavailable;
        };
        try_submit_message(sender, message)
    }

    /// 将 `/ask` 放入独立有界队列，避免同步 FFI 回调等待模型网络请求。
    pub fn submit_direct_ask(&self, task: DirectAskTask) -> DirectAskSubmitResult {
        let Ok(state) = self.state.lock() else {
            return DirectAskSubmitResult::Unavailable;
        };
        let Some(sender) = state.direct_ask_sender.as_ref() else {
            return DirectAskSubmitResult::Unavailable;
        };
        try_submit_direct_ask(sender, task)
    }

    /// 启动后台任务（压缩、反思等）。
    pub fn start_background_tasks(&self, config: AppConfig) -> Result<(), String> {
        self.start()?;

        let mut state = self
            .state
            .lock()
            .map_err(|_| "runtime 状态锁已损坏".to_string())?;
        let Some(runtime) = state.runtime.as_ref() else {
            return Err("runtime 尚未初始化".to_string());
        };
        let shutdown = state.shutdown.clone();
        let stop_notify = state.stop_notify.clone();
        let compaction_interval = Duration::from_secs(
            config
                .memories
                .compress_interval_hours
                .clamp(1, 168)
                .saturating_mul(3_600),
        );
        let reflection_interval = Duration::from_secs(
            config
                .memories
                .reflection_interval_hours
                .clamp(1, 168)
                .saturating_mul(3_600),
        );
        let interval = if config.memories.reflection_enabled {
            compaction_interval.min(reflection_interval)
        } else {
            compaction_interval
        };

        let task = runtime.handle().spawn(async move {
            log::info!("[AliceBot] 后台任务已启动");
            loop {
                if shutdown.load(Ordering::Acquire) {
                    break;
                }
                let compaction_succeeded = match crate::memory::compact::run_if_due(&config).await {
                    Ok(_) => true,
                    Err(error) => {
                        log::warn!("[AliceBot] scheduled compaction failed: {error}");
                        false
                    }
                };
                if compaction_succeeded {
                    match crate::memory::reflection::run_if_due(&config).await {
                        Ok(report) => {
                            if matches!(
                                report.action,
                                crate::memory::reflection::ReflectionAction::Applied
                                    | crate::memory::reflection::ReflectionAction::RolledBack
                            ) {
                                log::info!(
                                    "[AliceBot] behavior calibration completed: action={:?}, samples={}, cursor={}..{}",
                                    report.action,
                                    report.observed_samples,
                                    report.cursor_start,
                                    report.cursor_end
                                );
                            }
                        }
                        Err(error) => log::warn!("[AliceBot] scheduled reflection failed: {error}"),
                    }
                }
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {}
                    _ = stop_notify.notified() => break,
                }
            }
            log::info!("[AliceBot] 后台任务已停止");
        });
        state.background_tasks.push(task);
        Ok(())
    }

    /// 停止后台任务并销毁 runtime，保证动态库卸载前不再执行插件代码。
    pub fn shutdown(&self) {
        let (runtime, tasks, shutdown, stop_notify, message_sender, direct_ask_sender) =
            match self.state.lock() {
                Ok(mut state) => {
                    state.shutdown.store(true, Ordering::Release);
                    (
                        state.runtime.take(),
                        std::mem::take(&mut state.background_tasks),
                        state.shutdown.clone(),
                        state.stop_notify.clone(),
                        state.message_sender.take(),
                        state.direct_ask_sender.take(),
                    )
                }
                Err(_) => return,
            };

        drop(message_sender);
        drop(direct_ask_sender);
        shutdown.store(true, Ordering::Release);
        stop_notify.notify_waiters();
        let Some(runtime) = runtime else {
            return;
        };

        runtime.block_on(async move {
            for task in tasks {
                let _ = task.await;
            }
        });
        runtime.shutdown_timeout(Duration::from_secs(5));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageSubmitResult {
    Enqueued,
    Full,
    Unavailable,
}

/// `/ask` 队列提交结果，供命令回调返回即时可理解的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectAskSubmitResult {
    Enqueued,
    Full,
    Unavailable,
}

fn try_submit_message(sender: &mpsc::Sender<InMessage>, message: InMessage) -> MessageSubmitResult {
    match sender.try_send(message) {
        Ok(()) => MessageSubmitResult::Enqueued,
        Err(mpsc::error::TrySendError::Full(_)) => MessageSubmitResult::Full,
        Err(mpsc::error::TrySendError::Closed(_)) => MessageSubmitResult::Unavailable,
    }
}

fn try_submit_direct_ask(
    sender: &mpsc::Sender<DirectAskTask>,
    task: DirectAskTask,
) -> DirectAskSubmitResult {
    match sender.try_send(task) {
        Ok(()) => DirectAskSubmitResult::Enqueued,
        Err(mpsc::error::TrySendError::Full(_)) => DirectAskSubmitResult::Full,
        Err(mpsc::error::TrySendError::Closed(_)) => DirectAskSubmitResult::Unavailable,
    }
}

async fn message_dispatcher(
    mut receiver: mpsc::Receiver<InMessage>,
    shutdown: Arc<AtomicBool>,
    stop_notify: Arc<Notify>,
) {
    let permits = Arc::new(Semaphore::new(MESSAGE_CONCURRENCY));
    let mut tasks = JoinSet::new();

    loop {
        while tasks.try_join_next().is_some() {}
        let permit = tokio::select! {
            _ = stop_notify.notified() => break,
            permit = permits.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let next_message = tokio::select! {
            _ = stop_notify.notified() => break,
            message = receiver.recv() => message,
        };
        let Some(message) = next_message else {
            break;
        };

        let task_shutdown = shutdown.clone();
        let task_notify = stop_notify.clone();
        let event_key = message.event_key.clone();
        tasks.spawn(async move {
            let _permit = permit;
            if task_shutdown.load(Ordering::Acquire) {
                crate::pipeline::mark_record_only(&event_key, "shutdown");
                return;
            }
            tokio::select! {
                _ = task_notify.notified() => {
                    crate::pipeline::mark_record_only(&event_key, "shutdown");
                }
                _ = crate::pipeline::process_recorded_message(message) => {}
            }
        });
    }

    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            log::debug!("[AliceBot] message worker stopped: {error}");
        }
    }
}

/// 有界执行后台 `/ask`，关闭时取消未完成任务，避免动态库卸载后继续发送。
async fn direct_ask_dispatcher(
    mut receiver: mpsc::Receiver<DirectAskTask>,
    shutdown: Arc<AtomicBool>,
    stop_notify: Arc<Notify>,
) {
    let permits = Arc::new(Semaphore::new(DIRECT_ASK_CONCURRENCY));
    let mut tasks = JoinSet::new();

    loop {
        while tasks.try_join_next().is_some() {}
        let permit = tokio::select! {
            _ = stop_notify.notified() => break,
            permit = permits.clone().acquire_owned() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let next_task = tokio::select! {
            _ = stop_notify.notified() => break,
            task = receiver.recv() => task,
        };
        let Some(task) = next_task else {
            break;
        };

        let task_shutdown = shutdown.clone();
        let task_notify = stop_notify.clone();
        tasks.spawn(async move {
            let _permit = permit;
            if task_shutdown.load(Ordering::Acquire) {
                return;
            }
            tokio::select! {
                _ = task_notify.notified() => {}
                _ = crate::pipeline::process_direct_ask(task) => {}
            }
        });
    }

    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            log::debug!("[AliceBot] /ask worker stopped: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_can_restart_after_shutdown() {
        let runtime = PluginRuntime::new();

        runtime.start().expect("runtime should start");
        assert_eq!(runtime.block_on(async { 2 + 2 }), 4);
        runtime.shutdown();

        runtime.start().expect("runtime should restart");
        assert_eq!(runtime.block_on(async { 3 + 3 }), 6);
        runtime.shutdown();
    }

    #[test]
    fn bounded_message_sender_reports_full_without_dropping_contract() {
        let (sender, _receiver) = mpsc::channel(1);
        let message = test_message("queue-1");
        assert_eq!(
            try_submit_message(&sender, message.clone()),
            MessageSubmitResult::Enqueued
        );
        assert_eq!(
            try_submit_message(&sender, message),
            MessageSubmitResult::Full
        );
    }

    #[test]
    fn bounded_direct_ask_sender_reports_full_without_dropping_contract() {
        let (sender, _receiver) = mpsc::channel(1);
        let task = test_direct_ask();
        assert_eq!(
            try_submit_direct_ask(&sender, task.clone()),
            DirectAskSubmitResult::Enqueued
        );
        assert_eq!(
            try_submit_direct_ask(&sender, task),
            DirectAskSubmitResult::Full
        );
    }

    fn test_direct_ask() -> DirectAskTask {
        DirectAskTask {
            message: test_message("ask-1"),
            prompt: "test".to_string(),
        }
    }

    fn test_message(event_key: &str) -> InMessage {
        InMessage {
            event_key: event_key.to_string(),
            protocol: "onebot11".to_string(),
            bot_account_id: String::new(),
            session_type: "group".to_string(),
            session_id: "group-1".to_string(),
            sender_id: "user-1".to_string(),
            sender_name: "user".to_string(),
            message_id: event_key.to_string(),
            reply_to_id: String::new(),
            content: "test".to_string(),
            media: Vec::new(),
            has_media: false,
            at_me: false,
            timestamp: 1,
            safe_raw_json: "{}".to_string(),
        }
    }
}
