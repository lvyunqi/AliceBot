//! 自建 tokio runtime 管理。
//!
//! 动态插件回调是同步 FFI，不能把宿主请求引用带入异步任务。runtime 只接收
//! 已经拥有所有字段的任务，并在 shutdown 时停止和等待插件自己的后台任务。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::config::AppConfig;

struct RuntimeState {
    runtime: Option<Runtime>,
    shutdown: Arc<AtomicBool>,
    stop_notify: Arc<Notify>,
    background_tasks: Vec<JoinHandle<()>>,
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
            state.runtime =
                Some(Runtime::new().map_err(|e| format!("创建 tokio runtime 失败: {e}"))?);
            state.shutdown = Arc::new(AtomicBool::new(false));
            state.stop_notify = Arc::new(Notify::new());
            state.background_tasks.clear();
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

    /// 投递一个已经拥有输入数据的异步任务。
    pub fn spawn<F>(&self, future: F) -> bool
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        let Some(runtime) = state.runtime.as_ref() else {
            return false;
        };
        runtime.spawn(future);
        true
    }

    /// 启动后台任务（压缩、反思等）。
    pub fn start_background_tasks(&self, _config: AppConfig) -> Result<(), String> {
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

        let task = runtime.handle().spawn(async move {
            log::info!("[AliceBot] 后台任务已启动");
            loop {
                if shutdown.load(Ordering::Acquire) {
                    break;
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(60)) => {}
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
        let (runtime, tasks, shutdown, stop_notify) = match self.state.lock() {
            Ok(mut state) => {
                state.shutdown.store(true, Ordering::Release);
                (
                    state.runtime.take(),
                    std::mem::take(&mut state.background_tasks),
                    state.shutdown.clone(),
                    state.stop_notify.clone(),
                )
            }
            Err(_) => return,
        };

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
}
