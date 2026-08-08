//! AliceBot — 最强学习力拟人 QQ 机器人
//!
//! QimenBot 动态插件 (API 0.6)。自建 tokio runtime 管理 LLM 调用、数据库和后台任务。
//! 同步 FFI 回调只做轻量调度，耗时操作交给 runtime 内的异步流水线。

#![deny(unsafe_code)]

use abi_stable_host_api::{
    CommandRequest, CommandResponse, InterceptorRequest, InterceptorResponse, PluginInitConfig,
    PluginInitResult,
};
use qimen_dynamic_plugin_derive::dynamic_plugin;

mod config;
mod db;
mod decision;
mod media;
mod pipeline;
mod runtime;
mod send;

pub mod llm;
pub mod memory;
pub mod stickers;

// 全局状态：插件自建的 tokio runtime 和共享上下文
// 生命周期由 #[init] / #[shutdown] 管理
static RUNTIME: std::sync::LazyLock<runtime::PluginRuntime> =
    std::sync::LazyLock::new(runtime::PluginRuntime::new);

/// 在命令处理完成后写入 journal，并把非命令消息提交给异步处理队列。
fn accept_inbound_message(req: &InterceptorRequest) {
    let rt = &*RUNTIME;
    let event = pipeline::InboundEvent::from_request(req);
    match pipeline::record_inbound(event) {
        Ok(Some(message)) => {
            if pipeline::take_command_suppression(&message.event_key) {
                pipeline::mark_record_only(&message.event_key, "command_handled");
                return;
            }

            match rt.submit_message(message.clone()) {
                runtime::MessageSubmitResult::Enqueued => {}
                runtime::MessageSubmitResult::Full => {
                    pipeline::mark_record_only(&message.event_key, "queue_full");
                    log::warn!(
                        "[AliceBot] 入站处理队列已满，保留 record_only event_key={}",
                        message.event_key
                    );
                }
                runtime::MessageSubmitResult::Unavailable => {
                    pipeline::mark_record_only(&message.event_key, "runtime_unavailable");
                    log::warn!(
                        "[AliceBot] runtime 不可用，保留 record_only event_key={}",
                        message.event_key
                    );
                }
            }
        }
        Ok(None) => {}
        Err(error) => {
            log::error!("[AliceBot] 入站消息 journal 失败: {error}");
        }
    }
}

// ─── 动态插件描述符 ────────────────────────────────────────────

#[dynamic_plugin(
    id = "alicebot",
    version = "0.1.0",
    api = "0.6",
    config_schema = "../config.schema.json",
    config_ui = "../config.ui.json",
    config_version = 9,
    config_apply = "reload"
)]
mod plugin {
    use super::*;

    // ── 初始化 ────────────────────────────────────────────────

    #[init]
    fn init(config: PluginInitConfig) -> PluginInitResult {
        log::info!("[AliceBot] 初始化...");

        // 解析配置
        let cfg = match config::parse_and_validate_config(config.config_json.as_str()) {
            Ok(c) => c,
            Err(e) => {
                log::error!("[AliceBot] 配置解析失败: {}", e);
                return PluginInitResult::err(&format!("配置解析失败: {}", e));
            }
        };

        // 启动 runtime（如果尚未启动）
        let rt = &*RUNTIME;
        if let Err(e) = rt.start() {
            log::error!("[AliceBot] runtime 启动失败: {e}");
            return PluginInitResult::err(&format!("runtime 启动失败: {e}"));
        }

        let data_dir = config.data_dir.as_str();
        pipeline::set_config(
            cfg.clone(),
            std::path::PathBuf::from(data_dir).join("stickers"),
        );

        // 初始化数据库
        let db_path = format!("{}/alicebot.db", data_dir);
        match rt.block_on(db::init_database(&db_path)) {
            Ok(db) => {
                pipeline::set_db(db);
                log::info!("[AliceBot] 数据库初始化完成: {}", db_path);
            }
            Err(e) => {
                log::error!("[AliceBot] 数据库初始化失败: {}", e);
                pipeline::clear_config();
                rt.shutdown();
                return PluginInitResult::err(&format!("数据库初始化失败: {}", e));
            }
        }

        match memory::restore_short_context() {
            Ok(report) => log::info!(
                "[AliceBot] 短期上下文恢复完成: sessions={}, messages={}, inbound={}, outbound={}",
                report.sessions,
                report.messages,
                report.inbound_messages,
                report.outbound_messages
            ),
            Err(error) => log::warn!("[AliceBot] 短期上下文恢复失败，使用空缓存继续启动: {error}"),
        }

        // 启动后台任务（压缩、反思等）
        if let Err(e) = rt.start_background_tasks(cfg) {
            pipeline::clear_db();
            pipeline::clear_config();
            rt.shutdown();
            log::error!("[AliceBot] 后台任务启动失败: {e}");
            return PluginInitResult::err(&format!("后台任务启动失败: {e}"));
        }

        log::info!("[AliceBot] 初始化完成");
        PluginInitResult::ok()
    }

    // ── 关闭 ──────────────────────────────────────────────────

    #[shutdown]
    fn shutdown() {
        log::info!("[AliceBot] 关闭...");
        let rt = &*RUNTIME;
        rt.shutdown();
        decision::clear_runtime_state();
        memory::clear_runtime_state();
        pipeline::clear_command_suppressions();
        pipeline::clear_db();
        pipeline::clear_config();
        log::info!("[AliceBot] 已关闭");
    }

    // ── 消息拦截 (pre_handle) ──────────────────────────────────

    #[pre_handle]
    fn pre_handle(_req: &InterceptorRequest) -> InterceptorResponse {
        // 命令回调会在 after_completion 之前运行，届时才能可靠地抑制自主回复。
        InterceptorResponse::allow()
    }

    #[after_completion]
    fn after_completion(req: &InterceptorRequest) {
        accept_inbound_message(req);
    }

    // ── 命令 ───────────────────────────────────────────────────

    #[command(
        name = "ask",
        description = "直接问 AliceBot 问题",
        aliases = "问,艾特",
        category = "ai"
    )]
    fn cmd_ask(req: &CommandRequest) -> CommandResponse {
        pipeline::suppress_autonomous_reply_for_command(req);
        let Some(task) = pipeline::DirectAskTask::from_command(req) else {
            return CommandResponse::text("想问什么呀～直接说就行");
        };

        let rt = &*RUNTIME;
        match rt.submit_direct_ask(task) {
            runtime::DirectAskSubmitResult::Enqueued => {
                CommandResponse::text("收到啦，我想一下再回复你～")
            }
            runtime::DirectAskSubmitResult::Full => {
                CommandResponse::text("我这会儿有点忙，等一下再问我吧～")
            }
            runtime::DirectAskSubmitResult::Unavailable => {
                CommandResponse::text("我还没有初始化好，等一下再问我吧～")
            }
        }
    }

    #[command(
        name = "forget",
        description = "让 AliceBot 忘记某件事",
        aliases = "忘记",
        category = "ai",
        scope = "private"
    )]
    fn cmd_forget(req: &CommandRequest) -> CommandResponse {
        pipeline::suppress_autonomous_reply_for_command(req);
        let text = req.args.as_str();
        if text.is_empty() {
            return CommandResponse::text("想让我忘记什么呀？说个关键词～");
        }
        let message = pipeline::normalize_command_message(req, text);
        if message.session_type != "private" {
            return CommandResponse::text("为了避免误删共享记忆，请私聊我使用这个命令～");
        }
        // 委托给 runtime 执行短事务，命令回调不访问模型网络。
        let rt = &*RUNTIME;
        let result = rt.block_on(async {
            memory::forget_by_keyword(&message.protocol, &message.sender_id, text).await
        });
        CommandResponse::text(&result)
    }

    #[command(
        name = "status",
        description = "查看 AliceBot 状态",
        aliases = "状态,stats",
        category = "tools",
        role = "admin"
    )]
    fn cmd_status(req: &CommandRequest) -> CommandResponse {
        pipeline::suppress_autonomous_reply_for_command(req);
        let rt = &*RUNTIME;
        let status = rt.block_on(async { pipeline::get_status().await });
        CommandResponse::text(&status)
    }

    // ── 配置验证 (API 0.6) ──────────────────────────────────

    #[validate_config]
    fn validate(
        request: &abi_stable_host_api::PluginConfigRequest,
    ) -> abi_stable_host_api::PluginConfigResult {
        match config::parse_and_validate_config(request.config_json.as_str()) {
            Ok(_) => abi_stable_host_api::PluginConfigResult::ok(),
            Err(error) => abi_stable_host_api::PluginConfigResult::err(&error),
        }
    }
}

#[cfg(test)]
mod plugin_contract_tests {
    #[test]
    #[allow(unsafe_code)]
    fn command_descriptor_limits_status_to_administrators() {
        // The macro emits this C ABI symbol; inspect the generated descriptor rather than
        // duplicating its authorization metadata in a separate Rust constant.
        let descriptor = unsafe { super::qimen_plugin_descriptor() };
        let status = descriptor
            .commands
            .iter()
            .find(|command| command.name.as_str() == "status")
            .expect("status command should be registered");
        assert_eq!(status.required_role.as_str(), "admin");
        assert!(status.scope.as_str().is_empty());

        let forget = descriptor
            .commands
            .iter()
            .find(|command| command.name.as_str() == "forget")
            .expect("forget command should be registered");
        assert!(forget.required_role.as_str().is_empty());
        assert_eq!(forget.scope.as_str(), "private");
    }
}
