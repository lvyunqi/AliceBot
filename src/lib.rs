//! AliceBot — 最强学习力拟人 QQ 机器人
//!
//! QimenBot 动态插件 (API 0.6)。自建 tokio runtime 管理 LLM 调用、数据库和后台任务。
//! 同步 FFI 回调只做轻量调度，耗时操作交给 runtime 内的异步流水线。

#![deny(unsafe_code)]

use abi_stable_host_api::{
    CommandRequest, CommandResponse, InterceptorRequest, InterceptorResponse, NoticeRequest,
    NoticeResponse, PluginInitConfig, PluginInitResult,
};
use qimen_dynamic_plugin_derive::dynamic_plugin;

mod config;
mod db;
mod decision;
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

// ─── 动态插件描述符 ────────────────────────────────────────────

#[dynamic_plugin(
    id = "alicebot",
    version = "0.1.0",
    api = "0.6",
    config_schema = "../config.schema.json",
    config_ui = "../config.ui.json",
    config_version = 2,
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

        pipeline::set_config(cfg.clone());

        // 初始化数据库
        let data_dir = config.data_dir.as_str();
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
        pipeline::clear_db();
        pipeline::clear_config();
        log::info!("[AliceBot] 已关闭");
    }

    // ── 消息拦截 (pre_handle) ──────────────────────────────────

    #[pre_handle]
    fn pre_handle(_req: &InterceptorRequest) -> InterceptorResponse {
        // 不做全量拦截，只做快速过滤
        // 实际消息处理交给 route 回调
        InterceptorResponse::allow()
    }

    // ── 消息路由 ───────────────────────────────────────────────

    #[route(kind = "notice", events = "GroupMessage,PrivateMessage")]
    fn on_message(req: &NoticeRequest) -> NoticeResponse {
        let rt = &*RUNTIME;
        let event = pipeline::NoticeEvent::from_request(req);
        match pipeline::record_inbound(event) {
            Ok(Some(message)) => match rt.submit_message(message.clone()) {
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
            },
            Ok(None) => {}
            Err(error) => {
                log::error!("[AliceBot] 入站消息 journal 失败: {error}");
            }
        }
        NoticeResponse {
            action: abi_stable_host_api::DynamicActionResponse::ignore(),
        }
    }

    // ── 命令 ───────────────────────────────────────────────────

    #[command(
        name = "ask",
        description = "直接问 AliceBot 问题",
        aliases = "问,艾特",
        category = "ai"
    )]
    fn cmd_ask(req: &CommandRequest) -> CommandResponse {
        let text = req.args.as_str();
        if text.is_empty() {
            return CommandResponse::text("想问什么呀～直接说就行");
        }

        let rt = &*RUNTIME;
        let reply = rt.block_on(async { pipeline::direct_ask(text, req).await });
        CommandResponse::text(&reply)
    }

    #[command(
        name = "forget",
        description = "让 AliceBot 忘记某件事",
        aliases = "忘记",
        category = "ai"
    )]
    fn cmd_forget(req: &CommandRequest) -> CommandResponse {
        let text = req.args.as_str();
        if text.is_empty() {
            return CommandResponse::text("想让我忘记什么呀？说个关键词～");
        }
        // 委托给 runtime 异步处理
        let rt = &*RUNTIME;
        let result = rt.block_on(async { memory::forget_by_keyword(text).await });
        CommandResponse::text(&result)
    }

    #[command(
        name = "status",
        description = "查看 AliceBot 状态",
        aliases = "状态,stats",
        category = "tools"
    )]
    fn cmd_status(_req: &CommandRequest) -> CommandResponse {
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
