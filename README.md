# AliceBot

AliceBot 是一个 QimenBot 动态插件，提供自主对话、异步 `/ask` 命令、按作用域隔离的记忆，以及隐私感知的贴纸收集。它以 Rust `cdylib` 形式分发，使用 QimenBot 动态 ABI API `0.6`。

## 兼容性

- QimenBot `v0.1.18` 或更高版本。
- 动态 ABI API `0.6`。
- `abi-stable-host-api` 和 `qimen-dynamic-plugin-derive` 均为 `0.1.13`。
- Windows x64：`x86_64-pc-windows-msvc`。
- Linux x64 GNU：`x86_64-unknown-linux-gnu`，在 Debian 11 CI 中构建，glibc 上限为 2.31。
- musl 宿主不受支持，因为 QimenBot 无法在 musl 环境中动态加载插件。

代码中的测试样例已覆盖 OneBot v11 与官方 QQ Bot 的事件规范化、隐私边界和不受支持媒体的降级行为。真实 OneBot/官方 QQ Bot 的消息触发、出站发送和媒体上传仍需在接入机器人账号后验证，当前版本不将其声明为经过发布测试的能力。

## 命令

| 命令 | 权限 | 行为 |
|---|---|---|
| `/ask <文本>` | 所有人 | 将一次有界 LLM 请求加入队列，并通过配置指定的稳定 `account_id` 发送最终结果。 |
| `/forget <关键词>` | 私聊 | 仅删除发起者对应主体下与关键词匹配的记忆和知识条目。 |
| `/status` | 管理员 | 返回固定的聚合健康计数，不包含消息正文、ID、URL、提示词、响应或密钥。 |

`/ask` 不会阻塞动态 FFI 回调。插件使用有界工作队列，并记录已接收、被拒绝和投递结果不确定的状态，以便实现幂等重试。

## 安装

1. 下载与 QimenBot 宿主 target 匹配的发布资产。
2. 将动态库复制或重命名到宿主的动态插件目录：

   - Windows：`plugins/bin/alicebot.dll`
   - Linux GNU：`plugins/bin/libalicebot.so`

3. 配置宿主的 `official_host.plugin_bin_dir`，并确保 `plugin_config_dir` 可持久化。AliceBot 的配置文件通常为 `config/plugins/alicebot.toml`。
4. 在 QimenBot Web 管理面板中重新加载动态插件，或使用管理员令牌调用 `POST /api/v1/plugins/reload`。

插件提供 API 0.6 配置 Schema，`config_apply = "reload"`。API 密钥是只写密钥：通过密钥专用更新通道设置；省略字段会保留原值；明确提交 `null` 会清空密钥。清空后的提供商仍保留在配置中，但不会参与 LLM 调用。

在通过 Web 表单添加提供商前，可使用以下最小安全配置：

```toml
[persona]
name = "Alice"

[llm]
enabled = false

[send]
account_id = ""
```

不要提交 `config/plugins/alicebot.toml`、API 密钥、宿主管理令牌、SQLite 数据库、日志或已下载的贴纸媒体。

## 数据与网络行为

- SQLite 状态保存在 QimenBot 为插件提供的数据目录中。
- LLM 请求使用配置的 OpenAI 兼容或 Anthropic 兼容 HTTPS 端点。完整提示词、响应、响应体和 API 密钥不会被记录在日志或诊断数据中。
- 贴纸缓存仅接受来自已验证公网地址、大小受限的 HTTPS 媒体。带签名的媒体 URL 会被脱敏，缓存完成前不会被发送。
- 插件不会读取 QQ Bot 凭据，也不会自行上传官方 QQ 媒体；相关平台凭据和上传操作由 QimenBot 宿主负责。

## 构建与发布

```powershell
cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked --all-targets
cargo build --release --locked --target x86_64-pc-windows-msvc
.\scripts\package-release.ps1 -Target x86_64-pc-windows-msvc -InputPath target\x86_64-pc-windows-msvc\release\alicebot.dll
```

`.github/workflows/release.yml` 是发布流水线。它会校验版本元数据，执行格式化检查、检查、测试和 Clippy，构建原生 Windows x64 与 Debian 11 Linux x64 GNU 资产，记录 SHA256 和 Linux glibc 要求，为二进制生成构建来源证明，并且只在 `vX.Y.Z` tag 上发布 GitHub 发布版本。

发布资产名包含完整 target：

- `qimen_dynamic_plugin_alicebot-x86_64-pc-windows-msvc.dll`
- `libqimen_dynamic_plugin_alicebot-x86_64-unknown-linux-gnu.so`

## 许可证

MIT，详见 [LICENSE](LICENSE)。
