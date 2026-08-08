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
| `/ask <文本>` | 所有人 | 将一次带会话历史和只读工具查询的有界 LLM 请求加入队列，并通过配置指定的稳定 `account_id` 发送最终结果。 |
| `/sticker [关键词]` | 所有人 | 从当前协议已验证的收藏中确定性选择并发送一张表情包；只有宿主接受图片发送后才会回复确认文本。 |
| `/forget <关键词>` | 私聊 | 仅删除发起者对应主体下与关键词匹配的记忆和知识条目。 |
| `/status` | 管理员 | 返回固定的聚合健康计数，不包含消息正文、ID、URL、提示词、响应或密钥。 |

`/ask` 不会阻塞动态 FFI 回调。插件使用有界工作队列，并记录已接收、被拒绝和投递结果不确定的状态，以便实现幂等重试。

## 图片与表情包

默认情况下，模型只接收文本。为某个 provider 显式设置 `supports_vision = true` 后，AliceBot 会把当前消息中最多 4 张图片，以及命中明确历史图片指代的最近图片，作为多模态内容块发送给该 provider；可用 `vision_model` 为图片消息单独选择模型。公开 HTTPS 图片直接发送，带签名或临时凭据的 QQ 图片会在本地通过大小和 SSRF 检查后转成 Base64 发送，凭据不会转发给模型；下载失败时不会交给文本模型猜测。未开启视觉能力的主备模型会被跳过，不会被当成识图模型。

OpenAI 兼容 provider 使用 `image_url` 内容块，Anthropic provider 使用 Messages API 图片内容块。请只为实际支持视觉输入的模型开启该开关，例如视觉版 OpenAI 兼容模型或 Claude 视觉模型。

官方 QQ 的引用消息会按事件中的 `author` 识别当前发言者；引用对象的作者只作为上下文，不会被当成当前发言者。引用图片会在本轮按视觉输入处理，无法下载时不会降级给文本模型猜测。

用户 `@` 机器人要求表情包，或使用 `/sticker [关键词]` 时，插件不再让模型用文字虚构“已发图”：它会先走宿主图片发送队列。当前只有 OneBot v11 URL 图片发送标记为已支持；官方 QQ Bot 仍是未验证状态，会如实说明而不会声称图片已送达。

当消息提到“我发的图你看明白了吗”“刚才”“上条”“这张图”等明确指代时，插件会先确定性查询当前会话的短期历史；“我发的图”会优先选取当前发言者最近发送的图片。工具调用仍可在启用 `llm.agent_enabled` 后补充查询会话历史、长期记忆和近期媒体状态，但正确识别历史图片不依赖模型是否主动调用工具。工具有 `llm.agent_max_steps` 轮上限，不执行发送、改配置或写库操作。

带签名的 QQ 图片地址只保留在当前插件进程的短期会话中，不会写入数据库；插件重载或地址过期后，无法重新下载时会明确说明图片未加载，而不会说用户没有发图。

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
