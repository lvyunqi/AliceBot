# AliceBot

AliceBot 是一个 QimenBot 动态插件，提供自主对话、原生 Agent 工具、异步 `/ask` 命令、按作用域隔离的记忆，以及隐私感知的表情包收藏。它以 Rust `cdylib` 形式分发，当前版本为 `0.2.6`，使用 QimenBot 动态 ABI API `0.6`。

## 默认表达

默认回复以一两句自然短句为主，先回答问题，不使用客服腔、刻意卖萌、连续语气词或无关的情绪铺垫。Emoji 默认目标频率为 `0.1`，只在确有语义需要时使用。`persona.speaking_style` 只影响用词，不能覆盖这些基础表达规则。

已经保存的插件配置不会自动改写；若旧配置仍设置了较高的 `behavior.emoji_usage` 或 `behavior.allow_typos = true`，可在配置页将其调整为 `0.1` 和 `false`。

同样地，已有配置中的 `stickers.collect_probability = 0.3` 不会被自动覆盖。新安装默认值为 `1.0`；升级后请在配置页将该值设为 `1.0`，避免合规图片被随机跳过。

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
| `/ask <文本>` | 所有人 | 将一次带会话历史和原生 Agent 工具的有界 LLM 请求加入队列，并通过配置指定的稳定 `account_id` 发送最终结果。 |
| `/sticker [关键词]` | 所有人 | 从当前协议已验证的收藏中确定性选择并发送一张表情包；只有宿主接受图片发送后才会回复确认文本。 |
| `/forget <关键词>` | 私聊 | 仅删除发起者对应主体下与关键词匹配的记忆和知识条目。 |
| `/status` | 管理员 | 返回固定的聚合健康计数，不包含消息正文、ID、URL、提示词、响应或密钥。 |

`/ask` 不会阻塞动态 FFI 回调。插件使用有界工作队列，并记录已接收、被拒绝和投递结果不确定的状态，以便实现幂等重试。

## 图片与表情包

默认情况下，模型只接收文本。为某个 provider 显式设置 `supports_vision = true` 后，AliceBot 会把当前消息中最多 4 张图片，以及命中明确历史图片指代的最近图片，作为多模态内容块发送给该 provider；可用 `vision_model` 为图片消息单独选择模型。公开 HTTPS 图片直接发送，带签名或临时凭据的 QQ 图片会在本地通过大小和 SSRF 检查后转成 Base64 发送，凭据不会转发给模型；下载结果会按 JPEG、PNG、WebP、GIF 文件特征校验，错误页不会伪装成图片内容。未开启视觉能力的主备模型会被跳过，不会被当成识图模型。

OpenAI 兼容 provider 使用 `image_url` 内容块，Anthropic provider 使用 Messages API 图片内容块；图片块会排在文字前，兼容部分按首块决定是否启用视觉编码器的网关。图片实际附加后，模型若仍声称没有看到图片，插件会用同一图片进行一次无工具的聚焦重试。调试级别日志会输出 provider、模型名和 `remote_images` / `inline_images` 数量，不输出图片 URL、Base64 或用户内容。

官方 QQ 的引用消息会按事件中的 `author` 识别当前发言者，并用 `msg_idx/ref_msg_idx` 精确关联被引用的历史消息。引用对象的作者只作为上下文，不会被当成当前发言者；即使后面出现了更近的其他图片，也优先使用真正被引用的图片。引用图片会在本轮按视觉输入处理，无法读取时不会降级给文本模型猜测。

用户 `@` 机器人要求表情包，或使用 `/sticker [关键词]` 时，插件不再让模型用文字虚构“已发图”：它会先走宿主图片发送队列。OneBot v11 与官方 QQ Bot 均使用已收藏的长期 HTTPS URL 发送图片；只有宿主接受投递后才会回复确认文本。含短期凭据、需要缓存的媒体 URL 不会被直接复用。

Agent 现在有三个真实动作工具：`collect_recent_image` 收藏当前、引用或当前发言者最近发送的图片；`send_recent_image` 把对应图片重新发回当前会话；`send_sticker` 按关键词从收藏中选择并发送。用户明确说“收藏一下”“复制发给我”“再发一次”时，插件会强制执行对应原生工具，即使模型没有主动发起工具调用，也不会退化成口头承诺。收藏以数据库事务为准，发送以 QimenBot 宿主回执为准；失败结果不会被改写成成功。

合规的入站图片会以默认概率 `1.0` 收藏，纯图片消息也不会因缺少文字说明被评分规则漏掉。敏感内容、每日收藏上限、HTTPS/公网地址校验和缓存体积上限仍然生效。每次判定会保存不含原始 URL 的结果记录；用户 `@` 机器人问“收藏了吗”“收到了吗”时，插件直接查询这条记录和当前缓存状态，回复“已收藏”“缓存中”“缓存失败”或明确的跳过原因，不交给模型猜测。

当消息提到“我发的什么图”“我发的图你看明白了吗”“刚才”“上条”“这张图”“看到没”等明确指代时，插件会先确定性查询当前会话的短期历史；没有精确引用时优先选取当前发言者最近发送的图片。启用 `llm.agent_enabled` 后，模型还可以调用 `search_history`、`search_memory`、`recent_media_status`、`sticker_status` 和 `search_stickers` 查询真实状态，再按需调用三个动作工具。查询结果不返回 URL、临时凭据或图片正文；动作工具同一轮不会重复执行。若模型没有成功调用动作工具却声称“已收藏”或“已发送”，回复会被替换为真实失败结果。工具有 `llm.agent_max_steps` 轮上限。

带签名的 QQ 图片地址只保留在当前插件进程的短期会话中，不会写入数据库。其脱敏身份和受限本地缓存会持久化；插件热重载后若缓存仍在，历史图片仍可作为 Base64 视觉输入。原始 URL 已失效且没有缓存时会明确说明无法读取或重发，不会说用户没有发图，也不会把本地缓存冒充成官方 QQ 可访问的 URL。

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
- Agent 工具审计只保存工具名、成功/失败状态、会话路由和时间，不保存参数、工具结果、消息正文或图片 URL；`/status` 只返回成功与失败的聚合计数。
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
