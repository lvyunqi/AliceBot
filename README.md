# AliceBot

AliceBot is a QimenBot dynamic plugin that provides autonomous conversational
behavior, an asynchronous `/ask` command, scoped memory, and privacy-aware
sticker collection. It is distributed as a Rust `cdylib` and uses QimenBot's
dynamic ABI API `0.6`.

## Compatibility

- QimenBot `v0.1.18` or newer.
- Dynamic ABI API `0.6`.
- `abi-stable-host-api` and `qimen-dynamic-plugin-derive` `0.1.13`.
- Windows x64: `x86_64-pc-windows-msvc`.
- Linux x64 GNU: `x86_64-unknown-linux-gnu`, built in Debian 11 CI with a
  glibc 2.31 ceiling.
- musl hosts are not supported because QimenBot dynamic loading is unavailable
  there.

The source fixtures cover OneBot v11 and official QQ event normalization,
privacy boundaries, and unsupported media fallbacks. Real OneBot/official QQ
message triggering, outbound sending, and media upload still require a bot
account and have not been claimed as release-tested capabilities.

## Commands

| Command | Access | Behavior |
|---|---|---|
| `/ask <text>` | all | Queues one bounded LLM request and sends the eventual result through the configured stable `account_id`. |
| `/forget <keyword>` | private | Forgets only the requesting subject's matching memory and knowledge entries. |
| `/status` | admin | Returns fixed aggregate health counts without message text, IDs, URLs, prompts, responses, or secrets. |

`/ask` does not block the dynamic FFI callback. The plugin uses a bounded worker
queue and records accepted, rejected, and uncertain delivery outcomes for
idempotent retry behavior.

## Installation

1. Download the release asset matching the QimenBot host target.
2. Copy or rename the library into the host's dynamic plugin directory:

   - Windows: `plugins/bin/alicebot.dll`
   - Linux GNU: `plugins/bin/libalicebot.so`

3. Configure the host's `official_host.plugin_bin_dir` and keep its
   `plugin_config_dir` persistent. AliceBot's configuration file is normally
   `config/plugins/alicebot.toml`.
4. Reload dynamic plugins from the QimenBot web administration panel or through
   `POST /api/v1/plugins/reload` with an administrator token.

The plugin exposes an API 0.6 configuration schema with `config_apply =
"reload"`. API keys are write-only secrets: set them through the secret update
channel, omit them to preserve the current value, or submit `null` to clear a
key. A cleared provider stays in the configuration but is excluded from LLM
calls.

Minimal safe configuration, before adding a provider through the web form:

```toml
[persona]
name = "Alice"

[llm]
enabled = false

[send]
account_id = ""
```

Do not commit `config/plugins/alicebot.toml`, API keys, host access tokens,
SQLite databases, logs, or downloaded sticker media.

## Data and network behavior

- SQLite state is created in the plugin data directory supplied by QimenBot.
- LLM requests use configured OpenAI-compatible or Anthropic-compatible HTTPS
  endpoints. Full prompts, responses, response bodies, and API keys are not
  logged or stored in diagnostics.
- Sticker caching accepts only bounded HTTPS media from validated public
  addresses. Signed media URLs are redacted and are not sent until cached.
- The plugin never reads QQ Bot credentials or performs official QQ media
  uploads itself; QimenBot owns those platform credentials and uploads.

## Build and release

```powershell
cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked --all-targets
cargo build --release --locked --target x86_64-pc-windows-msvc
.\scripts\package-release.ps1 -Target x86_64-pc-windows-msvc -InputPath target\x86_64-pc-windows-msvc\release\alicebot.dll
```

`.github/workflows/release.yml` is the release pipeline. It validates version
metadata, runs formatting/check/test/Clippy, builds native Windows x64 and
Debian 11 Linux x64 GNU assets, records SHA256 and Linux glibc requirements,
attests the binaries, and publishes a GitHub Release only for a `vX.Y.Z` tag.

Release asset names include the complete target:

- `qimen_dynamic_plugin_alicebot-x86_64-pc-windows-msvc.dll`
- `libqimen_dynamic_plugin_alicebot-x86_64-unknown-linux-gnu.so`

## License

MIT. See [LICENSE](LICENSE).
