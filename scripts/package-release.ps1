[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet(
        "x86_64-pc-windows-msvc",
        "aarch64-pc-windows-msvc",
        "x86_64-unknown-linux-gnu",
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin"
    )]
    [string]$Target,

    [string]$InputPath,

    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\dist")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$platform = switch -Regex ($Target) {
    "windows" { "windows"; break }
    "linux" { "linux"; break }
    "darwin" { "macos"; break }
    default { throw "Unsupported target: $Target" }
}

$sourceName = switch ($platform) {
    "windows" { "alicebot.dll" }
    "linux" { "libalicebot.so" }
    "macos" { "libalicebot.dylib" }
}
$assetName = switch ($platform) {
    "windows" { "qimen_dynamic_plugin_alicebot-$Target.dll" }
    "linux" { "libqimen_dynamic_plugin_alicebot-$Target.so" }
    "macos" { "libqimen_dynamic_plugin_alicebot-$Target.dylib" }
}

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    $inputDirectory = if ($Target -eq "x86_64-pc-windows-msvc") {
        Join-Path $repoRoot "target\release"
    } else {
        Join-Path $repoRoot "target\$Target\release"
    }
    $InputPath = Join-Path $inputDirectory $sourceName
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw "Built library is missing: $InputPath"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$assetPath = Join-Path $OutputDirectory $assetName
Copy-Item -LiteralPath $InputPath -Destination $assetPath -Force

$asset = Get-Item -LiteralPath $assetPath
$manifest = [ordered]@{
    plugin_id = "alicebot"
    target = $Target
    asset_name = $asset.Name
    size_bytes = $asset.Length
    sha256 = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}
$manifestPath = "$assetPath.manifest.json"
$manifest | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding utf8

$manifest | ConvertTo-Json
