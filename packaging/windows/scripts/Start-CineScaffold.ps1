[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8080
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".runtime"
$Python = Join-Path $Runtime "venv\Scripts\python.exe"
$RuntimeBin = Join-Path $Runtime "bin"
$Config = Join-Path $Root ".cinescaffold.conf"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "尚未安装 CineScaffold；请先双击 Install-CineScaffold.cmd。"
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
$env:UV_PYTHON_BIN_DIR = $RuntimeBin
$env:UV_TOOL_DIR = Join-Path $Runtime "tools"
$env:UV_TOOL_BIN_DIR = $RuntimeBin
$env:PATH = "$RuntimeBin;$env:PATH"

Set-Location -LiteralPath $Root
& $Python -m cinescaffold ui --config $Config --host "127.0.0.1" --port $Port
if ($LASTEXITCODE -ne 0) {
    throw "CineScaffold Studio 退出码：$LASTEXITCODE"
}
