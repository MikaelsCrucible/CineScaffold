[CmdletBinding()]
param(
    [switch]$InstallMcp
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root ".runtime"
$Uv = Join-Path $Root "tools\uv.exe"
$Venv = Join-Path $Runtime "venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$RuntimeBin = Join-Path $Runtime "bin"
$Config = Join-Path $Root ".cinescaffold.conf"
$ConfigExample = Join-Path $Root ".cinescaffold.example.conf"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败（退出码 $LASTEXITCODE）：$FilePath $($Arguments -join ' ')"
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "此发行包只支持 Windows x64。"
}
if (-not (Test-Path -LiteralPath $Uv -PathType Leaf)) {
    throw "发行包不完整：缺少 tools\uv.exe。"
}

New-Item -ItemType Directory -Force -Path $Runtime, $RuntimeBin | Out-Null
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
$env:UV_PYTHON_BIN_DIR = $RuntimeBin
$env:UV_TOOL_DIR = Join-Path $Runtime "tools"
$env:UV_TOOL_BIN_DIR = $RuntimeBin
$env:PATH = "$RuntimeBin;$env:PATH"

Write-Host "[1/4] 正在准备隔离的 Python 3.12 环境……" -ForegroundColor Cyan
Invoke-Checked $Uv "venv" "--python" "3.12" "--clear" $Venv

Write-Host "[2/4] 正在安装锁定的 CineScaffold 依赖……" -ForegroundColor Cyan
$Lock = Join-Path $Root "app\requirements-windows.lock"
Invoke-Checked $Uv "pip" "install" "--python" $Python "--require-hashes" "--requirements" $Lock
$Wheels = @(Get-ChildItem -LiteralPath (Join-Path $Root "app") -Filter "cinescaffold-*.whl")
if ($Wheels.Count -ne 1) {
    throw "发行包必须且只能包含一个 CineScaffold wheel。"
}
Invoke-Checked $Uv "pip" "install" "--python" $Python "--no-deps" $Wheels[0].FullName

Write-Host "[3/4] 正在验证应用安装……" -ForegroundColor Cyan
Invoke-Checked $Python "-m" "cinescaffold" "--version"
if (-not (Test-Path -LiteralPath $Config)) {
    Copy-Item -LiteralPath $ConfigExample -Destination $Config
}

Write-Host "[4/4] 正在检查可选 Blender MCP Server……" -ForegroundColor Cyan
if (-not $InstallMcp) {
    Write-Host "默认使用后台 Blender，已跳过可选 MCP Server。" -ForegroundColor DarkGray
} elseif (Get-Command git.exe -ErrorAction SilentlyContinue) {
    $McpSource = "git+https://projects.blender.org/lab/blender_mcp.git@4309a39646e644261624bfcd2bca669b343b7621#subdirectory=mcp"
    Invoke-Checked $Uv "tool" "install" "--python" "3.12" "--force" "--with" "mcp[cli]>=1.2,<2" "--from" $McpSource "blender-mcp"
} else {
    Write-Warning "未找到 Git，无法安装可选 Blender MCP Server。"
}

Write-Host ""
Write-Host "CineScaffold 已安装到发行包自己的隔离目录。" -ForegroundColor Green
Write-Host "下一步：确认 Blender 5.2.1 LTS 已安装，然后双击 Start-CineScaffold.cmd。"
