# CineScaffold Windows x64

这是 CineScaffold Studio 的 Windows x64 便携安装包。应用、Python 和依赖都会安装在当前文件夹内，不修改系统 Python，也不需要管理员权限。

## 第一次使用

1. 将整个 ZIP 解压到普通本地目录，例如 `C:\CineScaffold`。不要直接在压缩包内运行。
2. 安装 **Blender 5.2.1 LTS**。
3. 按官方说明在 Blender 中安装并启用 **Blender MCP Add-on 1.0.0**。
4. 双击 `Install-CineScaffold.cmd`。首次安装需要联网下载 Python 和依赖。
5. 如果安装程序提示缺少 Git，请安装 Git for Windows 后再次运行安装程序；它用于安装固定提交的官方 Blender MCP Server。
6. 双击 `Start-CineScaffold.cmd`，浏览器会自动打开 Studio。
7. 在“设置”中填写 Provider、模型和 API Key。OpenAI/DeepSeek Base URL 留空即可使用官方地址。

## 发行边界

- 包内包含：CineScaffold wheel、Windows x64 依赖锁、固定版本 `uv.exe`、Prompt、Schema、示例和启动脚本。
- 包内不包含：Blender、Blender MCP Add-on、模型 API Key。
- Blender MCP Server 由安装脚本从官方仓库的固定提交独立安装，不作为 CineScaffold 的组成部分重新分发。
- 当前版本只支持 Windows x64 与 Python 3.12，不支持 Windows ARM64。

## 常见问题

### 浏览器没有自动打开

访问 `http://127.0.0.1:8080/`。如果端口被占用，可在 PowerShell 中运行：

```powershell
.\scripts\Start-CineScaffold.ps1 -Port 8081
```

### 找不到 Blender 或 blender-mcp.exe

在 Studio 的“设置 → Blender 与渲染”中填写实际路径。典型 Blender 路径为：

```text
C:\Program Files\Blender Foundation\Blender 5.2\blender.exe
```

安装脚本创建的 MCP 命令通常位于：

```text
.runtime\bin\blender-mcp.exe
```

### 重新安装

直接再次运行 `Install-CineScaffold.cmd`。它会重建隔离虚拟环境，但保留 `.cinescaffold.conf` 和 `runs` 结果。

## 安全说明

- Studio 默认只监听本机 `127.0.0.1`。
- `.cinescaffold.conf` 存放 API Key，请勿发送给他人。
- 校验发行 ZIP 旁边 `.sha256` 文件中的 SHA-256 后再分发。

官方依赖说明：

- Blender MCP：<https://www.blender.org/lab/mcp-server/>
- uv Windows/Python 管理：<https://docs.astral.sh/uv/getting-started/installation/>
