# Changelog

## Unreleased

### Added

- 新增稳定的 `cinescaffold.api` 嵌入入口，公开管线运行、Scene IR 执行、配置与公共数据类型，外部应用无需依赖 CLI/UI 或内部模块。

### Changed

- 版本提升至 0.8.0；Prompt 与 JSON Schema 迁入 `src/cinescaffold/resources/` 并随 wheel 分发。
- Runtime、CLI、UI 与 Planning 的缺省资源路径统一通过 `importlib.resources` 从已安装包解析，不再要求当前目录为 CineScaffold 源码仓库；显式资源覆盖仍保留。
- 新增仓库外 wheel 验收：隔离 Python 3.12 环境中的 Mock parse/plan、公共 API、Blender 5.2.1 LTS 构建和 180 帧预览渲染均通过。
- Scene IR 构建默认改为直接启动无窗口 Blender，不再先创建 factory template 或要求 Blender MCP。
- 构建与渲染分别支持 `background` 和 `mcp` 后端；MCP 保留为显式兼容选项，并延迟到实际使用时初始化。
- 当构建和渲染都使用后台后端时，默认由单个无窗口 Blender 进程连续完成构建、Runtime Validation、保存和渲染；`split` 诊断模式及所有 MCP 组合保留分阶段执行。
- Fused 模式在构建结果落盘时即时切换渲染进度，并分别执行构建与渲染超时；超时会清理 Blender 子进程，同时保留已完成的构建产物。
- 执行 manifest 升级到 v0.3，记录独立后端、请求/实际进程模式、后台结果和日志产物。
- Windows 安装默认跳过 MCP Server；只有显式传入 `-InstallMcp` 时才安装可选兼容后端。

### Verification

- Python 3.12 完整离线回归 242 项测试通过。
- Blender 5.2.1 LTS 真实后台 Fused/Split 冒烟均通过：10 秒/240 帧 Scene IR 完成构建、零 violation Runtime Validation，并输出 120 帧 640×360 Workbench 诊断视频；产物中的 Runtime Snapshot 与 Validation 逐字节一致。Fused CLI 在构建落盘后即时显示渲染阶段，0.01 秒真实渲染超时测试正确返回 `render_failed` 并保留已验证的 `scene.blend`。冷态单次曾测得约 4.50 秒对 2.31 秒，但后续热态短任务未显示稳定墙钟优势，因此只承诺消除一次进程启动和 `.blend` 重载，不承诺固定加速比例。
- 同一样本的 MCP 构建兼容冒烟通过，Runtime Snapshot 与 Validation 和后台构建逐字节一致；单次端到端约 4.3 秒，因此当前测量不足以声称有渲染性能提升。

## 0.7.0 — 2026-09-02

### Added

- 本地浏览器界面 CineScaffold Studio，支持自然语言、文本六维、Cinematic Brief JSON 与 Scene IR JSON 四种起点。
- 文本六维领域对象、固定六标题校验、自然语言运行时双写文本六维和机器权威 JSON 六维。
- UI 设置中心、文件导入、实时进度、事件日志、分阶段 token/成本、视频预览与本地产物入口。
- macOS、Windows、Linux 的 Blender/MCP 默认路径和打开文件边界。
- 可选 `nicegui==3.16.0` 依赖及继承核心版本冻结的 `requirements-ui.lock`。
- Windows x64 便携安装包、双击安装/启动脚本、平台专用依赖锁与原生 GitHub Actions 构建。

### Changed

- CLI 与 UI 统一使用异步 `WorkflowRunner`，共享阶段、失败状态、覆盖规则和 `pipeline_summary.json`。
- 配置保存改为校验后原子替换、备份与 POSIX `0600` 权限；浏览器不会收到已有 API Key 明文。
- 语义解析与规划 Agent 可分别配置模型价格，避免不同供应商的成本串用。
- 文本六维 UI 使用六个不可删除的固定标签和独立正文框；`.txt` 导入先校验再分项填充。

### Verified

- 226 个测试与 5 个 subtests 通过。
- 真实浏览器完成桌面、700 px 响应式、设置与四入口交互检查。
- Mock 端到端 UI 冒烟通过语义、规划、Blender MCP、Runtime Validation、Workbench 渲染和视频回放。
- Windows 发行包在 `windows-latest` 中安装锁定依赖、运行回归并组装带 SHA-256 的 ZIP。
