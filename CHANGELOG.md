# Changelog

## 0.7.0 — 2026-09-02

### Added

- 本地浏览器界面 CineScaffold Studio，支持自然语言、文本六维、Cinematic Brief JSON 与 Scene IR JSON 四种起点。
- 文本六维领域对象、固定六标题校验、自然语言运行时双写文本六维和机器权威 JSON 六维。
- UI 设置中心、文件导入、实时进度、事件日志、分阶段 token/成本、视频预览与本地产物入口。
- macOS、Windows、Linux 的 Blender/MCP 默认路径和打开文件边界。
- 可选 `nicegui==3.16.0` 依赖及继承核心版本冻结的 `requirements-ui.lock`。

### Changed

- CLI 与 UI 统一使用异步 `WorkflowRunner`，共享阶段、失败状态、覆盖规则和 `pipeline_summary.json`。
- 配置保存改为校验后原子替换、备份与 POSIX `0600` 权限；浏览器不会收到已有 API Key 明文。
- 语义解析与规划 Agent 可分别配置模型价格，避免不同供应商的成本串用。

### Verified

- 224 个测试与 5 个 subtests 通过。
- 真实浏览器完成桌面、700 px 响应式、设置与四入口交互检查。
- Mock 端到端 UI 冒烟通过语义、规划、Blender MCP、Runtime Validation、Workbench 渲染和视频回放。
