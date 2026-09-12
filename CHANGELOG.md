# Changelog

## Unreleased

### Added

- Cinematic Brief v0.6 增加主体静态/动态分类、稳定动作 ID、关键叙事标记、按实体独立的稀疏动作时间线和类型化事件关系；顺序描述不再机械等分总时长。
- 新增 Narrative Fidelity Gate，简化交付仍必须保留类型化动作、容纳/显隐后置状态和全部 explicit 要求。

- 新增跨实体、Constraint、Motion 与 Camera 的原子 `apply_candidate_patch`；提案先在不可见 Store preview 中完整验证，退化时不产生 revision。
- Design Option 预测摘要增加按根因去重的有限 hard violation hints，Agent-facing 验证增加保留完整明细、主因同样去重的 `repair_focus`。
- 默认规划路径新增结构化 Recovery Context；Commit Gate 拒绝、Agent 过早声明不可行/不支持和可恢复的 Agent 异常会在原预算中续跑。
- 将 Full Fidelity Gate 与 Execution Safety Gate 分离；完整恢复用尽后优先保留最佳 Agent Candidate，否则从 Objective Brief 确定性构建不增加 Provider 请求的简化交付。
- 规划与 Pipeline Summary 新增 `standard/recovered/simplified` 交付等级，简化交付保留完整 fidelity violations 供产品提示和审计。
- Runtime Validation 通过后默认最佳努力生成无压缩 `scene.glb` 与 `viewer_manifest.json`；清单绑定 Scene IR canonical hash、GLB SHA-256、稳定 Entity/Camera 节点、时间轴、显隐区间和 MP4 降级策略。
- `ExecutionResult.viewer` 与 Pipeline artifact map 公开交互预览状态和产物；GLB 导出或结构校验失败不会阻止白模视频渲染。

### Changed

- Semantic 输出的 `action_kind` 收敛为 hold/locomotion/interact/other；Planning 只依据明确方向、路径、载体和后置状态建立几何，不再把 arrive/depart/transport 等叙事标签二次加工为目标或方向。
- Scene Planner 必须先联合检查每个实体的完整阶段时间线；未明确转向或反转的线性运动跨静止阶段继承同一路线方向，事件末端会合关系不再把另一个参与者变成运动目标。
- Design Option 只向 Agent 暴露完整 hard-pass 候选，并在应用时再次完整复验。
- 外层恢复轮次改用不含上一轮 thinking 的新 `RepairPacket`；确定性工具不支持的首个 hard error 会立即开放受限原子 Candidate Patch。
- 动作方向验证改为检查施事主体自身的定向位移，目标移动不再能替代“靠近/离开”；等待和停留会写入保持关键点，已隐藏主体不再生成冗余 carried 轨道。
- 靠近、抵达、离开、进入/退出和绕行增加屏幕结果检查；世界动作保持 hard，屏幕呈现默认记录 warning。

### Fixed

- 修复接载叙事中车辆被错误要求朝乘客移动、随后又远离同一乘客的问题；乘员进入可由隐藏后置状态和紧接的 carried 关系满足，不再强制人物代理移动到载体中心。
- 确定性后备规划器不再根据 boarding/departure 等事件名制造方向；旧版接车 Brief 仍可在无专用动作种类的情况下生成 hard-pass Design Option。
- Semantic 时间关系允许独立主体动作以“开始先后但区间重叠”的方式表达；无明确数值间隔时，会把与事件区间冲突的关系标签确定性规范化为 `starts_before/starts_after/starts_with/ends_with/during/overlaps`，不再因可修复的标签歧义让整个 Pipeline 在 Planning 前失败。明确的最小/最大时间间隔仍失败关闭。

- Entity 更新改为真正的 merge-patch：省略字段保持现有值，显式空值才执行字段契约中的清除语义。
- 正常 Agent 在确定性修复无解后只看到一个组合 Candidate Patch；四个分散低层 Patch 保留给兼容和诊断，不再共同占用正常工具表。坐标、时间、来源与运动语义等一次规划所需 Prompt 信息暂不删减，等待冻结 A/B 后再决定。
- `suggest_repairs` 在当前 revision 确定性搜索穷尽后重新开放受控手工 Mutation，仍不允许绕过 Schema、Validator 或 Commit Gate。

### Verification

- Python 3.12 完整离线回归 289 项通过；旧版道路接车 Brief 的 Mock 端到端规划和已取消真实失败结构的离线 Design 重放均成功。
- `uv build` 成功生成 Core 0.8.9 的 sdist 与 wheel。
- Blender 5.2.1 LTS 真实后台验证静态主体/推镜和人物/车辆运动两类 Scene IR；GLB 约 122 KB，单次导出约 0.03 秒，稳定节点映射、摄影机、TRS 动画与 Scene IR 显隐补偿均通过结构检查。

## 0.8.3 — 2026-09-11

### Added

- Pipeline 可接收同币种 Provider 金额上限，并以每次响应的实际 token usage 和冻结价格快照累计费用；达到或超过上限后，在下一次 Provider 请求前停止。
- Semantic 与 Planning 每个已返回 usage 的请求都会发出独立 `provider_cost_incurred` 事件，供宿主应用即时持久化；Semantic 事件在结构化内容校验前发出，避免合法 HTTP 响应因后续解析失败而漏记。

### Verification

- Python 3.12 完整离线回归 254 项通过；新增回归覆盖 Semantic 首次响应越线后不进入 Planning、Semantic 响应结构非法仍先记账、Planning 越线后不再发出下一次模型请求，以及纯 Scene IR 执行不要求无关价格。

## 0.8.2 — 2026-09-11

### Fixed

- Design 物化优先使用 `subject_motion` 自身的精确时间段，避免多个动作共用一个大事件时把车辆到达和人物上车错误合并。
- 同一运动在 `action` / `direction` / `trajectory` / 空间关系 / 时间事件中的重复 explicit 表达共享映射证据；匹配的远景层与远距离关系同样处理，仍保留字段类型和实体绑定检查。
- 用户明确要求主体可见时生成 hard `keep_in_frame` 约束。该约束只证明代理几何至少部分进入画面，不声称已验证遮挡关系。
- `board` / `disembark` 的具体位移语义优先于泛化 `local_interaction` 标签，上下载人物不再被静止检查误拒。
- 解析轨道在用户未明确指定相机高度或视角时忽略 Brief 中的系统默认高度，自动采用可辨识闭合轨道的斜俯视机位。

### Verification

- Python 3.12 完整离线回归 249 项通过。
- 三条真实 `deepseek-flash` 规划结果在不增加模型调用的情况下离线重放，荒漠飞船、10 秒道路接车和嵌套公转均以 0 hard violation 通过 Commit Gate；Blender 5.2.1 LTS 实际构建后 Runtime Validation 均为 0 violation，并生成 640×360 / 12 fps H.264 预览。
- 三条付费调用按 DeepSeek V4.1 Flash 当时非繁忙时段官方 token 价格估算分别为 ¥0.104007、¥0.185781 和 ¥0.234332，合计 ¥0.524120；账户余额的两位小数结算变化为 ¥0.53。

## 0.8.1 — 2026-09-11

### Fixed

- 为 DeepSeek 官方 `deepseek-flash` 别名补充 thinking/tool calling 能力覆盖。PydanticAI 2.36 尚未识别该别名，原先会发送 thinking 模式不支持的 `tool_choice=required`；现在保留 DeepSeek reasoning 字段协议，并让框架按 Profile 自动降级为 `tool_choice=auto`。

### Verification

- Python 3.12 完整离线回归 246 项通过；`deepseek-flash` 构造回归确认 thinking 可用且不要求 `tool_choice=required`，sdist 与 wheel 构建通过。
- Studio 受控真实回归在约 145.3 秒内完成 DeepSeek Semantic/Planning、Blender、Artifact 与数据库终态链路；Planning 的 17 次请求和 14 次工具调用证明 thinking + tools 多轮协议可用。

## 0.8.0 — 2026-09-07

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
