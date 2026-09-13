# Changelog

## Unreleased

### Added

- Toolkit v0.36 为首次 Design 全部 hard-fail 的情况增加单独 `repair_baseline`：只保留通过 Execution Safety 的最佳失败候选，绝不混入正常 Options；`begin_design_repair` 物化后立即开放受限原子 Candidate Patch。
- Scene Skeleton 新增 `local_transform` 和 `parabolic` 表达。局部互动会生成可观察的旋转/尺度关键点并由 Validator 复验；抛物线会生成含弧顶的 Transform 轨迹。

- Toolkit v0.32 实现 `collision_clearance`、`visibility_fraction` 与 `negative_space` Validator；静态构图会把主要巨物画幅比例、至少一次实际入框和负空间与主体画幅比例一起建立为可验证约束。
- Toolkit v0.29 新增 `surface_clearance_range`：按两代理沿关系方向的外边界测量净空，并以较大代理的方向完整尺寸归一化；支持有界最小、偏好和最大比例以及 world/ground-plane 测量空间。
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

- Toolkit v0.36 收紧 Semantic→Skeleton→Design→Validator 契约：动态场景必须真实包含主体状态变化，关键 `motion_id` 必须由相容阶段实现；明确画幅比例和水平/垂直画面位置成为 hard constraint；只有线性主体运动启用缺省 20° 斜视规则。
- 摄影机 `static/push_in/pull_out/follow/orbit/lateral` 现在均有实际执行实现；明确俯仰、数值高度和常用焦段会改变真实 Camera 并接受 hard 复验。未识别的显式摄影机语义保持未映射并进入修复，而不再仅凭 `source_ref` 假通过。
- Agent 的 Constraint Patch 枚举收窄为 Toolkit 真正支持的 19 类；屏幕运动不可读保持非阻断 warning，不再错误进入只处理 hard violation 的确定性 Repair 列表。

### Fixed

- Core 0.8.17 / Semantic Parser v0.9 修复 `timeline.events` 时间字段的传输容错与确定性规则互相矛盾：模型为静态场景多生成事件并把范围留空时，Core 会在不增加 Provider 请求的情况下归一化为完整镜头；与已有主体动作绑定的事件会从动作范围确定性恢复。Prompt 同时明确持续静态镜头无需虚构事件，已输出事件不得使用空范围。完整离线回归 331 项通过。
- 修复 `environmental_motion` 因路径前缀过宽而被错误绑定到地面实体的问题。环境特效运动现在明确留给最终视频生成层，白模规划会记录忽略原因。
- 修复 deterministic fallback 把局部互动降级为 hold、丢失抛物线路径、只实现 push-in 摄影机以及忽略 `view_relation_to_motion/match_subject` 的跨层漂移。
- 修复摄影机 `focus_target_id` 与 `view_relation_to_motion` 仅检查来源标签、候选 Patch 后不复验真实机位的问题；相对视角现在按实际主体运动轴求解并做几何 hard 验证。
- Planning Trace 的 Objective 字段补回 `scene_dynamics`；Studio 可据此把无 `planning` 字样的 Core 规划事件稳定归类到规划阶段。
- Core 0.8.16 完整离线回归 329 项通过，并成功构建对应 sdist 与 wheel。

- Toolkit v0.35 移除未指定机位时没有规范来源的 35°/55° 偏航。静态或仅状态变化的场景沿规范场景纵深轴观察；有主体空间运动但 Agent 未选择具体观察关系时，只采用 Validator 冻结的最小可读斜角。情绪量化表正式移除未被下游兑现且会与机位高度、注视点或左右语义冲突的固定俯仰、地平线百分比和主体水平位置；用户 explicit 构图语义仍保留。
- Semantic Translation v0.5 / Rules v0.9 会把只出现在 `spatial_layers` 的明确远景/后景主体归一化为带近景参照的 `far_from` 关系，避免 Semantic 模型没有重复填写 `relationships` 时丢失纵深意图。
- Toolkit v0.34 统一 Semantic、Planning Prompt、Scene Skeleton Schema、Design Options 和 Repair 的运动术语：`scene_dynamics` 只判断主体状态，`subject_spatial_motion` 只表示主体位移，`camera_motion` 独立。无主体位移时不再暴露或采用 `maximize_motion_readability`，未明确观察关系的静态主体场景恢复为普通 35° 斜侧机位。
- Agent 骨架的方向枚举由容易误解成屏幕坐标的 `screen_left_to_right/screen_right_to_left` 改为真实执行语义 `world_right/world_left`；目标、载体、路径族、静止阶段和方向字段的矛盾组合在数值 Design 前直接拒绝。
- 情绪表生成的主要物体画幅比例无论整组 composition 来源如何都保持 inferred；只有字段级 explicit 画幅要求可主动移动摄影机。开放环境代理地面按摄影机轨迹扩展为至少 1000 m 的渲染背景，语义场景范围和主体距离不变。
- 线性主体运动的共线错误更名为 `VIEW_SUBJECT_MOTION_NEAR_COLLINEAR`，避免把“摄影机观察方向与主体运动方向接近共线”误读成摄影机自身运动；提示词、任务相关能力和 Repair 说明使用同一名称与适用条件。
- Toolkit v0.33 把场景语义深度与摄影机方位角解耦：普通 far/background 沿稳定场景纵深轴摆放，摄影机仍可从斜侧观察；镜头角度不再把背景主体无依据地移到参考主体侧后方。
- “远处/背景”不再使用较大主体尺寸归一化净空。Design 沿稳定场景纵深轴布置前后景，以环境代理的方向范围确定场景级绝对净空；主体尺寸只用于把中心位置换算为真实表面边界。静态摄影机根据全部 projected-size 上界整体后移，保持原推镜行程与速度。
- Design Options 先按完整 Validator 的 soft score 排序，再使用请求的策略顺序破同分；构图与实际入框问题也可进入确定性摄影机 Repair 搜索。
- 显式摄影机运动现在先选择匹配的确定性运动模板，并覆盖冲突的情绪缺省数值；旧版带量化快照的 Brief 在 Planning 投影时也会重新对齐。
- Design Options 按实际 Candidate 状态去重；相同 Design 硬失败连续两次后熔断当前完整 thinking 会话，以紧凑 RepairPacket 开始新的恢复轮次。
- 嵌套公转半径按完整子系统包络递归计算，并预留冻结的表面净空。
- Scene Skeleton 的摄影机焦点改为可空；未指定拍摄主体时使用稳定场景锚点完成初始取景。静态焦点不再隐式逐帧跟踪运动实体，持续跟踪必须由 `look_at` Track 明确表达。
- 构图 projected-size Constraint 按 Cinematic Brief `visual_scales[*].subject_id` 和对应百分比建立，不再依赖 Scene Skeleton 的实体插入顺序。
- 明确的“远处/背景”关系现在必须同时具有 `depth_order` 与 `surface_clearance_range` hard 证据。Design Option、确定性布局 Solver 和逐帧 Validator 共用同一方向包围体计算；距离档位相对物体尺寸而非固定米数，Agent 可在受限 Patch 中调整比例范围但不能绕过门禁直接写最终坐标。
- Semantic 输出的 `action_kind` 收敛为 hold/locomotion/interact/other；Planning 只依据明确方向、路径、载体和后置状态建立几何，不再把 arrive/depart/transport 等叙事标签二次加工为目标或方向。
- Scene Planner 必须先联合检查每个实体的完整阶段时间线；未明确转向或反转的线性运动跨静止阶段继承同一路线方向，事件末端会合关系不再把另一个参与者变成运动目标。
- Design Option 只向 Agent 暴露完整 hard-pass 候选，并在应用时再次完整复验。
- 外层恢复轮次改用不含上一轮 thinking 的新 `RepairPacket`；确定性工具不支持的首个 hard error 会立即开放受限原子 Candidate Patch。
- 动作方向验证改为检查施事主体自身的定向位移，目标移动不再能替代“靠近/离开”；等待和停留会写入保持关键点，已隐藏主体不再生成冗余 carried 轨道。
- 靠近、抵达、离开、进入/退出和绕行增加屏幕结果检查；世界动作保持 hard，屏幕呈现默认记录 warning。

### Fixed

- 修复静态前景—远景场景在用户未指定水平机位时仍被固定 35° 偏航拍成侧面，以及“远处”只写入空间层时 Planning 没有获得纵深关系的问题。
- 修复未指定视角时的默认斜侧摄影机被错误复用为世界空间背景方向，导致背景飞船和人物形成无语义依据的斜向排列；飞船自身朝向仍保持独立。
- 修复“远处”净空随单个巨物尺寸同比放大，导致越强调巨大越被推离镜头的问题；同时修复语义层已有 `major_object_frame_ratio` 与 `negative_space_ratio`，Design 却只落实人物画幅比例、允许主要物体全程处于画外的问题。
- 修复明确要求“缓慢推近”时，语义文本保留推近但量化相机仍继承中性情绪的 `static / 0 m/s`，导致 Planning 长时间求解不可能速度约束的问题。
- 修复内层轨道只按直接父子实体尺寸定半径、外层轨道没有容纳整个子系统，因而月亮可能穿过太阳的问题；解析公转新增实体相交 hard 检查。
- 修复无摄影机指令的多主体动态场景仍被迫绑定首个/任意实体，以及 `static` 镜头位置不动却逐帧旋转追踪该实体的问题。
- 修复语义层已把主体画幅比例绑定到特定 ID，Design 却把该约束错误施加给 Skeleton 第一个非地面实体的问题。
- 修复巨型代理仅以中心深度满足“远处”，导致中心相距十余米但人物距飞船边缘不足一米仍被 Commit Gate 接受的问题；旋转后的长轴也会进入方向相关净空计算。
- 修复 Semantic 模型把“同时结束”误写成 `meets`，并以 inferred `maximum_gap_seconds=0` 阻止关系规范化的问题；推断零间隔现在随错误标签一起清除并按实际区间改为 `ends_with` 等精确关系，用户明确数值间隔继续失败关闭。
- 修复接载叙事中车辆被错误要求朝乘客移动、随后又远离同一乘客的问题；乘员进入可由隐藏后置状态和紧接的 carried 关系满足，不再强制人物代理移动到载体中心。
- 确定性后备规划器不再根据 boarding/departure 等事件名制造方向；旧版接车 Brief 仍可在无专用动作种类的情况下生成 hard-pass Design Option。
- Semantic 时间关系允许独立主体动作以“开始先后但区间重叠”的方式表达；无明确数值间隔时，会把与事件区间冲突的关系标签确定性规范化为 `starts_before/starts_after/starts_with/ends_with/during/overlaps`，不再因可修复的标签歧义让整个 Pipeline 在 Planning 前失败。明确的最小/最大时间间隔仍失败关闭。

- Entity 更新改为真正的 merge-patch：省略字段保持现有值，显式空值才执行字段契约中的清除语义。
- 正常 Agent 在确定性修复无解后只看到一个组合 Candidate Patch；四个分散低层 Patch 保留给兼容和诊断，不再共同占用正常工具表。坐标、时间、来源与运动语义等一次规划所需 Prompt 信息暂不删减，等待冻结 A/B 后再决定。
- `suggest_repairs` 在当前 revision 确定性搜索穷尽后重新开放受控手工 Mutation，仍不允许绕过 Schema、Validator 或 Commit Gate。

### Verification

- Python 3.12 完整离线回归 317 项通过；新增回归覆盖静态纵深对齐、远景层确定性关系归一化和退役字段不再进入量化快照。
- Python 3.12 完整离线回归 309 项通过；新增回归覆盖斜侧镜头下背景主体仍沿场景正后方摆放，且实体朝向不受摄影机影响。
- `setup.py sdist bdist_wheel` 成功生成 Core 0.8.15 的 sdist 与 wheel；隔离式 `uv build` 因当前沙箱不能访问 PyPI 的 setuptools 构建依赖而未作为本轮验证入口。
- Python 3.12 完整离线回归 308 项通过；新增回归证明放大背景代理不会同比扩大表面净空，荒漠静态构图同时满足人物 1%–5%、主要物体 10%–30%、实际入框与至少 70% 负空间。
- `uv build` 成功生成 Core 0.8.14 的 sdist 与 wheel。
- Python 3.12 完整离线回归 307 项通过；荒漠飞船、道路接车、嵌套公转三条标准场景均直接 hard pass 并通过 Commit Gate，未调用 Provider。
- `uv build` 成功生成 Core 0.8.13 的 sdist 与 wheel。
- Python 3.12 完整离线回归 301 项通过；新增回归覆盖 Scene Skeleton 空焦点、静态实体焦点不隐式跟踪、固定场景锚点编译和 visual-scale 主体 ID 映射。
- Python 3.12 完整离线回归 297 项通过；新增回归覆盖超大自定义代理、旋转长方体、far explicit 双约束完整性、比例范围校验和统一 Solver/Validator 行为。
- Python 3.12 完整离线回归 290 项通过；旧版道路接车 Brief 的 Mock 端到端规划、inferred zero-gap 时间关系和已取消真实失败结构的离线 Design 重放均成功。
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
