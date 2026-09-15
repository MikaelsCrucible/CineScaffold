# CineScaffold

CineScaffold 是一个面向论文研究的自然语言到三维白模视频生成管线。它把非专业用户的描述转换为结构化电影语义，再由工具增强的场景规划 Agent 生成可验证的 Scene IR，最后通过 Blender 构建空间脚手架和摄影机预演。

> 当前状态：研究原型。核心管线和首版四要素到六维转换规则已经可以运行，但目标视频模型适配和正式实验协议尚未冻结，不建议作为生产工具使用。

## 研究动机

纯文本视频生成经常难以稳定复现相对位置、视觉尺度、运动轨迹和摄影机运动。CineScaffold 研究一种中间控制方式：先用可验证的三维代理场景明确这些关系，再把白模视频交给后续生成模型。

项目当前关注的问题是：

> 相比直接提示词或仅由 LLM 扩写提示词，加入六维电影语义、工具增强的场景规划和 Blender 白模控制，能否提高最终视频对空间与运动要求的忠实度？

## 工作流程

```text
自然语言
  -> Semantic Parser
  -> 文本六维（人类可读投影）
  -> Cinematic Brief（六维电影语义）
  -> Scene Planning Agent
  -> Scene Skeleton（无数值符号关系与稀疏路径点）
  -> Planning Toolkit Design Options
  -> Candidate + Validator + Commit Gate
  -> Constraint Plan + Scene IR
  -> 确定性 ExecutionRunner
  -> Background Blender Executor（MCP 可选）
  -> .blend 场景与白模视频
  -> 视频生成模型（后续阶段）
```

四个核心表示各自承担不同职责：

- `Cinematic Brief`：记录用户想表达什么，以及信息来自明确描述、推断还是默认值。
- `文本六维`：用六个固定中文标题呈现同一电影语义，方便非技术协作者阅读、修改和交流；严格 JSON 仍是机器权威表示。
- `Scene Skeleton`：记录实体类别、空间关系、动作阶段、稀疏符号路径点、摄影机意图以及定性的尺度/三轴比例，不含坐标、距离、速度、米制尺寸或焦距。
- `Constraint Plan`：记录 Agent 选择了哪些可执行空间、运动和摄影机策略。
- `Scene IR`：精确描述 Blender 应创建和渲染什么，可验证、可重放、可比较。

## 当前能力

- 支持 OpenAI、DeepSeek 和离线 Mock Provider。
- 使用 LLM 提取“谁、在哪、做什么、感觉”，再通过版本化规则表生成可复现的主体、运动、场景、摄影机、构图和光源量化快照。
- 将 Cinematic Brief 中的客观空间、运动、构图和摄影机要求交给规划 Agent。
- 未指定水平机位不再触发固定 35°/55° 偏航：静态或仅状态变化场景沿规范纵深轴观察，有线性场景实体运动时使用 Agent 的类型化观察选择，未选择则只采用 Validator 冻结的最小可读斜角。`spatial_layers` 只保留原文画面分层，不再自行选择参照实体并制造 `far_from`；双实体距离必须由独立类型化关系明确表达。旧情绪表中的固定俯仰、地平线百分比和场景实体水平位置已退出自动量化；用户明确俯仰会转换成实际机位并硬验证，明确水平画面位置会转换成 `screen_region` hard constraint。地平线百分比因没有独立于机位、焦点和场景几何的稳定语义，正式不作为自动或显式数值接口。
- 规划 Agent 先做符号化拆解，包括相对大小、flat/wide/tall 等形体比例，以及必要时跨完整主体时间线选择的最少路径点；路径点把非闭合运动绑定到已有、带时间事件的空间关系，不重复填写动作边界、数值时间或坐标。一个 Semantic 动作可以引用多个事件路径点，Toolkit 会把它们物化成同一连续 Track 内的多个关键帧，而不是伪造新的语义动作；同一 proximity 事件的多个运动参与者会作为一组联合求出接近但不碰撞的位置，结果不依赖实体构建顺序。普通直线运动按 Semantic Translation 冻结的速度范围和实际阶段时长生成位移，并以线性插值匀速通过非停顿路径点；只有明确的静止、速度变化、转向或更高优先级时空条件才改变运动状态。路线仍可选引用场景中已有道路、轨道或平台的主轴。Toolkit 再联合冻结 Profile 与完整 Validator 求解坐标并给出少量数值候选、可行范围和任务相关接口。内置比例不足时，Agent 可向建议接口提交受约束的三轴尺寸范围和理由，候选仍由 Toolkit 原子物化，避免绕过门禁直接改 IR。明确的“远处/背景”关系同时生成摄影机深度与绝对表面净空 hard 约束；净空按环境代理所表达的场景参考范围冻结，主体尺寸只用于求出真实表面边界，不再把“远”随某个巨物自身同比放大。只有用户明确给出的画幅比例能主动重排摄影机；情绪映射得到的 inferred/default 比例仍保留为 soft 评分，不再单独把巨物推到不可见。开放环境地面可独立扩展为至少 1000 m 的渲染背景以覆盖摄影机视野，但不改变场景参考尺度或主体间距。未指定拍摄主体时 Scene Skeleton 可保留空焦点，由 Toolkit 选择固定场景锚点；静态摄影机不会把初始取景主体变成逐帧跟踪目标。
- 正常规划只向 Agent 暴露符号骨架与已验证数值候选；只有确定性候选和修复都不足时，才开放单个受限组合 Patch。该 Patch 可在一个预演事务中联合调整实体尺寸/完整米制 Transform、运动、约束和摄影机，新增参照物不会再因隐藏坐标而成为不可提交的 unresolved 实体。实体 `look_at` 以本地 `+Y` 为前向轴真实编译，摄影机的初始焦点只决定初始取景；持续跟踪必须由独立 `look_at` Track 表达，互相注视只读取目标位置而不会递归求解朝向。所有布尔显隐变化按关键时间点 step 执行；短暂出画仍是可压缩的逐帧 warning，但任何应渲染实体在其可见窗口内一次都无法投影会阻止交付。
- Cinematic Brief v0.7 先区分主体静态/动态场景；动态场景按实体记录稀疏、独立且带数值起止时间的动作区间。事件可以位于镜头内任意数值区间，九种类型化时间关系表达相接、先后、包含、同始/同终和重叠；空间关系的 `throughout/at_start/at_end` 只是“所引用事件内部”的生效范围，不是全局仅有三个时间点。全流程明确区分“主体状态发生变化”“主体发生空间位移”和“摄影机发生运动”：运镜不能把主体静态场景变成动态，也不能触发主体运动可读性策略；只有 `path_move/carried` 才属于主体空间运动。所有独立位移统一为 `path_move`；公转只由 `relative_to_target + target_id + circle/ellipse` 表达，载运只由 `carried + carrier_id` 表达，普通人物/车辆贴地由 Planning 根据地面、代理类型和类型化动作确定性生成，三者都不再要求 Semantic 重复提交关系。局部互动必须物化为可观察的旋转/尺度变化，不能用静止轨道假装完成；抛物线路径会物化为带弧顶的关键点。稳定 `motion_id` 必须绑定对应的类型化动作来源，不能通过改写 `source_ref` 绕过验证。
- 当前版本只接受 Cinematic Brief v0.7、Constraint Plan v0.2 与 Scene IR v0.2。旧版中间产物不会被猜测、补字段或迁移；需要重新生成。已经交付的 MP4/GLB 仍是普通成品文件，不受这条版本边界影响。
- Semantic 每次解析固定执行两次 Provider 请求：第一轮生成初稿，第二轮依据原文、完整初稿、确定性诊断和版本化 `ReviewPacket` 做独立审查，并返回完整替换 JSON。Core 0.8.33 会根据稳定错误码/字段路径、初稿结构以及原文与初稿中的概念提示召回最多五条重点规则；`confirmed_error` 与 `review_risk` 严格区分，提示词只负责选择审查问题，绝不直接生成动作、方向、目标、关系或时间答案。完整契约始终有效，实际召回规则、证据、目录版本与 SHA-256 写入 Brief provenance。这是有上限的一次修正，不是网络重试，也不增加第三次模型调用；第一轮若在取得可用 JSON 前发生超时或 Provider 错误，不会盲目重复请求。第二轮仍不满足 Schema、类型字段一致性、引用或时间区间约束时直接失败。两次请求分别报告进度和费用，该流程与可选 Provider thinking mode 相互独立。
- 复合叙事动作不会扩张为场景专用实体、专用执行原语或隐含几何方向。当前输出只使用 hold/locomotion/interact，Schema 中的 `other` 保留值不作为绕过分类的出口；动作文字本身不会自动生成相对于事件参与者的方向或目标。规划器只读取类型字段来生成几何，原始动作文字仅用于审计上游遗漏。每个场景实体拥有独立数值时间区间；时间关系必须与区间一致，任何矛盾都会进入第二轮审查，最终仍矛盾就失败，不由代码静默改名。
- 通过类型化 Toolkit、候选 revision、Solver、Validator 和 Commit Gate 生成 Scene IR。
- Validator 按冻结时间线逐帧复验地面、投影、点式空间约束和类型化运动语义；摄影机也必须在整段轨迹中保持在水平环境地面的安全侧，连续穿地只形成一条时间段错误。静态构图后移在用户未明确俯仰时保持绝对机位高度，不能因高大背景主体抬升场景焦点而把镜头拉入地下。明确要求的来源不仅要映射到正确字段，还必须绑定 Brief 指定的实体。Commit Gate 会再核对编译后逐帧 Scene IR 与已验证 Candidate 的位置、旋转、尺度、显隐、摄影机和焦距等价。
- 用户明确要求“可见”时，白模阶段以 hard `keep_in_frame` 证明代理几何至少部分进入画面；这不等价于已验证真实遮挡、材质透明或最终生成视频的可见性。
- Design Option 在交给 Agent 前必须通过完整 Validator 的全部 hard 约束以及 Agent 自己声明的全部符号路径点，应用时再次复验；已知错误候选绝不混入正常 `options`。如果首次求解没有合法 Option，Toolkit 只保留一个已通过 Execution Safety 的最佳失败候选，并以明确不可交付的 `repair_baseline` 暴露；Agent 物化后立即进入受限原子 Patch。数值状态完全相同的策略候选会折叠为一个。
- Validator 对靠近、抵达、离开、进入/退出和绕行动作同时记录世界空间结果与屏幕质心、方向性尺度、出入画或显隐结果。动作主体的世界位移和状态改变是 hard，目标自身移动不能替代主体动作；屏幕表现默认作为 warning，不会为了可读性篡改真实动作。
- 默认规划的外层恢复轮次不再回放上一轮完整 thinking；每轮获得新的紧凑 `RepairPacket`，只含冻结目标时间线、关键动作、当前 Candidate 摘要、压缩 violation、能力缺口和剩余尝试。同一 revision 的同类确定性工具失败连续出现两次时，不再开启新的付费模型轮次：Runner 会先用确定性 Skeleton→Design 路径区分 Agent 原地打转与框架契约冲突，再直接进入产品后备交付或诊断失败。第一次出现确定性工具不支持的 hard violation 就立即开放受 Schema 和 Validator 限制的原子 Candidate Patch。
- 逐帧 Validator 仍在 Candidate、checkpoint 和诊断产物中保留完整证据；发给 Agent 的工具返回与 RepairPacket 会按错误签名和相邻帧合并为连续时间段，只携带首个、最严重和最后样本，且去除 Envelope 中重复的 validation 列表，避免同一连续错误耗尽上下文。
- Toolkit 采用渐进披露而不是删掉解题信息：Design Option 在选择前给出任务相关能力；遇到确定性工具不支持的 hard error 时，Agent 获得一个可跨实体、约束、运动和摄影机联合修改的原子 Candidate Patch。建议修复领域只负责排序注意力，不限制可用字段；完整压缩错误仍保留，提案还会先在不可见副本中验证，退化时不写入 revision。
- Agent 的 Entity Patch 不再清空其 Schema 中不可见的已求解 Transform；Design 生成的 `ground_plane` 自动携带语言无关的环境地面身份。Design Option、全部 Mutation、历史恢复和 Validator 现在共用同一组引用、地面、时间、轨道唯一性与参考系不变量，避免一个入口生成、另一个入口拒绝同一状态。
- 工具参数不再“接受后忽略”：`inspect_candidate` 会实际执行适用的实体、摄影机、约束和时间过滤，并拒绝与视图不相容的过滤器；Agent 只看到当前真正实现的 layout/camera 启发式 Solver 选项。轨道会拒绝执行器不用的变体字段、重复关键帧时间和落在自身时间段外的关键帧。
- 完整修复用尽后，规划器优先保留并提交同时通过执行安全与叙事完整性门禁的最佳 Candidate；若还没有合格 Candidate，则从已类型 Objective Brief 确定性生成简化方案，不再请求 Provider。简化交付可以放宽构图和屏幕表现，但仍必须保留全部关键动作、显隐/容纳后置状态和 explicit 要求。
- 支持世界、局部、目标相对和摄影机相对参考系。
- 支持直线、抛物线、圆、椭圆、平滑样条和 8 字等代理运动轨迹；摄影机 static、push-in、pull-out、follow、orbit、lateral、pan 均有实际执行轨迹。`pan` 在指定时间前保持初始机位与朝向，随后保持位置不动并平滑旋转到运动目标；`follow` 才使用目标相对轨迹随主体平移。当前 Cinematic Brief 表达一个主动运镜区间，因此支持“先固定、后执行一次运镜”，尚不表达任意数量的异类运镜段落。
- 嵌套公转按完整子系统包络计算外层轨道半径；Validator 会逐帧检查公转实体与所有非地面代理的相交，并把连续帧汇总为实体对级错误。
- Agent 可见接口固定右手 Z-up、路径方向和屏幕坐标约定；推断机位下的解析轨道需通过投影可读性门禁。
- 每个量化为移动的主体阶段都会经过通用投影结果检查；仅有世界坐标位移、但屏幕轨迹和尺度变化都过小时记录 warning，供后续质量优化和评测使用。
- 未明确要求迎面或背面跟拍时，线性主体运动与摄影机视线必须保留至少 20° 的中位斜视夹角，避免 Agent 仅靠迎面尺寸变化通过运动可读性门禁；明确机位保留用户要求并记录 warning。
- 默认通过无窗口 Blender CLI 调用固定 Executor；官方 Blender Lab MCP 仅作为显式兼容后端。两种路径都不把任意 Blender Python 暴露给模型。
- 输出 `.blend`、运行时验证、GLB、执行 manifest，并可选生成 H.264 白模视频；一键 Pipeline 用稳定逻辑 ID `scene_blend` 把已验证的 `.blend` 交给宿主，由宿主决定受控下载策略。
- CLI 实时显示 Agent 请求、工具调用、token、revision、验证和渲染进度。
- 可从自然语言、文本六维、Cinematic Brief 或 Scene IR 一键运行到白模视频。
- CLI 的一键命令由独立异步 Workflow 服务编排；后续 UI 复用同一阶段、事件、失败语义和 `pipeline_summary.json`，不会维护第二套生成逻辑。

## 研究范围

当前原型主要面向单镜头、有限刚性代理实体和基础摄影机运动。系统没有写死主体数量，也不会仅因“运镜复杂”而拒绝请求；实际能力边界由 Scene IR、Toolkit、Validator 和运行资源共同决定。

暂未覆盖：

- 多轮追问和交互式修改；
- 多镜头、转场和长视频编排；
- 骨骼级人物表演、表情与口型；
- 布料、流体、破坏和动态拓扑；
- 精细资产、材质与完整灯光设计；
- 最终视频生成模型的统一 Adapter 与正式评测管线。

## 环境要求

- Python `3.12.x`
- Blender `5.2.1 LTS`
- 可选：官方 Blender Lab MCP `1.0.0`（仅使用 `mcp` 构建或渲染后端时需要）
- macOS 原型环境使用 `uv` 管理 Python 与工具依赖

## 安装

```bash
uv python install 3.12
uv venv --python 3.12 --clear .venv
uv pip install --python .venv/bin/python \
  --require-hashes --requirements requirements.lock
uv pip install --python .venv/bin/python --no-deps --editable .
source .venv/bin/activate
```

需要浏览器 UI 时，用包含核心依赖与 NiceGUI 的锁文件替代 `requirements.lock`：

```bash
uv pip install --python .venv/bin/python \
  --require-hashes --requirements requirements-ui.lock
uv pip install --python .venv/bin/python --no-deps --editable .
```

UI 是可选依赖；仅使用 CLI 时不需要安装它。未采用锁文件的开发环境也可用 `pip install -e ".[ui]"`，正式测试仍建议使用锁文件。

### 作为 Python 核心库安装

CineScaffold 的运行时 Prompt 与 JSON Schema 已随 wheel 安装，不要求调用方位于源码仓库，也不依赖当前工作目录。另一个仓库在本地联调时可以使用可编辑安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ../CineScaffold
python -c "import cinescaffold; print(cinescaffold.__version__)"
```

生产环境应依赖正式 tag 或完整 commit，而不是 `main`：

```text
cinescaffold @ git+https://github.com/MikaelsCrucible/CineScaffold.git@v0.8.19
```

嵌入式调用只依赖 [`cinescaffold.api`](src/cinescaffold/api.py) 的公共入口；`planning`、`execution` 等子模块属于内部实现：

```python
import asyncio
import json
from pathlib import Path

from cinescaffold.api import (
    PipelineSource,
    build_pipeline_run_config,
    load_config,
    run_pipeline,
)


async def main() -> None:
    scene_ir = json.loads(Path("scene_ir.json").read_text(encoding="utf-8"))
    config = build_pipeline_run_config(
        load_config(),
        output_dir=Path("output"),
        include_semantic=False,
        overwrite=False,
    )
    result = await run_pipeline(
        PipelineSource(kind="scene_ir", payload=scene_ir),
        config,
    )
    print(result["status"])


asyncio.run(main())
```

需要替换 Prompt 或 Schema 的研究工具仍可显式构造 `RuntimeResourcePaths`；缺省始终使用 wheel 内冻结资源。

### Windows x64 发行包

面向非技术协作者的 Windows 包由 GitHub Actions 在真实 `windows-latest` 环境构建。下载并完整解压 `CineScaffold-Windows-x64-v*.zip` 后：

1. 双击 `Install-CineScaffold.cmd`，自动建立包内隔离的 Python 3.12 环境并安装 Windows 锁定依赖。
2. 按包内 `README-Windows.md` 安装 Blender 5.2.1 LTS。
3. 双击 `Start-CineScaffold.cmd` 启动 Studio。

该包不要求用户预装 Python，也不修改系统 Python；首次安装依赖联网。Blender 因体积和许可证边界不随 CineScaffold 打包。Windows 专用锁保持与核心/UI 锁相同版本，只用 `pywin32` 相关包替代非 Windows 的 `uvloop`。

默认链路不需要 MCP。只有显式配置 `execution_build_backend = mcp` 或 `execution_render_backend = mcp` 时，才需要把 Blender MCP 作为外部工具单独安装：

```bash
uv tool install --python 3.12 \
  --with "mcp[cli]>=1.2,<2" \
  --from "git+https://projects.blender.org/lab/blender_mcp.git@4309a39646e644261624bfcd2bca669b343b7621#subdirectory=mcp" \
  blender-mcp
```

该兼容模式还需要在 Blender 中安装并启用官方 MCP Add-on。上游安装说明见 [Blender Lab MCP](https://www.blender.org/lab/mcp-server/)。

## 配置文件

复制示例并用任意文本编辑器填写：

```bash
cp .cinescaffold.example.conf .cinescaffold.conf
```

最小配置只有三行：

```text
provider = deepseek
model = 填写模型 ID
api_key = 填写 API KEY
```

`.cinescaffold.conf` 和 `cinescaffold.local.conf` 已加入 Git Ignore，不会被正常提交。自定义其他文件名时需要自行确认其未被 Git 跟踪。

配置也可以分别指定 `semantic_provider/model` 和 `planning_provider/model`。完整可选字段及注释见 [`.cinescaffold.example.conf`](.cinescaffold.example.conf)。命令行参数优先于配置文件；未找到配置值时，API Key 仍可从 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY` 环境变量读取。

配置还可以保存 Agent 预算、分阶段成本单价、Blender 路径、构建/渲染后端与超时。构建和渲染默认都是 `background`；`mcp` 是显式兼容选项。macOS、Windows 与 Linux 会分别寻找常见 Blender 和可选 `blender-mcp` 可执行文件；显式路径始终优先。UI 设置中心与 CLI 共用同一配置校验器，保存时不会把已配置 API Key 回传到浏览器，并在覆盖前生成本地 `.bak`。

OpenAI Semantic Parser 使用 Responses API。`semantic_reasoning_effort` 会作为 `reasoning.effort` 实际发送；GPT-5.6 可使用 `none/low/medium/high/xhigh/max`。例如：

```text
semantic_provider = openai
semantic_model = gpt-5.6-luna
semantic_reasoning_effort = medium
semantic_max_tokens = 65536
```

DeepSeek V4 Pro 的思考模式缺省为启用，缺省强度为 `high`。早期回归中，模型读取完整 Toolkit 能力后的首次规划曾产生上万 reasoning tokens，并让单次请求持续数分钟。当前正常流程不再先返回整本能力手册，而是依次暴露 Scene Skeleton、Design Options 和必要的后备工具；模型仍可能因任务与服务负载产生长推理。若实验优先考虑低延迟与稳定工具调用，可在配置中明确填写：

```text
planning_thinking_mode = disabled
planning_model_max_tokens = 8192
```

上述设置只控制复杂的 Scene Planning Agent。自然语言 Semantic Parser 是单轮 JSON 提取任务，DeepSeek V4 的思考模式在此阶段缺省关闭，防止 reasoning 占满 JSON 输出预算；这不会改变 Agent 1 的 thinking 设置。需要专门做语义推理消融时可以单独配置：

```text
semantic_thinking_mode = enabled
semantic_reasoning_effort = low
semantic_max_tokens = 65536
```

也可使用 `--semantic-thinking-mode`、`--semantic-reasoning-effort` 和 `--max-tokens` 临时覆盖。只提供“10 秒以内”等时长上限时，Semantic Parser 保留原范围，并确定性选择上限作为本次时长；完全未提供时长才使用 15 秒缺省值。

论文实验应明确冻结思考模式、强度和 token 上限，不能把供应商缺省值视为不变条件。普通规划 Trace 记录 reasoning token 数、请求耗时、停止原因与工具调用，不保存对话/思维原文；`--full-power-diagnostic` 模式额外记录完整对话与逐字思维链（受 `TraceConfig` 字节/字符上限约束，可用 `--trace-max-event-bytes`/`--trace-max-string-chars` 调大）。

若要启用低强度思考，开关和强度必须分别填写：

```text
planning_thinking_mode = enabled
planning_reasoning_effort = low
```

`planning_thinking_mode = low` 是无效配置；thinking 开关只接受 `enabled/disabled`。

DeepSeek 在 thinking 与 tools 同时启用时要求后续请求完整回传历史 `reasoning_content`。CineScaffold 依赖 PydanticAI 的 DeepSeek Profile 完成字段映射，并在这种模式下保留完整工具思考历史，不再应用 13-message 压缩窗口；普通工具前置正文仍会省略。对于 PydanticAI 2.36 尚未内置识别的官方别名 `deepseek-flash`，CineScaffold 0.8.1 还会显式声明 thinking 支持并禁用强制 `tool_choice=required`，让工具选择按 DeepSeek thinking 协议使用 `auto`。该协议可能明显增加后续请求上下文，正式实验需要把思考模式和上下文预算一起冻结。

常规规划默认使用面向长思考工具循环的研究预算：48 次模型请求、80 次工具调用、单请求 128K 输入上下文、整轮累计 200K 输出 token、20 分钟墙钟时间和 5 次外层规划尝试。外层尝试同时覆盖 Commit Gate 拒绝、Agent 过早声明不可行/不支持，以及可恢复的供应商返回异常。累计输入和输入输出总量默认不另设上限，但仍受上述单项门禁与供应商限制保护。每项均可通过 `--max-requests`、`--max-tool-calls`、`--max-context-tokens`、`--max-output-tokens`、`--max-seconds` 和 `--max-commit-attempts` 临时覆盖；论文批量实验应显式记录或冻结这些值。

嵌入式宿主可以在 `PipelineRunConfig` 设置 `max_provider_cost`、`provider_cost_currency`，并为 Semantic 与 Planning 提供同币种的冻结 `CostRates`。Core 在每次 Provider 响应返回 usage 后发出 `provider_cost_incurred`，累计达到或超过上限后在下一次请求前停止；所以这是事后门禁，最后一个已经发生的请求可能越线。模型响应前收到明确的非计费 HTTP 请求拒绝时，同一事件以零金额和 `billing_resolution=confirmed_not_billed` 关闭预留；超时、网络中断和服务端错误仍保持未知。价格快照、跨任务/跨进程的日额度、预留和未知账单结算由宿主负责，Core 不把估算成本冒充 Provider 账单。

## 使用

### 浏览器 UI

安装 `requirements-ui.lock` 后，在项目根目录运行：

```bash
cinescaffold ui
```

界面默认只监听 `127.0.0.1:8080`，并自动打开浏览器。它支持从自然语言、文本六维、Cinematic Brief JSON 或 Scene IR JSON 开始；JSON/文本既可粘贴，也可从本地文件导入。文本六维入口把六个标签固定在界面中，用户只编辑各项正文；导入 `.txt` 时会先校验标题再分别填入，避免误删标签。运行中会显示语义解析、规划 Agent、Blender 构建与渲染进度，结束后展示文本六维、JSON 六维、token、分阶段估算成本、视频和本地产物入口。

设置面板可以临时应用或安全写回 `.cinescaffold.conf`。现有 API Key 不会被读回页面；密码框留空表示保留，填写新值才会替换。若语义模型与规划模型不同，应分别填写 `semantic_*_cost_per_million` 与 `planning_*_cost_per_million`，避免成本串用。

Windows 使用同一套 UI 和 Workflow，无需重写前端；激活命令改为：

```powershell
.venv\Scripts\Activate.ps1
cinescaffold ui
```

Windows 仍需单独安装 Blender，并在设置面板确认可执行文件路径；默认不需要 Blender MCP。程序会使用 Windows 的 `explorer` 打开输出目录；macOS 使用 `open`，Linux 使用 `xdg-open`。如不希望自动打开浏览器，可使用 `cinescaffold ui --no-open`。

### 一键运行

先在项目根目录激活虚拟环境。终端提示符通常会出现 `(.venv)`：

```bash
source .venv/bin/activate
cinescaffold --version
```

如果不想激活，也可以把下列命令中的 `cinescaffold` 替换为 `.venv/bin/cinescaffold`。

`run` 缺省会自动读取当前目录的 `.cinescaffold.conf`：

```bash
# 从自然语言开始
cinescaffold run \
  --text "一个男人站在荒漠里，远处有巨大的飞船。" \
  --output-dir runs/example/full

# 从六个固定部分的人类可读文本开始
cinescaffold run \
  --text-six-file examples/textual_six/example.txt \
  --output-dir runs/example/from-textual-six

# 从 Cinematic Brief 开始
cinescaffold run \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/full

# 从 Scene IR 开始
cinescaffold run \
  --ir runs/example/final_scene_ir.json \
  --output-dir runs/example/execution-only
```

使用其他配置文件时传入 `--config cinescaffold.local.conf`。一键运行会写出 `pipeline_summary.json`，记录语义解析与规划阶段各自的 usage、可选价格快照和估算成本，并在任一阶段失败时停止，不会把不合法 IR 继续交给 Blender。

### 分阶段运行

`parse`、`plan`、`execute` 仍然保留，方便研究者检查、更换或复用中间产物。

#### 1. 自然语言转 Cinematic Brief

Provider、模型和 API Key 直接复用 `.cinescaffold.conf`：

```bash
cinescaffold parse \
  --text "一个男人站在荒漠里，远处有巨大的飞船，镜头慢慢推近。" \
  --output runs/example/cinematic_brief.json
```

Mock Provider 不理解文本，只返回指定的模拟响应。未传入 `--mock-response` 时，它仅用于检查 Schema 和 CLI 接口。

语义说明位于 [`resources/prompts/semantic_parser/rules.md`](src/cinescaffold/resources/prompts/semantic_parser/rules.md)，固定数值位于 [`resources/prompts/semantic_parser/translation_rules.json`](src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json)。新解析结果使用 Cinematic Brief v0.5：语义模型直接输出类型化的动作、运动模式、目标、载体、路径、动作后置状态，以及摄影机相对主体运动的 `front/rear/side/three_quarter/unspecified` 关系。Provenance 额外记录输入来自自然语言还是文本六维，并用 SHA-256 绑定同步保存的人类可读文本。确定性代码只校验这些字段并查表量化，不再用关键词子串重新猜测动作或机位含义。旧 v0.1–v0.4 Brief 仍可直接进入规划。用户明确要求始终优先于规则推导和缺省值；径向推近/后拉的速度由起止距离和实际时长计算。光源量化只留给后续视频生成阶段，Blender 白模仍使用中性、无阴影的技术照明。

规划门禁不仅检查实体是否在三维世界中移动，还检查每个移动阶段在摄影机投影中的轨迹范围和尺度变化。默认推断机位若让运动在白模里近似静止，Agent 必须调整运动方向、机位或距离；用户明确指定摄影机设计时保留其要求，并把同项降为 warning。

#### 2. Cinematic Brief 转 Scene IR

```bash
cinescaffold plan \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/planning
```

规划阶段同样读取 `.cinescaffold.conf`。如需临时实验覆盖，可以额外传入 `--provider` 和 `--model`，但不会修改配置文件。

正常规划的前三个工具阶段固定为：提交无数值 Scene Skeleton、请求经 Validator 预测的 Design Options、按 `option_id` 原子应用一个候选。只有候选仍有结构化 violation 或 capability gap 时，低层 Patch、Solver 与修复建议工具才会按状态开放；从 checkpoint 恢复时则直接从已有 Candidate 继续。

`planning_summary.json` 用 `delivery_tier=standard|recovered|simplified` 区分首次完整通过、外层恢复后完整通过和叙事保真简化交付。RepairPacket 和简化交付遗留的非阻断质量问题保存在 Summary/Trace 中。研究用 `--full-power-diagnostic` 保留原始严格观测语义，不自动转为简化交付。常规产品运行若达到请求、工具调用、输入、上下文或输出 Token 上限，会停止继续调用 Provider，并尝试用已有 Candidate 或 Objective Brief 的确定性方案通过 Execution Safety 与 Narrative Fidelity Gate；成功时仍交付 `simplified`。墙钟超时和 Provider 金额上限保持失败关闭，不在预算事件后扩大运行范围。

定位供应商长推理或流式停顿时，可以显式开启一次性诊断特例：

```bash
cinescaffold plan \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/diagnostics/example-max \
  --full-power-diagnostic
```

该选项强制启用 `max` 推理强度和流式遥测，并关闭 CineScaffold 与 PydanticAI 的时间、请求、工具调用、token、提交尝试及 HTTP 请求超时上限。供应商自身的上下文、输出、速率和服务可用性限制仍然存在。Trace 会记录推理 chunk 数、字符数、时序、停顿、usage 和工具调用，并把流式 `reasoning_content` 与正文原文随遥测事件写入 Trace（受 `TraceConfig` 上限约束）。该模式可能无限运行并产生不可预估费用，只用于人工监督的故障诊断，不应混入正式论文样本。

#### 3. Scene IR 转 Blender 白模视频

```bash
cinescaffold execute \
  --scene-ir runs/example/planning/final_scene_ir.json \
  --output-dir runs/example/execution
```

默认 `preview` 使用 Workbench 快速生成诊断视频。需要完整分辨率和逐帧正式控制输出时使用：

```bash
cinescaffold execute \
  --scene-ir runs/example/planning/final_scene_ir.json \
  --output-dir runs/example/control-execution \
  --render-profile control
```

当构建与渲染都使用默认 `background` 后端时，执行阶段默认只启动一个无窗口 Blender 进程：从 factory startup 读取规范化 Scene IR，构建场景、运行 Runtime Validation、立即保存 `scene.blend`，再在同一进程中渲染视频。这样同时省去第二次 Blender 启动和 `.blend` 重载，以及 MCP Server、Add-on、工具发现和中间 template 文件。父进程会等待完整构建结果落盘，再立刻切换 UI/CLI 到渲染阶段；构建与渲染分别受各自超时约束，任一阶段超时都会终止 Blender 子进程。构建结果先于渲染落盘，因此渲染失败或超时仍可保留已验证的场景并被正确归类。

嵌入式宿主可在公共 `ExecutionConfig` 中设置 `render_video=False`，或通过配置项 `execution_render_video = 0`，只完成场景构建、Runtime Validation、`.blend`、GLB 和 Viewer Manifest。此模式不会启动逐帧视频渲染，并强制使用单个 Build-only Blender 进程；执行 manifest 的 `render_video` 为 `false`、`render` 为 `null`，成功状态表示构建与 Runtime Validation 成功。GLB 是否足以构成交付仍由宿主自己的产品合同决定。

Runtime Validation 通过后，固定 Executor 还会在视频渲染前最佳努力生成 `scene.glb` 与 `viewer_manifest.json`。GLB 由 Blender 官方 glTF 2.0 导出器生成，保留代理网格、简单 PBR 材质、摄影机、TRS 动画和 `cinescaffold_id` Extras；初版不启用 Draco 或 Meshopt。Manifest 用 Scene IR canonical hash 与 GLB 文件 SHA-256 绑定二者，并记录时间轴、Entity/Camera 节点映射、显隐区间、能力边界和 MP4 降级策略。显隐及动态焦距仍以 Scene IR 为准，不假装 GLB 能完整表达。GLB 导出或结构检查失败只会把 `ExecutionResult.viewer` 标记为 `unavailable`，不会阻止已验证场景继续渲染视频。

需要隔离日志或定位构建/渲染问题时可传入 `--process-mode split`，恢复两个后台 Blender 进程。只要构建或渲染任一阶段选择 `mcp`，系统也会自动使用 Split；兼容性测试可分别传入 `--build-backend mcp`、`--render-backend mcp`。对应配置项为 `execution_process_mode = fused|split`。

## CLI 输出

默认模式面向人工演示：进度写入 `stderr`，最终摘要写入 `stdout`。所有阶段都支持：

- `--no-color`：关闭终端颜色；
- `--quiet`：隐藏阶段进度；
- `--json --quiet`：只输出机器可读 JSON，适合测试和批处理。

运行 `cinescaffold --help` 或 `cinescaffold <命令> --help` 查看完整参数。

规划进度使用“模型请求”这一中性名称，不表示 Provider 一定启用了 thinking。若模型在必须调用工具时返回纯正文，Runner 会把该正文压缩为固定短标记、记录“异常正文”事件并继续重试，避免单次失控输出污染后续上下文。

## 主要产物

| 阶段 | 主要输出 |
| --- | --- |
| `parse` | `cinematic_brief.json`、`textual_six_dimensions.txt` |
| `plan` | Agent Trace、checkpoint、Constraint Plan、验证报告、`final_scene_ir.json` |
| `execute` | `scene.blend`、`scene.glb`、`viewer_manifest.json`、Runtime Snapshot、Runtime Validation、执行 manifest、白模 MP4 |
| `run` | 所有适用阶段的产物、`textual_six_dimensions.txt` 及 `pipeline_summary.json` |

运行目录默认拒绝覆盖已有产物。`run --overwrite` 会在新管线开始前清理该输出目录中的 `planning/`、`execution/`、`pipeline_summary.json`，以及由 `--text`/`--text-six-file` 生成的根级 Brief 与文本六维，但保留根目录中的其他文件；独立 `plan` 仍要求使用空输出目录。执行阶段只有显式传入 `--overwrite` 才会覆盖其固定输出。

## 开发与验证

运行离线测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Provider HTTP 边界使用无网络模拟回归覆盖认证拒绝（`401`）、限流（`429`）、余额不足（`402`）、服务繁忙（`503`）、网络失败与超时；HTTP 拒绝保留类型化状态码且底层不会自动重试。HTTP 响应按固定总墙钟截止时间增量读取，DeepSeek 非流式空行保活可以继续传输，但不能刷新或延长 `semantic_timeout`。Pipeline Summary 额外携带稳定、无 Secret 的 `failure_code`，供宿主区分过载、限流、认证、余额、网络与超时；原始 Provider 正文仍只保留在内部诊断产物。模型响应前的明确请求拒绝会结算零费用，超时与 5xx 继续保留账单不确定性；金额越线另由 Workflow 测试确认 Semantic 已产生的费用先记录，随后在 Planning 前停止。

JSON Schema 与 Prompt 位于 [`src/cinescaffold/resources/`](src/cinescaffold/resources/)，核心实现位于 [`src/cinescaffold/`](src/cinescaffold/)。这些资源是源码运行与 wheel 安装共用的唯一权威副本。

## 项目状态

v0.8 在 v0.7 统一 CLI/UI 管线基础上补齐可嵌入核心库边界：运行资源随 wheel 分发、默认路径不依赖源码工作目录，并提供稳定的 `cinescaffold.api`。下一阶段重点是用审核样本评估并冻结规则、扩充通用 Validator、输出 Depth/Object ID 控制素材，并建立可重复的视频模型对照实验。

## 许可证

仓库目前尚未附带 CineScaffold 自身的开源许可证；公开可见不等于已经授予复制、修改或分发许可。Blender 与可选 Blender MCP 使用各自的许可证，后者作为外部工具独立安装。
