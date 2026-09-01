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
  -> Scene Skeleton（无数值符号关系）
  -> Planning Toolkit Design Options
  -> Candidate + Validator + Commit Gate
  -> Constraint Plan + Scene IR
  -> 确定性 ExecutionRunner
  -> Blender MCP + Blender Executor
  -> .blend 场景与白模视频
  -> 视频生成模型（后续阶段）
```

四个核心表示各自承担不同职责：

- `Cinematic Brief`：记录用户想表达什么，以及信息来自明确描述、推断还是默认值。
- `文本六维`：用六个固定中文标题呈现同一电影语义，方便非技术协作者阅读、修改和交流；严格 JSON 仍是机器权威表示。
- `Scene Skeleton`：记录实体类别、空间关系、动作阶段、摄影机意图以及定性的尺度/三轴比例，不含坐标、距离、速度、米制尺寸或焦距。
- `Constraint Plan`：记录 Agent 选择了哪些可执行空间、运动和摄影机策略。
- `Scene IR`：精确描述 Blender 应创建和渲染什么，可验证、可重放、可比较。

## 当前能力

- 支持 OpenAI、DeepSeek 和离线 Mock Provider。
- 使用 LLM 提取“谁、在哪、做什么、感觉”，再通过版本化规则表生成可复现的主体、运动、场景、摄影机、构图和光源量化快照。
- 将 Cinematic Brief 中的客观空间、运动、构图和摄影机要求交给规划 Agent。
- 规划 Agent 先做符号化拆解，包括相对大小和 flat/wide/tall 等形体比例；Toolkit 再联合冻结 Profile 与完整 Validator 给出少量数值候选、可行范围和任务相关接口。内置比例不足时，Agent 可向建议接口提交受约束的三轴尺寸范围，候选仍由 Toolkit 原子物化，避免绕过门禁直接改 IR。
- 复合叙事动作不会扩张为场景专用实体或专用执行原语；例如“上车”由通用的目标相对直线移动、事件末端接近约束和明确要求下的可见性转场组合表达，事件名称只负责时间编排。
- 通过类型化 Toolkit、候选 revision、Solver、Validator 和 Commit Gate 生成 Scene IR。
- Validator 按冻结时间线逐帧复验地面、投影、点式空间约束和类型化运动语义；明确要求的来源不仅要映射到正确字段，还必须绑定 Brief 指定的实体。Commit Gate 会再核对编译后逐帧 Scene IR 与已验证 Candidate 的位置、旋转、尺度、显隐、摄影机和焦距等价。
- Validator 发现共线机位、屏幕运动不可读、主体出画或投影尺寸不合格时，Toolkit 会确定性搜索少量经复验的摄影机策略；Agent 选择整体方案，不再逐项猜坐标和焦距。
- 支持世界、局部、目标相对和摄影机相对参考系。
- 支持直线、圆、椭圆、平滑样条和 8 字等代理运动轨迹。
- Agent 可见接口固定右手 Z-up、路径方向和屏幕坐标约定；推断机位下的解析轨道需通过投影可读性门禁。
- 每个量化为移动的主体阶段还需通过通用投影运动可读性门禁；仅有世界坐标位移、但屏幕轨迹和尺度变化都过小的 Candidate 不得提交。
- 未明确要求迎面或背面跟拍时，线性主体运动与摄影机视线必须保留至少 20° 的中位斜视夹角，避免 Agent 仅靠迎面尺寸变化通过运动可读性门禁；明确机位保留用户要求并记录 warning。
- 通过官方 Blender Lab MCP 调用固定 Blender Executor，不把任意 Blender Python 暴露给模型。
- 输出 `.blend`、运行时验证、执行 manifest 和 H.264 白模视频。
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
- 官方 Blender Lab MCP `1.0.0`
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

Blender MCP 作为外部工具单独安装：

```bash
uv tool install --python 3.12 \
  --with "mcp[cli]>=1.2,<2" \
  --from "git+https://projects.blender.org/lab/blender_mcp.git@4309a39646e644261624bfcd2bca669b343b7621#subdirectory=mcp" \
  blender-mcp
```

还需要在 Blender 中安装并启用官方 MCP Add-on。上游安装说明见 [Blender Lab MCP](https://www.blender.org/lab/mcp-server/)。

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

配置还可以保存 Agent 预算、成本单价、Blender/MCP 路径和渲染选项。macOS、Windows 与 Linux 会分别寻找常见 Blender 和 `blender-mcp` 可执行文件；显式路径始终优先。后续 UI 设置中心与 CLI 共用同一配置校验器，保存时不会把已配置 API Key 回传到浏览器，并在覆盖前生成本地 `.bak`。

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

DeepSeek 在 thinking 与 tools 同时启用时要求后续请求完整回传历史 `reasoning_content`。CineScaffold 依赖 PydanticAI 的 DeepSeek Profile 完成字段映射，并在这种模式下保留完整工具思考历史，不再应用 13-message 压缩窗口；普通工具前置正文仍会省略。该协议可能明显增加后续请求上下文，正式实验需要把思考模式和上下文预算一起冻结。

常规规划默认使用面向长思考工具循环的研究预算：48 次模型请求、80 次工具调用、单请求 128K 输入上下文、整轮累计 200K 输出 token、20 分钟墙钟时间和 5 次 Commit Gate 尝试。累计输入和输入输出总量默认不另设上限，但仍受上述单项门禁与供应商限制保护。每项均可通过 `--max-requests`、`--max-tool-calls`、`--max-context-tokens`、`--max-output-tokens`、`--max-seconds` 和 `--max-commit-attempts` 临时覆盖；论文批量实验应显式记录或冻结这些值。

## 使用

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

语义说明位于 [`prompts/semantic_parser/rules.md`](prompts/semantic_parser/rules.md)，固定数值位于 [`prompts/semantic_parser/translation_rules.json`](prompts/semantic_parser/translation_rules.json)。新解析结果使用 Cinematic Brief v0.5：语义模型直接输出类型化的动作、运动模式、目标、载体、路径、动作后置状态，以及摄影机相对主体运动的 `front/rear/side/three_quarter/unspecified` 关系。Provenance 额外记录输入来自自然语言还是文本六维，并用 SHA-256 绑定同步保存的人类可读文本。确定性代码只校验这些字段并查表量化，不再用关键词子串重新猜测动作或机位含义。旧 v0.1–v0.4 Brief 仍可直接进入规划。用户明确要求始终优先于规则推导和缺省值；径向推近/后拉的速度由起止距离和实际时长计算。光源量化只留给后续视频生成阶段，Blender 白模仍使用中性、无阴影的技术照明。

规划门禁不仅检查实体是否在三维世界中移动，还检查每个移动阶段在摄影机投影中的轨迹范围和尺度变化。默认推断机位若让运动在白模里近似静止，Agent 必须调整运动方向、机位或距离；用户明确指定摄影机设计时保留其要求，并把同项降为 warning。

#### 2. Cinematic Brief 转 Scene IR

```bash
cinescaffold plan \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/planning
```

规划阶段同样读取 `.cinescaffold.conf`。如需临时实验覆盖，可以额外传入 `--provider` 和 `--model`，但不会修改配置文件。

正常规划的前三个工具阶段固定为：提交无数值 Scene Skeleton、请求经 Validator 预测的 Design Options、按 `option_id` 原子应用一个候选。只有候选仍有结构化 violation 或 capability gap 时，低层 Patch、Solver 与修复建议工具才会按状态开放；从 checkpoint 恢复时则直接从已有 Candidate 继续。

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
| `execute` | `scene.blend`、Runtime Snapshot、Runtime Validation、执行 manifest、白模 MP4 |
| `run` | 所有适用阶段的产物、`textual_six_dimensions.txt` 及 `pipeline_summary.json` |

运行目录默认拒绝覆盖已有产物。`run --overwrite` 会在新管线开始前清理该输出目录中的 `planning/`、`execution/`、`pipeline_summary.json`，以及由 `--text`/`--text-six-file` 生成的根级 Brief 与文本六维，但保留根目录中的其他文件；独立 `plan` 仍要求使用空输出目录。执行阶段只有显式传入 `--overwrite` 才会覆盖其固定输出。

## 开发与验证

运行离线测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

JSON Schema 位于 [`schemas/`](schemas/)，Prompt 位于 [`prompts/`](prompts/)，核心实现位于 [`src/cinescaffold/`](src/cinescaffold/)。

## 项目状态

已完成自然语言到 Scene IR、Scene IR 到 Blender 场景和白模视频的首条研究管线，并实现首版四要素到六维规则。下一阶段重点是用审核样本评估并冻结该规则、扩充通用 Validator、输出 Depth/Object ID 控制素材，并建立可重复的视频模型对照实验。

## 许可证

仓库目前尚未附带 CineScaffold 自身的开源许可证；公开可见不等于已经授予复制、修改或分发许可。Blender 与 Blender MCP 使用各自的许可证，后者作为外部工具独立安装。
