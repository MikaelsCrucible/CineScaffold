# CineScaffold

面向论文研究的自然语言到三维白模视频生成管线。项目研究非专业用户的一次性自然语言描述，经过电影语义解析、Agent 场景规划、可验证 Scene IR 和 Blender 白模预演后，是否能提升最终生成视频对空间布局、主体运动和摄影机运动的忠实度。

## 当前研究问题

> 在单镜头、单轮、非交互条件下，相比直接使用小白提示词或仅由 LLM 扩写提示词，引入六维电影语义、工具增强的场景规划 Agent 和 Blender 白模控制，能否更准确地还原目标视频的空间关系与运动？

本阶段研究系统，不以商业产品为目标。追问、用户迭代、多镜头长视频、复杂骨骼动作、布料/流体和精细资产暂不进入 v0.1。

论文核心样本会优先采用简单到中等复杂度的单镜头场景，但系统不在 Prompt 或场景逻辑中写死主体数量，也不以“复杂运镜”这类主观标签拒绝请求。实际能力由版本化 Scene IR、Toolkit 能力清单、验证器和运行资源配置共同界定；无法表达或求解的要求返回可诊断结果。

## v0.1 管线

```text
小白自然语言
  -> LLM Semantic Parser
  -> Cinematic Brief（六维、人类可读、保留来源与不确定性）
  -> Scene Planning Agent（Agent 1）
  -> 调用 Scene Planning Toolkit 进行计划、尝试、验证和修复
  -> Constraint Plan + 通过验证的 Scene IR
  -> 确定性 ExecutionRunner
  -> 官方 Blender Lab MCP Adapter + 确定性 Blender Executor
  -> Blender 真实场景构建、运行时验证与 H.264 渲染
  -> 白模视频与控制素材
  -> 视频生成模型
  -> 匿名专家评测与统计分析
```

核心分工：

- Agent 1 负责开放语义理解、电影策略选择和根据结构化错误修复 Scene IR Candidate。
- ExecutionRunner 通过固定状态机执行不可变 Scene IR，不需要第二个 LLM Agent 参与正常路径。
- Agent 2 仅保留为未来可选的异常诊断与恢复实验层；是否引入由无 Agent 基线的消融实验决定。
- Scene IR 完整描述正式实验中的场景状态，禁止产生 IR 之外的隐藏状态。
- Blender Adapter 只负责确定性映射；底层选择官方 Blender Lab MCP，调用内容由项目固定生成，不接受模型提供的任意 Python。
- Blender 负责确定性执行、真实资产/场景验证和渲染。

## 当前状态

已实现从自然语言到白模视频的首条确定性研究管线：

```text
自然语言
  -> 可替换规则 Prompt
  -> OpenAI / DeepSeek / Mock Provider
  -> Cinematic Brief v0.1
  -> 本地 Schema 验证与来源记录
  -> 客观语义投影（剥离 mood、摘要和原始提示词）
  -> OpenAI / DeepSeek / Mock Scene Planning Agent
  -> 九个 Scene Planning Toolkit 接口
  -> Candidate 求解、验证、修复与 Commit Gate
  -> Constraint Plan + 逐帧 Scene IR v0.1
  -> 执行前跨字段验证
  -> 官方 Blender MCP 固定入口
  -> Blender 5.2 确定性 Executor + Runtime Validator
  -> `.blend` + H.264 白模视频 + 可复现 manifest
```

规则文件当前有意保持为空，等待项目成员提供正式转换规则。

Agent 1 的客观语义投影、Candidate revision store、九个 Planning Toolkit 接口、确定性启发式 Solver、结构化 Validator 和 Scene IR Commit Gate 已经实现。模型调用前只保留主体、运动、空间关系、可数值化构图、摄影机和时间线，`mood` 与混合摘要不会进入规划模型上下文；Commit Gate 会把规划轨迹烘焙为完整逐帧 Scene IR。当前确定性约束能力以 `get_capabilities` 返回为准，尚未实现的约束会产生明确 capability gap；同一响应会提供冻结的 FPS、时长、半开时间域和最后一帧时间，避免 Agent 通过失败调用猜测时间边界。Transform Keyframe 使用关闭额外字段的类型化 Schema，错误字段会在工具调用边界被拒绝。

Agent 协议由 Runner 确定性约束：闭集参数直接进入 Tool Schema 枚举；每个上下文只能读取一次能力清单，同一 revision 的相同 inspect 会被拒绝；实体和摄影机的每个状态通道最多存在一条 Track；`commit_ready=true` 后不再允许继续调用工具。Checkpoint 只在当前 Candidate revision 真正变化时写入，读取历史 revision 不会制造伪 checkpoint。Toolkit v0.2 会拒绝载入旧 Toolkit checkpoint，避免把旧验证语义静默带入新实验。

真实 Agent 回归暴露的规划缺陷已经进入确定性门禁：`push_in/pull_out` 按摄影机到目标的距离变化验证，不绑定世界轴；`speed_range` 独立表达移动速度，来源兼容性检查会拒绝用摄影机距离冒充“缓慢”；屏幕构图按旋转后代理体包围盒计算并使用冻结数值容差。Solver 不会为了制造侧面可见性而擅自旋转实体；三维代理是否真实构建由 Blender Runtime Validator 读取实际 mesh 拓扑和局部包围盒验证，与摄影机投影视角解耦。

无 Agent 2 的执行基线也已实现：`cinescaffold execute` 在修改 Blender 前验证 IR，创建 factory template，经官方 Blender MCP 的 `execute_blender_code_for_cli` 调用固定 Executor，从空场景生成代理几何、逐帧实体/摄影机状态、白模材质和灯光，回读 Runtime Snapshot 并渲染 H.264。未给定灯光语义时，Compiler 使用版本化的摄影机相对对称无影灯组，不推断世界光源方向；Runtime Validator 会核对灯组用途、模式、父级、旋转、能量和阴影开关。两次相同 IR 重建得到字节一致的规范化 Runtime Snapshot；两实体、144 帧黄金场景已在 Blender 5.2.1 LTS 上完成 0 violation 构建和视频渲染。

## 使用

项目固定使用 Python 3.12.x；`.python-version` 记录解释器系列，`requirements.lock` 锁定完整 Python 依赖及哈希。推荐使用 [uv](https://docs.astral.sh/uv/) 建立环境：

```bash
uv python install 3.12
uv venv --python 3.12 --clear .venv
uv pip install --python .venv/bin/python \
  --require-hashes --requirements requirements.lock
uv pip install --python .venv/bin/python --no-deps --editable .
source .venv/bin/activate
```

运行时依赖采用最小化的 `pydantic-ai-slim[openai,mcp]==2.36.0`，同时覆盖 OpenAI、DeepSeek 与 MCP stdio Client，不安装 UI、Logfire 或其他未使用组件。

### Blender 与 Blender MCP

原型环境要求：

- Blender `5.2.1 LTS`。
- 官方 Blender Lab MCP `1.0.0`，源码固定到 commit `4309a39646e644261624bfcd2bca669b343b7621`。
- Blender MCP Server 的 MCP Python SDK 固定为 `>=1.2,<2`。官方 1.0.0 源码仍使用 `mcp.server.fastmcp.FastMCP`，与 MCP SDK 2.x 不兼容。
- Blender 中安装并启用官方 MCP Add-on；交互模式需要用 `blender --online-mode` 启动，或由用户明确开启 Blender 的全局在线访问偏好。

Blender MCP Server 应作为独立工具安装，避免把 GPL Server 代码并入 CineScaffold Python 包：

```bash
uv tool install --python 3.12 \
  --with "mcp[cli]>=1.2,<2" \
  --from "git+https://projects.blender.org/lab/blender_mcp.git@4309a39646e644261624bfcd2bca669b343b7621#subdirectory=mcp" \
  blender-mcp
```

Add-on 的官方安装说明见 [Blender Lab MCP](https://www.blender.org/lab/mcp-server/)。正式实验前仍需把 Blender、Add-on、Server、MCP SDK 和 Executor hash 一起写入运行 manifest。

使用 Mock Provider 进行离线测试：

```bash
cinescaffold parse \
  --provider mock \
  --text "一个男人站在荒漠里，远处有飞船。" \
  --output runs/example/cinematic_brief.json
```

将 Cinematic Brief 通过 Agent 1 转换为 Scene IR：

```bash
cinescaffold plan \
  --provider mock \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/planning
```

真实规划模型需要显式指定模型 ID：

```bash
export OPENAI_API_KEY="..."
cinescaffold plan \
  --provider openai \
  --model <model-id> \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/openai-planning

export DEEPSEEK_API_KEY="..."
cinescaffold plan \
  --provider deepseek \
  --model deepseek-v4-pro \
  --thinking-mode enabled \
  --reasoning-effort high \
  --model-max-tokens 8192 \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/deepseek-planning
```

每次规划都会写出 `planning_agent_tool_trace.jsonl`、`constraint_plan.json`、`planning_validation.json`、`planning_summary.json`；成功时额外写出 `final_scene_ir.json`。Trace 记录每轮模型请求、模型设置、工具参数/结果、revision、验证错误和耗时，但不保存模型 thinking/reasoning 内容。`planning_summary.json` 分开记录累计输入 token、缓存命中、未缓存输入和最大单次上下文；累计输入会把每轮重复上下文相加，不能解释为模型上下文窗口大小。DeepSeek 的单次输出上限会通过其 Chat Completions 所需的 `max_tokens` 字段发送；其他兼容 Provider 使用 PydanticAI 的通用设置。

每个成功 Mutation 还会写入 `checkpoints/revision_NNNN.json` 并更新 `checkpoint_latest.json`。预算或网络中断后，应使用同一份 Brief 在新的输出目录建立短上下文续跑；Checkpoint 会校验 Brief、Toolkit、Profile 与 Candidate hash：

```bash
cinescaffold plan \
  --provider deepseek \
  --model <model-id> \
  --brief runs/example/cinematic_brief.json \
  --resume-from runs/example/failed-planning/checkpoint_latest.json \
  --output-dir runs/example/resumed-planning
```

将已提交 Scene IR 确定性执行为 Blender 场景和白模视频：

```bash
cinescaffold execute \
  --scene-ir runs/example/planning/final_scene_ir.json \
  --output-dir runs/example/execution
```

输出包括 `scene.blend`、`runtime_snapshot.json`、`runtime_validation.json`、`clay_preview.mp4`、MCP/Blender 日志与 `execution_manifest.json`。已有同名产物时默认拒绝覆盖；只有显式传入 `--overwrite` 才会清理本次执行的固定产物。Blender 5.2 的 Metal/EEVEE 后台执行在受限沙箱中可能无法初始化，CI 或桌面 Agent 环境需要给予 Blender 正常的 GPU/进程权限。

Provider 价格会变化，因此代码不硬编码价格。实验运行时可把当日价格作为快照显式传入：

```bash
cinescaffold plan \
  --provider openai \
  --model <model-id> \
  --brief <brief.json> \
  --output-dir <run-dir> \
  --input-cost-per-million <price> \
  --output-cost-per-million <price> \
  --price-source "provider pricing page YYYY-MM-DD"
```

使用 OpenAI Responses API：

```bash
export OPENAI_API_KEY="..."
cinescaffold parse --provider openai --model <model-id> --text "..."
```

使用 DeepSeek Chat Completions API：

```bash
export DEEPSEEK_API_KEY="..."
cinescaffold parse --provider deepseek --model <model-id> --text "..."
```

模型名称必须显式传入，避免 API 别名变化导致实验条件静默漂移。规则可直接编辑 [rules.md](prompts/semantic_parser/rules.md)，也可通过 `--rules` 指定其他文件。

运行离线测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

API 实现依据 [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[OpenAI Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)、[DeepSeek 首次调用](https://api-docs.deepseek.com/guides/function_calling/)和 [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/) 官方文档。

## 下一步

三个版本化 Schema 当前状态：

1. `Cinematic Brief v0.1`：已完成首稿和离线验证。
2. `Constraint Plan v0.1`：机器 Schema 已由领域模型生成。
3. `Scene IR v0.1`：机器 Schema、坐标、逐帧状态、相机和 Blender 映射基线已实现。

下一步是补齐 Runtime Validator 的投影、可见性、遮挡、材质、灯光与输出 Pass 检查，并输出 RGB/Depth/Object ID Control Bundle；随后以确定性 ExecutionRunner 为基线，评估可选 Agent 2 是否能在故障恢复任务上带来可测收益。

## 名称

`CineScaffold` 指将非专业创作意图落地为可验证的电影空间脚手架。名称不绑定 Blender、MCP、特定 LLM 或视频生成模型。
