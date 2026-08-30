# CineScaffold

CineScaffold 是一个面向论文研究的自然语言到三维白模视频生成管线。它把非专业用户的描述转换为结构化电影语义，再由工具增强的场景规划 Agent 生成可验证的 Scene IR，最后通过 Blender 构建空间脚手架和摄影机预演。

> 当前状态：研究原型。核心管线已经可以运行，但语义转换规则、目标视频模型适配和正式实验协议尚未冻结，不建议作为生产工具使用。

## 研究动机

纯文本视频生成经常难以稳定复现相对位置、视觉尺度、运动轨迹和摄影机运动。CineScaffold 研究一种中间控制方式：先用可验证的三维代理场景明确这些关系，再把白模视频交给后续生成模型。

项目当前关注的问题是：

> 相比直接提示词或仅由 LLM 扩写提示词，加入六维电影语义、工具增强的场景规划和 Blender 白模控制，能否提高最终视频对空间与运动要求的忠实度？

## 工作流程

```text
自然语言
  -> Semantic Parser
  -> Cinematic Brief（六维电影语义）
  -> Scene Planning Agent
  -> Planning Toolkit + Commit Gate
  -> Constraint Plan + Scene IR
  -> 确定性 ExecutionRunner
  -> Blender MCP + Blender Executor
  -> .blend 场景与白模视频
  -> 视频生成模型（后续阶段）
```

三个核心表示各自承担不同职责：

- `Cinematic Brief`：记录用户想表达什么，以及信息来自明确描述、推断还是默认值。
- `Constraint Plan`：记录 Agent 选择了哪些可执行空间、运动和摄影机策略。
- `Scene IR`：精确描述 Blender 应创建和渲染什么，可验证、可重放、可比较。

## 当前能力

- 支持 OpenAI、DeepSeek 和离线 Mock Provider。
- 将 Cinematic Brief 中的客观空间、运动、构图和摄影机要求交给规划 Agent。
- 通过类型化 Toolkit、候选 revision、Solver、Validator 和 Commit Gate 生成 Scene IR。
- 支持世界、局部、目标相对和摄影机相对参考系。
- 支持直线、圆、椭圆、平滑样条和 8 字等代理运动轨迹。
- 通过官方 Blender Lab MCP 调用固定 Blender Executor，不把任意 Blender Python 暴露给模型。
- 输出 `.blend`、运行时验证、执行 manifest 和 H.264 白模视频。
- CLI 实时显示 Agent 请求、工具调用、token、revision、验证和渲染进度。
- 可从自然语言、Cinematic Brief 或 Scene IR 一键运行到白模视频。

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

DeepSeek V4 Pro 的思考模式缺省为启用，缺省强度为 `high`。真实回归中，模型读取完整 Toolkit 能力后的首次规划可能产生上万 reasoning tokens，并让单次请求持续数分钟。若实验优先考虑低延迟与稳定工具调用，可在配置中明确填写：

```text
planning_thinking_mode = disabled
planning_model_max_tokens = 8192
```

论文实验应明确冻结思考模式、强度和 token 上限，不能把供应商缺省值视为不变条件。规划 Trace 记录 reasoning token 数、请求耗时、停止原因与工具调用，但不保存逐字思维链。

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

# 从 Cinematic Brief 开始
cinescaffold run \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/full

# 从 Scene IR 开始
cinescaffold run \
  --ir runs/example/final_scene_ir.json \
  --output-dir runs/example/execution-only
```

使用其他配置文件时传入 `--config cinescaffold.local.conf`。一键运行会写出 `pipeline_summary.json`，并在任一阶段失败时停止，不会把不合法 IR 继续交给 Blender。

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

正式使用前需要由项目成员补充并评审 [`prompts/semantic_parser/rules.md`](prompts/semantic_parser/rules.md) 中的六维转换规则；该文件当前有意留空。

#### 2. Cinematic Brief 转 Scene IR

```bash
cinescaffold plan \
  --brief runs/example/cinematic_brief.json \
  --output-dir runs/example/planning
```

规划阶段同样读取 `.cinescaffold.conf`。如需临时实验覆盖，可以额外传入 `--provider` 和 `--model`，但不会修改配置文件。

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

## 主要产物

| 阶段 | 主要输出 |
| --- | --- |
| `parse` | `cinematic_brief.json` |
| `plan` | Agent Trace、checkpoint、Constraint Plan、验证报告、`final_scene_ir.json` |
| `execute` | `scene.blend`、Runtime Snapshot、Runtime Validation、执行 manifest、白模 MP4 |
| `run` | 所有适用阶段的产物及 `pipeline_summary.json` |

运行目录默认拒绝覆盖已有产物；执行阶段只有显式传入 `--overwrite` 才会覆盖其固定输出。

## 开发与验证

运行离线测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

JSON Schema 位于 [`schemas/`](schemas/)，Prompt 位于 [`prompts/`](prompts/)，核心实现位于 [`src/cinescaffold/`](src/cinescaffold/)。

## 项目状态

已完成自然语言到 Scene IR、Scene IR 到 Blender 场景和白模视频的首条研究管线。下一阶段重点是冻结语义转换规则、扩充通用 Validator、输出 Depth/Object ID 控制素材，并建立可重复的视频模型对照实验。

## 许可证

仓库目前尚未附带 CineScaffold 自身的开源许可证；公开可见不等于已经授予复制、修改或分发许可。Blender 与 Blender MCP 使用各自的许可证，后者作为外部工具独立安装。
