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
  -> Blender Execution Agent（Agent 2）
  -> 调用 Blender Execution Toolkit 编排构建、检查、恢复和渲染
  -> 官方 Blender Lab MCP Adapter + 确定性 Blender Executor
  -> Blender 真实场景构建、运行时验证与 Artifact Commit Gate
  -> 白模视频与控制素材
  -> 视频生成模型
  -> 匿名专家评测与统计分析
```

核心分工：

- Agent 1 负责开放语义理解、电影策略选择和根据结构化错误修复 Scene IR Candidate。
- Agent 2 负责不可变 Scene IR 的 Blender 执行编排、运行时诊断、重试恢复和 Control Bundle 渲染，不重新解释电影语义。
- 两套 Toolkit 分别负责三维规划状态和 Blender 执行状态；两个 Agent 都不能绕过各自 Commit Gate。
- Scene IR 完整描述正式实验中的场景状态，禁止产生 IR 之外的隐藏状态。
- Blender Adapter 只负责确定性映射；底层选择官方 Blender Lab MCP，原始任意代码工具不直接暴露给 Agent。
- Blender 负责确定性执行、真实资产/场景验证和渲染。

## 当前状态

已实现第一段研究管线：

```text
自然语言
  -> 可替换规则 Prompt
  -> OpenAI / DeepSeek / Mock Provider
  -> Cinematic Brief v0.1
  -> 本地 Schema 验证与来源记录
```

规则文件当前有意保持为空，等待项目成员提供正式转换规则。

双 Agent、Scene IR v0.1 和 Blender 执行接口目前已经完成设计基线，但尚未实现。Blender MCP 原型计划采用官方 Blender Lab MCP 1.0.0，并优先在 Blender 5.2 LTS 上验证；依赖尚未加入项目。

## 使用

项目要求 Python 3.9 或更高版本，当前无运行时第三方依赖。项目使用 `setup.cfg + setup.py`，使 macOS 自带的 pip 21.x 也能使用 editable install。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

使用 Mock Provider 进行离线测试：

```bash
cinescaffold parse \
  --provider mock \
  --text "一个男人站在荒漠里，远处有飞船。" \
  --output runs/example/cinematic_brief.json
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

下一步是继续定义并评审三个版本化 Schema：

1. `Cinematic Brief v0.1`：已完成首稿和离线验证。
2. `Constraint Plan v0.1`：待开发。
3. `Scene IR v0.1`：语义、坐标、帧状态、相机和 Blender 映射基线已确定；JSON Schema 与样例待开发。

在安装 Agent/MCP 依赖前，将先评审双 Agent 权限和工具契约，并用“荒漠中的男人、远处巨型飞船、镜头缓慢推近”建立 Brief、Constraint Plan、Scene IR 三层完整样例。

## 名称

`CineScaffold` 指将非专业创作意图落地为可验证的电影空间脚手架。名称不绑定 Blender、MCP、特定 LLM 或视频生成模型。
