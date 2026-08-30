from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO


TOOL_LABELS = {
    "get_capabilities": "读取工具能力",
    "inspect_candidate": "检查当前场景",
    "apply_entity_patch": "创建或调整场景物体",
    "apply_constraint_patch": "添加空间与动作约束",
    "apply_motion_patch": "编排物体运动",
    "apply_camera_patch": "设置摄影机",
    "solve_candidate": "求解场景参数",
    "validate_candidate": "验证候选场景",
    "restore_candidate": "恢复历史候选",
}


class TerminalReporter:
    """把结构化运行事件转换成适合演示的中文终端输出。"""

    def __init__(
        self,
        *,
        quiet: bool = False,
        color: bool = True,
        stream: TextIO | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.color = color and self.stream.isatty() and "NO_COLOR" not in os.environ
        self.started = time.monotonic()

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.quiet:
            return
        handler = getattr(self, f"_on_{event_type}", None)
        if handler is not None:
            handler(payload)

    def stage(self, title: str, message: str) -> None:
        self._line("◆", title, message, "cyan")

    def success(self, title: str, message: str) -> None:
        self._line("✓", title, message, "green")

    def warning(self, title: str, message: str) -> None:
        self._line("!", title, message, "yellow")

    def failure(self, title: str, message: str) -> None:
        self._line("✗", title, message, "red")

    def _line(self, symbol: str, title: str, message: str, color: str | None = None) -> None:
        if self.quiet:
            return
        elapsed = time.monotonic() - self.started
        prefix = f"[{_format_elapsed(elapsed)}] {symbol} {title}"
        if self.color and color:
            prefix = f"\033[{_color_code(color)}m{prefix}\033[0m"
        print(f"{prefix}  {message}", file=self.stream, flush=True)

    def _on_run_started(self, payload: dict[str, Any]) -> None:
        self.stage(
            "场景规划",
            f"启动 {payload.get('provider')}/{payload.get('model')}，准备 Agent 工具循环",
        )

    def _on_objective_projection_completed(self, payload: dict[str, Any]) -> None:
        self.success(
            "客观语义",
            f"提取 {payload.get('explicit_requirement_count', 0)} 条明确要求；主观字段已隔离",
        )

    def _on_duration_frozen_from_brief(self, payload: dict[str, Any]) -> None:
        self._duration(payload.get("resolution", {}), "解析并冻结时间线")

    def _on_duration_resolution_reused(self, payload: dict[str, Any]) -> None:
        self._duration(payload.get("resolution", {}), "复用 checkpoint 时间线")

    def _duration(self, resolution: dict[str, Any], prefix: str) -> None:
        self.success(
            "时间线",
            f"{prefix}：{resolution.get('resolved_duration_seconds', '?')} 秒，"
            f"{resolution.get('frame_count', '?')} 帧",
        )

    def _on_candidate_checkpoint_loaded(self, payload: dict[str, Any]) -> None:
        self.success("恢复进度", f"已载入 Candidate revision {payload.get('revision', '?')}")

    def _on_planning_attempt_started(self, payload: dict[str, Any]) -> None:
        self.stage(
            "Agent 循环",
            f"第 {payload.get('attempt', '?')} 次提交尝试，当前 revision "
            f"{payload.get('current_revision', '?')}",
        )

    def _on_model_request_started(self, payload: dict[str, Any]) -> None:
        self._line(
            "…",
            "模型思考",
            f"第 {payload.get('request_index', '?')} 次请求，"
            f"上下文 {payload.get('message_count', '?')} 条消息",
            "blue",
        )

    def _on_model_request_completed(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage", {})
        self.success(
            "模型响应",
            f"第 {payload.get('request_index', '?')} 次完成 · "
            f"输入 {_integer(usage.get('input_tokens'))} · "
            f"输出 {_integer(usage.get('output_tokens'))} tokens · "
            f"{_milliseconds(payload.get('duration_ms'))}",
        )

    def _on_model_request_failed(self, payload: dict[str, Any]) -> None:
        self.failure("模型请求", str(payload.get("error", "未知错误")))

    def _on_tool_call_started(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("tool_name", "unknown"))
        self._line("→", "Agent 工具", TOOL_LABELS.get(name, name), "blue")

    def _on_tool_call_completed(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("tool_name", "unknown"))
        status = str(payload.get("status", "unknown"))
        message = (
            f"{TOOL_LABELS.get(name, name)}：{_status_label(status)} · "
            f"revision {payload.get('revision_before', '?')} → "
            f"{payload.get('revision_after', '?')} · {_milliseconds(payload.get('duration_ms'))}"
        )
        if status in {"ok", "no_change"}:
            self.success("工具结果", message)
        else:
            self.warning("工具结果", message)

    def _on_model_history_compacted(self, payload: dict[str, Any]) -> None:
        self._line(
            "↳",
            "上下文整理",
            f"保留 {payload.get('messages_after', '?')}/{payload.get('messages_before', '?')} "
            f"条消息；权威状态 revision {payload.get('authoritative_revision', '?')}",
        )

    def _on_agent_terminal_received(self, payload: dict[str, Any]) -> None:
        terminal = payload.get("terminal", {})
        self.success("Agent 决策", f"返回 {_terminal_label(terminal.get('type'))}")

    def _on_commit_gate_completed(self, payload: dict[str, Any]) -> None:
        validation = payload.get("validation", {})
        message = (
            f"{_status_label(str(payload.get('status', 'unknown')))} · "
            f"hard pass={validation.get('hard_pass', False)} · "
            f"soft score={_decimal(validation.get('soft_score'))} · "
            f"violations={len(payload.get('violations', []))}"
        )
        if payload.get("status") == "success":
            self.success("Commit Gate", message)
        else:
            self.warning("Commit Gate", message)

    def _on_run_failed(self, payload: dict[str, Any]) -> None:
        error = payload.get("error", {})
        self.failure("规划失败", str(error.get("message", payload.get("status", "未知错误"))))

    def _on_run_finished(self, payload: dict[str, Any]) -> None:
        duration = _milliseconds(payload.get("duration_ms"))
        if payload.get("status") == "success":
            self.success("规划完成", f"revision {payload.get('final_revision', '?')} · {duration}")
        else:
            self.warning("规划结束", f"状态 {_status_label(str(payload.get('status')))} · {duration}")

    def _on_execution_validation_started(self, payload: dict[str, Any]) -> None:
        self.stage("执行前检查", "读取并验证 Scene IR")

    def _on_execution_validation_completed(self, payload: dict[str, Any]) -> None:
        self.success(
            "执行前检查",
            f"{payload.get('entity_count', '?')} 个实体 · {payload.get('frame_count', '?')} 帧 · "
            f"{payload.get('duration_seconds', '?')} 秒",
        )

    def _on_execution_validation_failed(self, payload: dict[str, Any]) -> None:
        self.failure("执行前检查", str(payload.get("error", "Scene IR 校验失败")))

    def _on_execution_workspace_started(self, payload: dict[str, Any]) -> None:
        self._line("→", "运行目录", "准备本次执行产物", "blue")

    def _on_execution_workspace_completed(self, payload: dict[str, Any]) -> None:
        self.success("运行目录", str(payload.get("output_dir", "已就绪")))

    def _on_factory_template_started(self, payload: dict[str, Any]) -> None:
        self._line("→", "Blender", "创建干净的 factory template", "blue")

    def _on_factory_template_completed(self, payload: dict[str, Any]) -> None:
        self.success("Blender", "factory template 已就绪")

    def _on_factory_template_failed(self, payload: dict[str, Any]) -> None:
        self.failure("Blender", str(payload.get("error", "模板创建失败")))

    def _on_mcp_build_started(self, payload: dict[str, Any]) -> None:
        self._line("→", "Blender MCP", "根据 Scene IR 构建代理场景", "blue")

    def _on_mcp_build_completed(self, payload: dict[str, Any]) -> None:
        message = (
            f"Blender {payload.get('blender_version', '?')} · "
            f"Runtime validation={payload.get('validation_passed', False)} · "
            f"violations={payload.get('violation_count', '?')}"
        )
        if payload.get("validation_passed"):
            self.success("Blender MCP", message)
        else:
            self.warning("Blender MCP", message)

    def _on_mcp_build_failed(self, payload: dict[str, Any]) -> None:
        self.failure("Blender MCP", str(payload.get("error", "构建失败")))

    def _on_render_started(self, payload: dict[str, Any]) -> None:
        self.stage(
            "视频渲染",
            f"{payload.get('profile')} / {payload.get('backend')} · "
            f"源时间线 {payload.get('frame_count', '?')} 帧 · "
            f"超时 {payload.get('timeout_seconds', '?')} 秒",
        )

    def _on_render_completed(self, payload: dict[str, Any]) -> None:
        self.success(
            "视频渲染",
            f"{payload.get('rendered_frame_count', '?')} 帧 · "
            f"{payload.get('resolution_x', '?')}×{payload.get('resolution_y', '?')} · "
            f"{payload.get('fps', '?')} fps",
        )

    def _on_render_failed(self, payload: dict[str, Any]) -> None:
        self.failure("视频渲染", str(payload.get("error", "渲染失败")))

    def _on_execution_finished(self, payload: dict[str, Any]) -> None:
        message = (
            f"{_status_label(str(payload.get('status')))} · "
            f"{_seconds(payload.get('elapsed_seconds'))}"
        )
        if payload.get("status") == "success":
            self.success("执行完成", message)
        else:
            self.warning("执行结束", message)


def print_planning_summary(result: Any, run_dir: Path, *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    usage = result.usage
    tokens = usage.get("tokens", {})
    context = usage.get("context", {})
    print("\nCineScaffold 规划结果", file=output)
    print("=" * 34, file=output)
    print(f"状态      {_status_label(result.status)}", file=output)
    print(f"模型      {result.provider}/{result.model}", file=output)
    print(f"Revision  {result.final_revision}", file=output)
    print(f"请求/工具 {tokens.get('requests', 0)} / {tokens.get('tool_calls', 0)}", file=output)
    print(
        f"Tokens    输入 {_integer(tokens.get('input_tokens'))} · "
        f"输出 {_integer(tokens.get('output_tokens'))} · "
        f"缓存 {_integer(tokens.get('cache_read_tokens'))}",
        file=output,
    )
    print(f"最大上下文 {_integer(context.get('max_request_input_tokens'))} tokens", file=output)
    cost = usage.get("estimated_cost")
    if cost:
        print(f"估算成本  {cost.get('amount')} {cost.get('currency')}", file=output)
    elif tokens.get("cost") is not None:
        print(f"供应商 cost 字段  {tokens.get('cost')}（币种未冻结）", file=output)
    if result.scene_ir_hash:
        print(f"IR Hash   {result.scene_ir_hash}", file=output)
    print(f"输出目录  {run_dir.resolve()}", file=output)
    scene_ir = result.artifacts.get("scene_ir")
    if scene_ir:
        print(f"下一步    cinescaffold execute --scene-ir {run_dir / scene_ir} --output-dir <目录>", file=output)
    if result.error:
        print(f"错误      {result.error.get('message', result.error)}", file=output)


def print_execution_summary(result: Any, *, stream: TextIO | None = None) -> None:
    output = stream or sys.stdout
    render = result.render or {}
    print("\nCineScaffold Blender 执行结果", file=output)
    print("=" * 34, file=output)
    print(f"状态      {_status_label(result.status)}", file=output)
    print(f"耗时      {result.elapsed_seconds:.2f} 秒", file=output)
    print(
        f"Runtime   Blender {result.build.get('blender_version', '?')} · "
        f"violations={result.build.get('violation_count', '?')}",
        file=output,
    )
    if render:
        print(
            f"视频      {render.get('rendered_frame_count', '?')} 帧 · "
            f"{render.get('resolution_x', '?')}×{render.get('resolution_y', '?')} · "
            f"{render.get('fps', '?')} fps",
            file=output,
        )
        if render.get("artifact"):
            print(f"打开视频  {render['artifact']}", file=output)
    if result.error:
        print(f"错误      {result.error}", file=output)


def print_parse_summary(brief: dict[str, Any], output_path: Path | None, *, stream: TextIO) -> None:
    content = brief.get("content", {})
    subjects = content.get("subjects", [])
    names = [
        item.get("category", {}).get("value")
        for item in subjects
        if item.get("category", {}).get("value")
    ]
    timeline = content.get("timeline", {})
    print("\nCineScaffold 六维语义结果", file=stream)
    print("=" * 34, file=stream)
    print(f"主体      {', '.join(names) if names else '未识别'}", file=stream)
    print(f"动作      {len(content.get('subject_motion', []))} 段", file=stream)
    duration = timeline.get("duration_seconds")
    print(f"时长      {duration} 秒" if duration else "时长      待定", file=stream)
    if output_path:
        print(f"输出文件  {output_path.resolve()}", file=stream)
        if duration:
            print(
                f"下一步    cinescaffold plan --brief {output_path} --output-dir <目录>",
                file=stream,
            )
        else:
            print("下一步    先在 Semantic Parser 阶段解析并确认正数时长", file=stream)


def _format_elapsed(seconds: float) -> str:
    minutes, remainder = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{remainder:02d}"


def _milliseconds(value: Any) -> str:
    try:
        return f"{float(value) / 1000.0:.2f}s"
    except (TypeError, ValueError):
        return "?s"


def _seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f} 秒"
    except (TypeError, ValueError):
        return "耗时未知"


def _integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _decimal(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "?"


def _status_label(status: str) -> str:
    return {
        "ok": "成功",
        "success": "成功",
        "no_change": "无需修改",
        "rejected": "被拒绝，等待修正",
        "failed": "失败",
        "budget_exhausted": "预算耗尽",
        "commit_rejected": "提交未通过",
        "unsupported": "当前不支持",
        "infeasible": "不可行",
        "execution_failed": "构建失败",
        "runtime_mismatch": "Runtime 不一致",
        "render_failed": "渲染失败",
    }.get(status, status)


def _terminal_label(value: Any) -> str:
    return {
        "commit_request": "提交请求",
        "unsupported": "能力缺口",
        "infeasible": "不可行结论",
    }.get(str(value), str(value or "未知结论"))


def _color_code(color: str) -> str:
    return {"red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}[color]
