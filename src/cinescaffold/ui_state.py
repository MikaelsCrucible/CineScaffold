from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cinescaffold.config import LoadedConfig, merge_config, save_config
from cinescaffold.runtime import RuntimeResourcePaths, build_pipeline_run_config
from cinescaffold.workflow import PipelineRunConfig, PipelineSource


SourceKind = Literal["text", "textual_six", "brief", "scene_ir"]


@dataclass
class UiRunInputs:
    source_kind: SourceKind = "text"
    natural_text: str = ""
    textual_six: str = ""
    brief_json: str = ""
    scene_ir_json: str = ""
    output_dir: str = ""
    render_profile: Literal["preview", "control"] = "preview"
    overwrite: bool = False


@dataclass
class UiRunMetrics:
    elapsed_seconds: float = 0.0
    semantic_tokens: int = 0
    planning_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    semantic_cost: str = "—"
    planning_cost: str = "—"

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> UiRunMetrics:
        stages = summary.get("stages", {})
        semantic = stages.get("semantic") or {}
        planning = stages.get("planning") or {}
        semantic_usage = semantic.get("usage") or {}
        semantic_tokens = semantic_usage.get("tokens") or {}
        planning_usage = planning.get("usage") or {}
        planning_tokens = planning_usage.get("tokens") or {}
        return cls(
            elapsed_seconds=float(summary.get("elapsed_seconds") or 0),
            semantic_tokens=_token_total(semantic_tokens),
            planning_tokens=_token_total(planning_tokens),
            requests=int(planning_tokens.get("requests") or 0),
            tool_calls=int(planning_tokens.get("tool_calls") or 0),
            semantic_cost=_cost_text(semantic_usage.get("estimated_cost")),
            planning_cost=_cost_text(planning_usage.get("estimated_cost")),
        )


@dataclass
class UiSessionController:
    loaded_config: LoadedConfig
    config_path: Path
    project_root: Path
    updates: dict[str, str | None] = field(default_factory=dict)

    def apply_settings(self, updates: dict[str, str | None]) -> LoadedConfig:
        merged = merge_config(self.loaded_config, updates)
        self.updates = dict(updates)
        return merged

    def save_settings(self, updates: dict[str, str | None]) -> LoadedConfig:
        self.loaded_config = save_config(self.config_path, updates)
        self.updates = {}
        return self.loaded_config

    def active_config(self) -> LoadedConfig:
        return merge_config(self.loaded_config, self.updates)

    def build_run(
        self,
        inputs: UiRunInputs,
    ) -> tuple[PipelineSource, PipelineRunConfig]:
        source = source_from_inputs(inputs)
        if not inputs.output_dir.strip():
            raise ValueError("请填写输出目录")
        output = Path(inputs.output_dir).expanduser()
        config = build_pipeline_run_config(
            self.active_config(),
            output_dir=output,
            include_semantic=inputs.source_kind in {"text", "textual_six"},
            overwrite=inputs.overwrite,
            resources=RuntimeResourcePaths.from_root(self.project_root),
            render_profile=inputs.render_profile,
        )
        return source, config


def source_from_inputs(inputs: UiRunInputs) -> PipelineSource:
    if inputs.source_kind == "text":
        return PipelineSource(kind="text", text=_required(inputs.natural_text, "自然语言"))
    if inputs.source_kind == "textual_six":
        return PipelineSource(
            kind="textual_six",
            text=_required(inputs.textual_six, "文本六维"),
        )
    if inputs.source_kind == "brief":
        return PipelineSource(kind="brief", payload=_json_object(inputs.brief_json, "Cinematic Brief"))
    return PipelineSource(kind="scene_ir", payload=_json_object(inputs.scene_ir_json, "Scene IR"))


def event_progress(event_type: str, current: float) -> float:
    fixed = {
        "pipeline_started": 0.03,
        "pipeline_semantic_started": 0.08,
        "pipeline_semantic_completed": 0.27,
        "pipeline_input_ready": 0.27,
        "pipeline_planning_started": 0.32,
        "pipeline_execution_started": 0.73,
        "render_started": 0.88,
        "render_completed": 0.97,
        "pipeline_finished": 1.0,
    }
    if event_type in fixed:
        return max(current, fixed[event_type])
    if event_type == "model_request_completed":
        return min(0.68, max(current, 0.36) + 0.025)
    if event_type == "tool_call_completed":
        return min(0.70, max(current, 0.36) + 0.015)
    return current


def event_message(event_type: str, payload: dict[str, Any]) -> str | None:
    messages = {
        "pipeline_started": "管线已启动",
        "pipeline_semantic_started": "正在把输入转换为六维语义",
        "pipeline_semantic_completed": "文本六维与 Cinematic Brief 已完成",
        "pipeline_planning_started": "场景规划 Agent 已启动",
        "pipeline_execution_started": "正在通过 Blender MCP 构建场景",
        "execution_validation_completed": (
            f"Scene IR 已校验：{payload.get('entity_count', '?')} 个实体，"
            f"{payload.get('frame_count', '?')} 帧"
        ),
        "mcp_build_completed": (
            f"Blender 场景已构建：violations={payload.get('violation_count', '?')}"
        ),
        "render_started": f"开始渲染 {payload.get('profile', '')} 白模视频",
        "render_completed": (
            f"渲染完成：{payload.get('rendered_frame_count', '?')} 帧，"
            f"{payload.get('resolution_x', '?')}×{payload.get('resolution_y', '?')}"
        ),
        "pipeline_finished": f"管线结束：{payload.get('status', 'unknown')}",
    }
    if event_type == "model_request_completed":
        usage = payload.get("usage") or {}
        return (
            f"模型响应 #{payload.get('request_index', '?')}："
            f"输入 {usage.get('input_tokens', 0)} / 输出 {usage.get('output_tokens', 0)} tokens"
        )
    if event_type == "tool_call_completed":
        return f"工具 {payload.get('tool_name', 'unknown')}：{payload.get('status', 'unknown')}"
    if event_type == "pipeline_failed":
        return f"{payload.get('stage', 'unknown')} 失败：{payload.get('error', '未知错误')}"
    return messages.get(event_type)


def _required(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"请填写{label}")
    return text


def _json_object(value: str, label: str) -> dict[str, Any]:
    text = _required(value, label)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON 无效：{error.msg}（第 {error.lineno} 行）") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 根节点必须是对象")
    return payload


def _token_total(tokens: dict[str, Any]) -> int:
    return int(tokens.get("input_tokens") or 0) + int(tokens.get("output_tokens") or 0)


def _cost_text(cost: Any) -> str:
    if not isinstance(cost, dict):
        return "—"
    return f"{cost.get('amount', '—')} {cost.get('currency', '')}".strip()
