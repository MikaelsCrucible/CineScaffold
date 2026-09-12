from __future__ import annotations

import json
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition

from cinescaffold.planning.domain import (
    AgentTerminal,
    CameraStatic,
    CandidateState,
    ConstraintKind,
    GroundInteractionSpec,
    ProxyGeometry,
    StrictModel,
    TrackKeyframe,
)
from cinescaffold.planning.design import EntitySizeRequest, SceneSkeleton
from cinescaffold.planning.toolkit import TOOLKIT_VERSION, ScenePlanningToolkit
from cinescaffold.planning.trace import TraceRecorder


class EntityPatchInput(StrictModel):
    entity_id: str
    label: str | None = Field(
        default=None,
        description="更新时省略表示保持原值；显式 null 表示清除标签",
    )
    role: str | None = Field(
        default=None,
        description="新实体必填；更新时省略表示保持原值",
    )
    proxy: ProxyGeometry | None = Field(
        default=None,
        description="新实体必填；更新时省略表示保持现有代理几何",
    )
    parent_id: str | None = Field(
        default=None,
        description="更新时省略表示保持原值；显式 null 表示移除父级",
    )
    tags: list[str] | None = Field(
        default=None,
        description="更新时省略表示保持原值；空数组表示清空",
    )
    source_refs: list[str] | None = Field(
        default=None,
        description="更新时省略表示保持原值；空数组表示清空",
    )
    ground_interaction: GroundInteractionSpec | None = Field(
        default=None,
        description="更新时省略表示保持现有地面策略",
    )


class ConstraintPatchInput(StrictModel):
    """保持工具 schema 紧凑，领域模型仍由 Toolkit 严格复验。"""

    constraint_id: str = Field(min_length=1)
    type: ConstraintKind
    strength: Literal["hard", "soft"]
    weight: float = Field(default=1.0, gt=0)
    subjects: list[str] = Field(default_factory=list)
    time_range_seconds: tuple[float, float]
    parameters: dict[str, Any] = Field(
        description="字段必须遵循 Design Option relevant_capabilities 或工具错误返回的对应契约"
    )
    source_status: Literal["explicit", "inferred", "default", "agent_selected", "unknown"]
    source_ref: str


class PathPatchInput(StrictModel):
    """将关闭路径联合展平，避免同一联合在多个工具中重复展开。"""

    representation: Literal[
        "polyline",
        "sampled",
        "circle",
        "ellipse",
        "catmull_rom",
        "lemniscate",
    ] = Field(description="路径几何类型；普通公转使用 circle/ellipse")
    space: Literal["world", "local", "camera", "target_relative"] = Field(
        default="world",
        description="相对运动必须用 target_relative；所有世界坐标遵循右手 +Z-up",
    )
    target_id: str | None = Field(
        default=None,
        description="target_relative 路径的动态参照实体",
    )
    closed: bool | None = Field(
        default=None,
        description="解析圆/椭圆自动闭合；其他路径按语义显式设置",
    )
    cycle_count: float | None = Field(
        default=None,
        gt=0,
        description="Track 时间段内循环次数；省略时为 1",
    )
    parameterization: Literal["normalized_time", "arc_length"] | None = Field(
        default=None,
        description="省略时为 normalized_time；匀速经过非解析路径可选 arc_length",
    )
    orientation_mode: Literal["keep"] | None = Field(
        default=None,
        description="当前只支持 keep，不自动让实体沿切线旋转",
    )
    control_points: list[tuple[float, float, float]] | None = Field(
        default=None,
        description="位于 path.space 的米制 [x,y,z] 控制点",
    )
    center_offset_m: tuple[float, float, float] | None = Field(
        default=None,
        description="解析路径中心相对 target_id 的米制偏移；省略为 [0,0,0]",
    )
    plane_normal: tuple[float, float, float] | None = Field(
        default=None,
        description="解析路径平面法线；省略为 +Z，即规范 XY 水平面",
    )
    axis_direction: tuple[float, float, float] | None = Field(
        default=None,
        description="解析路径 0 度方向；省略为 +X，且不得平行 plane_normal",
    )
    initial_phase_degrees: float | None = Field(
        default=None,
        description="从 axis_direction 起算的初相位；此字段明确使用角度而非弧度",
    )
    direction: Literal["counterclockwise", "clockwise"] | None = Field(
        default=None,
        description="从 +plane_normal 一侧朝路径中心观察时的方向；省略为 counterclockwise",
    )
    radius_m: float | None = Field(default=None, gt=0)
    semi_major_axis_m: float | None = Field(default=None, gt=0)
    semi_minor_axis_m: float | None = Field(default=None, gt=0)
    width_m: float | None = Field(default=None, gt=0)
    height_m: float | None = Field(default=None, gt=0)


class TrackPatchInput(StrictModel):
    """Agent 侧紧凑输入；Toolkit 转回严格 TrackSpec。"""

    track_id: str = Field(min_length=1)
    target_entity_id: str | None = None
    type: Literal["transform", "path_follow", "visibility", "look_at", "focal_length"]
    time_range_seconds: tuple[float, float] = Field(
        description="Track 生效的半开秒区间 [start,end)；不得用末帧时间替代 end",
    )
    keyframes: list[TrackKeyframe] = Field(default_factory=list)
    path: PathPatchInput | None = None
    target_id: str | None = None
    interpolation: Literal["step", "linear", "smooth"] = "linear"
    source_ref: str | None = None

    def to_domain_payload(self) -> dict[str, Any]:
        # 省略未使用的变体字段，由关闭的领域联合补默认值并拒绝错配。
        return self.model_dump(mode="json", exclude_none=True)


class CameraPatchInput(StrictModel):
    """Camera portion of an escalated, atomic Candidate repair."""

    camera_id: str
    projection: Literal["perspective"] = "perspective"
    active: bool = True
    static: CameraStatic
    tracks: list[TrackPatchInput] = Field(default_factory=list)
    remove_track_ids: list[str] = Field(default_factory=list)

    def to_tool_payload(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "projection": self.projection,
            "active": self.active,
            "static": self.static.model_dump(mode="json"),
            "tracks": [item.to_domain_payload() for item in self.tracks],
            "remove_track_ids": self.remove_track_ids,
        }


MAX_AGENT_HISTORY_MESSAGES = 13
NON_TOOL_TEXT_MARKER = "[已省略不符合协议的无工具正文；请按重试指令调用工具。]"


def compact_agent_payload(
    value: Any,
    *,
    frame_interval_seconds: float,
) -> Any:
    """Project verbose sampled violations into a bounded Agent-facing view.

    The authoritative Candidate validation is intentionally left untouched. This
    projection is used only for model tool returns and recovery prompts, while
    checkpoints and validation artifacts retain every sampled violation.
    """

    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            if (
                (key == "violations" or key.endswith("_violations"))
                and isinstance(item, list)
            ):
                compacted[key] = _compact_sampled_violations(
                    item,
                    frame_interval_seconds=frame_interval_seconds,
                )
            else:
                compacted[key] = compact_agent_payload(
                    item,
                    frame_interval_seconds=frame_interval_seconds,
                )
        data = compacted.get("data")
        top_level_violations = compacted.get("violations")
        if (
            isinstance(data, dict)
            and isinstance(top_level_violations, list)
            and data.get("violations") == top_level_violations
        ):
            data.pop("violations")
            data["violation_count"] = len(top_level_violations)
            data["violations_location"] = "top_level_violations"
        if isinstance(data, dict) and isinstance(top_level_violations, list):
            data["repair_focus"] = _repair_focus(top_level_violations)
        return compacted
    if isinstance(value, list):
        return [
            compact_agent_payload(
                item,
                frame_interval_seconds=frame_interval_seconds,
            )
            for item in value
        ]
    return deepcopy(value)


def _repair_focus(violations: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank likely root causes while retaining every compacted violation."""

    def priority(item: dict[str, Any]) -> tuple[int, int, str]:
        severity = {"hard": 0, "soft": 1, "warning": 2}.get(
            str(item.get("severity")),
            3,
        )
        code = str(item.get("code", ""))
        structural_markers = (
            "UNRESOLVED",
            "MISSING",
            "UNKNOWN",
            "REFERENCE",
            "TIMELINE",
            "HIERARCHY",
            "TRANSFORM",
            "GROUND",
        )
        if any(marker in code for marker in structural_markers):
            cause = 0
        elif "EXPLICIT_REQUIREMENT" in code or code == "UNMAPPED_EXPLICIT_REQUIREMENT":
            cause = 1
        elif "MOTION" in code or "HOLD" in code or "SPEED" in code:
            cause = 2
        else:
            cause = 3
        return severity, cause, code

    ranked = sorted(
        (item for item in violations if isinstance(item, dict)),
        key=priority,
    )
    distinct: list[dict[str, Any]] = []
    seen_groups: set[tuple[Any, ...]] = set()
    for item in ranked:
        group_key = (
            str(item.get("code", "")),
            str(item.get("constraint_id", "")),
            tuple(str(value) for value in item.get("entity_ids", [])),
            tuple(str(value) for value in item.get("adjustable_variables", [])),
        )
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        distinct.append(item)
    primary = distinct[:3]
    domains: set[str] = set()
    for item in primary:
        for variable in item.get("adjustable_variables", []):
            lowered = str(variable).lower()
            if "camera" in lowered:
                domains.add("camera")
            if "motion" in lowered or "track" in lowered:
                domains.add("motion_tracks")
            if "constraint" in lowered:
                domains.add("constraints")
            if "entity" in lowered or "ground" in lowered:
                domains.add("entities")
            if "transform" in lowered or "translation" in lowered:
                domains.add("layout_solver")
    return {
        "primary_violation_ids": [str(item.get("id", "")) for item in primary],
        "primary_codes": [str(item.get("code", "")) for item in primary],
        "recommended_domains": sorted(domains),
        "secondary_violation_count": max(0, len(ranked) - len(primary)),
        "distinct_secondary_cause_count": max(0, len(distinct) - len(primary)),
        "all_compacted_violations_retained": True,
    }


def _compact_sampled_violations(
    violations: list[Any],
    *,
    frame_interval_seconds: float,
) -> list[Any]:
    groups: dict[str, list[tuple[int, float, dict[str, Any]]]] = {}
    output: list[tuple[int, Any]] = []
    for index, raw in enumerate(violations):
        if not isinstance(raw, dict):
            output.append((index, deepcopy(raw)))
            continue
        point = _point_violation_time(raw)
        if point is None:
            output.append((index, deepcopy(raw)))
            continue
        groups.setdefault(_sampled_violation_signature(raw), []).append(
            (index, point, raw)
        )

    maximum_gap = frame_interval_seconds * 1.5 + 1e-9
    for records in groups.values():
        records.sort(key=lambda item: (item[1], item[0]))
        segment: list[tuple[int, float, dict[str, Any]]] = []
        for record in records:
            if segment and record[1] - segment[-1][1] > maximum_gap:
                output.extend(_summarize_sampled_segment(segment))
                segment = []
            segment.append(record)
        output.extend(_summarize_sampled_segment(segment))

    return [item for _, item in sorted(output, key=lambda pair: pair[0])]


def _point_violation_time(violation: dict[str, Any]) -> float | None:
    value = violation.get("time_range_seconds")
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], (int, float))
        or not isinstance(value[1], (int, float))
        or abs(float(value[0]) - float(value[1])) > 1e-9
    ):
        return None
    return float(value[0])


def _sampled_violation_signature(violation: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in violation.items()
        if key not in {"id", "time_range_seconds", "actual"}
    }
    stable["actual_shape"] = _categorical_shape(violation.get("actual"))
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _categorical_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _categorical_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_categorical_shape(item) for item in value]
    if isinstance(value, tuple):
        return [_categorical_shape(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return "<number>"
    return f"<{type(value).__name__}>"


def _summarize_sampled_segment(
    segment: list[tuple[int, float, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    if len(segment) < 2:
        index, _, violation = segment[0]
        return [(index, deepcopy(violation))]

    first = segment[0]
    last = segment[-1]
    worst = max(segment, key=lambda item: _violation_evidence_score(item[2]))
    summary = deepcopy(worst[2])
    summary["time_range_seconds"] = [first[1], last[1]]
    summary["actual"] = {
        "summary_kind": "sampled_time_range",
        "sample_count": len(segment),
        "first_sample": {
            "time_seconds": first[1],
            "actual": deepcopy(first[2].get("actual")),
        },
        "worst_sample": {
            "time_seconds": worst[1],
            "actual": deepcopy(worst[2].get("actual")),
        },
        "last_sample": {
            "time_seconds": last[1],
            "actual": deepcopy(last[2].get("actual")),
        },
    }
    return [(min(item[0] for item in segment), summary)]


def _violation_evidence_score(violation: dict[str, Any]) -> tuple[float, float]:
    actual = violation.get("actual")
    if not isinstance(actual, dict):
        return (0.0, 0.0)
    priority_values: list[float] = []
    fallback_values: list[float] = []
    for key, raw in actual.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = abs(float(raw))
        fallback_values.append(value)
        if any(
            marker in key
            for marker in (
                "penetration",
                "clearance",
                "error",
                "deviation",
                "delta",
                "violation",
            )
        ):
            priority_values.append(value)
    return (
        max(priority_values, default=0.0),
        max(fallback_values, default=0.0),
    )


def _count_violation_items(value: Any) -> int:
    if isinstance(value, dict):
        return sum(
            (
                len(item)
                if (key == "violations" or key.endswith("_violations"))
                and isinstance(item, list)
                else _count_violation_items(item)
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return sum(_count_violation_items(item) for item in value)
    return 0


def _compact_tool_call_history(
    ctx: RunContext[PlanningDeps],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """保留必要协议字段；允许时再裁剪为近期完整工具轮。"""

    compacted: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            has_tool_call = any(isinstance(part, ToolCallPart) for part in message.parts)
            text_parts = [part for part in message.parts if isinstance(part, TextPart)]
            if has_tool_call:
                # DeepSeek thinking+tools 要求后续请求完整回传 reasoning_content。
                parts = [part for part in message.parts if not isinstance(part, TextPart)]
                message = replace(message, parts=parts)
            elif text_parts:
                omitted_chars = sum(len(part.content) for part in text_parts)
                parts = [part for part in message.parts if not isinstance(part, TextPart)]
                parts.append(TextPart(NON_TOOL_TEXT_MARKER))
                message = replace(message, parts=parts)
                response_key = message.provider_response_id or str(message.timestamp)
                if response_key not in ctx.deps.compacted_non_tool_responses:
                    ctx.deps.compacted_non_tool_responses.add(response_key)
                    ctx.deps.trace.record(
                        "model_non_tool_text_compacted",
                        omitted_chars=omitted_chars,
                        authoritative_revision=ctx.deps.toolkit.store.current_revision,
                    )
        compacted.append(message)
    if ctx.deps.preserve_complete_thinking_history:
        return compacted
    if len(compacted) <= MAX_AGENT_HISTORY_MESSAGES:
        return compacted

    first = compacted[0]
    start = len(compacted) - (MAX_AGENT_HISTORY_MESSAGES - 1)
    if (
        start > 1
        and isinstance(compacted[start], ModelRequest)
        and any(isinstance(part, ToolReturnPart) for part in compacted[start].parts)
    ):
        # 不留下缺少对应 ToolCall 的孤立 ToolReturn。
        start -= 1
    bounded = [first, *compacted[start:]]
    ctx.deps.trace.record(
        "model_history_compacted",
        messages_before=len(compacted),
        messages_after=len(bounded),
        authoritative_revision=ctx.deps.toolkit.store.current_revision,
    )
    return bounded


@dataclass
class PlanningDeps:
    toolkit: ScenePlanningToolkit
    trace: TraceRecorder
    deadline_monotonic: float | None
    checkpoint_writer: Callable[[CandidateState], Path] | None = None
    capabilities_read: bool = False
    inspected_calls: set[str] = field(default_factory=set)
    compacted_non_tool_responses: set[str] = field(default_factory=set)
    preserve_complete_thinking_history: bool = False

    def call_tool(self, name: str, arguments: dict[str, Any], operation) -> dict[str, Any]:
        if (
            self.deadline_monotonic is not None
            and time.monotonic() >= self.deadline_monotonic
        ):
            raise TimeoutError("Agent 1 已超过运行时间预算")
        started = time.monotonic()
        revision_before = self.toolkit.store.current_revision
        self.trace.record(
            "tool_call_started",
            tool_name=name,
            revision_before=revision_before,
            arguments=arguments,
        )
        rejection = self._protocol_rejection(name, arguments, revision_before)
        if rejection is not None:
            self.trace.record(
                "tool_call_completed",
                tool_name=name,
                revision_before=revision_before,
                revision_after=revision_before,
                result_revision=revision_before,
                status="rejected",
                protocol_rejected=True,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                result=rejection,
            )
            return rejection
        try:
            result = operation()
        except Exception as error:
            self.trace.record(
                "tool_call_failed",
                tool_name=name,
                revision_before=revision_before,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        current_revision_after = self.toolkit.store.current_revision
        self.trace.record(
            "tool_call_completed",
            tool_name=name,
            revision_before=revision_before,
            revision_after=current_revision_after,
            result_revision=result.get("revision_after"),
            status=result.get("status"),
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            result=result,
        )
        if name == "get_capabilities" and result.get("status") == "ok":
            self.capabilities_read = True
        if name == "inspect_candidate" and result.get("status") == "ok":
            self.inspected_calls.add(self._inspect_key(arguments, revision_before))
        if self.checkpoint_writer is not None and current_revision_after != revision_before:
            checkpoint_path = self.checkpoint_writer(self.toolkit.store.get())
            self.trace.record(
                "candidate_checkpoint_written",
                revision=current_revision_after,
                checkpoint=checkpoint_path.name,
            )
        timeline = self.toolkit.store.get().timeline
        agent_result = compact_agent_payload(
            result,
            frame_interval_seconds=(
                timeline.fps_denominator / timeline.fps_numerator
            ),
        )
        full_violation_count = _count_violation_items(result)
        agent_violation_count = _count_violation_items(agent_result)
        if agent_violation_count < full_violation_count:
            self.trace.record(
                "agent_violation_payload_compacted",
                tool_name=name,
                full_violation_count=full_violation_count,
                agent_violation_count=agent_violation_count,
                authoritative_revision=current_revision_after,
            )
        return agent_result

    def _protocol_rejection(
        self,
        name: str,
        arguments: dict[str, Any],
        revision: int,
    ) -> dict[str, Any] | None:
        if name == "get_capabilities" and self.capabilities_read:
            return _protocol_rejected(
                revision,
                "能力清单在本次上下文中已经读取，不得重复调用",
                ["使用已有能力结果继续构造或提交"],
            )
        if name == "get_capabilities":
            # 仅保留给兼容测试和显式低层诊断；正常 Agent 工具表不会暴露。
            return None
        if not self.capabilities_read and not self.toolkit.design_option_applied:
            if not self.toolkit.has_scene_skeleton and name != "submit_scene_skeleton":
                return _protocol_rejected(
                    revision,
                    "必须先提交只含符号关系的 Scene Skeleton",
                    ["调用 submit_scene_skeleton"],
                )
            if (
                self.toolkit.has_scene_skeleton
                and not self.toolkit.has_design_options
                and name
                not in (
                    {"request_design_options", "submit_scene_skeleton"}
                    if self.toolkit.design_search_failed
                    else {"request_design_options"}
                )
            ):
                return _protocol_rejected(
                    revision,
                    (
                        "当前 Scene Skeleton 未生成合法候选；请修订骨架或调整尺寸请求"
                        if self.toolkit.design_search_failed
                        else "Scene Skeleton 已接受；下一步必须请求数值设计选项"
                    ),
                    (
                        ["修订后调用 submit_scene_skeleton，或调整后调用 request_design_options"]
                        if self.toolkit.design_search_failed
                        else ["调用 request_design_options"]
                    ),
                )
            if self.toolkit.has_design_options and name not in {
                "request_design_options",
                "apply_design_option",
            }:
                return _protocol_rejected(
                    revision,
                    "Design Options 已生成；请选择 option，或带不同尺寸范围重新请求",
                    ["调用 apply_design_option 或调整后调用 request_design_options"],
                )
        validation = self.toolkit.store.get().validation
        if (
            name != "get_capabilities"
            and validation is not None
            and validation.hard_pass
            and validation.soft_score >= self.toolkit.profile.minimum_soft_score
        ):
            return _protocol_rejected(
                revision,
                "当前 revision 已满足 Commit Gate 阈值，不允许继续调用工具",
                [f"立即返回 CommitRequest(candidate_revision={revision})"],
            )
        if name == "inspect_candidate":
            key = self._inspect_key(arguments, revision)
            if key in self.inspected_calls:
                return _protocol_rejected(
                    revision,
                    "同一 revision 的相同 inspect 请求已经执行",
                    ["使用已有结果，或先产生新 revision"],
                )
        return None

    @staticmethod
    def _inspect_key(arguments: dict[str, Any], current_revision: int) -> str:
        normalized = dict(arguments)
        normalized["revision"] = (
            current_revision if arguments.get("revision") is None else arguments["revision"]
        )
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _protocol_rejected(revision: int, message: str, next_actions: list[str]) -> dict[str, Any]:
    return {
        "tool_version": TOOLKIT_VERSION,
        "status": "rejected",
        "revision_before": revision,
        "revision_after": revision,
        "changes": [],
        "data": {},
        "violations": [],
        "warnings": [message],
        "capability_gaps": [],
        "next_actions": next_actions,
    }


async def _prepare_capabilities_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    # 正常流程由 Design Options 返回任务相关能力，不再付费读取整本手册。
    del ctx, tool_definition
    return None


async def _prepare_scene_skeleton_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    if ctx.deps.capabilities_read or ctx.deps.toolkit.design_option_applied:
        return None
    return (
        tool_definition
        if not ctx.deps.toolkit.has_scene_skeleton
        or ctx.deps.toolkit.design_search_failed
        else None
    )


async def _prepare_design_options_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    toolkit = ctx.deps.toolkit
    if ctx.deps.capabilities_read or toolkit.design_option_applied:
        return None
    return tool_definition if toolkit.has_scene_skeleton else None


async def _prepare_design_apply_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    toolkit = ctx.deps.toolkit
    if ctx.deps.capabilities_read or toolkit.design_option_applied:
        return None
    return tool_definition if toolkit.has_design_options else None


async def _prepare_candidate_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    # 首次能力读取前只暴露能力工具；可提交后只允许结构化终止。
    if not (ctx.deps.capabilities_read or ctx.deps.toolkit.design_option_applied):
        return None
    validation = ctx.deps.toolkit.store.get().validation
    if (
        validation is not None
        and validation.hard_pass
        and validation.soft_score >= ctx.deps.toolkit.profile.minimum_soft_score
    ):
        return None
    return tool_definition


async def _prepare_manual_mutation_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    # Component mutations remain registered for compatibility and direct
    # diagnostics. The normal Agent receives one atomic Candidate repair tool so
    # coupled fixes do not require unsafe intermediate revisions.
    del ctx, tool_definition
    return None


async def _prepare_escalated_candidate_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    prepared = await _prepare_candidate_tool(ctx, tool_definition)
    if prepared is None:
        return None
    toolkit = ctx.deps.toolkit
    if toolkit.has_unrepairable_hard_violations:
        return prepared
    if toolkit.has_current_repair_suggestions:
        return None
    if toolkit.has_repairable_violations and not toolkit.has_exhausted_repair_search:
        return None
    return prepared


async def _prepare_candidate_patch_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    prepared = await _prepare_escalated_candidate_tool(ctx, tool_definition)
    if prepared is None:
        return None
    return replace(
        prepared,
        description=(
            (prepared.description or "")
            + " 当前 Validator 建议优先关注："
            + ", ".join(ctx.deps.toolkit.recommended_manual_repair_domains)
            + "。这些是建议而非硬隐藏；必要时可在同一事务中组合多个领域修改。"
        ),
    )


async def _prepare_repair_suggestion_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    prepared = await _prepare_candidate_tool(ctx, tool_definition)
    if prepared is None:
        return None
    if ctx.deps.toolkit.has_current_repair_suggestions:
        return None
    return prepared if ctx.deps.toolkit.has_repairable_violations else None


async def _prepare_repair_apply_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    prepared = await _prepare_candidate_tool(ctx, tool_definition)
    if prepared is None:
        return None
    return prepared if ctx.deps.toolkit.has_current_repair_suggestions else None


def create_planning_agent(model: Model, system_prompt: str) -> Agent[PlanningDeps, AgentTerminal]:
    agent: Agent[PlanningDeps, AgentTerminal] = Agent(
        model,
        output_type=AgentTerminal,
        deps_type=PlanningDeps,
        instructions=system_prompt,
        name="cinescaffold_scene_planner",
        retries=2,
        end_strategy="exhaustive",
        capabilities=[ProcessHistory(_compact_tool_call_history)],
    )

    @agent.tool(sequential=True, prepare=_prepare_scene_skeleton_tool)
    async def submit_scene_skeleton(
        ctx: RunContext[PlanningDeps],
        skeleton: SceneSkeleton,
    ) -> dict[str, Any]:
        """提交实体、关系、动作阶段、符号路径点和摄影机意图；不得包含坐标或尺寸。"""
        arguments = {"skeleton": skeleton.model_dump(mode="json")}
        return ctx.deps.call_tool(
            "submit_scene_skeleton",
            arguments,
            lambda: ctx.deps.toolkit.submit_scene_skeleton(arguments["skeleton"]),
        )

    @agent.tool(sequential=True, prepare=_prepare_design_options_tool)
    async def request_design_options(
        ctx: RunContext[PlanningDeps],
        preference: Literal[
            "balanced",
            "preserve_composition",
            "maximize_motion_readability",
        ] = "balanced",
        max_options: int = 3,
        custom_size_requests: list[EntitySizeRequest] | None = None,
    ) -> dict[str, Any]:
        """让 Toolkit 联合求解数值候选；可请求受约束的三轴尺寸范围。"""
        arguments = {
            "preference": preference,
            "max_options": max_options,
            "custom_size_requests": [
                item.model_dump(mode="json")
                for item in (custom_size_requests or [])
            ],
        }
        return ctx.deps.call_tool(
            "request_design_options",
            arguments,
            lambda: ctx.deps.toolkit.request_design_options(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_design_apply_tool)
    async def apply_design_option(
        ctx: RunContext[PlanningDeps],
        base_revision: int,
        option_id: str,
    ) -> dict[str, Any]:
        """按 option_id 原子物化数值 Candidate，不手抄范围或坐标。"""
        arguments = {"base_revision": base_revision, "option_id": option_id}
        return ctx.deps.call_tool(
            "apply_design_option",
            arguments,
            lambda: ctx.deps.toolkit.apply_design_option(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_capabilities_tool)
    async def get_capabilities(
        ctx: RunContext[PlanningDeps],
        sections: list[
            Literal[
                "entities",
                "constraints",
                "tracks",
                "camera",
                "validators",
                "limits",
                "timeline",
                "profiles",
                "constraint_types",
                "resources",
                "mcp",
                "blender",
                "executor",
                "scene_ir",
                "render",
            ]
        ],
    ) -> dict[str, Any]:
        """读取版本化能力、约束、轨道、验证器和当前资源状态。"""
        return ctx.deps.call_tool(
            "get_capabilities",
            {"sections": sections},
            lambda: ctx.deps.toolkit.get_capabilities(sections),
        )

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
    async def inspect_candidate(
        ctx: RunContext[PlanningDeps],
        view: Literal[
            "summary",
            "entities",
            "camera",
            "constraints",
            "violations",
            "timeline",
            "diff",
            "full_ir",
        ] = "summary",
        revision: int | None = None,
        entity_ids: list[str] | None = None,
        camera_ids: list[str] | None = None,
        constraint_ids: list[str] | None = None,
        time_range_seconds: tuple[float, float] | None = None,
        compare_to_revision: int | None = None,
    ) -> dict[str, Any]:
        """按枚举视图读取当前或历史 Candidate，不修改状态。"""
        arguments = {
            "view": view,
            "revision": revision,
            "entity_ids": entity_ids or [],
            "camera_ids": camera_ids or [],
            "constraint_ids": constraint_ids or [],
            "time_range_seconds": time_range_seconds,
            "compare_to_revision": compare_to_revision,
        }
        return ctx.deps.call_tool(
            "inspect_candidate",
            arguments,
            lambda: ctx.deps.toolkit.inspect_candidate(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_manual_mutation_tool)
    async def apply_entity_patch(
        ctx: RunContext[PlanningDeps],
        upserts: list[EntityPatchInput],
        remove_ids: list[str],
    ) -> dict[str, Any]:
        """原子创建、更新或删除实体；更新时保留 Agent 不可见的已求解 Transform。"""
        # The Agent is intentionally not allowed to author solved coordinates.
        # Omitting this hidden field lets Toolkit preserve it for existing entities
        # while keeping it unresolved for genuinely new entities.
        payload = [
            item.model_dump(mode="json", exclude_unset=True) for item in upserts
        ]
        arguments = {"upserts": payload, "remove_ids": remove_ids}
        return ctx.deps.call_tool(
            "apply_entity_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_entity_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_manual_mutation_tool)
    async def apply_constraint_patch(
        ctx: RunContext[PlanningDeps],
        upserts: list[ConstraintPatchInput],
        remove_ids: list[str],
    ) -> dict[str, Any]:
        """原子新增、更新或删除声明式客观约束。"""
        arguments = {
            "upserts": [item.model_dump(mode="json") for item in upserts],
            "remove_ids": remove_ids,
        }
        return ctx.deps.call_tool(
            "apply_constraint_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_constraint_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_manual_mutation_tool)
    async def apply_motion_patch(
        ctx: RunContext[PlanningDeps],
        upserts: list[TrackPatchInput],
        remove_ids: list[str],
    ) -> dict[str, Any]:
        """原子创建、更新或删除实体运动与可见性轨道。"""
        arguments = {
            "upserts": [item.to_domain_payload() for item in upserts],
            "remove_ids": remove_ids,
        }
        return ctx.deps.call_tool(
            "apply_motion_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_motion_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_manual_mutation_tool)
    async def apply_camera_patch(
        ctx: RunContext[PlanningDeps],
        camera_id: str,
        projection: Literal["perspective"],
        active: bool,
        static: CameraStatic,
        tracks: list[TrackPatchInput],
        remove_track_ids: list[str],
    ) -> dict[str, Any]:
        """创建或修改活动透视摄影机及其运动、观察和焦距轨道。"""
        arguments = {
            "camera_id": camera_id,
            "projection": projection,
            "active": active,
            "static": static.model_dump(mode="json"),
            "tracks": [item.to_domain_payload() for item in tracks],
            "remove_track_ids": remove_track_ids,
        }
        return ctx.deps.call_tool(
            "apply_camera_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_camera_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_candidate_patch_tool)
    async def apply_candidate_patch(
        ctx: RunContext[PlanningDeps],
        base_revision: int,
        entity_upserts: list[EntityPatchInput] | None = None,
        entity_remove_ids: list[str] | None = None,
        constraint_upserts: list[ConstraintPatchInput] | None = None,
        constraint_remove_ids: list[str] | None = None,
        motion_upserts: list[TrackPatchInput] | None = None,
        motion_remove_ids: list[str] | None = None,
        camera_patch: CameraPatchInput | None = None,
    ) -> dict[str, Any]:
        """必要时在一个预演事务中组合实体、约束、运动和摄影机修复。"""

        arguments = {
            "base_revision": base_revision,
            "entity_upserts": [
                item.model_dump(mode="json", exclude_unset=True)
                for item in (entity_upserts or [])
            ],
            "entity_remove_ids": entity_remove_ids or [],
            "constraint_upserts": [
                item.model_dump(mode="json") for item in (constraint_upserts or [])
            ],
            "constraint_remove_ids": constraint_remove_ids or [],
            "motion_upserts": [
                item.to_domain_payload() for item in (motion_upserts or [])
            ],
            "motion_remove_ids": motion_remove_ids or [],
            "camera_patch": (
                camera_patch.to_tool_payload() if camera_patch is not None else None
            ),
        }
        return ctx.deps.call_tool(
            "apply_candidate_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_candidate_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_escalated_candidate_tool)
    async def solve_candidate(
        ctx: RunContext[PlanningDeps],
        scope: Literal["layout", "camera", "all"] = "all",
        constraint_ids: list[str] | None = None,
        locked_variables: list[str] | None = None,
        strategy: Literal["auto", "heuristic"] = "auto",
    ) -> dict[str, Any]:
        """用当前确定性启发式求解器补齐未定布局和摄影机变量。"""
        arguments = {
            "scope": scope,
            "constraint_ids": constraint_ids or [],
            "locked_variables": locked_variables or [],
            "strategy": strategy,
        }
        return ctx.deps.call_tool(
            "solve_candidate",
            arguments,
            lambda: ctx.deps.toolkit.solve_candidate(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
    async def validate_candidate(
        ctx: RunContext[PlanningDeps],
        revision: int | None = None,
        checks: list[
            Literal[
                "schema",
                "references",
                "timeline",
                "hierarchy",
                "transforms",
                "projection",
                "composition",
                "visibility",
                "motion",
                "camera",
                "hard_semantics",
                "rebuildability",
            ]
        ]
        | None = None,
        sampling_profile: Literal["research_default"] = "research_default",
    ) -> dict[str, Any]:
        """确定性验证 Candidate，并返回稳定错误码和数值证据。"""
        arguments = {
            "revision": revision,
            "checks": checks or [],
            "sampling_profile": sampling_profile,
        }
        return ctx.deps.call_tool(
            "validate_candidate",
            arguments,
            lambda: ctx.deps.toolkit.validate_candidate(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_repair_suggestion_tool)
    async def suggest_repairs(
        ctx: RunContext[PlanningDeps],
        revision: int | None = None,
        violation_ids: list[str] | None = None,
        preference: Literal[
            "balanced",
            "maximize_motion_readability",
            "minimize_change",
            "preserve_composition",
        ] = "balanced",
        max_options: int = 3,
    ) -> dict[str, Any]:
        """为投影或机位 violation 搜索一至三个经复验的整体策略，不修改 Candidate。"""
        arguments = {
            "revision": revision,
            "violation_ids": violation_ids or [],
            "preference": preference,
            "max_options": max_options,
        }
        return ctx.deps.call_tool(
            "suggest_repairs",
            arguments,
            lambda: ctx.deps.toolkit.suggest_repairs(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_repair_apply_tool)
    async def apply_repair(
        ctx: RunContext[PlanningDeps],
        base_revision: int,
        suggestion_id: str,
    ) -> dict[str, Any]:
        """按 suggestion_id 原子应用具体数值并复验；只接受未过期的当前 revision。"""
        arguments = {
            "base_revision": base_revision,
            "suggestion_id": suggestion_id,
        }
        return ctx.deps.call_tool(
            "apply_repair",
            arguments,
            lambda: ctx.deps.toolkit.apply_repair(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
    async def restore_candidate(
        ctx: RunContext[PlanningDeps],
        source_revision: int,
        reason: str,
    ) -> dict[str, Any]:
        """把历史 revision 复制成新的当前 revision，不删除失败历史。"""
        arguments = {"source_revision": source_revision, "reason": reason}
        return ctx.deps.call_tool(
            "restore_candidate",
            arguments,
            lambda: ctx.deps.toolkit.restore_candidate(**arguments),
        )

    return agent
