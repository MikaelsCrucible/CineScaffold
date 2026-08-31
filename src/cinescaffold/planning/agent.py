from __future__ import annotations

import json
import time
from collections.abc import Callable
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
from cinescaffold.planning.toolkit import TOOLKIT_VERSION, ScenePlanningToolkit
from cinescaffold.planning.trace import TraceRecorder


class EntityPatchInput(StrictModel):
    entity_id: str
    label: str | None = None
    role: str
    proxy: ProxyGeometry
    parent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    locked_fields: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    ground_interaction: GroundInteractionSpec = Field(default_factory=GroundInteractionSpec)


class ConstraintPatchInput(StrictModel):
    """保持工具 schema 紧凑，领域模型仍由 Toolkit 严格复验。"""

    constraint_id: str = Field(min_length=1)
    type: ConstraintKind
    strength: Literal["hard", "soft"]
    weight: float = Field(default=1.0, gt=0)
    subjects: list[str] = Field(default_factory=list)
    time_range_seconds: tuple[float, float]
    parameters: dict[str, Any] = Field(
        description="字段必须遵循 get_capabilities.constraint_parameter_schemas 中对应 type 的契约"
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
    locked_components: list[str] = Field(default_factory=list)
    source_ref: str | None = None

    def to_domain_payload(self) -> dict[str, Any]:
        # 省略未使用的变体字段，由关闭的领域联合补默认值并拒绝错配。
        return self.model_dump(mode="json", exclude_none=True)


MAX_AGENT_HISTORY_MESSAGES = 13
NON_TOOL_TEXT_MARKER = "[已省略不符合协议的无工具正文；请按重试指令调用工具。]"
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
        return result

    def _protocol_rejection(
        self,
        name: str,
        arguments: dict[str, Any],
        revision: int,
    ) -> dict[str, Any] | None:
        if name != "get_capabilities" and not self.capabilities_read:
            return _protocol_rejected(revision, "必须先读取一次 get_capabilities", ["调用 get_capabilities"])
        if name == "get_capabilities" and self.capabilities_read:
            return _protocol_rejected(
                revision,
                "能力清单已经读取，不得重复调用",
                ["使用已有能力结果继续构造 Candidate"],
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
    return None if ctx.deps.capabilities_read else tool_definition


async def _prepare_candidate_tool(
    ctx: RunContext[PlanningDeps],
    tool_definition: ToolDefinition,
) -> ToolDefinition | None:
    # 首次能力读取前只暴露能力工具；可提交后只允许结构化终止。
    if not ctx.deps.capabilities_read:
        return None
    validation = ctx.deps.toolkit.store.get().validation
    if (
        validation is not None
        and validation.hard_pass
        and validation.soft_score >= ctx.deps.toolkit.profile.minimum_soft_score
    ):
        return None
    # 实体创建前不暴露其他大 Schema，避免读取能力后的首次 Mutation 发生全局规划。
    if not ctx.deps.toolkit.store.get().entities and tool_definition.name != "apply_entity_patch":
        return None
    return tool_definition


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

    @agent.tool(sequential=True, prepare=_prepare_capabilities_tool)
    async def get_capabilities(
        ctx: RunContext[PlanningDeps],
    ) -> dict[str, Any]:
        """一次读取完整 Scene Planning 能力；随后先创建实体，再逐步构造与求解。"""
        sections = ["entities", "constraints", "tracks", "camera", "validators", "limits"]
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

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
    async def apply_entity_patch(
        ctx: RunContext[PlanningDeps],
        upserts: list[EntityPatchInput],
        remove_ids: list[str],
    ) -> dict[str, Any]:
        """原子创建、更新或删除刚性代理实体和静态层级。"""
        payload = [
            item.model_dump(mode="json") | {"solved_transform": {}}
            for item in upserts
        ]
        arguments = {"upserts": payload, "remove_ids": remove_ids}
        return ctx.deps.call_tool(
            "apply_entity_patch",
            arguments,
            lambda: ctx.deps.toolkit.apply_entity_patch(**arguments),
        )

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
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

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
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

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
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

    @agent.tool(sequential=True, prepare=_prepare_candidate_tool)
    async def solve_candidate(
        ctx: RunContext[PlanningDeps],
        scope: Literal["layout", "camera", "motion", "all"] = "all",
        constraint_ids: list[str] | None = None,
        allowed_variables: list[str] | None = None,
        locked_variables: list[str] | None = None,
        profile: Literal["research_default"] = "research_default",
        strategy: Literal["auto", "heuristic", "numeric", "hybrid"] = "auto",
    ) -> dict[str, Any]:
        """确定性求解未定布局、摄影机和运动变量。"""
        arguments = {
            "scope": scope,
            "constraint_ids": constraint_ids or [],
            "allowed_variables": allowed_variables or [],
            "locked_variables": locked_variables or [],
            "profile": profile,
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
