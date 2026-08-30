from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition

from cinescaffold.planning.domain import (
    AgentTerminal,
    CameraStatic,
    CandidateState,
    ConstraintSpec,
    ProxyGeometry,
    StrictModel,
    TrackSpec,
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


@dataclass
class PlanningDeps:
    toolkit: ScenePlanningToolkit
    trace: TraceRecorder
    deadline_monotonic: float
    checkpoint_writer: Callable[[CandidateState], Path] | None = None
    capabilities_read: bool = False
    inspected_calls: set[str] = field(default_factory=set)

    def call_tool(self, name: str, arguments: dict[str, Any], operation) -> dict[str, Any]:
        if time.monotonic() >= self.deadline_monotonic:
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
                "能力清单在本次上下文中已经读取，不得重复调用",
                ["使用已有能力结果继续构造或提交"],
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
    # 能力读取后从模型工具表移除，避免重复调用浪费请求。
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
        upserts: list[ConstraintSpec],
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
        upserts: list[TrackSpec],
        remove_ids: list[str],
    ) -> dict[str, Any]:
        """原子创建、更新或删除实体运动与可见性轨道。"""
        arguments = {
            "upserts": [item.model_dump(mode="json") for item in upserts],
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
        tracks: list[TrackSpec],
        remove_track_ids: list[str],
    ) -> dict[str, Any]:
        """创建或修改活动透视摄影机及其运动、观察和焦距轨道。"""
        arguments = {
            "camera_id": camera_id,
            "projection": projection,
            "active": active,
            "static": static.model_dump(mode="json"),
            "tracks": [item.model_dump(mode="json") for item in tracks],
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
