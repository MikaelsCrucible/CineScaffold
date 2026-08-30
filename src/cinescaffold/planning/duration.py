from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from cinescaffold.planning.domain import (
    DurationResolution,
    ExactDurationRequest,
    InferredDurationRequest,
    RangeDurationRequest,
    StrictModel,
)
from cinescaffold.planning.objective import ObjectivePlanningBrief


class DurationProposal(StrictModel):
    duration_seconds: float = Field(gt=0)
    reason: str = Field(min_length=1, max_length=240)


@dataclass(frozen=True)
class DurationResolverDeps:
    minimum_seconds: float | None
    maximum_seconds: float | None


def create_duration_resolver_agent(
    model: Model,
    system_prompt: str,
) -> Agent[DurationResolverDeps, DurationProposal]:
    agent: Agent[DurationResolverDeps, DurationProposal] = Agent(
        model,
        output_type=DurationProposal,
        deps_type=DurationResolverDeps,
        instructions=system_prompt,
        name="cinescaffold_duration_resolver",
        retries=2,
    )

    @agent.output_validator
    async def validate_duration(
        ctx: RunContext[DurationResolverDeps],
        proposal: DurationProposal,
    ) -> DurationProposal:
        # 用户给出范围时，模型必须在范围内选择，不能静默裁剪。
        if ctx.deps.minimum_seconds is not None and proposal.duration_seconds < ctx.deps.minimum_seconds:
            raise ModelRetry(f"时长不得短于 {ctx.deps.minimum_seconds:g} 秒")
        if ctx.deps.maximum_seconds is not None and proposal.duration_seconds > ctx.deps.maximum_seconds:
            raise ModelRetry(f"时长不得长于 {ctx.deps.maximum_seconds:g} 秒")
        return proposal

    return agent


def duration_request(
    timeline: dict[str, Any],
) -> tuple[Literal["exact", "range", "inferred"], float | None, float | None]:
    exact = timeline.get("duration_seconds")
    duration_range = timeline.get("duration_range_seconds")
    if exact is not None and duration_range is not None:
        raise ValueError("timeline 不能同时给出精确时长和时长范围")
    if exact is not None:
        if (
            not isinstance(exact, (int, float))
            or isinstance(exact, bool)
            or not math.isfinite(exact)
            or exact <= 0
        ):
            raise ValueError("timeline.duration_seconds 必须为有限正数")
        return "exact", float(exact), float(exact)
    if duration_range is not None:
        if not isinstance(duration_range, dict):
            raise ValueError("timeline.duration_range_seconds 必须为对象或 null")
        minimum = duration_range.get("minimum_seconds")
        maximum = duration_range.get("maximum_seconds")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in (minimum, maximum)
        ):
            raise ValueError("时长范围上下界必须为有限正数")
        if minimum > maximum:
            raise ValueError("时长范围下界不得大于上界")
        return "range", float(minimum), float(maximum)
    return "inferred", None, None


def duration_prompt(
    brief: ObjectivePlanningBrief,
    request_mode: str,
    minimum: float | None,
    maximum: float | None,
) -> str:
    payload = {
        "request_mode": request_mode,
        "minimum_seconds": minimum,
        "maximum_seconds": maximum,
        "objective_scene": {
            "subjects": brief.subjects,
            "subject_motion": brief.subject_motion,
            "scene_design": brief.scene_design,
            "composition": brief.composition,
            "camera": brief.camera,
            "events": brief.timeline.get("events", []),
        },
    }
    return "为这个单镜头客观场景选择足以表达动作与运镜的时长。\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def freeze_duration(
    *,
    request_mode: Literal["exact", "range", "inferred"],
    proposed_seconds: float,
    reason: str,
    fps_numerator: int,
    fps_denominator: int,
    minimum_seconds: float | None = None,
    maximum_seconds: float | None = None,
) -> DurationResolution:
    frame_rate = fps_numerator / fps_denominator
    if request_mode == "range":
        assert minimum_seconds is not None and maximum_seconds is not None
        minimum_frames = math.ceil(minimum_seconds * frame_rate - 1e-12)
        maximum_frames = math.floor(maximum_seconds * frame_rate + 1e-12)
        if minimum_frames > maximum_frames:
            raise ValueError("用户时长范围窄于一个可用帧采样点")
        frame_count = min(max(round(proposed_seconds * frame_rate), minimum_frames), maximum_frames)
    else:
        frame_count = max(1, round(proposed_seconds * frame_rate))
    resolved = frame_count / frame_rate
    method = {
        "exact": "user_exact",
        "range": "agent_within_user_range",
        "inferred": "agent_inferred",
    }[request_mode]
    request = (
        ExactDurationRequest(mode="exact", seconds=proposed_seconds)
        if request_mode == "exact"
        else RangeDurationRequest(
            mode="range",
            minimum_seconds=minimum_seconds,
            maximum_seconds=maximum_seconds,
        )
        if request_mode == "range"
        else InferredDurationRequest(mode="inferred")
    )
    return DurationResolution(
        request=request,
        resolution_method=method,
        proposed_duration_seconds=proposed_seconds,
        resolved_duration_seconds=resolved,
        frame_count=frame_count,
        reason=reason,
    )


def attach_duration_resolution(
    brief: ObjectivePlanningBrief,
    resolution: DurationResolution | dict[str, Any],
) -> ObjectivePlanningBrief:
    resolution = DurationResolution.model_validate(resolution)
    timeline = dict(brief.timeline)
    timeline["duration_resolution"] = resolution.model_dump(mode="json")
    return brief.model_copy(update={"timeline": timeline}, deep=True)
