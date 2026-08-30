from __future__ import annotations

import math
from typing import Any

from cinescaffold.planning.domain import BriefDurationRequest, DurationResolution
from cinescaffold.planning.objective import ObjectivePlanningBrief


def freeze_brief_duration(
    timeline: dict[str, Any],
    *,
    fps_numerator: int,
    fps_denominator: int,
) -> DurationResolution:
    """只冻结第一步已经解析完成的时长，不在规划层推断。"""
    duration = timeline.get("duration_seconds")
    source_status = timeline.get("duration_source_status")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError(
            "Cinematic Brief 必须在自然语言解析阶段给出正数 duration_seconds；"
            "规划层不再调用独立模型推断时长"
        )
    if source_status not in {"explicit", "inferred", "default"}:
        raise ValueError("已解析时长必须标记 explicit、inferred 或 default 来源")

    duration_range = timeline.get("duration_range_seconds")
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
        if minimum > maximum or not minimum <= duration <= maximum:
            raise ValueError("第一步解析出的 duration_seconds 必须位于用户时长范围内")

    frame_rate = fps_numerator / fps_denominator
    frame_count = max(1, round(float(duration) * frame_rate))
    resolved = frame_count / frame_rate
    return DurationResolution(
        request=BriefDurationRequest(
            mode="cinematic_brief",
            source_status=source_status,
            seconds=float(duration),
        ),
        resolution_method=f"brief_{source_status}",
        proposed_duration_seconds=float(duration),
        resolved_duration_seconds=resolved,
        frame_count=frame_count,
        reason="采用 Cinematic Brief 已解析的时长并按冻结 FPS 对齐。",
    )


def attach_duration_resolution(
    brief: ObjectivePlanningBrief,
    resolution: DurationResolution | dict[str, Any],
) -> ObjectivePlanningBrief:
    resolution = DurationResolution.model_validate(resolution)
    timeline = dict(brief.timeline)
    timeline["duration_resolution"] = resolution.model_dump(mode="json")
    return brief.model_copy(update={"timeline": timeline}, deep=True)
