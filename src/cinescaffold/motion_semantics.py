from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

PlanningMotionKind = Literal[
    "hold",
    "path_move",
    "carried",
    "local_transform",
]
PlanningDirectionMode = Literal[
    "none",
    "world_forward",
    "toward_target",
    "away_from_target",
    "world_left",
    "world_right",
    "relative_to_target",
]
PlanningPathFamily = Literal[
    "stationary",
    "linear",
    "circle",
    "ellipse",
    "catmull_rom",
    "lemniscate",
    "parabolic",
]


@dataclass(frozen=True)
class PlanningMotionShape:
    """Canonical Planning representation of one typed Semantic motion."""

    kind: PlanningMotionKind
    direction_mode: PlanningDirectionMode
    target_id: str | None
    carrier_id: str | None
    path_family: PlanningPathFamily


def planning_motion_shape(semantics: dict[str, Any]) -> PlanningMotionShape | None:
    """Translate the closed Semantic vocabulary into Planning's closed vocabulary.

    ``None`` is reserved for legacy Briefs without a typed ``motion_mode``. Invalid
    combinations raise here so every builder and validator shares one contract.
    """

    motion_mode = semantics.get("motion_mode")
    direction_mode = semantics.get("direction_mode") or "none"
    target_id = semantics.get("target_id")
    carrier_id = semantics.get("carrier_id")
    path_type = semantics.get("path_type")

    if motion_mode is None:
        return None
    if motion_mode == "stationary":
        return PlanningMotionShape("hold", "none", None, None, "stationary")
    if motion_mode == "local_interaction":
        return PlanningMotionShape("local_transform", "none", None, None, "stationary")
    if motion_mode == "carried":
        return PlanningMotionShape(
            "carried",
            "none",
            None,
            carrier_id if isinstance(carrier_id, str) else None,
            "stationary",
        )
    if motion_mode != "self_propelled":
        raise ValueError(f"未知 motion_mode：{motion_mode}")

    if path_type in {"circular", "elliptical"}:
        if direction_mode != "relative_to_target" or not isinstance(target_id, str):
            raise ValueError("相对闭合路径必须提供 relative_to_target 与 target_id")
        return PlanningMotionShape(
            "path_move",
            "relative_to_target",
            target_id,
            None,
            "ellipse" if path_type == "elliptical" else "circle",
        )

    if direction_mode == "relative_to_target":
        raise ValueError("relative_to_target 只用于 circular/elliptical 闭合路径")
    if direction_mode not in {
        "none",
        "world_forward",
        "toward_target",
        "away_from_target",
    }:
        raise ValueError(f"自主运动不支持 direction_mode={direction_mode}")
    if direction_mode in {"toward_target", "away_from_target"} and not isinstance(
        target_id, str
    ):
        raise ValueError(f"{direction_mode} 必须提供 target_id")
    if direction_mode in {"none", "world_forward"}:
        target_id = None

    path_family = {
        "s_curve": "catmull_rom",
        "figure_eight": "lemniscate",
        "parabolic": "parabolic",
        "linear": "linear",
        "unspecified": "linear",
        None: "linear",
    }.get(path_type)
    if path_family is None:
        raise ValueError(f"自主运动不支持 path_type={path_type}")
    return PlanningMotionShape(
        "path_move",
        direction_mode,
        target_id if isinstance(target_id, str) else None,
        None,
        path_family,
    )
