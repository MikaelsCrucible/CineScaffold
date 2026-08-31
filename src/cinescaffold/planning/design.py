from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from cinescaffold.planning.domain import CandidateState, PlanningProfile, StrictModel
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.planning.store import canonical_hash


SourceStatus = Literal["explicit", "inferred", "default", "agent_selected"]


class SkeletonEntity(StrictModel):
    entity_id: str = Field(min_length=1)
    semantic_type: str = Field(min_length=1)
    role: str = Field(min_length=1)
    proxy_family: Literal[
        "ground_plane",
        "human_capsule",
        "vehicle_box",
        "celestial_sphere",
        "generic_box",
    ]
    scale_intent: Literal[
        "tiny", "small", "human", "large", "huge", "unspecified"
    ] = "unspecified"
    source_refs: list[str] = Field(default_factory=list)


class SkeletonRelation(StrictModel):
    relation_id: str = Field(min_length=1)
    kind: Literal[
        "ground_support",
        "camera_depth_order",
        "relative_position",
        "proximity",
        "scale_dominance",
        "orbit_around",
        "carried_by",
    ]
    subject_id: str
    reference_id: str
    direction: Literal[
        "left", "right", "front", "behind", "below", "above"
    ] | None = None
    timeline_event_id: str | None = None
    source_status: SourceStatus
    source_ref: str

    @model_validator(mode="after")
    def validate_relation_shape(self) -> SkeletonRelation:
        if self.kind == "relative_position" and self.direction is None:
            raise ValueError("relative_position 必须提供 direction")
        if self.kind != "relative_position" and self.direction is not None:
            raise ValueError(f"{self.kind} 不接受 direction")
        return self


class SkeletonMotionPhase(StrictModel):
    phase_id: str = Field(min_length=1)
    subject_id: str
    kind: Literal[
        "hold",
        "linear_move",
        "orbit",
        "board",
        "carried",
        "visibility",
    ]
    timeline_event_id: str | None = None
    target_id: str | None = None
    carrier_id: str | None = None
    direction_mode: Literal[
        "none",
        "toward_target",
        "away_from_target",
        "screen_left_to_right",
        "screen_right_to_left",
        "orbit_around",
    ] = "none"
    path_family: Literal[
        "stationary",
        "linear",
        "circle",
        "ellipse",
        "catmull_rom",
        "lemniscate",
    ] = "stationary"
    source_status: SourceStatus
    source_ref: str

    @model_validator(mode="after")
    def validate_motion_shape(self) -> SkeletonMotionPhase:
        if self.kind in {"orbit", "board"} and self.target_id is None:
            raise ValueError(f"{self.kind} 必须提供 target_id")
        if self.kind == "carried" and self.carrier_id is None:
            raise ValueError("carried 必须提供 carrier_id")
        if self.kind == "orbit" and self.path_family not in {"circle", "ellipse"}:
            raise ValueError("普通 orbit 必须使用 circle 或 ellipse")
        if self.kind == "hold" and self.path_family != "stationary":
            raise ValueError("hold 必须使用 stationary")
        return self


class SkeletonCameraIntent(StrictModel):
    movement: Literal[
        "static", "push_in", "pull_out", "follow", "orbit", "lateral"
    ]
    focus_target_id: str
    view_relation_to_motion: Literal[
        "front", "rear", "side", "three_quarter", "unspecified"
    ] = "unspecified"
    source_status: SourceStatus
    source_ref: str


class SceneSkeleton(StrictModel):
    entities: list[SkeletonEntity] = Field(min_length=1)
    relations: list[SkeletonRelation] = Field(default_factory=list)
    motion_phases: list[SkeletonMotionPhase] = Field(default_factory=list)
    camera_intent: SkeletonCameraIntent

    @model_validator(mode="after")
    def validate_references(self) -> SceneSkeleton:
        entity_ids = [item.entity_id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Scene Skeleton Entity ID 不得重复")
        known = set(entity_ids)
        if self.camera_intent.focus_target_id not in known:
            raise ValueError("摄影机观察目标不在 Scene Skeleton 中")
        for relation in self.relations:
            if relation.subject_id not in known or relation.reference_id not in known:
                raise ValueError(f"Relation 引用了未知 Entity：{relation.relation_id}")
        for phase in self.motion_phases:
            references = [phase.subject_id, phase.target_id, phase.carrier_id]
            if any(item is not None and item not in known for item in references):
                raise ValueError(f"Motion Phase 引用了未知 Entity：{phase.phase_id}")
        return self


@dataclass(frozen=True)
class DesignOption:
    option_id: str
    base_revision: int
    skeleton_hash: str
    strategy: str
    candidate: CandidateState
    numeric_envelopes: dict[str, Any]
    assumptions: tuple[str, ...]
    relevant_capabilities: dict[str, Any]
    predicted: dict[str, Any]


def validate_scene_skeleton(
    objective: ObjectivePlanningBrief,
    value: SceneSkeleton,
) -> None:
    """只验证符号结构与 Brief 身份，不在这里推断数值。"""

    objective_subject_ids = {
        str(item.get("id"))
        for item in objective.subjects
        if item.get("id") is not None
    }
    skeleton_ids = {item.entity_id for item in value.entities}
    missing = sorted(objective_subject_ids - skeleton_ids)
    if missing:
        raise ValueError(f"Scene Skeleton 遗漏 Brief 主体：{', '.join(missing)}")

    valid_explicit_refs = {item.path for item in objective.explicit_requirements}
    for source_ref, status in _skeleton_sources(value):
        if status == "explicit" and source_ref not in valid_explicit_refs:
            raise ValueError(f"Scene Skeleton explicit source_ref 不存在：{source_ref}")


def skeleton_hash(value: SceneSkeleton) -> str:
    return canonical_hash(value.model_dump(mode="json"))


def design_option_id(
    *,
    skeleton_sha256: str,
    base_revision: int,
    strategy: str,
    candidate: CandidateState,
) -> str:
    identity = {
        "skeleton_hash": skeleton_sha256,
        "base_revision": base_revision,
        "strategy": strategy,
        "candidate": candidate.model_dump(
            mode="json",
            exclude={"revision", "validation"},
        ),
    }
    return f"design_{canonical_hash(identity).removeprefix('sha256:')[:16]}"


def task_capability_slice(
    skeleton: SceneSkeleton,
    profile: PlanningProfile,
) -> dict[str, Any]:
    """只返回当前骨架会用到的约定，避免回传整本能力手册。"""

    path_families = sorted(
        {
            phase.path_family
            for phase in skeleton.motion_phases
            if phase.path_family != "stationary"
        }
    )
    relation_kinds = sorted({item.kind for item in skeleton.relations})
    return {
        "coordinate_system": {
            "linear_unit": "meter",
            "handedness": "right",
            "up_axis": "+Z",
            "screen_coordinates": "[0,1]，左上原点",
        },
        "timeline": {
            "time_domain": "half_open",
            "fps": profile.fps_numerator / profile.fps_denominator,
        },
        "required_relation_kinds": relation_kinds,
        "required_path_families": path_families,
        "acceptance": {
            "requires_hard_pass": True,
            "minimum_soft_score": profile.minimum_soft_score,
            "minimum_projected_motion_extent": profile.minimum_projected_motion_extent,
            "minimum_camera_motion_obliqueness_degrees": (
                profile.minimum_camera_motion_obliqueness_degrees
            ),
        },
        "next_tool": "apply_design_option",
    }


def _skeleton_sources(value: SceneSkeleton) -> list[tuple[str, SourceStatus]]:
    result: list[tuple[str, SourceStatus]] = []
    for entity in value.entities:
        result.extend((source_ref, "explicit") for source_ref in entity.source_refs)
    result.extend((item.source_ref, item.source_status) for item in value.relations)
    result.extend((item.source_ref, item.source_status) for item in value.motion_phases)
    result.append((value.camera_intent.source_ref, value.camera_intent.source_status))
    return result
