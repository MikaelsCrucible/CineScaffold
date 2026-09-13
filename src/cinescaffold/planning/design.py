from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from cinescaffold.planning.domain import (
    CameraCandidate,
    CameraStatic,
    CandidateState,
    ConstraintSpec,
    EntitySpec,
    GroundInteractionSpec,
    PlanningProfile,
    StrictModel,
    TrackKeyframe,
    TrackSpec,
    TransformValue,
)
from cinescaffold.planning.geometry import (
    look_at_camera_quaternion,
    normalize,
    sample_transform_track,
    surface_clearance_target_distance,
)
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
    ] = Field(
        default="unspecified",
        description="不含米制数值的绝对尺度档位；语义上可判断大小时不得全部留空",
    )
    proportion_intent: Literal[
        "isotropic", "flat", "wide", "tall", "elongated", "unspecified"
    ] = Field(
        default="unspecified",
        description="不含米制数值的三轴比例意图；薄表面使用 flat",
    )
    source_refs: list[str] = Field(default_factory=list)


class EntitySizeRequest(StrictModel):
    entity_id: str = Field(min_length=1)
    minimum_xyz_m: tuple[float, float, float] = Field(
        description="X/Y/Z 三轴完整包围盒尺寸下界，单位米"
    )
    maximum_xyz_m: tuple[float, float, float] = Field(
        description="X/Y/Z 三轴完整包围盒尺寸上界，单位米"
    )
    preferred_xyz_m: tuple[float, float, float] | None = Field(
        default=None,
        description="可选偏好尺寸，必须位于上下界内",
    )
    rationale: str = Field(
        min_length=1,
        description="说明为何内置尺度与比例意图不足，便于 Trace 审计",
    )

    @model_validator(mode="after")
    def validate_range(self) -> EntitySizeRequest:
        for axis, (minimum, maximum) in enumerate(
            zip(self.minimum_xyz_m, self.maximum_xyz_m)
        ):
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError("自定义尺寸范围必须是有限数")
            if minimum <= 0 or maximum <= 0 or minimum > maximum:
                raise ValueError(f"自定义尺寸第 {axis + 1} 轴范围无效")
        if self.preferred_xyz_m is not None:
            for axis, (preferred, minimum, maximum) in enumerate(
                zip(
                    self.preferred_xyz_m,
                    self.minimum_xyz_m,
                    self.maximum_xyz_m,
                )
            ):
                if (
                    not math.isfinite(preferred)
                    or not minimum <= preferred <= maximum
                ):
                    raise ValueError(f"自定义尺寸第 {axis + 1} 轴偏好值越界")
        return self


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
    temporal_mode: Literal["throughout", "at_start", "at_end"] = "throughout"
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
    motion_id: str | None = Field(
        default=None,
        description="Cinematic Brief v0.6 的稳定动作身份，用于叙事完整性校验",
    )
    subject_id: str
    kind: Literal[
        "hold",
        "linear_move",
        "orbit",
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
    speed_intent: Literal[
        "stationary", "slow", "medium", "fast", "unspecified"
    ] = "unspecified"
    speed_source_status: SourceStatus | None = None
    speed_source_ref: str | None = None
    visibility_state: Literal["visible", "hidden"] | None = None
    transition_at: Literal["at_start", "at_end"] | None = None
    source_status: SourceStatus
    source_ref: str
    narrative_required: bool = False

    @model_validator(mode="after")
    def validate_motion_shape(self) -> SkeletonMotionPhase:
        if self.kind == "orbit" and self.target_id is None:
            raise ValueError(f"{self.kind} 必须提供 target_id")
        if self.kind == "carried" and self.carrier_id is None:
            raise ValueError("carried 必须提供 carrier_id")
        if self.kind == "orbit" and self.path_family not in {"circle", "ellipse"}:
            raise ValueError("普通 orbit 必须使用 circle 或 ellipse")
        if self.kind == "hold" and self.path_family != "stationary":
            raise ValueError("hold 必须使用 stationary")
        visibility_fields = (self.visibility_state, self.transition_at)
        if self.kind == "visibility" and any(item is None for item in visibility_fields):
            raise ValueError("visibility 必须提供 visibility_state 与 transition_at")
        if self.kind == "visibility" and (
            self.path_family != "stationary"
            or self.direction_mode != "none"
            or self.speed_intent != "unspecified"
        ):
            raise ValueError("visibility 只能改变可见性，不接受路径、方向或速度")
        if self.kind != "visibility" and any(item is not None for item in visibility_fields):
            raise ValueError(f"{self.kind} 不接受 visibility_state 或 transition_at")
        if (self.speed_source_status is None) != (self.speed_source_ref is None):
            raise ValueError("speed_source_status 与 speed_source_ref 必须同时提供")
        return self


class SkeletonRouteAnchor(StrictModel):
    anchor_id: str = Field(min_length=1)
    phase_id: str = Field(
        min_length=1,
        description="路径点所属的线性运动阶段",
    )
    boundary: Literal["at_start", "at_end"] = Field(
        description="路径点约束该运动阶段的起点或终点",
    )
    relation_id: str = Field(
        min_length=1,
        description="该路径点必须满足的既有 proximity 或 relative_position 关系",
    )


class SkeletonRouteIntent(StrictModel):
    route_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    anchors: list[SkeletonRouteAnchor] = Field(
        min_length=1,
        description="按主体时间线排列的稀疏符号路径点，不填写米制坐标",
    )
    axis_reference_id: str | None = Field(
        default=None,
        description="可选的既有空间参照实体；Toolkit 使用其水平长轴作为路线主轴",
    )
    continuity: Literal["preserve_direction", "allow_turns"] = Field(
        default="preserve_direction",
        description="跨阶段保持总体前进方向，或允许由多个路径点产生转向",
    )


class SkeletonCameraIntent(StrictModel):
    movement: Literal[
        "static", "push_in", "pull_out", "follow", "orbit", "lateral"
    ]
    focus_target_id: str
    view_relation_to_motion: Literal[
        "front", "rear", "side", "three_quarter", "unspecified"
    ] = "unspecified"
    speed_intent: Literal[
        "stationary", "slow", "medium", "fast", "unspecified"
    ] = "unspecified"
    speed_source_status: SourceStatus | None = None
    speed_source_ref: str | None = None
    source_status: SourceStatus
    source_ref: str

    @model_validator(mode="after")
    def validate_speed_source(self) -> SkeletonCameraIntent:
        if (self.speed_source_status is None) != (self.speed_source_ref is None):
            raise ValueError("speed_source_status 与 speed_source_ref 必须同时提供")
        return self


class SceneSkeleton(StrictModel):
    entities: list[SkeletonEntity] = Field(min_length=1)
    relations: list[SkeletonRelation] = Field(default_factory=list)
    motion_phases: list[SkeletonMotionPhase] = Field(default_factory=list)
    route_intents: list[SkeletonRouteIntent] = Field(
        default_factory=list,
        description="由 Planning Agent 选择的稀疏符号路径；Toolkit 再求解实际坐标",
    )
    camera_intent: SkeletonCameraIntent

    @model_validator(mode="after")
    def validate_references(self) -> SceneSkeleton:
        entity_ids = [item.entity_id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Scene Skeleton Entity ID 不得重复")
        known = set(entity_ids)
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("Scene Skeleton Relation ID 不得重复")
        phase_ids = [item.phase_id for item in self.motion_phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("Scene Skeleton Motion Phase ID 不得重复")
        route_ids = [item.route_id for item in self.route_intents]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("Scene Skeleton Route ID 不得重复")
        route_subjects = [item.subject_id for item in self.route_intents]
        if len(route_subjects) != len(set(route_subjects)):
            raise ValueError("同一主体只能提交一条 Route Intent")
        if self.camera_intent.focus_target_id not in known:
            raise ValueError("摄影机观察目标不在 Scene Skeleton 中")
        proxy_families = {item.entity_id: item.proxy_family for item in self.entities}
        ground_subjects: dict[str, str] = {}
        for relation in self.relations:
            if relation.subject_id not in known or relation.reference_id not in known:
                raise ValueError(f"Relation 引用了未知 Entity：{relation.relation_id}")
            if relation.subject_id == relation.reference_id:
                raise ValueError(f"Relation 不得自引用：{relation.relation_id}")
            if relation.kind == "ground_support":
                if proxy_families[relation.reference_id] != "ground_plane":
                    raise ValueError(
                        f"ground_support 必须指向 ground_plane：{relation.relation_id}"
                    )
                previous_ground = ground_subjects.get(relation.subject_id)
                if previous_ground is not None and previous_ground != relation.reference_id:
                    raise ValueError(
                        f"同一实体的 ground_support 不得指向多个地面：{relation.subject_id}"
                    )
                ground_subjects[relation.subject_id] = relation.reference_id
        for phase in self.motion_phases:
            references = [phase.subject_id, phase.target_id, phase.carrier_id]
            if any(item is not None and item not in known for item in references):
                raise ValueError(f"Motion Phase 引用了未知 Entity：{phase.phase_id}")
        phases_by_id = {item.phase_id: item for item in self.motion_phases}
        relations_by_id = {item.relation_id: item for item in self.relations}
        for route in self.route_intents:
            if route.subject_id not in known:
                raise ValueError(f"Route Intent 引用了未知主体：{route.route_id}")
            if route.axis_reference_id is not None:
                if route.axis_reference_id not in known:
                    raise ValueError(f"Route Intent 引用了未知路线参照：{route.route_id}")
                if route.axis_reference_id == route.subject_id:
                    raise ValueError(f"Route Intent 不得把运动主体用作路线参照：{route.route_id}")
            anchor_ids = [item.anchor_id for item in route.anchors]
            if len(anchor_ids) != len(set(anchor_ids)):
                raise ValueError(f"Route Anchor ID 不得重复：{route.route_id}")
            phase_boundaries = [(item.phase_id, item.boundary) for item in route.anchors]
            if len(phase_boundaries) != len(set(phase_boundaries)):
                raise ValueError(f"同一运动阶段边界只能有一个 Route Anchor：{route.route_id}")
            for anchor in route.anchors:
                phase = phases_by_id.get(anchor.phase_id)
                if phase is None:
                    raise ValueError(f"Route Anchor 引用了未知 Motion Phase：{anchor.anchor_id}")
                if phase.subject_id != route.subject_id or phase.kind != "linear_move":
                    raise ValueError(
                        f"Route Anchor 只能绑定同一主体的 linear_move：{anchor.anchor_id}"
                    )
                relation = relations_by_id.get(anchor.relation_id)
                if relation is None:
                    raise ValueError(f"Route Anchor 引用了未知 Relation：{anchor.anchor_id}")
                if relation.kind not in {"proximity", "relative_position"}:
                    raise ValueError(
                        f"Route Anchor 只支持可落为路径点的空间关系：{anchor.anchor_id}"
                    )
                if route.subject_id not in {relation.subject_id, relation.reference_id}:
                    raise ValueError(
                        f"Route Anchor 的空间关系不包含运动主体：{anchor.anchor_id}"
                    )
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

    known_event_ids = {
        str(item.get("id"))
        for item in objective.timeline.get("events", [])
        if item.get("id") is not None
    }
    referenced_event_ids = {
        item.timeline_event_id
        for item in [*value.relations, *value.motion_phases]
        if item.timeline_event_id is not None
    }
    unknown_events = sorted(referenced_event_ids - known_event_ids)
    if unknown_events:
        raise ValueError(f"Scene Skeleton 引用了未知事件：{', '.join(unknown_events)}")

    if objective.schema_version == "0.6":
        objective_motion_subjects = {
            str(item.get("motion_id")): str(item.get("subject_id"))
            for item in objective.subject_motion
            if isinstance(item, dict) and item.get("motion_id") is not None
        }
        for phase in value.motion_phases:
            if phase.motion_id is None:
                continue
            expected_subject = objective_motion_subjects.get(phase.motion_id)
            if expected_subject is None:
                raise ValueError(
                    f"Scene Skeleton 引用了未知 motion_id：{phase.motion_id}"
                )
            if expected_subject != phase.subject_id:
                raise ValueError(
                    f"Scene Skeleton motion_id 与主体不一致：{phase.motion_id}"
                )
        required_motion_ids = {
            str(item.get("motion_id"))
            for item in objective.subject_motion
            if isinstance(item, dict)
            and isinstance(item.get("motion_semantics"), dict)
            and item["motion_semantics"].get("narrative_required") is True
        }
        skeleton_motion_ids = {
            item.motion_id
            for item in value.motion_phases
            if item.motion_id is not None
        }
        missing_motion_ids = sorted(required_motion_ids - skeleton_motion_ids)
        if missing_motion_ids:
            raise ValueError(
                "Scene Skeleton 遗漏关键叙事动作："
                + ", ".join(missing_motion_ids)
            )

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
    speed_intents = {
        item.speed_intent
        for item in skeleton.motion_phases
        if item.speed_intent not in {"stationary", "unspecified"}
    }
    if skeleton.camera_intent.speed_intent not in {"stationary", "unspecified"}:
        speed_intents.add(skeleton.camera_intent.speed_intent)
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
        "route_planning": {
            "decision_owner": "planning_agent",
            "coordinate_owner": "toolkit",
            "anchor_relations": ["proximity", "relative_position"],
            "anchor_boundaries": ["at_start", "at_end"],
            "continuity_modes": ["preserve_direction", "allow_turns"],
            "axis_reference": "optional entity horizontal long axis",
        },
        "speed_ranges_mps": {
            intent: list(_speed_range(intent, profile))
            for intent in sorted(speed_intents)
        },
        "size_design": {
            "scale_intents": [
                "tiny", "small", "human", "large", "huge", "unspecified"
            ],
            "proportion_intents": [
                "isotropic", "flat", "wide", "tall", "elongated", "unspecified"
            ],
            "custom_size_requests": {
                "unit": "meter",
                "value": "三轴完整包围盒尺寸范围，不是半尺寸",
                "selection": "Toolkit 按候选策略在范围内选择并完整验证",
            },
        },
        "inspect_views": [
            "summary",
            "entities",
            "camera",
            "constraints",
            "violations",
            "timeline",
            "diff",
            "full_ir",
        ],
        "acceptance": {
            "requires_hard_pass": True,
            "minimum_soft_score": profile.minimum_soft_score,
            "soft_score_role": "advisory_quality_metric",
            "minimum_projected_motion_extent": profile.minimum_projected_motion_extent,
            "minimum_camera_motion_obliqueness_degrees": (
                profile.minimum_camera_motion_obliqueness_degrees
            ),
        },
        "next_tool": "apply_design_option",
    }


def build_design_candidate(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    base: CandidateState,
    profile: PlanningProfile,
    strategy: Literal[
        "balanced", "preserve_composition", "maximize_motion_readability"
    ],
    size_requests: dict[str, EntitySizeRequest] | None = None,
) -> tuple[CandidateState, dict[str, Any], tuple[str, ...]]:
    """把符号骨架物化为确定性数值候选，不让模型填写坐标。"""

    candidate = base.model_copy(deep=True)
    candidate.entities = _build_entities(
        objective,
        skeleton,
        strategy,
        size_requests or {},
    )
    _place_entities(candidate, skeleton, profile)
    candidate.constraints = _build_relation_constraints(
        objective,
        skeleton,
        candidate,
        profile,
    )
    candidate.motion_tracks = _build_motion(
        objective,
        skeleton,
        candidate,
        profile,
    )
    candidate.camera = _build_camera(
        objective,
        skeleton,
        candidate,
        profile,
        strategy,
    )
    _add_speed_constraints(objective, skeleton, candidate, profile)
    _add_composition_constraints(objective, skeleton, candidate)
    assumptions = _design_assumptions(
        objective,
        skeleton,
        strategy,
        size_requests or {},
    )
    return (
        candidate,
        _numeric_envelopes(
            objective,
            skeleton,
            candidate,
            size_requests or {},
        ),
        assumptions,
    )


def _build_entities(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    strategy: str,
    size_requests: dict[str, EntitySizeRequest],
) -> dict[str, EntitySpec]:
    subject_parameters = _subject_parameters(objective)
    ground_links = {
        item.subject_id: item
        for item in skeleton.relations
        if item.kind == "ground_support"
    }
    entities: dict[str, EntitySpec] = {}
    for item in skeleton.entities:
        parameters = subject_parameters.get(item.entity_id, {})
        source_refs = set(item.source_refs)
        source_refs.update(_entity_explicit_refs(objective, item.entity_id))
        if item.proxy_family == "ground_plane":
            source_refs.update(
                requirement.path
                for requirement in objective.explicit_requirements
                if requirement.path.startswith("content.scene_design.environment")
            )
        ground = ground_links.get(item.entity_id)
        ground_interaction = GroundInteractionSpec(
            mode="must_touch" if ground else "must_be_above",
            ground_entity_id=ground.reference_id if ground else None,
            source_status=(ground.source_status if ground else "default"),
            source_ref=(ground.source_ref if ground and ground.source_status == "explicit" else None),
        )
        tags = [item.semantic_type, item.scale_intent]
        if item.proxy_family == "ground_plane":
            # Downstream mutation and validation gates use this language-neutral
            # marker to distinguish an environment ground from an arbitrary plane.
            # Skeleton roles and semantic labels are user/model-authored and may be
            # in any language, so they cannot be the sole identity contract.
            tags.extend(["environment", "ground"])
        entities[item.entity_id] = EntitySpec(
            entity_id=item.entity_id,
            label=item.semantic_type,
            role=item.role,
            proxy=_proxy_geometry(
                objective,
                item,
                parameters,
                strategy,
                size_requests.get(item.entity_id),
            ),
            tags=list(dict.fromkeys(tags)),
            source_refs=sorted(source_refs),
            ground_interaction=ground_interaction,
        )
    return entities


def _proxy_geometry(
    objective: ObjectivePlanningBrief,
    entity: SkeletonEntity,
    parameters: dict[str, Any],
    strategy: str,
    size_request: EntitySizeRequest | None,
) -> dict[str, Any]:
    if size_request is not None:
        dimensions = _select_requested_dimensions(size_request, strategy)
        _validate_requested_dimensions(entity, parameters, dimensions)
        if entity.proxy_family == "celestial_sphere":
            return {"type": "sphere", "radius_m": sum(dimensions) / 6.0}
        if entity.proxy_family == "human_capsule":
            radius = (dimensions[0] + dimensions[1]) / 4.0
            return {
                "type": "capsule",
                "radius_m": radius,
                "segment_length_m": max(0.01, dimensions[2] - radius * 2.0),
                "axis": "+Z",
            }
        if entity.proxy_family in {"vehicle_box", "generic_box"}:
            return {"type": "box", "size_xyz_m": list(dimensions)}
    if entity.proxy_family == "ground_plane":
        dimensions = (
            (objective.translation_parameters or {}).get("scene", {}).get("dimensions_m")
            or [100.0, 100.0]
        )
        return {"type": "plane", "size_xy_m": [float(dimensions[0]), float(dimensions[1])]}
    if entity.proxy_family == "human_capsule":
        height = float(parameters.get("reference_height_m") or 1.75)
        radius = max(0.2, height * 0.17)
        return {
            "type": "capsule",
            "radius_m": radius,
            "segment_length_m": max(0.2, height - radius * 2.0),
            "axis": "+Z",
        }
    if entity.proxy_family == "celestial_sphere":
        radius = {
            "tiny": 0.35,
            "small": 0.6,
            "human": 0.9,
            "large": 1.8,
            "huge": 3.2,
            "unspecified": 1.0,
        }[entity.scale_intent]
        return {"type": "sphere", "radius_m": radius}
    if entity.proxy_family == "vehicle_box":
        footprint = parameters.get("minimum_footprint_m")
        if isinstance(footprint, list) and len(footprint) == 2:
            multiplier = parameters.get("dominant_scale_multiplier_range") or [1.0, 1.0]
            scale = (float(multiplier[0]) + float(multiplier[1])) / 2.0
            width = max(4.5, float(footprint[0]) * scale)
            depth = max(2.0, float(footprint[1]) * scale)
            height = max(1.6, min(width, depth) * 0.45)
        elif entity.scale_intent == "huge":
            width, depth, height = 30.0, 12.0, 8.0
        else:
            width, depth, height = 4.5, 2.0, 1.6
        return {"type": "box", "size_xyz_m": [width, depth, height]}
    edge = {
        "tiny": 0.5,
        "small": 1.0,
        "human": 1.75,
        "large": 4.0,
        "huge": 12.0,
        "unspecified": 2.0,
    }[entity.scale_intent]
    dimensions = {
        "isotropic": (edge, edge, edge),
        "flat": (edge * 4.0, edge * 2.0, max(0.05, edge * 0.025)),
        "wide": (edge * 2.5, edge * 1.5, edge),
        "tall": (edge, edge, edge * 3.0),
        "elongated": (edge * 3.0, edge, edge),
        "unspecified": (edge, edge, edge),
    }[entity.proportion_intent]
    return {"type": "box", "size_xyz_m": list(dimensions)}


def _select_requested_dimensions(
    request: EntitySizeRequest,
    strategy: str,
) -> tuple[float, float, float]:
    if request.preferred_xyz_m is not None:
        return request.preferred_xyz_m
    ratio = {
        "preserve_composition": 0.25,
        "balanced": 0.5,
        "maximize_motion_readability": 0.75,
    }[strategy]
    return tuple(
        minimum + (maximum - minimum) * ratio
        for minimum, maximum in zip(request.minimum_xyz_m, request.maximum_xyz_m)
    )


def _validate_requested_dimensions(
    entity: SkeletonEntity,
    parameters: dict[str, Any],
    dimensions: tuple[float, float, float],
) -> None:
    if entity.proxy_family == "ground_plane":
        raise ValueError("ground_plane 尺寸由场景平面控制，不接受三轴自定义尺寸")
    if entity.proxy_family == "celestial_sphere":
        if max(dimensions) - min(dimensions) > max(dimensions) * 0.05:
            raise ValueError("celestial_sphere 的三轴尺寸必须近似相等")
    if entity.proxy_family == "human_capsule":
        if abs(dimensions[0] - dimensions[1]) > max(dimensions[:2]) * 0.05:
            raise ValueError("human_capsule 的 X/Y 尺寸必须近似相等")
        if dimensions[2] <= max(dimensions[:2]):
            raise ValueError("human_capsule 高度必须大于横向尺寸")
    footprint = parameters.get("minimum_footprint_m")
    if isinstance(footprint, list) and len(footprint) == 2:
        if (
            dimensions[0] < float(footprint[0])
            or dimensions[1] < float(footprint[1])
        ):
            raise ValueError("自定义尺寸小于 Brief 明确的最小占地尺寸")
    minimum_height = parameters.get("minimum_height_m")
    if minimum_height is not None and dimensions[2] < float(minimum_height):
        raise ValueError("自定义尺寸小于 Brief 明确的最小高度")
    reference_height = parameters.get("reference_height_m")
    if (
        parameters.get("height_source_status") == "explicit"
        and reference_height is not None
        and not math.isclose(
            dimensions[2],
            float(reference_height),
            rel_tol=0.15,
        )
    ):
        raise ValueError("自定义尺寸与 Brief 明确的参考高度冲突")


def _place_entities(
    candidate: CandidateState,
    skeleton: SceneSkeleton,
    profile: PlanningProfile,
) -> None:
    for index, entity in enumerate(candidate.entities.values()):
        if entity.proxy.type == "plane":
            position = (0.0, 0.0, 0.0)
        else:
            position = (float(index * 2), 0.0, _proxy_half_height(entity))
        entity.solved_transform = TransformValue(
            translation_m=position,
            rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )

    for relation in skeleton.relations:
        subject = candidate.entities[relation.subject_id]
        reference = candidate.entities[relation.reference_id]
        subject_position = list(subject.solved_transform.translation_m or (0.0, 0.0, 0.0))
        reference_position = reference.solved_transform.translation_m or (0.0, 0.0, 0.0)
        if relation.kind == "camera_depth_order":
            # A fixed center gap collapses for large proxies. Place explicit
            # foreground/background relations by edge clearance relative to size.
            direction = normalize(
                (
                    subject_position[0] - reference_position[0],
                    profile.default_depth_gap_m,
                    0.0,
                )
            )
            center_distance = surface_clearance_target_distance(
                reference.proxy,
                reference.solved_transform,
                subject.proxy,
                subject.solved_transform,
                direction,
                profile.far_clearance_preferred_ratio,
            )
            x_delta = subject_position[0] - reference_position[0]
            y_delta = math.sqrt(max(center_distance**2 - x_delta**2, 0.0))
            subject_position[1] = reference_position[1] + y_delta
        elif relation.kind == "relative_position" and relation.direction:
            axis, sign = {
                "left": (0, -1.0),
                "right": (0, 1.0),
                "front": (1, -1.0),
                "behind": (1, 1.0),
                "below": (2, -1.0),
                "above": (2, 1.0),
            }[relation.direction]
            subject_position[axis] = reference_position[axis] + sign * profile.default_depth_gap_m
        subject.solved_transform = subject.solved_transform.model_copy(
            update={"translation_m": tuple(subject_position)}
        )


def _build_relation_constraints(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> dict[str, ConstraintSpec]:
    constraints: dict[str, ConstraintSpec] = {}
    duration = candidate.timeline.duration_seconds
    for relation in skeleton.relations:
        time_range = _relation_time_range(
            objective,
            relation,
            candidate,
            duration,
        )
        strength = "hard" if relation.source_status == "explicit" else "soft"
        common = {
            "constraint_id": f"skeleton_{relation.relation_id}",
            "strength": strength,
            "weight": 1.0,
            "subjects": [relation.subject_id, relation.reference_id],
            "time_range_seconds": time_range,
            "source_status": relation.source_status,
            "source_ref": relation.source_ref,
        }
        payloads: list[dict[str, Any]] = []
        if relation.kind == "camera_depth_order":
            minimum_ratio, maximum_ratio = profile.far_clearance_ratio_range
            payloads = [
                common | {
                    "type": "depth_order",
                    "parameters": {
                        "near_entity_id": relation.reference_id,
                        "far_entity_id": relation.subject_id,
                        "camera_id": "camera_main",
                        "minimum_depth_gap_meters": profile.default_depth_gap_m * 0.5,
                    },
                },
                common | {
                    "constraint_id": f"skeleton_{relation.relation_id}_clearance",
                    "type": "surface_clearance_range",
                    "parameters": {
                        "entity_ids": [relation.reference_id, relation.subject_id],
                        "minimum_ratio": minimum_ratio,
                        "preferred_ratio": profile.far_clearance_preferred_ratio,
                        "maximum_ratio": maximum_ratio,
                        "scale_basis": "larger_directional_extent",
                        "space": "ground_plane",
                    },
                },
            ]
        elif relation.kind == "relative_position":
            payloads = [
                common
                | {
                    "type": "relative_position",
                    "parameters": {
                        "subject_id": relation.subject_id,
                        "reference_id": relation.reference_id,
                        "relation": relation.direction,
                        "space": "world",
                        "minimum_gap": 0.5,
                        "maximum_gap": profile.default_depth_gap_m * 2.0,
                    },
                }
            ]
        elif relation.kind == "proximity":
            payloads = [
                common
                | {
                    "type": "distance_range",
                    "parameters": {
                        "entity_ids": [relation.subject_id, relation.reference_id],
                        "minimum_meters": 0.0,
                        "maximum_meters": 3.0,
                    },
                }
            ]
        elif relation.kind == "scale_dominance":
            payloads = [
                common
                | {
                    "type": "projected_scale_ratio",
                    "parameters": {
                        "numerator_entity_id": relation.subject_id,
                        "denominator_entity_id": relation.reference_id,
                        "measurement": "height",
                        "minimum_ratio": 2.0,
                        "maximum_ratio": 20.0,
                    },
                }
            ]
        elif relation.kind == "orbit_around":
            subject = candidate.entities[relation.subject_id]
            reference = candidate.entities[relation.reference_id]
            radius = max(
                2.0,
                _proxy_bounding_radius(subject)
                + _proxy_bounding_radius(reference)
                + 2.0,
            )
            payloads = [
                common
                | {
                    "type": "distance_range",
                    "parameters": {
                        "entity_ids": [relation.subject_id, relation.reference_id],
                        "minimum_meters": radius * 0.95,
                        "maximum_meters": radius * 1.05,
                    },
                }
            ]
        else:
            continue
        for payload in payloads:
            constraint = ConstraintSpec.model_validate(payload)
            constraints[constraint.constraint_id] = constraint

    for phase in skeleton.motion_phases:
        common = {
            "constraint_id": f"skeleton_motion_{phase.phase_id}",
            "strength": "hard" if phase.source_status == "explicit" else "soft",
            "subjects": [phase.subject_id],
            "time_range_seconds": _phase_range(objective, phase, duration),
            "source_status": phase.source_status,
            "source_ref": phase.source_ref,
        }
        if phase.kind == "hold":
            payload = common | {
                "type": "hold",
                "parameters": {
                    "target_id": phase.subject_id,
                    "components": ["translation", "rotation", "scale"],
                },
            }
        elif phase.kind == "linear_move" and phase.direction_mode in {
            "screen_left_to_right",
            "screen_right_to_left",
        }:
            payload = common | {
                "type": "motion_direction",
                "parameters": {
                    "target_id": phase.subject_id,
                    "direction": (
                        "left"
                        if phase.direction_mode == "screen_right_to_left"
                        else "right"
                    ),
                    "space": "world",
                    "minimum_displacement_m": 0.5,
                },
            }
        elif (
            phase.kind == "linear_move"
            and phase.target_id is not None
            and phase.direction_mode in {"toward_target", "away_from_target"}
        ):
            target_follows_subject = any(
                item.kind == "carried"
                and item.subject_id == phase.target_id
                and item.carrier_id == phase.subject_id
                and item.timeline_event_id == phase.timeline_event_id
                for item in skeleton.motion_phases
            )
            if phase.direction_mode == "away_from_target" and target_follows_subject:
                # 载体离开时，乘员会同步移动，二者距离不应被要求增大。
                payload = common | {
                    "type": "motion_direction",
                    "parameters": {
                        "target_id": phase.subject_id,
                        "direction": (
                            "forward"
                            if _objective_motion_direction(objective, phase)
                            == "world_forward"
                            else "right"
                        ),
                        "space": "world",
                        "minimum_displacement_m": 0.5,
                    },
                }
            else:
                start, end = _phase_range(objective, phase, duration)
                frame_step = (
                    candidate.timeline.fps_denominator
                    / candidate.timeline.fps_numerator
                )
                toward = phase.direction_mode == "toward_target"
                payload = common | {
                    "type": "distance_range",
                    "subjects": [phase.subject_id, phase.target_id],
                    "time_range_seconds": [max(start, end - frame_step), end],
                    "parameters": {
                        "entity_ids": [phase.subject_id, phase.target_id],
                        "minimum_meters": 0.0 if toward else 5.0,
                        "maximum_meters": 3.0 if toward else 1000000.0,
                    },
                }
        else:
            # 其他阶段没有可独立验证的位移方向。
            continue
        constraint = ConstraintSpec.model_validate(payload)
        constraints[constraint.constraint_id] = constraint
    return constraints


def _build_motion(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> dict[str, TrackSpec]:
    tracks: dict[str, TrackSpec] = {}
    duration = candidate.timeline.duration_seconds
    orbit_subjects = {
        phase.subject_id for phase in skeleton.motion_phases if phase.kind == "orbit"
    }
    grouped: dict[str, list[SkeletonMotionPhase]] = {}
    for phase in skeleton.motion_phases:
        grouped.setdefault(phase.subject_id, []).append(phase)

    grouped_items = sorted(
        grouped.items(),
        key=lambda item: min(
            (
                _phase_range(objective, phase, duration)[0]
                for phase in item[1]
                if phase.kind == "linear_move"
            ),
            default=math.inf,
        ),
    )
    for subject_id, phases in grouped_items:
        orbit = next((item for item in phases if item.kind == "orbit"), None)
        if orbit is not None and orbit.target_id is not None:
            depth = 2 if orbit.target_id in orbit_subjects else 1
            subject = candidate.entities[subject_id]
            reference = candidate.entities[orbit.target_id]
            radius = max(
                2.0,
                _proxy_bounding_radius(subject) + _proxy_bounding_radius(reference) + 2.0,
            )
            track = TrackSpec.model_validate(
                {
                    "track_id": f"design_orbit_{subject_id}",
                    "target_entity_id": subject_id,
                    "type": "path_follow",
                    "time_range_seconds": _phase_range(objective, orbit, duration),
                    "path": {
                        "representation": orbit.path_family,
                        "space": "target_relative",
                        "target_id": orbit.target_id,
                        "closed": True,
                        "cycle_count": 3.0 if depth == 2 else 1.0,
                        "radius_m": radius,
                    }
                    if orbit.path_family == "circle"
                    else {
                        "representation": "ellipse",
                        "space": "target_relative",
                        "target_id": orbit.target_id,
                        "closed": True,
                        "cycle_count": 3.0 if depth == 2 else 1.0,
                        "semi_major_axis_m": radius,
                        "semi_minor_axis_m": radius * 0.75,
                    },
                    "interpolation": "linear",
                    "source_ref": orbit.source_ref,
                }
            )
            tracks[track.track_id] = track
            continue

        moving = [item for item in phases if item.kind == "linear_move"]
        if moving:
            moving = sorted(
                moving,
                key=lambda item: _phase_range(objective, item, duration)[0],
            )
            position = list(
                candidate.entities[subject_id].solved_transform.translation_m
                or (0.0, 0.0, _proxy_half_height(candidate.entities[subject_id]))
            )
            first = moving[0]
            route = next(
                (
                    item
                    for item in skeleton.route_intents
                    if item.subject_id == subject_id
                ),
                None,
            )
            route_direction = _selected_route_direction(moving)
            if (
                route is not None
                and route.axis_reference_id is not None
                and all(item.direction_mode == "none" for item in moving)
            ):
                route_direction = _reference_route_direction(
                    candidate.entities[route.axis_reference_id]
                )
            route_anchors = (
                {(item.phase_id, item.boundary): item for item in route.anchors}
                if route is not None
                else {}
            )
            first_end_anchor = next(
                (
                    (index, route_anchors[(phase.phase_id, "at_end")])
                    for index, phase in enumerate(moving)
                    if (phase.phase_id, "at_end") in route_anchors
                ),
                None,
            )
            if first.direction_mode in {
                "none",
                "screen_left_to_right",
                "screen_right_to_left",
            } and first_end_anchor is not None:
                anchor_index, anchor = first_end_anchor
                waypoint = _route_anchor_position(
                    skeleton,
                    candidate,
                    subject_id,
                    anchor,
                    route_direction,
                    profile,
                )
                travel_x, travel_y = route_direction
                # Put the subject far enough before its first declared endpoint
                # that every preceding unanchored phase can advance along one route.
                position[0] = waypoint[0] - travel_x * 8.0 * (anchor_index + 1)
                position[1] = waypoint[1] - travel_y * 8.0 * (anchor_index + 1)
                candidate.entities[subject_id].solved_transform = candidate.entities[
                    subject_id
                ].solved_transform.model_copy(update={"translation_m": tuple(position)})
            if first.direction_mode == "toward_target" and first.target_id:
                # 让“接近”阶段拥有可见的初始距离，避免代理一开始已经贴着目标。
                target = candidate.entities[first.target_id]
                target_position = target.solved_transform.translation_m or (
                    0.0,
                    0.0,
                    0.0,
                )
                delta_x = position[0] - target_position[0]
                delta_y = position[1] - target_position[1]
                distance = math.hypot(delta_x, delta_y)
                clearance = (
                    _proxy_horizontal_radius(candidate.entities[subject_id])
                    + _proxy_horizontal_radius(target)
                )
                required_distance = clearance + 8.0
                if distance < required_distance:
                    if distance <= 1e-9:
                        delta_x, delta_y, distance = 1.0, 0.0, 1.0
                    position[0] = target_position[0] + delta_x / distance * required_distance
                    position[1] = target_position[1] + delta_y / distance * required_distance
                candidate.entities[subject_id].solved_transform = candidate.entities[
                    subject_id
                ].solved_transform.model_copy(update={"translation_m": tuple(position)})
            keyframes: list[TrackKeyframe] = []
            previous_direction = route_direction
            positional_phases = [
                item for item in phases if item.kind in {"hold", "linear_move"}
            ]
            for phase in sorted(
                positional_phases,
                key=lambda item: _phase_range(objective, item, duration)[0],
            ):
                start, end = _phase_range(objective, phase, duration)
                if phase.kind == "hold":
                    keyframes.extend(
                        [
                            TrackKeyframe(
                                time_seconds=start,
                                value=TransformValue(translation_m=tuple(position)),
                                interpolation="step",
                            ),
                            TrackKeyframe(
                                time_seconds=_track_end_time(candidate, end),
                                value=TransformValue(translation_m=tuple(position)),
                                interpolation="step",
                            ),
                        ]
                    )
                    continue
                start_anchor = route_anchors.get((phase.phase_id, "at_start"))
                if start_anchor is not None:
                    position = list(
                        _route_anchor_position(
                            skeleton,
                            candidate,
                            subject_id,
                            start_anchor,
                            previous_direction,
                            profile,
                        )
                    )
                    if phase is first:
                        candidate.entities[subject_id].solved_transform = candidate.entities[
                            subject_id
                        ].solved_transform.model_copy(
                            update={"translation_m": tuple(position)}
                        )
                keyframes.append(
                    TrackKeyframe(
                        time_seconds=start,
                        value=TransformValue(translation_m=tuple(position)),
                        interpolation="smooth",
                    )
                )
                hidden_at_end = any(
                    item.kind == "visibility"
                    and item.timeline_event_id == phase.timeline_event_id
                    and item.visibility_state == "hidden"
                    and item.transition_at == "at_end"
                    for item in phases
                )
                previous_position = tuple(position)
                end_anchor = route_anchors.get((phase.phase_id, "at_end"))
                if end_anchor is not None:
                    position = list(
                        _route_anchor_position(
                            skeleton,
                            candidate,
                            subject_id,
                            end_anchor,
                            previous_direction,
                            profile,
                        )
                    )
                else:
                    position = _linear_phase_endpoint(
                        candidate,
                        phase,
                        position,
                        tracks,
                        _track_end_time(candidate, end),
                        semantic_direction_mode=_objective_motion_direction(
                            objective,
                            phase,
                        ),
                        hidden_at_end=hidden_at_end,
                        fallback_direction=previous_direction,
                    )
                delta_x = position[0] - previous_position[0]
                delta_y = position[1] - previous_position[1]
                delta_length = math.hypot(delta_x, delta_y)
                if delta_length > 1e-9:
                    proposed_direction = (
                        delta_x / delta_length,
                        delta_y / delta_length,
                    )
                    if (
                        route is not None
                        and route.continuity == "preserve_direction"
                        and proposed_direction[0] * previous_direction[0]
                        + proposed_direction[1] * previous_direction[1]
                        <= 0.0
                    ):
                        raise ValueError(
                            f"Route Intent {route.route_id} 的路径点导致运动方向反转"
                        )
                    previous_direction = proposed_direction
                keyframes.append(
                    TrackKeyframe(
                        time_seconds=_track_end_time(candidate, end),
                        value=TransformValue(translation_m=tuple(position)),
                        interpolation="step",
                    )
                )
            track = TrackSpec(
                track_id=f"design_motion_{subject_id}",
                target_entity_id=subject_id,
                type="transform",
                time_range_seconds=(keyframes[0].time_seconds, duration),
                keyframes=_deduplicate_keyframes(keyframes),
                interpolation="smooth",
                source_ref=moving[0].source_ref,
            )
            tracks[track.track_id] = track

        carried_phases = [item for item in phases if item.kind == "carried"]
        for phase in sorted(
            carried_phases,
            key=lambda item: _phase_range(objective, item, duration)[0],
        ):
            if phase.carrier_id is None:
                continue
            start, end = _phase_range(objective, phase, duration)
            if _subject_hidden_at(phases, objective, duration, start):
                # 已隐藏的乘员无需再生成可见的跟随代理；显隐后置条件本身
                # 足以表达“已进入载体”，也避免两个代理卡在一起。
                continue
            subject = candidate.entities[subject_id]
            carrier = candidate.entities[phase.carrier_id]
            subject_z = (subject.solved_transform.translation_m or (0.0, 0.0, 0.0))[2]
            carrier_z = (carrier.solved_transform.translation_m or (0.0, 0.0, 0.0))[2]
            offset = (0.0, 0.0, subject_z - carrier_z)
            track = TrackSpec.model_validate(
                {
                    "track_id": f"design_carried_{subject_id}_{phase.phase_id}",
                    "target_entity_id": subject_id,
                    "type": "path_follow",
                    "time_range_seconds": [start, end],
                    "path": {
                        "representation": "polyline",
                        "space": "target_relative",
                        "target_id": phase.carrier_id,
                        "closed": False,
                        "control_points": [offset, offset],
                    },
                    "interpolation": "step",
                    "source_ref": phase.source_ref,
                }
            )
            tracks[track.track_id] = track

        visibility_phases = [item for item in phases if item.kind == "visibility"]
        if visibility_phases:
            transitions: list[tuple[float, bool, str]] = []
            for phase in visibility_phases:
                start, end = _phase_range(objective, phase, duration)
                transition_time = start if phase.transition_at == "at_start" else end
                transitions.append(
                    (
                        _track_end_time(candidate, transition_time),
                        phase.visibility_state == "visible",
                        phase.source_ref,
                    )
                )
            transitions.sort(key=lambda item: item[0])
            source_ref = transitions[0][2]
            track = TrackSpec(
                track_id=f"design_visibility_{subject_id}",
                target_entity_id=subject_id,
                type="visibility",
                time_range_seconds=(0.0, duration),
                keyframes=_deduplicate_keyframes(
                    [TrackKeyframe(time_seconds=0.0, value=True, interpolation="step")]
                    + [
                        TrackKeyframe(
                            time_seconds=time_seconds,
                            value=visible,
                            interpolation="step",
                        )
                        for time_seconds, visible, _ in transitions
                    ]
                ),
                interpolation="step",
                source_ref=source_ref,
            )
            tracks[track.track_id] = track
    return tracks


def _build_camera(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
    strategy: str,
) -> CameraCandidate:
    parameters = (objective.translation_parameters or {}).get("camera", {})
    focal = float(parameters.get("focal_length_mm") or profile.default_focal_length_mm)
    movement = skeleton.camera_intent.movement
    start_distance = float(parameters.get("start_distance_m") or profile.default_camera_distance_m)
    if parameters.get("end_distance_m") is not None:
        end_distance = float(parameters["end_distance_m"])
    elif movement == "push_in":
        end_distance = start_distance * 0.5
    elif movement == "pull_out":
        end_distance = start_distance * 1.5
    else:
        end_distance = start_distance
    distance_scale = {
        "balanced": 1.5,
        "preserve_composition": 2.4,
        "maximize_motion_readability": 1.15,
    }[strategy]
    start_distance *= distance_scale
    end_distance *= distance_scale
    height = float(parameters.get("height_m") or 1.5)
    has_orbit = any(item.kind == "orbit" for item in skeleton.motion_phases)
    if has_orbit and not _has_explicit_camera_elevation(objective):
        # 缺省轨道镜头提高俯视夹角，避免圆轨道投影成直线往返。
        focus_height = _proxy_half_height(
            candidate.entities[skeleton.camera_intent.focus_target_id]
        )
        height = focus_height + start_distance * 0.75
    view = skeleton.camera_intent.view_relation_to_motion
    yaw = {
        "front": 0.0,
        "rear": 180.0,
        "side": 90.0,
        "three_quarter": 45.0,
        "unspecified": 55.0 if strategy == "maximize_motion_readability" else 35.0,
    }[view]
    focus = candidate.entities[skeleton.camera_intent.focus_target_id]
    focus_point = focus.solved_transform.translation_m or (0.0, 0.0, 0.0)
    start = _camera_position(focus_point, start_distance, height, yaw)
    end = _camera_position(focus_point, end_distance, height, yaw)
    duration = candidate.timeline.duration_seconds
    tracks: dict[str, TrackSpec] = {}
    if movement in {"push_in", "pull_out"}:
        if movement == "pull_out":
            start, end = end, start
        track = TrackSpec(
            track_id="design_camera_transform",
            type="transform",
            time_range_seconds=(0.0, duration),
            keyframes=[
                TrackKeyframe(
                    time_seconds=0.0,
                    value=TransformValue(
                        translation_m=start,
                        rotation_quaternion_wxyz=look_at_camera_quaternion(start, focus_point),
                    ),
                    interpolation="smooth",
                ),
                TrackKeyframe(
                    time_seconds=_track_end_time(candidate, duration),
                    value=TransformValue(
                        translation_m=end,
                        rotation_quaternion_wxyz=look_at_camera_quaternion(end, focus_point),
                    ),
                    interpolation="smooth",
                ),
            ],
            interpolation="smooth",
            source_ref=skeleton.camera_intent.source_ref,
        )
        tracks[track.track_id] = track
    source_refs = sorted(
        {
            skeleton.camera_intent.source_ref,
            *(
                item.path
                for item in objective.explicit_requirements
                if item.path.startswith("content.camera")
            ),
        }
    )
    return CameraCandidate(
        static=CameraStatic(
            focal_length_mm=focal,
            focus_target_id=skeleton.camera_intent.focus_target_id,
            source_refs=source_refs,
        ),
        tracks=tracks,
        solved_transform=TransformValue(
            translation_m=start,
            rotation_quaternion_wxyz=look_at_camera_quaternion(start, focus_point),
            scale=(1.0, 1.0, 1.0),
        ),
    )


def _add_composition_constraints(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
) -> None:
    parameters = (objective.translation_parameters or {}).get("composition", {})
    duration = candidate.timeline.duration_seconds
    foreground = next(
        (
            item.entity_id
            for item in skeleton.entities
            if item.proxy_family not in {"ground_plane"}
        ),
        None,
    )
    if foreground and parameters.get("subject_frame_ratio"):
        minimum, maximum = parameters["subject_frame_ratio"]
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": "design_subject_projected_size",
                "type": "projected_size",
                "strength": "soft",
                "subjects": [foreground],
                "time_range_seconds": [0.0, duration],
                "parameters": {
                    "entity_id": foreground,
                    "measurement": "height",
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                },
                "source_status": parameters.get("source_status", "inferred"),
                "source_ref": "translation_parameters.composition.subject_frame_ratio",
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint

    for index, item in enumerate(objective.composition.get("visibility_requirements", [])):
        if not isinstance(item, dict):
            continue
        subject_id = item.get("subject_id")
        requirement = item.get("requirement")
        if (
            not isinstance(subject_id, str)
            or subject_id not in candidate.entities
            or not isinstance(requirement, dict)
            or requirement.get("source_status") != "explicit"
        ):
            continue
        source_ref = f"content.composition.visibility_requirements[{index}].requirement"
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": f"design_visibility_{subject_id}_{index}",
                "type": "keep_in_frame",
                "strength": "hard",
                "subjects": [subject_id],
                "time_range_seconds": [0.0, duration],
                "parameters": {
                    "entity_id": subject_id,
                    "minimum_inside_fraction": 0.01,
                },
                "source_status": "explicit",
                "source_ref": source_ref,
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint


def _add_speed_constraints(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> None:
    """把符号速度档位映射为冻结 Profile 范围。"""

    duration = candidate.timeline.duration_seconds
    for phase in skeleton.motion_phases:
        generic_explicit_motion = (
            phase.kind == "linear_move"
            and phase.speed_intent == "unspecified"
            and phase.source_status == "explicit"
        )
        if phase.speed_intent == "unspecified" and not generic_explicit_motion:
            continue
        minimum, maximum = (
            (0.01, profile.fast_speed_range_mps[1])
            if generic_explicit_motion
            else _speed_range(phase.speed_intent, profile)
        )
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": f"skeleton_speed_{phase.phase_id}",
                "type": "speed_range",
                "strength": (
                    "hard"
                    if generic_explicit_motion
                    or phase.speed_source_status == "explicit"
                    else "soft"
                ),
                "subjects": [phase.subject_id],
                "time_range_seconds": _phase_range(objective, phase, duration),
                "parameters": {
                    "target_id": phase.subject_id,
                    "minimum_mps": minimum,
                    "maximum_mps": maximum,
                    "space": "world",
                },
                "source_status": (
                    phase.source_status
                    if generic_explicit_motion
                    else phase.speed_source_status or phase.source_status
                ),
                "source_ref": (
                    phase.source_ref
                    if generic_explicit_motion
                    else phase.speed_source_ref or phase.source_ref
                ),
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint

    camera = skeleton.camera_intent
    if camera.speed_intent == "unspecified":
        return
    minimum, maximum = _speed_range(camera.speed_intent, profile)
    constraint = ConstraintSpec.model_validate(
        {
            "constraint_id": "skeleton_camera_speed",
            "type": "speed_range",
            "strength": "hard" if camera.speed_source_status == "explicit" else "soft",
            "subjects": [],
            "time_range_seconds": [0.0, duration],
            "parameters": {
                "target_id": "camera_main",
                "minimum_mps": minimum,
                "maximum_mps": maximum,
                "space": "world",
            },
            "source_status": camera.speed_source_status or camera.source_status,
            "source_ref": camera.speed_source_ref or camera.source_ref,
        }
    )
    candidate.constraints[constraint.constraint_id] = constraint


def _speed_range(
    intent: str,
    profile: PlanningProfile,
) -> tuple[float, float]:
    return {
        "stationary": (0.0, profile.stationary_speed_max_mps),
        "slow": profile.slow_speed_range_mps,
        "medium": profile.medium_speed_range_mps,
        "fast": profile.fast_speed_range_mps,
    }[intent]


def _numeric_envelopes(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    size_requests: dict[str, EntitySizeRequest],
) -> dict[str, Any]:
    camera = candidate.camera
    assert camera is not None
    camera_positions = [camera.solved_transform.translation_m]
    transform = next(
        (item for item in camera.tracks.values() if item.type == "transform"),
        None,
    )
    if transform:
        camera_positions = [
            item.value.translation_m
            for item in transform.keyframes
            if isinstance(item.value, TransformValue)
        ]
    focus = candidate.entities[skeleton.camera_intent.focus_target_id]
    focus_point = focus.solved_transform.translation_m or (0.0, 0.0, 0.0)
    distances = [
        math.dist(item, focus_point)
        for item in camera_positions
        if item is not None
    ]
    depth_gaps = [
        float(item.parameters.minimum_depth_gap_meters or 0.0)
        for item in candidate.constraints.values()
        if item.type == "depth_order"
    ]
    clearance_ratios = [
        {
            "entity_ids": list(item.parameters.entity_ids),
            "minimum": item.parameters.minimum_ratio,
            "preferred": item.parameters.preferred_ratio,
            "maximum": item.parameters.maximum_ratio,
            "scale_basis": item.parameters.scale_basis,
        }
        for item in candidate.constraints.values()
        if item.type == "surface_clearance_range"
    ]
    motion_speeds = [
        item.get("speed_range_mps")
        for item in (objective.translation_parameters or {}).get("motions", [])
        if item.get("speed_range_mps") is not None
    ]
    return {
        "camera_distance_m": _range_around(distances),
        "camera_focal_length_mm": _range_around(
            [camera.static.focal_length_mm or 35.0],
            relative_margin=0.15,
        ),
        "depth_gap_m": _range_around(depth_gaps),
        "surface_clearance_ratios": clearance_ratios,
        "motion_speed_ranges_mps": motion_speeds,
        "entity_size_ranges_m": {
            entity_id: (
                {
                    "minimum_xyz": list(size_requests[entity_id].minimum_xyz_m),
                    "maximum_xyz": list(size_requests[entity_id].maximum_xyz_m),
                    "selected_xyz": list(_entity_dimensions(entity)),
                    "source": "agent_requested_validated_range",
                }
                if entity_id in size_requests
                else _entity_size_range(entity)
            )
            for entity_id, entity in candidate.entities.items()
            if entity.proxy.type != "plane"
        },
    }


def _design_assumptions(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    strategy: str,
    size_requests: dict[str, EntitySizeRequest],
) -> tuple[str, ...]:
    assumptions = [
        f"数值策略采用 {strategy}",
        "未明确的世界方向由 Toolkit 选择斜侧可读方向",
        "所有范围都保留 explicit > inferred > default 的来源优先级",
    ]
    if not objective.translation_parameters:
        assumptions.append("旧版 Brief 缺少量化快照，使用冻结 Research Profile")
    if any(item.kind == "orbit" for item in skeleton.motion_phases):
        assumptions.append("未指定嵌套周期时，子轨道使用不同 cycle_count 避免同相锁定")
    assumptions.extend(
        f"实体 {entity_id} 使用 Agent 请求的受验证尺寸范围：{request.rationale}"
        for entity_id, request in sorted(size_requests.items())
    )
    return tuple(assumptions)


def _skeleton_sources(value: SceneSkeleton) -> list[tuple[str, SourceStatus]]:
    result: list[tuple[str, SourceStatus]] = []
    for entity in value.entities:
        result.extend((source_ref, "explicit") for source_ref in entity.source_refs)
    result.extend((item.source_ref, item.source_status) for item in value.relations)
    result.extend((item.source_ref, item.source_status) for item in value.motion_phases)
    result.extend(
        (item.speed_source_ref, item.speed_source_status)
        for item in value.motion_phases
        if item.speed_source_ref is not None and item.speed_source_status is not None
    )
    result.append((value.camera_intent.source_ref, value.camera_intent.source_status))
    if (
        value.camera_intent.speed_source_ref is not None
        and value.camera_intent.speed_source_status is not None
    ):
        result.append(
            (
                value.camera_intent.speed_source_ref,
                value.camera_intent.speed_source_status,
            )
        )
    return result


def _subject_parameters(objective: ObjectivePlanningBrief) -> dict[str, dict[str, Any]]:
    translated = (objective.translation_parameters or {}).get("subjects", [])
    result: dict[str, dict[str, Any]] = {}
    for index, subject in enumerate(objective.subjects):
        entity_id = subject.get("id")
        if entity_id is None:
            continue
        result[str(entity_id)] = (
            deepcopy(translated[index]) if index < len(translated) else {}
        )
    return result


def _entity_explicit_refs(
    objective: ObjectivePlanningBrief,
    entity_id: str,
) -> set[str]:
    index = next(
        (
            index
            for index, subject in enumerate(objective.subjects)
            if str(subject.get("id")) == entity_id
        ),
        None,
    )
    if index is None:
        return set()
    prefix = f"content.subjects[{index}]"
    return {
        item.path
        for item in objective.explicit_requirements
        if item.path.startswith(prefix)
    }


def _proxy_half_height(entity: EntitySpec) -> float:
    proxy = entity.proxy
    if proxy.type == "box":
        return proxy.size_xyz_m[2] / 2.0
    if proxy.type == "sphere":
        return proxy.radius_m
    if proxy.type == "capsule":
        return proxy.segment_length_m / 2.0 + proxy.radius_m
    if proxy.type in {"cylinder", "cone"}:
        return proxy.depth_m / 2.0
    return 0.0


def _proxy_bounding_radius(entity: EntitySpec) -> float:
    proxy = entity.proxy
    if proxy.type == "box":
        return math.sqrt(sum((item / 2.0) ** 2 for item in proxy.size_xyz_m))
    if proxy.type == "sphere":
        return proxy.radius_m
    if proxy.type == "capsule":
        return proxy.segment_length_m / 2.0 + proxy.radius_m
    if proxy.type == "cylinder":
        return math.hypot(proxy.radius_m, proxy.depth_m / 2.0)
    if proxy.type == "cone":
        return math.hypot(
            max(proxy.radius_bottom_m, proxy.radius_top_m),
            proxy.depth_m / 2.0,
        )
    return math.hypot(*[item / 2.0 for item in proxy.size_xy_m])


def _event_range(
    objective: ObjectivePlanningBrief,
    event_id: str | None,
    duration: float,
) -> tuple[float, float]:
    if event_id:
        for event in objective.timeline.get("events", []):
            if event.get("id") == event_id:
                start = float(event.get("start_time_seconds") or 0.0)
                end = float(event.get("end_time_seconds") or duration)
                if 0.0 <= start < end <= duration:
                    return (start, end)
    return (0.0, duration)


def _has_explicit_camera_elevation(objective: ObjectivePlanningBrief) -> bool:
    return any(
        requirement.path.startswith(
            ("content.camera.camera_height", "content.camera.view_angle")
        )
        for requirement in objective.explicit_requirements
    )


def _phase_range(
    objective: ObjectivePlanningBrief,
    phase: SkeletonMotionPhase,
    duration: float,
) -> tuple[float, float]:
    """Prefer a motion's narrower interval when one event contains multiple phases."""

    prefix = "content.subject_motion["
    try:
        if phase.source_ref.startswith(prefix):
            index = int(phase.source_ref[len(prefix):].split("]", 1)[0])
            motion = objective.subject_motion[index]
            start = float(motion.get("start_time_seconds") or 0.0)
            end = float(motion.get("end_time_seconds") or duration)
            if 0.0 <= start < end <= duration:
                return (start, end)
            raise ValueError(f"Motion Phase 引用了无效动作时间范围：{phase.phase_id}")
    except (IndexError, TypeError, ValueError, AttributeError):
        pass
    return _event_range(objective, phase.timeline_event_id, duration)


def _subject_hidden_at(
    phases: list[SkeletonMotionPhase],
    objective: ObjectivePlanningBrief,
    duration: float,
    time_seconds: float,
) -> bool:
    hidden = False
    transitions: list[tuple[float, bool]] = []
    for phase in phases:
        if phase.kind != "visibility" or phase.visibility_state is None:
            continue
        start, end = _phase_range(objective, phase, duration)
        transition = start if phase.transition_at == "at_start" else end
        transitions.append((transition, phase.visibility_state == "hidden"))
    for transition, becomes_hidden in sorted(transitions):
        if transition > time_seconds + 1e-9:
            break
        hidden = becomes_hidden
    return hidden


def _relation_time_range(
    objective: ObjectivePlanningBrief,
    relation: SkeletonRelation,
    candidate: CandidateState,
    duration: float,
) -> tuple[float, float]:
    start, end = _event_range(objective, relation.timeline_event_id, duration)
    frame_step = candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    if relation.temporal_mode == "at_start":
        return (start, min(end, start + frame_step))
    if relation.temporal_mode == "at_end":
        return (max(start, end - frame_step), end)
    return (start, end)


def _track_end_time(candidate: CandidateState, end: float) -> float:
    frame_step = (
        candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    )
    return min(end, candidate.timeline.duration_seconds - frame_step)


def _linear_phase_endpoint(
    candidate: CandidateState,
    phase: SkeletonMotionPhase,
    position: list[float],
    tracks: dict[str, TrackSpec],
    sample_time: float,
    *,
    semantic_direction_mode: str | None,
    hidden_at_end: bool,
    fallback_direction: tuple[float, float],
) -> list[float]:
    endpoint = list(position)
    if semantic_direction_mode == "world_forward":
        endpoint[1] -= 8.0
    elif phase.direction_mode == "toward_target" and phase.target_id:
        target_entity = candidate.entities[phase.target_id]
        target_track = tracks.get(f"design_motion_{phase.target_id}")
        target = sample_transform_track(
            target_track,
            sample_time,
            target_entity.solved_transform,
        ).translation_m
        if target is not None:
            delta_x = target[0] - position[0]
            delta_y = target[1] - position[1]
            distance = math.hypot(delta_x, delta_y)
            if distance > 1e-9:
                clearance = 0.0 if hidden_at_end else (
                    _proxy_horizontal_radius(candidate.entities[phase.subject_id])
                    + _proxy_horizontal_radius(target_entity)
                )
                endpoint[0] = target[0] - delta_x / distance * clearance
                endpoint[1] = target[1] - delta_y / distance * clearance
    elif phase.direction_mode == "away_from_target":
        endpoint[0] += 8.0
    elif phase.direction_mode == "screen_right_to_left":
        endpoint[0] -= 8.0
    elif phase.direction_mode == "none":
        endpoint[0] += fallback_direction[0] * 8.0
        endpoint[1] += fallback_direction[1] * 8.0
    else:
        endpoint[0] += 8.0
    return endpoint


def _selected_route_direction(
    phases: list[SkeletonMotionPhase],
) -> tuple[float, float]:
    """为同一主体全部未指定方向的阶段选择一条路线朝向。"""

    selected = {
        phase.direction_mode
        for phase in phases
        if phase.direction_mode in {"screen_left_to_right", "screen_right_to_left"}
    }
    if len(selected) > 1:
        raise ValueError("同一主体的未指定方向运动不能选择相反的局部路线")
    if selected == {"screen_right_to_left"}:
        return (-1.0, 0.0)
    return (1.0, 0.0)


def _reference_route_direction(entity: EntitySpec) -> tuple[float, float]:
    """Use a reference proxy's longest horizontal axis without exposing meters."""

    proxy = entity.proxy
    if proxy.type == "box":
        return (1.0, 0.0) if proxy.size_xyz_m[0] >= proxy.size_xyz_m[1] else (0.0, 1.0)
    if proxy.type == "plane":
        return (1.0, 0.0) if proxy.size_xy_m[0] >= proxy.size_xy_m[1] else (0.0, 1.0)
    return (1.0, 0.0)


def _route_anchor_position(
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    subject_id: str,
    anchor: SkeletonRouteAnchor,
    route_direction: tuple[float, float],
    profile: PlanningProfile,
) -> tuple[float, float, float]:
    """Resolve one symbolic route anchor into a validated world-space waypoint."""

    relation = next(
        item for item in skeleton.relations if item.relation_id == anchor.relation_id
    )
    other_id = (
        relation.reference_id
        if relation.subject_id == subject_id
        else relation.subject_id
    )
    subject = candidate.entities[subject_id]
    other = candidate.entities[other_id]
    subject_position = subject.solved_transform.translation_m or (
        0.0,
        0.0,
        _proxy_half_height(subject),
    )
    other_position = other.solved_transform.translation_m or (0.0, 0.0, 0.0)
    if relation.kind == "proximity":
        clearance = min(
            2.5,
            max(
                0.75,
                _proxy_lateral_radius(subject, route_direction)
                + _proxy_lateral_radius(other, route_direction)
                + 0.25,
            ),
        )
        return (
            other_position[0] - route_direction[1] * clearance,
            other_position[1] + route_direction[0] * clearance,
            subject_position[2],
        )
    if relation.kind == "relative_position" and relation.direction is not None:
        axis, sign = {
            "left": (0, -1.0),
            "right": (0, 1.0),
            "front": (1, -1.0),
            "behind": (1, 1.0),
            "below": (2, -1.0),
            "above": (2, 1.0),
        }[relation.direction]
        if relation.reference_id == subject_id:
            sign *= -1.0
        waypoint = list(other_position)
        waypoint[axis] += sign * profile.default_depth_gap_m
        if axis != 2:
            waypoint[2] = subject_position[2]
        return tuple(waypoint)
    raise ValueError(
        f"Route Anchor {anchor.anchor_id} 的空间关系无法求解为路径点"
    )


def _objective_motion_direction(
    objective: ObjectivePlanningBrief,
    phase: SkeletonMotionPhase,
) -> str | None:
    """按主体和事件取回 Brief 的类型化方向，避免符号骨架丢失世界轴语义。"""

    for motion in objective.subject_motion:
        if motion.get("subject_id") != phase.subject_id:
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        if semantics.get("timeline_event_id") != phase.timeline_event_id:
            continue
        direction = semantics.get("direction_mode")
        if isinstance(direction, str):
            return direction
    return None


def _proxy_horizontal_radius(entity: EntitySpec) -> float:
    proxy = entity.proxy
    if proxy.type == "box":
        return max(proxy.size_xyz_m[0], proxy.size_xyz_m[1]) / 2.0
    if proxy.type in {"sphere", "capsule", "cylinder"}:
        return proxy.radius_m
    if proxy.type == "cone":
        return max(proxy.radius_bottom_m, proxy.radius_top_m)
    return max(proxy.size_xy_m) / 2.0


def _proxy_lateral_radius(
    entity: EntitySpec,
    route_direction: tuple[float, float],
) -> float:
    """返回代理在水平路线垂直方向上的支撑半径。"""

    proxy = entity.proxy
    if proxy.type == "box":
        lateral_x = -route_direction[1]
        lateral_y = route_direction[0]
        return (
            abs(lateral_x) * proxy.size_xyz_m[0] / 2.0
            + abs(lateral_y) * proxy.size_xyz_m[1] / 2.0
        )
    if proxy.type in {"sphere", "capsule", "cylinder"}:
        return proxy.radius_m
    if proxy.type == "cone":
        return max(proxy.radius_bottom_m, proxy.radius_top_m)
    return max(proxy.size_xy_m) / 2.0


def _deduplicate_keyframes(values: list[TrackKeyframe]) -> list[TrackKeyframe]:
    by_time: dict[float, TrackKeyframe] = {}
    for item in values:
        by_time[item.time_seconds] = item
    return [by_time[key] for key in sorted(by_time)]


def _camera_position(
    focus: tuple[float, float, float],
    distance: float,
    height: float,
    yaw_degrees: float,
) -> tuple[float, float, float]:
    angle = math.radians(yaw_degrees)
    return (
        focus[0] + math.sin(angle) * distance,
        focus[1] - math.cos(angle) * distance,
        height,
    )


def _range_around(
    values: list[float],
    *,
    relative_margin: float = 0.2,
) -> dict[str, float] | None:
    if not values:
        return None
    minimum = min(values)
    maximum = max(values)
    lower = max(0.0, minimum * (1.0 - relative_margin))
    upper = maximum * (1.0 + relative_margin)
    return {"minimum": round(lower, 6), "maximum": round(upper, 6)}


def _entity_dimensions(entity: EntitySpec) -> tuple[float, float, float]:
    proxy = entity.proxy
    if proxy.type == "box":
        return proxy.size_xyz_m
    elif proxy.type == "sphere":
        return (proxy.radius_m * 2.0,) * 3
    elif proxy.type == "capsule":
        diameter = proxy.radius_m * 2.0
        height = proxy.segment_length_m + diameter
        return (diameter, diameter, height)
    elif proxy.type in {"cylinder", "cone"}:
        radius = (
            proxy.radius_m
            if proxy.type == "cylinder"
            else max(proxy.radius_bottom_m, proxy.radius_top_m)
        )
        return (radius * 2.0, radius * 2.0, proxy.depth_m)
    return (proxy.size_xy_m[0], proxy.size_xy_m[1], 0.0)


def _entity_size_range(entity: EntitySpec) -> dict[str, list[float]]:
    dimensions = _entity_dimensions(entity)
    return {
        "minimum_xyz": [round(item * 0.8, 6) for item in dimensions],
        "maximum_xyz": [round(item * 1.2, 6) for item in dimensions],
    }
