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
from cinescaffold.planning.geometry import look_at_camera_quaternion
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


def build_design_candidate(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    base: CandidateState,
    profile: PlanningProfile,
    strategy: Literal[
        "balanced", "preserve_composition", "maximize_motion_readability"
    ],
) -> tuple[CandidateState, dict[str, Any], tuple[str, ...]]:
    """把符号骨架物化为确定性数值候选，不让模型填写坐标。"""

    candidate = base.model_copy(deep=True)
    candidate.entities = _build_entities(objective, skeleton)
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
    _add_composition_constraints(objective, skeleton, candidate)
    assumptions = _design_assumptions(objective, skeleton, strategy)
    return candidate, _numeric_envelopes(objective, skeleton, candidate), assumptions


def _build_entities(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
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
        entities[item.entity_id] = EntitySpec(
            entity_id=item.entity_id,
            label=item.semantic_type,
            role=item.role,
            proxy=_proxy_geometry(objective, item, parameters),
            tags=[item.semantic_type, item.scale_intent],
            source_refs=sorted(source_refs),
            ground_interaction=ground_interaction,
        )
    return entities


def _proxy_geometry(
    objective: ObjectivePlanningBrief,
    entity: SkeletonEntity,
    parameters: dict[str, Any],
) -> dict[str, Any]:
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
    return {"type": "box", "size_xyz_m": [edge, edge, edge]}


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
            subject_position[1] = reference_position[1] + profile.default_depth_gap_m
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
        time_range = _event_range(objective, relation.timeline_event_id, duration)
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
        if relation.kind == "camera_depth_order":
            payload = common | {
                "type": "depth_order",
                "parameters": {
                    "near_entity_id": relation.reference_id,
                    "far_entity_id": relation.subject_id,
                    "camera_id": "camera_main",
                    "minimum_depth_gap_meters": profile.default_depth_gap_m * 0.5,
                },
            }
        elif relation.kind == "relative_position":
            payload = common | {
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
        elif relation.kind == "proximity":
            payload = common | {
                "type": "distance_range",
                "parameters": {
                    "entity_ids": [relation.subject_id, relation.reference_id],
                    "minimum_meters": 0.0,
                    "maximum_meters": 3.0,
                },
            }
        elif relation.kind == "scale_dominance":
            payload = common | {
                "type": "projected_scale_ratio",
                "parameters": {
                    "numerator_entity_id": relation.subject_id,
                    "denominator_entity_id": relation.reference_id,
                    "measurement": "height",
                    "minimum_ratio": 2.0,
                    "maximum_ratio": 20.0,
                },
            }
        else:
            continue
        constraint = ConstraintSpec.model_validate(payload)
        constraints[constraint.constraint_id] = constraint

    for phase in skeleton.motion_phases:
        if phase.kind != "hold":
            continue
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": f"skeleton_hold_{phase.phase_id}",
                "type": "hold",
                "strength": "hard" if phase.source_status == "explicit" else "soft",
                "subjects": [phase.subject_id],
                "time_range_seconds": _event_range(
                    objective,
                    phase.timeline_event_id,
                    duration,
                ),
                "parameters": {
                    "target_id": phase.subject_id,
                    "components": ["translation", "rotation", "scale"],
                },
                "source_status": phase.source_status,
                "source_ref": phase.source_ref,
            }
        )
        constraints[constraint.constraint_id] = constraint
    return constraints


def _build_motion(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> dict[str, TrackSpec]:
    del profile
    tracks: dict[str, TrackSpec] = {}
    duration = candidate.timeline.duration_seconds
    orbit_subjects = {
        phase.subject_id for phase in skeleton.motion_phases if phase.kind == "orbit"
    }
    grouped: dict[str, list[SkeletonMotionPhase]] = {}
    for phase in skeleton.motion_phases:
        grouped.setdefault(phase.subject_id, []).append(phase)

    for subject_id, phases in grouped.items():
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
                    "time_range_seconds": _event_range(
                        objective,
                        orbit.timeline_event_id,
                        duration,
                    ),
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
            position = list(
                candidate.entities[subject_id].solved_transform.translation_m
                or (0.0, 0.0, _proxy_half_height(candidate.entities[subject_id]))
            )
            first = moving[0]
            if first.direction_mode == "toward_target":
                position[0] -= 8.0
                candidate.entities[subject_id].solved_transform = candidate.entities[
                    subject_id
                ].solved_transform.model_copy(update={"translation_m": tuple(position)})
            keyframes: list[TrackKeyframe] = []
            for phase in sorted(
                moving,
                key=lambda item: _event_range(
                    objective,
                    item.timeline_event_id,
                    duration,
                )[0],
            ):
                start, end = _event_range(objective, phase.timeline_event_id, duration)
                keyframes.append(
                    TrackKeyframe(
                        time_seconds=start,
                        value=TransformValue(translation_m=tuple(position)),
                        interpolation="smooth",
                    )
                )
                position = _linear_phase_endpoint(candidate, phase, position)
                keyframes.append(
                    TrackKeyframe(
                        time_seconds=_track_end_time(candidate, end),
                        value=TransformValue(translation_m=tuple(position)),
                        interpolation="smooth",
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

        hidden_times: list[tuple[float, str]] = []
        for phase in phases:
            if phase.kind == "board":
                hidden_times.append(
                    (
                        _event_range(objective, phase.timeline_event_id, duration)[1],
                        phase.source_ref,
                    )
                )
            elif phase.kind == "carried":
                hidden_times.append(
                    (
                        _event_range(objective, phase.timeline_event_id, duration)[0],
                        phase.source_ref,
                    )
                )
        if hidden_times:
            hidden_at, source_ref = min(hidden_times)
            hidden_at = _track_end_time(candidate, hidden_at)
            track = TrackSpec(
                track_id=f"design_visibility_{subject_id}",
                target_entity_id=subject_id,
                type="visibility",
                time_range_seconds=(0.0, duration),
                keyframes=[
                    TrackKeyframe(time_seconds=0.0, value=True, interpolation="step"),
                    TrackKeyframe(time_seconds=hidden_at, value=False, interpolation="step"),
                ],
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
    start_distance = float(parameters.get("start_distance_m") or profile.default_camera_distance_m)
    end_distance = float(parameters.get("end_distance_m") or start_distance)
    distance_scale = {
        "balanced": 1.5,
        "preserve_composition": 2.4,
        "maximize_motion_readability": 1.15,
    }[strategy]
    start_distance *= distance_scale
    end_distance *= distance_scale
    height = float(parameters.get("height_m") or 1.5)
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
    movement = skeleton.camera_intent.movement
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


def _numeric_envelopes(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
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
        "motion_speed_ranges_mps": motion_speeds,
        "entity_size_ranges_m": {
            entity_id: _entity_size_range(entity)
            for entity_id, entity in candidate.entities.items()
            if entity.proxy.type != "plane"
        },
    }


def _design_assumptions(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    strategy: str,
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
    return tuple(assumptions)


def _skeleton_sources(value: SceneSkeleton) -> list[tuple[str, SourceStatus]]:
    result: list[tuple[str, SourceStatus]] = []
    for entity in value.entities:
        result.extend((source_ref, "explicit") for source_ref in entity.source_refs)
    result.extend((item.source_ref, item.source_status) for item in value.relations)
    result.extend((item.source_ref, item.source_status) for item in value.motion_phases)
    result.append((value.camera_intent.source_ref, value.camera_intent.source_status))
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


def _track_end_time(candidate: CandidateState, end: float) -> float:
    frame_step = (
        candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    )
    return min(end, candidate.timeline.duration_seconds - frame_step)


def _linear_phase_endpoint(
    candidate: CandidateState,
    phase: SkeletonMotionPhase,
    position: list[float],
) -> list[float]:
    endpoint = list(position)
    if phase.direction_mode == "toward_target" and phase.target_id:
        target = candidate.entities[phase.target_id].solved_transform.translation_m
        if target is not None:
            endpoint[0] = target[0] - 2.5
            endpoint[1] = target[1]
    elif phase.direction_mode == "away_from_target":
        endpoint[0] += 8.0
    elif phase.direction_mode == "screen_right_to_left":
        endpoint[0] -= 8.0
    else:
        endpoint[0] += 8.0
    return endpoint


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


def _entity_size_range(entity: EntitySpec) -> dict[str, list[float]]:
    proxy = entity.proxy
    if proxy.type == "box":
        dimensions = list(proxy.size_xyz_m)
    elif proxy.type == "sphere":
        dimensions = [proxy.radius_m * 2.0] * 3
    elif proxy.type == "capsule":
        diameter = proxy.radius_m * 2.0
        height = proxy.segment_length_m + diameter
        dimensions = [diameter, diameter, height]
    elif proxy.type in {"cylinder", "cone"}:
        radius = (
            proxy.radius_m
            if proxy.type == "cylinder"
            else max(proxy.radius_bottom_m, proxy.radius_top_m)
        )
        dimensions = [radius * 2.0, radius * 2.0, proxy.depth_m]
    else:
        dimensions = [proxy.size_xy_m[0], proxy.size_xy_m[1], 0.0]
    return {
        "minimum_xyz": [round(item * 0.8, 6) for item in dimensions],
        "maximum_xyz": [round(item * 1.2, 6) for item in dimensions],
    }
