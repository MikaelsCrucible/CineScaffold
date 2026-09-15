from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, model_validator

from cinescaffold.camera_semantics import (
    camera_lens_focal_length,
    classify_camera_movement,
    classify_camera_view_angle,
)
from cinescaffold.motion_semantics import planning_motion_shape
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
    add,
    directional_support_extent,
    dot,
    look_at_camera_quaternion,
    normalize,
    project_geometry_bounds,
    quaternion_multiply,
    rotate_vector,
    sample_path_track,
    sample_transform_track,
    subtract,
    surface_clearance_target_distance_m,
)
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.planning.store import canonical_hash
from cinescaffold.relationships import classify_relationship

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
    scale_intent: Literal["tiny", "small", "human", "large", "huge", "unspecified"] = (
        Field(
            default="unspecified",
            description="不含米制数值的绝对尺度档位；语义上可判断大小时不得全部留空",
        )
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
                if not math.isfinite(preferred) or not minimum <= preferred <= maximum:
                    raise ValueError(f"自定义尺寸第 {axis + 1} 轴偏好值越界")
        return self


class SkeletonRelation(StrictModel):
    relation_id: str = Field(min_length=1)
    kind: Literal[
        "camera_depth_order",
        "relative_position",
        "proximity",
        "scale_dominance",
    ]
    subject_id: str
    reference_id: str
    direction: Literal["left", "right", "front", "behind", "below", "above"] | None = (
        None
    )
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
        description="Cinematic Brief v0.7 的稳定动作身份，用于叙事完整性校验",
    )
    subject_id: str
    kind: Literal[
        "hold",
        "path_move",
        "carried",
        "local_transform",
        "visibility",
    ]
    timeline_event_id: str | None = None
    target_id: str | None = None
    carrier_id: str | None = None
    direction_mode: Literal[
        "none",
        "world_forward",
        "toward_target",
        "away_from_target",
        "world_left",
        "world_right",
        "relative_to_target",
    ] = Field(
        default="none",
        description=(
            "主体的几何运动方向；事件参与者不是 target。world_forward 表示规范世界 -Y，"
            "world_left/world_right 表示规范世界 -X/+X；这些名称不承诺最终屏幕方向"
        ),
    )
    path_family: Literal[
        "stationary",
        "linear",
        "circle",
        "ellipse",
        "catmull_rom",
        "lemniscate",
        "parabolic",
    ] = "stationary"
    local_components: list[Literal["rotation", "scale"]] = Field(
        default_factory=list,
        description="local_transform 的可观察局部变化通道；不表示世界空间位移",
    )
    speed_intent: Literal["stationary", "slow", "medium", "fast", "unspecified"] = (
        "unspecified"
    )
    speed_source_status: SourceStatus | None = None
    speed_source_ref: str | None = None
    visibility_state: Literal["visible", "hidden"] | None = None
    transition_at: Literal["at_start", "at_end"] | None = None
    source_status: SourceStatus
    source_ref: str
    narrative_required: bool = False

    @model_validator(mode="after")
    def validate_motion_shape(self) -> SkeletonMotionPhase:
        if self.target_id == self.subject_id:
            raise ValueError("Motion Phase target_id 不得指向运动主体自身")
        if self.carrier_id == self.subject_id:
            raise ValueError("Motion Phase carrier_id 不得指向被携带主体自身")
        if self.kind == "carried" and self.carrier_id is None:
            raise ValueError("carried 必须提供 carrier_id")
        if self.kind != "carried" and self.carrier_id is not None:
            raise ValueError(f"{self.kind} 不接受 carrier_id")
        if self.direction_mode in {
            "toward_target",
            "away_from_target",
            "relative_to_target",
        }:
            if self.target_id is None:
                raise ValueError(f"{self.direction_mode} 必须提供 target_id")
        elif self.target_id is not None:
            raise ValueError(f"{self.direction_mode} 不接受 target_id")
        if (
            self.kind in {"hold", "carried", "local_transform", "visibility"}
            and self.direction_mode != "none"
        ):
            raise ValueError(f"{self.kind} 必须使用 direction_mode=none")
        if self.kind == "path_move" and self.path_family == "stationary":
            raise ValueError("path_move 不能使用 stationary")
        if self.path_family in {"circle", "ellipse"} and (
            self.kind != "path_move"
            or self.direction_mode != "relative_to_target"
            or self.target_id is None
        ):
            raise ValueError(
                "circle/ellipse 公转必须使用 path_move + relative_to_target + target_id"
            )
        if self.direction_mode == "relative_to_target" and self.path_family not in {
            "circle",
            "ellipse",
        }:
            raise ValueError("relative_to_target 当前只支持 circle 或 ellipse")
        if self.kind == "hold" and self.path_family != "stationary":
            raise ValueError("hold 必须使用 stationary")
        if self.kind == "hold" and self.speed_intent not in {
            "stationary",
            "unspecified",
        }:
            raise ValueError("hold 不能同时声明非零位移速度")
        if self.kind == "path_move" and self.path_family not in {
            "linear",
            "circle",
            "ellipse",
            "catmull_rom",
            "lemniscate",
            "parabolic",
        }:
            raise ValueError("path_move 必须使用已实现的路径族")
        if self.kind == "path_move" and self.speed_intent == "stationary":
            raise ValueError("path_move 不能使用 stationary 速度")
        if (
            self.kind in {"carried", "local_transform", "visibility"}
            and self.path_family != "stationary"
        ):
            raise ValueError(
                f"{self.kind} 必须使用 stationary；世界运动由载体或显隐语义负责"
            )
        if self.kind == "local_transform" and not self.local_components:
            raise ValueError("local_transform 至少需要 rotation 或 scale 通道")
        if self.kind != "local_transform" and self.local_components:
            raise ValueError(f"{self.kind} 不接受 local_components")
        visibility_fields = (self.visibility_state, self.transition_at)
        if self.kind == "visibility" and any(
            item is None for item in visibility_fields
        ):
            raise ValueError("visibility 必须提供 visibility_state 与 transition_at")
        if self.kind == "visibility" and (
            self.path_family != "stationary"
            or self.direction_mode != "none"
            or self.speed_intent != "unspecified"
        ):
            raise ValueError("visibility 只能改变可见性，不接受路径、方向或速度")
        if self.kind != "visibility" and any(
            item is not None for item in visibility_fields
        ):
            raise ValueError(f"{self.kind} 不接受 visibility_state 或 transition_at")
        if (self.speed_source_status is None) != (self.speed_source_ref is None):
            raise ValueError("speed_source_status 与 speed_source_ref 必须同时提供")
        return self


class SkeletonRouteAnchor(StrictModel):
    anchor_id: str = Field(min_length=1)
    phase_id: str = Field(
        min_length=1,
        description="包含该事件路径点的线性运动阶段",
    )
    relation_id: str = Field(
        min_length=1,
        description=(
            "决定路径点时间与空间条件的既有 proximity 或 relative_position 关系；"
            "时间由该关系引用的 timeline event 与 temporal_mode 唯一确定"
        ),
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
        "static", "push_in", "pull_out", "follow", "orbit", "lateral", "pan"
    ] = Field(
        description="只描述摄影机自身运动，与 scene_dynamics 的主体静/动态分类彼此独立"
    )
    focus_target_id: str | None = Field(
        default=None,
        description=(
            "可选的初始取景主体；null 表示使用稳定的场景锚点，不得为了满足 Schema "
            "而臆造逐帧跟踪目标"
        ),
    )
    movement_target_id: str | None = Field(
        default=None,
        description=(
            "pan/follow/orbit 的运动目标，来自 camera.movement.target_id；"
            "与初始取景 focus_target_id 相互独立"
        ),
    )
    view_relation_to_motion: Literal[
        "front", "rear", "side", "three_quarter", "unspecified"
    ] = Field(
        default="unspecified",
        description=(
            "摄影机观察方向相对主体线性运动的关系；不存在主体线性运动或 Brief 未明确时使用 unspecified；"
            "它不描述摄影机自身如何移动"
        ),
    )
    speed_intent: Literal[
        "stationary", "slow", "medium", "fast", "match_subject", "unspecified"
    ] = "unspecified"
    speed_source_status: SourceStatus | None = None
    speed_source_ref: str | None = None
    source_status: SourceStatus
    source_ref: str

    @model_validator(mode="after")
    def validate_speed_source(self) -> SkeletonCameraIntent:
        if (self.speed_source_status is None) != (self.speed_source_ref is None):
            raise ValueError("speed_source_status 与 speed_source_ref 必须同时提供")
        if self.movement == "static" and self.speed_intent not in {
            "stationary",
            "unspecified",
        }:
            raise ValueError("静止摄影机不能同时声明非零运动速度")
        if self.movement != "static" and self.speed_intent == "stationary":
            raise ValueError("运动摄影机不能使用 stationary 速度")
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
        if (
            self.camera_intent.focus_target_id is not None
            and self.camera_intent.focus_target_id not in known
        ):
            raise ValueError("摄影机观察目标不在 Scene Skeleton 中")
        if (
            self.camera_intent.movement_target_id is not None
            and self.camera_intent.movement_target_id not in known
        ):
            raise ValueError("摄影机运动目标不在 Scene Skeleton 中")
        if (
            self.camera_intent.movement_target_id is not None
            and self.camera_intent.movement not in {"pan", "follow", "orbit"}
        ):
            raise ValueError("只有 pan/follow/orbit 可以设置摄影机运动目标")
        if (
            self.camera_intent.movement in {"pan", "follow", "orbit"}
            and self.camera_intent.movement_target_id is None
        ):
            raise ValueError(
                "pan/follow/orbit 摄影机必须单独提供 movement_target_id；"
                "focus_target_id 不能代替运动目标"
            )
        relation_shapes: set[tuple[object, ...]] = set()
        distance_meanings: dict[tuple[object, ...], set[str]] = {}
        far_orientations: dict[
            tuple[tuple[str, str], str | None, str],
            set[tuple[str, str]],
        ] = {}
        for relation in self.relations:
            if relation.subject_id not in known or relation.reference_id not in known:
                raise ValueError(f"Relation 引用了未知 Entity：{relation.relation_id}")
            if relation.subject_id == relation.reference_id:
                raise ValueError(f"Relation 不得自引用：{relation.relation_id}")
            if (
                relation.timeline_event_id is None
                and relation.temporal_mode != "throughout"
            ):
                raise ValueError(
                    f"at_start/at_end Relation 必须引用 timeline event：{relation.relation_id}"
                )
            unordered_pair = tuple(sorted((relation.subject_id, relation.reference_id)))
            pair: tuple[str, str] = (
                unordered_pair
                if relation.kind == "proximity"
                else (relation.subject_id, relation.reference_id)
            )
            timing = (relation.timeline_event_id, relation.temporal_mode)
            shape = (relation.kind, relation.direction, pair, *timing)
            if shape in relation_shapes:
                raise ValueError(
                    f"Scene Skeleton 包含语义重复的 Relation：{relation.relation_id}"
                )
            relation_shapes.add(shape)
            if relation.kind in {"camera_depth_order", "proximity"}:
                distance_meanings.setdefault((unordered_pair, *timing), set()).add(
                    relation.kind
                )
            if relation.kind == "camera_depth_order":
                far_orientations.setdefault(
                    (unordered_pair, *timing),
                    set(),
                ).add((relation.subject_id, relation.reference_id))
        if any(
            meanings == {"camera_depth_order", "proximity"}
            for meanings in distance_meanings.values()
        ):
            raise ValueError("同一实体对在同一时间范围内不能同时远离并靠近")
        if any(len(items) > 1 for items in far_orientations.values()):
            raise ValueError("同一实体对在同一时间范围内不能互相都位于远景")
        for phase in self.motion_phases:
            references = [phase.subject_id, phase.target_id, phase.carrier_id]
            if any(item is not None and item not in known for item in references):
                raise ValueError(f"Motion Phase 引用了未知 Entity：{phase.phase_id}")
        phases_by_subject: dict[str, list[SkeletonMotionPhase]] = {}
        for phase in self.motion_phases:
            phases_by_subject.setdefault(phase.subject_id, []).append(phase)
        for subject_id, phases in phases_by_subject.items():
            orbit_count = sum(_is_orbit_phase(phase) for phase in phases)
            carried_count = sum(phase.kind == "carried" for phase in phases)
            if orbit_count > 1:
                raise ValueError(f"同一主体暂不支持多个 orbit 阶段：{subject_id}")
            if orbit_count and any(
                (phase.kind == "path_move" and not _is_orbit_phase(phase))
                or phase.kind == "carried"
                for phase in phases
            ):
                raise ValueError(
                    f"同一主体不能在符号骨架中混用 orbit 与其他世界运动：{subject_id}"
                )
            if carried_count > 1:
                raise ValueError(f"同一主体暂不支持多个 carried 阶段：{subject_id}")
        if (
            self.camera_intent.movement == "pan"
            and self.camera_intent.focus_target_id
            == self.camera_intent.movement_target_id
            and not any(
                phase.subject_id == self.camera_intent.movement_target_id
                and phase.kind in {"path_move", "carried"}
                for phase in self.motion_phases
            )
        ):
            raise ValueError("pan 的静态起始注视点与结束目标相同，无法产生旋转")
        phases_by_id = {item.phase_id: item for item in self.motion_phases}
        relations_by_id = {item.relation_id: item for item in self.relations}
        for route in self.route_intents:
            if route.subject_id not in known:
                raise ValueError(f"Route Intent 引用了未知主体：{route.route_id}")
            if route.axis_reference_id is not None:
                if route.axis_reference_id not in known:
                    raise ValueError(
                        f"Route Intent 引用了未知路线参照：{route.route_id}"
                    )
                if route.axis_reference_id == route.subject_id:
                    raise ValueError(
                        f"Route Intent 不得把运动主体用作路线参照：{route.route_id}"
                    )
            anchor_ids = [item.anchor_id for item in route.anchors]
            if len(anchor_ids) != len(set(anchor_ids)):
                raise ValueError(f"Route Anchor ID 不得重复：{route.route_id}")
            anchor_relation_ids = [item.relation_id for item in route.anchors]
            if len(anchor_relation_ids) != len(set(anchor_relation_ids)):
                raise ValueError(
                    f"同一 Route Intent 不得重复锚定同一 Relation：{route.route_id}"
                )
            for anchor in route.anchors:
                phase = phases_by_id.get(anchor.phase_id)
                if phase is None:
                    raise ValueError(
                        f"Route Anchor 引用了未知 Motion Phase：{anchor.anchor_id}"
                    )
                if (
                    phase.subject_id != route.subject_id
                    or phase.kind != "path_move"
                    or _is_orbit_phase(phase)
                ):
                    raise ValueError(
                        f"Route Anchor 只能绑定同一主体的非公转 path_move：{anchor.anchor_id}"
                    )
                relation = relations_by_id.get(anchor.relation_id)
                if relation is None:
                    raise ValueError(
                        f"Route Anchor 引用了未知 Relation：{anchor.anchor_id}"
                    )
                if relation.kind not in {"proximity", "relative_position"}:
                    raise ValueError(
                        f"Route Anchor 只支持可落为路径点的空间关系：{anchor.anchor_id}"
                    )
                if route.subject_id not in {relation.subject_id, relation.reference_id}:
                    raise ValueError(
                        f"Route Anchor 的空间关系不包含运动主体：{anchor.anchor_id}"
                    )
                if relation.timeline_event_id is None:
                    raise ValueError(
                        f"Route Anchor 必须引用带 timeline event 的空间关系：{anchor.anchor_id}"
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
        str(item.get("id")) for item in objective.subjects if item.get("id") is not None
    }
    skeleton_ids = {item.entity_id for item in value.entities}
    missing = sorted(objective_subject_ids - skeleton_ids)
    if missing:
        raise ValueError(f"Scene Skeleton 遗漏 Brief 主体：{', '.join(missing)}")

    dynamics_mode = objective.scene_dynamics.get("mode")
    if dynamics_mode == "static":
        changing_phases = [
            item.phase_id for item in value.motion_phases if item.kind != "hold"
        ]
        if changing_phases:
            raise ValueError(
                "scene_dynamics=static 表示主体状态不变，不能提交主体运动或状态转换阶段："
                + ", ".join(changing_phases)
            )
    elif dynamics_mode == "dynamic" and not any(
        item.kind != "hold" for item in value.motion_phases
    ):
        raise ValueError(
            "scene_dynamics=dynamic 必须包含至少一个主体位移、局部变化、显隐或携带阶段"
        )

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

    duration_node = objective.timeline.get("duration_resolution")
    duration_value = (
        duration_node.get("resolved_duration_seconds")
        if isinstance(duration_node, dict)
        else objective.timeline.get("duration_seconds")
    )
    if (
        not isinstance(duration_value, (int, float))
        or isinstance(duration_value, bool)
        or duration_value <= 0
    ):
        raise ValueError("Objective Planning Brief 缺少有效冻结时长")
    phase_ranges = {
        phase.phase_id: _phase_range(objective, phase, float(duration_value))
        for phase in value.motion_phases
    }
    _validate_phase_overlap_support(value.motion_phases, phase_ranges)
    _camera_motion_range(objective, float(duration_value))
    frame_count = (
        duration_node.get("frame_count") if isinstance(duration_node, dict) else None
    )
    frame_step = (
        float(duration_value) / float(frame_count)
        if isinstance(frame_count, int)
        and not isinstance(frame_count, bool)
        and frame_count > 0
        else 1.0 / 24.0
    )
    relations_by_id = {item.relation_id: item for item in value.relations}
    anchors_by_subject_time: dict[tuple[str, float], str] = {}
    for route in value.route_intents:
        for anchor in route.anchors:
            phase = next(
                item for item in value.motion_phases if item.phase_id == anchor.phase_id
            )
            relation = relations_by_id[anchor.relation_id]
            anchor_time = route_anchor_time_seconds(
                objective,
                relation,
                float(duration_value),
                frame_step,
                phase=phase,
            )
            phase_start, phase_end = phase_ranges[phase.phase_id]
            phase_sample_end = max(
                phase_start,
                min(
                    phase_end - frame_step,
                    float(duration_value) - frame_step,
                ),
            )
            if not (
                phase_start - 1e-9
                <= anchor_time
                <= phase_sample_end + 1e-9
            ):
                raise ValueError(
                    "Route Anchor 的事件时刻不在所引用 Motion Phase 内："
                    f"{anchor.anchor_id}"
                )
            slot = (route.subject_id, round(anchor_time, 9))
            previous = anchors_by_subject_time.get(slot)
            if previous is not None:
                raise ValueError(
                    "同一主体在同一事件时刻只能有一个 Route Anchor："
                    f"{previous} / {anchor.anchor_id}"
                )
            anchors_by_subject_time[slot] = anchor.anchor_id
    distance_relations: dict[
        tuple[str, str],
        list[tuple[str, tuple[float, float], str]],
    ] = {}
    for relation in value.relations:
        if relation.kind not in {"camera_depth_order", "proximity"}:
            continue
        pair = tuple(sorted((relation.subject_id, relation.reference_id)))
        window = _relation_semantic_window(
            objective,
            relation,
            float(duration_value),
            frame_step,
        )
        distance_relations.setdefault(pair, []).append(
            (relation.kind, window, relation.relation_id)
        )
    for values in distance_relations.values():
        for index, first in enumerate(values):
            for second in values[index + 1 :]:
                if first[0] == second[0] or not _ranges_overlap(first[1], second[1]):
                    continue
                raise ValueError(
                    "同一实体对在重叠时间范围内不能同时远离并靠近："
                    f"{first[2]} / {second[2]}"
                )

    if objective.schema_version == "0.7":
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
        required_motions = {
            str(item.get("motion_id")): item
            for item in objective.subject_motion
            if isinstance(item, dict)
            and isinstance(item.get("motion_semantics"), dict)
            and item["motion_semantics"].get("narrative_required") is True
        }
        phases_by_motion_id: dict[str, list[SkeletonMotionPhase]] = {}
        for phase in value.motion_phases:
            if phase.motion_id is not None:
                phases_by_motion_id.setdefault(phase.motion_id, []).append(phase)
        missing_motion_ids = sorted(
            motion_id
            for motion_id, motion in required_motions.items()
            if not any(
                _phase_matches_motion_semantics(
                    phase, motion.get("motion_semantics", {})
                )
                for phase in phases_by_motion_id.get(motion_id, [])
            )
        )
        if missing_motion_ids:
            raise ValueError(
                "Scene Skeleton 遗漏或错误表达关键叙事动作："
                + ", ".join(missing_motion_ids)
            )

    valid_explicit_refs = {item.path for item in objective.explicit_requirements}
    for source_ref, status in _skeleton_sources(value):
        if status == "explicit" and source_ref not in valid_explicit_refs:
            raise ValueError(f"Scene Skeleton explicit source_ref 不存在：{source_ref}")

    if objective.schema_version == "0.7":
        _validate_typed_source_bindings(objective, value)

    camera = objective.camera
    movement_node = camera.get("movement", {}).get("type")
    if (
        isinstance(movement_node, dict)
        and movement_node.get("source_status") == "explicit"
    ):
        if value.camera_intent.source_status != "explicit":
            raise ValueError("Scene Skeleton 不得降低明确摄影机运动的来源等级")
        expected_movement = classify_camera_movement(movement_node.get("value"))
        if (
            expected_movement is not None
            and value.camera_intent.movement != expected_movement
        ):
            raise ValueError(
                "Scene Skeleton 摄影机运动与 Brief 明确要求不一致："
                f"{value.camera_intent.movement} != {expected_movement}"
            )
    movement_target_id = camera.get("movement", {}).get("target_id")
    if value.camera_intent.movement_target_id != movement_target_id:
        raise ValueError("Scene Skeleton 摄影机运动目标与 Brief 不一致")
    view_node = camera.get("view_relation_to_motion")
    if isinstance(view_node, dict) and view_node.get("source_status") == "explicit":
        expected_view = view_node.get("value")
        if value.camera_intent.view_relation_to_motion != expected_view:
            raise ValueError("Scene Skeleton 摄影机观察关系与 Brief 明确要求不一致")
        if expected_view != "unspecified" and not any(
            phase.kind == "path_move" and not _is_orbit_phase(phase)
            for phase in value.motion_phases
        ):
            raise ValueError("摄影机相对运动视角需要至少一个非公转 path_move 阶段")
    focus_node = camera.get("focus_target_id")
    if isinstance(focus_node, dict) and focus_node.get("source_status") == "explicit":
        if value.camera_intent.focus_target_id != focus_node.get("value"):
            raise ValueError("Scene Skeleton 摄影机焦点与 Brief 明确要求不一致")


def _phase_matches_motion_semantics(
    phase: SkeletonMotionPhase,
    semantics: dict[str, Any],
) -> bool:
    """Require a narrative motion ID to be represented by a compatible phase."""

    try:
        expected = planning_motion_shape(semantics)
    except ValueError:
        return False
    return phase.kind == expected.kind


def _is_orbit_phase(phase: SkeletonMotionPhase) -> bool:
    """An orbit is derived from path geometry and reference frame, not a second enum."""

    return (
        phase.kind == "path_move"
        and phase.direction_mode == "relative_to_target"
        and phase.target_id is not None
        and phase.path_family in {"circle", "ellipse"}
    )


def _validate_phase_overlap_support(
    phases: list[SkeletonMotionPhase],
    ranges: dict[str, tuple[float, float]],
) -> None:
    """Reject same-subject transform combinations the deterministic IR cannot mean."""

    transform_kinds = {"hold", "local_transform", "path_move", "carried"}
    by_subject: dict[str, list[SkeletonMotionPhase]] = {}
    for phase in phases:
        if phase.kind in transform_kinds:
            by_subject.setdefault(phase.subject_id, []).append(phase)
    for subject_id, subject_phases in by_subject.items():
        for index, first in enumerate(subject_phases):
            for second in subject_phases[index + 1 :]:
                if not _ranges_overlap(ranges[first.phase_id], ranges[second.phase_id]):
                    continue
                pair = {first.kind, second.kind}
                if "local_transform" in pair and len(pair) == 2 and "hold" not in pair:
                    continue
                raise ValueError(
                    "同一主体存在无法同时成立的重叠运动阶段："
                    f"{subject_id} / {first.phase_id} / {second.phase_id}"
                )


def _relation_semantic_window(
    objective: ObjectivePlanningBrief,
    relation: SkeletonRelation,
    duration: float,
    frame_step: float,
) -> tuple[float, float]:
    start, end = _event_range(objective, relation.timeline_event_id, duration)
    if relation.temporal_mode == "at_start":
        return (start, min(end, start + frame_step))
    if relation.temporal_mode == "at_end":
        return (max(start, end - frame_step), end)
    return (start, end)


def route_anchor_time_seconds(
    objective: ObjectivePlanningBrief,
    relation: SkeletonRelation,
    duration: float,
    frame_step: float,
    *,
    phase: SkeletonMotionPhase | None = None,
) -> float:
    """Resolve a symbolic relation to one renderable waypoint time."""

    if relation.timeline_event_id is None:
        raise ValueError(
            f"Route Anchor Relation 缺少 timeline event：{relation.relation_id}"
        )
    start, end = _event_range(objective, relation.timeline_event_id, duration)
    if relation.temporal_mode == "at_start":
        anchor_time = start
    elif relation.temporal_mode == "at_end":
        anchor_time = max(start, min(end - frame_step, duration - frame_step))
    else:
        anchor_time = min((start + end) * 0.5, duration - frame_step)
    if phase is None:
        return anchor_time
    phase_start, phase_end = _phase_range(objective, phase, duration)
    phase_sample_end = max(
        phase_start,
        min(phase_end - frame_step, duration - frame_step),
    )
    # A motion endpoint remains the entity's world position until another
    # translation phase changes it, so a later relation may legitimately bind
    # the endpoint of the phase that brought the entity there.
    return min(max(anchor_time, phase_start), phase_sample_end)


def route_anchor_phase_time_range(
    objective: ObjectivePlanningBrief,
    phase: SkeletonMotionPhase,
    duration: float,
) -> tuple[float, float]:
    """Expose the authoritative motion interval used by route validation."""

    return _phase_range(objective, phase, duration)


def _ranges_overlap(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _validate_typed_source_bindings(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
) -> None:
    """Keep v0.7 symbolic choices attached to the typed fact they cite."""

    relationship_prefix = "content.scene_design.relationships["
    objective_relationships = objective.scene_design.get("relationships", [])
    expected_relation_kinds = {
        "far": "camera_depth_order",
        "proximity": "proximity",
        "relative_position": "relative_position",
        "scale_dominance": "scale_dominance",
    }
    for relation in skeleton.relations:
        if not relation.source_ref.startswith(relationship_prefix):
            continue
        try:
            index = int(
                relation.source_ref[len(relationship_prefix) :].split("]", 1)[0]
            )
            source = objective_relationships[index]
        except (IndexError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(
                f"Relation source_ref 无法解析：{relation.relation_id}"
            ) from error
        meaning = classify_relationship(source) if isinstance(source, dict) else None
        expected_kind = (
            expected_relation_kinds.get(meaning.kind) if meaning is not None else None
        )
        expected_shape = (
            expected_kind,
            source.get("subject_id"),
            source.get("reference_id"),
            meaning.direction if meaning is not None else None,
            source.get("timeline_event_id"),
            source.get("temporal_mode", "throughout"),
        )
        actual_shape = (
            relation.kind,
            relation.subject_id,
            relation.reference_id,
            relation.direction,
            relation.timeline_event_id,
            relation.temporal_mode,
        )
        if expected_kind is None or actual_shape != expected_shape:
            raise ValueError(
                f"Relation 与 source_ref 的规范语义不一致：{relation.relation_id}"
            )
        if (
            source.get("source_status") == "explicit"
            and relation.source_status != "explicit"
        ):
            raise ValueError(
                f"Relation 不得降低明确空间关系的来源等级：{relation.relation_id}"
            )

    motion_prefix = "content.subject_motion["
    motion_indexes_by_id = {
        str(motion.get("motion_id")): index
        for index, motion in enumerate(objective.subject_motion)
        if isinstance(motion, dict) and motion.get("motion_id") is not None
    }
    explicit_refs = {item.path for item in objective.explicit_requirements}
    for phase in skeleton.motion_phases:
        source_index: int | None = None
        if phase.source_ref.startswith(motion_prefix):
            try:
                source_index = int(
                    phase.source_ref[len(motion_prefix) :].split("]", 1)[0]
                )
                source_motion = objective.subject_motion[source_index]
            except (IndexError, TypeError, ValueError, AttributeError) as error:
                raise ValueError(
                    f"Motion Phase source_ref 无法解析：{phase.phase_id}"
                ) from error
        else:
            source_motion = None

        motion_index = (
            motion_indexes_by_id.get(phase.motion_id)
            if phase.motion_id is not None
            else source_index
        )
        if motion_index is None:
            continue
        motion = objective.subject_motion[motion_index]
        if source_index is not None and source_index != motion_index:
            raise ValueError(
                f"Motion Phase 的 motion_id 与 source_ref 指向不同动作：{phase.phase_id}"
            )
        if phase.motion_id is not None and source_index is None:
            raise ValueError(
                f"Motion Phase 必须把 motion_id 绑定到对应 subject_motion 来源：{phase.phase_id}"
            )
        if source_motion is not None and source_motion is not motion:
            raise ValueError(
                f"Motion Phase 的 source_ref 与 motion_id 不一致：{phase.phase_id}"
            )
        if not isinstance(motion, dict) or motion.get("subject_id") != phase.subject_id:
            raise ValueError(
                f"Motion Phase 与 source_ref 的主体不一致：{phase.phase_id}"
            )
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        expected_motion_id = motion.get("motion_id")
        if phase.motion_id != expected_motion_id:
            raise ValueError(
                f"Motion Phase 与 source_ref 的 motion_id 不一致：{phase.phase_id}"
            )
        if phase.timeline_event_id != semantics.get("timeline_event_id"):
            raise ValueError(
                f"Motion Phase 与 source_ref 的事件绑定不一致：{phase.phase_id}"
            )
        explicit_motion_refs = {
            ref
            for ref in explicit_refs
            if ref.startswith(f"content.subject_motion[{motion_index}].")
            and ref.rsplit(".", 1)[-1]
            in {"action", "motion_semantics", "direction", "trajectory"}
        }
        if explicit_motion_refs and (
            phase.source_status != "explicit"
            or phase.source_ref not in explicit_motion_refs
        ):
            raise ValueError(
                f"Motion Phase 不得脱离对应的明确动作来源：{phase.phase_id}"
            )
        if (
            semantics.get("source_status") == "explicit"
            and phase.source_status != "explicit"
        ):
            raise ValueError(
                f"Motion Phase 不得降低明确动作的来源等级：{phase.phase_id}"
            )
        if phase.kind == "visibility":
            postconditions = semantics.get("postconditions")
            expected_visibility = (
                postconditions.get("external_visibility")
                if isinstance(postconditions, dict)
                else None
            )
            if phase.visibility_state != expected_visibility:
                raise ValueError(
                    f"Visibility Phase 与 source_ref 的后置状态不一致：{phase.phase_id}"
                )
            continue
        elif not _phase_matches_motion_semantics(phase, semantics):
            raise ValueError(
                f"Motion Phase 与 source_ref 的运动模式不一致：{phase.phase_id}"
            )
        expected_shape = planning_motion_shape(semantics)
        if expected_shape.kind == "carried" and (
            phase.carrier_id != expected_shape.carrier_id
        ):
            raise ValueError(
                f"Carried Phase 与 source_ref 的载体不一致：{phase.phase_id}"
            )
        if semantics.get("direction_mode") not in {None, "none"} and (
            phase.direction_mode != expected_shape.direction_mode
            or phase.target_id != expected_shape.target_id
        ):
            raise ValueError(
                f"Motion Phase 与 source_ref 的方向/目标不一致：{phase.phase_id}"
            )
        if semantics.get("path_type") not in {None, "unspecified"} and (
            phase.path_family != expected_shape.path_family
        ):
            raise ValueError(
                f"Motion Phase 与 source_ref 的路径族不一致：{phase.phase_id}"
            )


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
    if skeleton.camera_intent.speed_intent not in {
        "stationary",
        "match_subject",
        "unspecified",
    }:
        speed_intents.add(skeleton.camera_intent.speed_intent)
    has_subject_translation = any(
        phase.kind in {"path_move", "carried"} for phase in skeleton.motion_phases
    )
    has_linear_subject_motion = any(
        phase.kind == "path_move" and not _is_orbit_phase(phase)
        for phase in skeleton.motion_phases
    )
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
        "path_realization": {
            "linear": "transform keyframes",
            "parabolic": "transform keyframes with an interior apex",
            "circle": "analytic path_follow",
            "ellipse": "analytic path_follow",
            "catmull_rom": "path_follow",
            "lemniscate": "analytic path_follow",
        },
        "route_planning": {
            "decision_owner": "planning_agent",
            "coordinate_owner": "toolkit",
            "anchor_relations": ["proximity", "relative_position"],
            "anchor_timing": "referenced relation timeline event and temporal_mode",
            "interior_event_waypoints": True,
            "joint_moving_proximity": True,
            "continuity_modes": ["preserve_direction", "allow_turns"],
            "axis_reference": "optional entity horizontal long axis",
        },
        "speed_ranges_mps": {
            intent: list(_speed_range(intent, profile))
            for intent in sorted(speed_intents)
        },
        "camera_speed_policy": (
            "target_relative_follow"
            if skeleton.camera_intent.speed_intent == "match_subject"
            else "absolute_range"
        ),
        "size_design": {
            "scale_intents": ["tiny", "small", "human", "large", "huge", "unspecified"],
            "proportion_intents": [
                "isotropic",
                "flat",
                "wide",
                "tall",
                "elongated",
                "unspecified",
            ],
            "custom_size_requests": {
                "unit": "meter",
                "value": "三轴完整包围盒尺寸范围，不是半尺寸",
                "selection": "Toolkit 按候选策略在范围内选择并完整验证",
            },
        },
        "motion_readability": {
            "applies_to": "subject_spatial_motion_only",
            "subject_translation_present": has_subject_translation,
            "camera_motion_counts_as_subject_motion": False,
            "maximize_motion_readability_applicable": has_subject_translation,
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
            "minimum_view_subject_motion_obliqueness_degrees": (
                profile.minimum_view_subject_motion_obliqueness_degrees
                if has_linear_subject_motion
                else None
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
    orbit_radii = _orbit_radius_map(candidate, skeleton, profile)
    candidate.constraints = _build_relation_constraints(
        objective,
        skeleton,
        candidate,
        profile,
        orbit_radii,
    )
    candidate.motion_tracks = _build_motion(
        objective,
        skeleton,
        candidate,
        profile,
        orbit_radii,
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
    has_subject_translation = any(
        phase.kind in {"path_move", "carried"} for phase in skeleton.motion_phases
    )
    if objective.scene_dynamics.get("mode") == "static" or not has_subject_translation:
        _fit_static_camera_to_composition(objective, candidate, skeleton, profile)
    _fit_open_environment_ground_to_camera(objective, candidate, profile)
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
    ground_ids = [
        item.entity_id
        for item in skeleton.entities
        if item.proxy_family == "ground_plane"
    ]
    entities: dict[str, EntitySpec] = {}
    for item in skeleton.entities:
        parameters = subject_parameters.get(item.entity_id, {})
        source_refs = set(item.source_refs)
        source_refs.update(_entity_explicit_refs(objective, item.entity_id))
        if item.proxy_family == "ground_plane":
            source_refs.update(
                requirement.path
                for requirement in objective.explicit_requirements
                if requirement.path == "content.scene_design.environment"
            )
        ground_id, ground_mode = _deterministic_ground_interaction(
            objective,
            item,
            ground_ids,
        )
        ground_interaction = GroundInteractionSpec(
            mode=ground_mode,
            ground_entity_id=ground_id,
            source_status="inferred" if ground_id is not None else "default",
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


def _deterministic_ground_interaction(
    objective: ObjectivePlanningBrief,
    entity: SkeletonEntity,
    ground_ids: list[str],
) -> tuple[
    str | None,
    Literal["must_be_above", "must_touch"],
]:
    """Ground ordinary people/vehicles without asking either model to restate it."""

    if not ground_ids or entity.proxy_family not in {"human_capsule", "vehicle_box"}:
        return None, "must_be_above"
    semantics = [
        motion.get("motion_semantics")
        for motion in objective.subject_motion
        if isinstance(motion, dict) and motion.get("subject_id") == entity.entity_id
    ]
    # A per-entity contact mode cannot remain must_touch through an airborne,
    # jumping, or carried phase. Those paths still start from a grounded
    # deterministic transform and the default validator prevents penetration.
    has_non_ground_phase = any(
        isinstance(item, dict)
        and (
            item.get("motion_type") in {"flying", "jumping", "carried"}
            or item.get("motion_mode") == "carried"
        )
        for item in semantics
    )
    if has_non_ground_phase:
        return ground_ids[0], "must_be_above"
    return ground_ids[0], "must_touch"


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
        dimensions = (objective.translation_parameters or {}).get("scene", {}).get(
            "dimensions_m"
        ) or [100.0, 100.0]
        return {
            "type": "plane",
            "size_xy_m": [float(dimensions[0]), float(dimensions[1])],
        }
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
        if dimensions[0] < float(footprint[0]) or dimensions[1] < float(footprint[1]):
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


def _camera_yaw_degrees(
    view: str,
    *,
    motion_direction_xy: tuple[float, float] | None = None,
    subject_motion_readability_applicable: bool = True,
    minimum_motion_obliqueness_degrees: float = 20.0,
) -> float:
    base_yaw = 0.0
    if motion_direction_xy is not None:
        base_yaw = math.degrees(
            math.atan2(motion_direction_xy[0], -motion_direction_xy[1])
        )
    if view == "unspecified":
        # A static scene has no motion axis that could justify an arbitrary
        # oblique view. Align it with canonical scene depth; moving scenes keep
        # a safety margin because the camera aims at the scene focus while an
        # individual subject may cross that focus.
        return (
            base_yaw + min(89.0, minimum_motion_obliqueness_degrees + 25.0)
            if subject_motion_readability_applicable
            else 0.0
        )
    offset = {
        "front": 0.0,
        "rear": 180.0,
        "side": 90.0,
        "three_quarter": 45.0,
    }[view]
    return base_yaw + offset


def _primary_linear_motion_direction_xy(
    candidate: CandidateState,
    skeleton: SceneSkeleton,
) -> tuple[float, float] | None:
    """Resolve camera-relative wording against realized motion, not world axes."""

    ordered_subjects = [
        phase.subject_id
        for phase in skeleton.motion_phases
        if phase.kind == "path_move" and not _is_orbit_phase(phase)
    ]
    focus_id = skeleton.camera_intent.focus_target_id
    if focus_id in ordered_subjects:
        ordered_subjects.remove(focus_id)
        ordered_subjects.insert(0, focus_id)
    for subject_id in dict.fromkeys(ordered_subjects):
        best: tuple[float, float] | None = None
        best_length = 0.0
        for track in candidate.motion_tracks.values():
            if track.target_entity_id != subject_id or track.type != "transform":
                continue
            translations = [
                keyframe.value.translation_m
                for keyframe in track.keyframes
                if isinstance(keyframe.value, TransformValue)
                and keyframe.value.translation_m is not None
            ]
            for start, end in zip(translations, translations[1:]):
                delta = (end[0] - start[0], end[1] - start[1])
                distance = math.hypot(*delta)
                if distance > best_length:
                    best = (delta[0] / distance, delta[1] / distance)
                    best_length = distance
        if best is not None:
            return best
    return None


def _scene_background_ground_direction() -> tuple[float, float, float]:
    """Keep semantic scene depth independent from the observing camera azimuth."""

    return (0.0, 1.0, 0.0)


def _scene_reference_clearance_m(
    candidate: CandidateState,
    direction: tuple[float, float, float],
    profile: PlanningProfile,
) -> float:
    """Scale a far/background gap from the enclosing scene, never one subject."""

    scene_extents = [
        2.0
        * directional_support_extent(
            entity.proxy,
            entity.solved_transform,
            direction,
        )
        for entity in candidate.entities.values()
        if entity.proxy.type == "plane" or "environment" in entity.tags
    ]
    if not scene_extents:
        return profile.default_depth_gap_m
    return max(
        profile.default_depth_gap_m,
        max(scene_extents) * profile.far_scene_extent_ratio,
    )


def _place_entities(
    candidate: CandidateState,
    skeleton: SceneSkeleton,
    profile: PlanningProfile,
) -> None:
    # Initial layout must not change when an equivalent Skeleton serializes its
    # entity array in another order.
    for index, entity_id in enumerate(sorted(candidate.entities)):
        entity = candidate.entities[entity_id]
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
        subject_position = list(
            subject.solved_transform.translation_m or (0.0, 0.0, 0.0)
        )
        reference_position = reference.solved_transform.translation_m or (0.0, 0.0, 0.0)
        if relation.kind == "camera_depth_order":
            # "Far" belongs to the scene reference frame. Object size only
            # contributes its surface support; it must never enlarge the requested
            # empty gap merely because the object itself is large.
            direction = _scene_background_ground_direction()
            clearance_m = _scene_reference_clearance_m(
                candidate,
                direction,
                profile,
            )
            center_distance = surface_clearance_target_distance_m(
                reference.proxy,
                reference.solved_transform,
                subject.proxy,
                subject.solved_transform,
                direction,
                clearance_m,
            )
            subject_position[0] = reference_position[0] + direction[0] * center_distance
            subject_position[1] = reference_position[1] + direction[1] * center_distance
        elif relation.kind == "relative_position" and relation.direction:
            axis, sign = {
                "left": (0, -1.0),
                "right": (0, 1.0),
                "front": (1, -1.0),
                "behind": (1, 1.0),
                "below": (2, -1.0),
                "above": (2, 1.0),
            }[relation.direction]
            subject_position[axis] = (
                reference_position[axis] + sign * profile.default_depth_gap_m
            )
        elif relation.kind == "proximity" and relation.temporal_mode in {
            "throughout",
            "at_start",
        }:
            direction = (1.0, 0.0, 0.0)
            center_distance = surface_clearance_target_distance_m(
                reference.proxy,
                reference.solved_transform,
                subject.proxy,
                subject.solved_transform,
                direction,
                0.5,
            )
            subject_position[0] = reference_position[0] + center_distance
            subject_position[1] = reference_position[1]
        subject.solved_transform = subject.solved_transform.model_copy(
            update={"translation_m": tuple(subject_position)}
        )


def _orbit_radius_map(
    candidate: CandidateState,
    skeleton: SceneSkeleton,
    profile: PlanningProfile,
) -> dict[tuple[str, str], float]:
    """为整棵嵌套轨道预留包络，避免内层天体穿过外层中心天体。"""
    pairs = {
        (phase.subject_id, phase.target_id)
        for phase in skeleton.motion_phases
        if _is_orbit_phase(phase) and phase.target_id is not None
    }
    parent_by_child: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    for child_id, parent_id in sorted(pairs):
        previous_parent = parent_by_child.get(child_id)
        if previous_parent is not None and previous_parent != parent_id:
            raise ValueError(f"Orbit Entity 不能同时围绕多个目标：{child_id}")
        parent_by_child[child_id] = parent_id
        children.setdefault(parent_id, []).append(child_id)

    radii: dict[tuple[str, str], float] = {}
    extents: dict[str, float] = {}
    visiting: set[str] = set()

    def subsystem_extent(entity_id: str) -> float:
        if entity_id in extents:
            return extents[entity_id]
        if entity_id in visiting:
            raise ValueError(f"Orbit 关系形成循环：{entity_id}")
        visiting.add(entity_id)
        body_extent = _proxy_bounding_radius(candidate.entities[entity_id])
        envelope = body_extent
        for child_id in children.get(entity_id, []):
            child_extent = subsystem_extent(child_id)
            radius = max(
                2.0,
                envelope + child_extent + profile.orbit_surface_clearance_m,
            )
            radii[(child_id, entity_id)] = radius
            envelope = max(envelope, radius + child_extent)
        visiting.remove(entity_id)
        extents[entity_id] = envelope
        return envelope

    for entity_id in sorted(candidate.entities):
        subsystem_extent(entity_id)
    return radii


def _build_relation_constraints(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
    orbit_radii: dict[tuple[str, str], float],
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
            direction = _scene_background_ground_direction()
            clearance_m = _scene_reference_clearance_m(
                candidate,
                direction,
                profile,
            )
            payloads = [
                common
                | {
                    "type": "depth_order",
                    "parameters": {
                        "near_entity_id": relation.reference_id,
                        "far_entity_id": relation.subject_id,
                        "camera_id": "camera_main",
                        "minimum_depth_gap_meters": profile.default_depth_gap_m * 0.5,
                    },
                },
                common
                | {
                    "constraint_id": f"skeleton_{relation.relation_id}_collision_clearance",
                    "type": "collision_clearance",
                    "parameters": {
                        "entity_ids": [relation.reference_id, relation.subject_id],
                        "minimum_meters": clearance_m,
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
            subject = candidate.entities[relation.subject_id]
            reference = candidate.entities[relation.reference_id]
            conservative_contact_distance = _proxy_horizontal_radius(
                subject
            ) + _proxy_horizontal_radius(reference)
            payloads = [
                common
                | {
                    "type": "distance_range",
                    "parameters": {
                        "entity_ids": [relation.subject_id, relation.reference_id],
                        "minimum_meters": 0.0,
                        "maximum_meters": conservative_contact_distance + 1.0,
                    },
                }
            ]
            if not _relation_allows_containment_overlap(objective, relation):
                payloads.append(
                    common
                    | {
                        "constraint_id": f"skeleton_{relation.relation_id}_collision_clearance",
                        "type": "collision_clearance",
                        "parameters": {
                            "entity_ids": [
                                relation.subject_id,
                                relation.reference_id,
                            ],
                            "minimum_meters": 0.0,
                            "space": "ground_plane",
                        },
                    }
                )
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
        payloads: list[dict[str, Any]]
        if phase.kind == "hold":
            payloads = [
                common
                | {
                    "type": "hold",
                    "parameters": {
                        "target_id": phase.subject_id,
                        "components": ["translation", "rotation", "scale"],
                    },
                }
            ]
        elif phase.kind == "local_transform":
            payloads = [
                common
                | {
                    "type": "hold",
                    "parameters": {
                        "target_id": phase.subject_id,
                        "components": ["translation"],
                    },
                }
            ]
        elif phase.kind == "path_move" and phase.direction_mode in {
            "world_forward",
            "world_left",
            "world_right",
        }:
            payloads = [
                common
                | {
                    "type": "motion_direction",
                    "parameters": {
                        "target_id": phase.subject_id,
                        "direction": {
                            "world_forward": "forward",
                            "world_left": "left",
                            "world_right": "right",
                        }[phase.direction_mode],
                        "space": "world",
                        "minimum_displacement_m": 0.5,
                    },
                }
            ]
        elif (
            phase.kind == "path_move"
            and phase.target_id is not None
            and phase.direction_mode in {"toward_target", "away_from_target"}
        ):
            target_follows_subject = _phase_target_follows_subject(
                skeleton,
                phase,
            )
            if phase.direction_mode == "away_from_target" and target_follows_subject:
                # The target is transported by this actor, so their mutual
                # distance and an invented world axis cannot express departure.
                # Typed-motion validation still requires visible displacement.
                continue
            else:
                start, end = _phase_range(objective, phase, duration)
                frame_step = (
                    candidate.timeline.fps_denominator
                    / candidate.timeline.fps_numerator
                )
                toward = phase.direction_mode == "toward_target"
                subject = candidate.entities[phase.subject_id]
                reference = candidate.entities[phase.target_id]
                contact_distance = _proxy_horizontal_radius(
                    subject
                ) + _proxy_horizontal_radius(reference)
                containment = toward and _phase_allows_containment_overlap(
                    objective,
                    phase,
                )
                payloads = [
                    common
                    | {
                        "type": "distance_range",
                        "subjects": [phase.subject_id, phase.target_id],
                        "time_range_seconds": [max(start, end - frame_step), end],
                        "parameters": {
                            "entity_ids": [phase.subject_id, phase.target_id],
                            "minimum_meters": (
                                0.0 if toward else contact_distance + 5.0
                            ),
                            "maximum_meters": (
                                max(3.0, contact_distance)
                                if containment
                                else contact_distance + 3.0
                                if toward
                                else 1000000.0
                            ),
                        },
                    }
                ]
        else:
            # 其他阶段没有可独立验证的位移方向。
            continue
        for payload in payloads:
            constraint = ConstraintSpec.model_validate(payload)
            constraints[constraint.constraint_id] = constraint
    return constraints


def _relation_allows_containment_overlap(
    objective: ObjectivePlanningBrief,
    relation: SkeletonRelation,
) -> bool:
    """Containment transitions may overlap the container at their boundary."""

    pair = {relation.subject_id, relation.reference_id}
    for motion in objective.subject_motion:
        if not isinstance(motion, dict) or motion.get("subject_id") not in pair:
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        postconditions = semantics.get("postconditions")
        contained_by_id = (
            postconditions.get("contained_by_id")
            if isinstance(postconditions, dict)
            else None
        )
        if contained_by_id not in pair or contained_by_id == motion.get("subject_id"):
            continue
        if semantics.get("timeline_event_id") == relation.timeline_event_id:
            return True
    return False


def _phase_allows_containment_overlap(
    objective: ObjectivePlanningBrief,
    phase: SkeletonMotionPhase,
) -> bool:
    for motion in objective.subject_motion:
        if not isinstance(motion, dict) or motion.get("subject_id") != phase.subject_id:
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        postconditions = semantics.get("postconditions")
        if not isinstance(postconditions, dict):
            continue
        if (
            semantics.get("timeline_event_id") == phase.timeline_event_id
            and postconditions.get("contained_by_id") == phase.target_id
        ):
            return True
    return False


def _build_motion(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
    orbit_radii: dict[tuple[str, str], float],
) -> dict[str, TrackSpec]:
    tracks: dict[str, TrackSpec] = {}
    duration = candidate.timeline.duration_seconds
    orbit_subjects = {
        phase.subject_id for phase in skeleton.motion_phases if _is_orbit_phase(phase)
    }
    grouped: dict[str, list[SkeletonMotionPhase]] = {}
    for phase in skeleton.motion_phases:
        grouped.setdefault(phase.subject_id, []).append(phase)

    routes_by_subject = {item.subject_id: item for item in skeleton.route_intents}
    route_directions: dict[str, tuple[float, float]] = {}
    for subject_id, phases in grouped.items():
        moving_phases = [
            item
            for item in phases
            if item.kind == "path_move" and not _is_orbit_phase(item)
        ]
        if not moving_phases:
            continue
        direction = _selected_route_direction(moving_phases)
        route = routes_by_subject.get(subject_id)
        if (
            route is not None
            and route.axis_reference_id is not None
            and all(item.direction_mode == "none" for item in moving_phases)
        ):
            direction = _reference_route_direction(
                candidate.entities[route.axis_reference_id]
            )
        route_directions[subject_id] = direction
    route_anchor_schedules = _route_anchor_schedules(
        objective,
        skeleton,
        candidate,
    )
    joint_route_positions = _joint_proximity_route_positions(
        skeleton,
        candidate,
        route_anchor_schedules,
        route_directions,
    )

    grouped_items = _motion_group_build_order(
        objective,
        grouped,
        duration,
    )
    for subject_id, phases in grouped_items:
        _reject_unrepresentable_reference_frame_switch(
            objective,
            phases,
            duration,
        )
        orbit = next((item for item in phases if _is_orbit_phase(item)), None)
        if orbit is not None and orbit.target_id is not None:
            depth = 2 if orbit.target_id in orbit_subjects else 1
            radius = orbit_radii[(subject_id, orbit.target_id)]
            orbit_start, orbit_end = _phase_range(objective, orbit, duration)
            target_at_start = _design_entity_transform_at(
                candidate,
                tracks,
                orbit.target_id,
                orbit_start,
            )
            local_start = TransformValue(
                translation_m=(radius, 0.0, 0.0),
                rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                space="target_relative",
                target_id=orbit.target_id,
            )
            orbit_world_start = _compose_design_transform(target_at_start, local_start)
            candidate.entities[subject_id].solved_transform = candidate.entities[
                subject_id
            ].solved_transform.model_copy(
                update={
                    "translation_m": orbit_world_start.translation_m,
                    "rotation_quaternion_wxyz": (
                        orbit_world_start.rotation_quaternion_wxyz
                    ),
                    "scale": orbit_world_start.scale,
                }
            )
            track = TrackSpec.model_validate(
                {
                    "track_id": f"design_orbit_{subject_id}",
                    "target_entity_id": subject_id,
                    "type": "path_follow",
                    "time_range_seconds": (orbit_start, orbit_end),
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

        moving = [
            item
            for item in phases
            if item.kind == "path_move" and not _is_orbit_phase(item)
        ]
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
            route = routes_by_subject.get(subject_id)
            route_direction = route_directions[subject_id]
            scheduled_anchors = route_anchor_schedules.get(subject_id, [])
            anchors_by_phase: dict[
                str,
                list[tuple[float, SkeletonRouteAnchor]],
            ] = {}
            for anchor_time, anchor in scheduled_anchors:
                anchors_by_phase.setdefault(anchor.phase_id, []).append(
                    (anchor_time, anchor)
                )
            first_motion_start = _phase_range(objective, first, duration)[0]
            has_prior_authored_state = any(
                phase is not first
                and phase.kind != "visibility"
                and _phase_range(objective, phase, duration)[0]
                < first_motion_start - 1e-9
                for phase in phases
            )
            can_stage_initial = (
                first_motion_start <= 1e-9 or not has_prior_authored_state
            )
            first_route_anchor = next(
                (
                    (index, anchor_time, anchor)
                    for index, phase in enumerate(moving)
                    for anchor_time, anchor in anchors_by_phase.get(phase.phase_id, [])
                ),
                None,
            )
            if (
                can_stage_initial
                and first_route_anchor is not None
                and first_route_anchor[1] > first_motion_start + 1e-9
            ):
                anchor_index, anchor_time, anchor = first_route_anchor
                waypoint = _route_anchor_position(
                    skeleton,
                    candidate,
                    tracks,
                    subject_id,
                    anchor,
                    route_direction,
                    profile,
                    anchor_time,
                    joint_route_positions,
                )
                travel_x, travel_y = route_direction
                # Put the subject far enough before its first declared endpoint
                # that every preceding unanchored phase can advance along one route.
                position[0] = waypoint[0] - travel_x * 8.0 * (anchor_index + 1)
                position[1] = waypoint[1] - travel_y * 8.0 * (anchor_index + 1)
                candidate.entities[subject_id].solved_transform = candidate.entities[
                    subject_id
                ].solved_transform.model_copy(update={"translation_m": tuple(position)})
            if (
                first.direction_mode == "toward_target"
                and first.target_id
                and first_route_anchor is None
                and can_stage_initial
            ):
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
                clearance = _proxy_horizontal_radius(
                    candidate.entities[subject_id]
                ) + _proxy_horizontal_radius(target)
                required_distance = clearance + 8.0
                if distance < required_distance:
                    if distance <= 1e-9:
                        delta_x, delta_y, distance = 1.0, 0.0, 1.0
                    position[0] = (
                        target_position[0] + delta_x / distance * required_distance
                    )
                    position[1] = (
                        target_position[1] + delta_y / distance * required_distance
                    )
                candidate.entities[subject_id].solved_transform = candidate.entities[
                    subject_id
                ].solved_transform.model_copy(update={"translation_m": tuple(position)})
            keyframes: list[TrackKeyframe] = []
            previous_direction = route_direction
            positional_phases = [
                item
                for item in phases
                if item.kind == "hold"
                or (item.kind == "path_move" and not _is_orbit_phase(item))
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
                                time_seconds=_track_end_time(candidate, end, start),
                                value=TransformValue(translation_m=tuple(position)),
                                interpolation="step",
                            ),
                        ]
                    )
                    continue
                sample_end = _track_end_time(candidate, end, start)
                phase_anchors = sorted(
                    anchors_by_phase.get(phase.phase_id, []),
                    key=lambda item: item[0],
                )
                start_anchor = next(
                    (
                        anchor
                        for anchor_time, anchor in phase_anchors
                        if math.isclose(anchor_time, start, abs_tol=1e-9)
                    ),
                    None,
                )
                if start_anchor is not None:
                    position = list(
                        _route_anchor_position(
                            skeleton,
                            candidate,
                            tracks,
                            subject_id,
                            start_anchor,
                            previous_direction,
                            profile,
                            start,
                            joint_route_positions,
                        )
                    )
                    if phase is first:
                        candidate.entities[
                            subject_id
                        ].solved_transform = candidate.entities[
                            subject_id
                        ].solved_transform.model_copy(
                            update={"translation_m": tuple(position)}
                        )
                phase_points: list[
                    tuple[float, tuple[float, float, float], SkeletonRouteAnchor | None]
                ] = [(start, tuple(position), start_anchor)]
                hidden_at_end = any(
                    item.kind == "visibility"
                    and item.timeline_event_id == phase.timeline_event_id
                    and item.visibility_state == "hidden"
                    and item.transition_at == "at_end"
                    for item in phases
                )
                for anchor_time, anchor in phase_anchors:
                    if anchor_time <= start + 1e-9:
                        continue
                    waypoint = _route_anchor_position(
                        skeleton,
                        candidate,
                        tracks,
                        subject_id,
                        anchor,
                        previous_direction,
                        profile,
                        anchor_time,
                        joint_route_positions,
                    )
                    phase_points.append((anchor_time, waypoint, anchor))
                if not math.isclose(
                    phase_points[-1][0],
                    sample_end,
                    abs_tol=1e-9,
                ):
                    endpoint_start = list(phase_points[-1][1])
                    endpoint_phase = phase
                    if (
                        phase.direction_mode == "away_from_target"
                        and _phase_target_follows_subject(skeleton, phase)
                    ):
                        endpoint_phase = phase.model_copy(
                            update={"direction_mode": "none", "target_id": None}
                        )
                    endpoint = _linear_phase_endpoint(
                        candidate,
                        endpoint_phase,
                        endpoint_start,
                        tracks,
                        sample_end,
                        semantic_direction_mode=_objective_motion_direction(
                            objective,
                            phase,
                        ),
                        hidden_at_end=hidden_at_end,
                        fallback_direction=previous_direction,
                    )
                    phase_points.append((sample_end, tuple(endpoint), None))

                for point_index, (point_time, point_position, _) in enumerate(
                    phase_points
                ):
                    if point_index == 0:
                        keyframes.append(
                            TrackKeyframe(
                                time_seconds=point_time,
                                value=TransformValue(
                                    translation_m=point_position
                                ),
                                interpolation="smooth",
                            )
                        )
                        continue
                    previous_time, previous_position, _ = phase_points[
                        point_index - 1
                    ]
                    if phase.path_family == "parabolic":
                        midpoint_time = previous_time + (
                            point_time - previous_time
                        ) * 0.5
                        horizontal_distance = math.hypot(
                            point_position[0] - previous_position[0],
                            point_position[1] - previous_position[1],
                        )
                        midpoint = (
                            (previous_position[0] + point_position[0]) * 0.5,
                            (previous_position[1] + point_position[1]) * 0.5,
                            max(previous_position[2], point_position[2])
                            + max(1.0, horizontal_distance * 0.25),
                        )
                        keyframes.append(
                            TrackKeyframe(
                                time_seconds=midpoint_time,
                                value=TransformValue(translation_m=midpoint),
                                interpolation="smooth",
                            )
                        )
                    delta_x = point_position[0] - previous_position[0]
                    delta_y = point_position[1] - previous_position[1]
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
                                f"Route Intent {route.route_id} 的路径点导致运动方向反转："
                                f"previous={previous_direction}, proposed={proposed_direction}, "
                                f"phase={phase.phase_id}"
                            )
                        previous_direction = proposed_direction
                    keyframes.append(
                        TrackKeyframe(
                            time_seconds=point_time,
                            value=TransformValue(translation_m=point_position),
                            interpolation=(
                                "step"
                                if math.isclose(
                                    point_time,
                                    sample_end,
                                    abs_tol=1e-9,
                                )
                                else "smooth"
                            ),
                        )
                    )
                position = list(phase_points[-1][1])
            track = TrackSpec(
                track_id=f"design_motion_{subject_id}",
                target_entity_id=subject_id,
                type="transform",
                time_range_seconds=(keyframes[0].time_seconds, duration),
                keyframes=_deduplicate_keyframes(keyframes),
                interpolation="smooth",
                source_ref=moving[0].source_ref,
                source_refs=sorted(
                    {
                        source_ref
                        for phase in positional_phases
                        for source_ref in (phase.source_ref, phase.speed_source_ref)
                        if source_ref is not None
                    }
                    - {moving[0].source_ref}
                ),
            )
            tracks[track.track_id] = track

        local_phases = [item for item in phases if item.kind == "local_transform"]
        if local_phases:
            track_id = f"design_motion_{subject_id}"
            existing = tracks.get(track_id)
            fallback = candidate.entities[subject_id].solved_transform
            keyframes = list(existing.keyframes) if existing is not None else []
            for phase in sorted(
                local_phases,
                key=lambda item: _phase_range(objective, item, duration)[0],
            ):
                start, end = _phase_range(objective, phase, duration)
                sample_end = _track_end_time(candidate, end, start)
                midpoint = start + (sample_end - start) * 0.5
                baseline_start = sample_transform_track(existing, start, fallback)
                baseline_mid = sample_transform_track(existing, midpoint, fallback)
                baseline_end = sample_transform_track(existing, sample_end, fallback)
                angle = math.radians(15.0) / 2.0
                changed_rotation = (
                    quaternion_multiply(
                        baseline_mid.rotation_quaternion_wxyz,
                        # Use the canonical up axis for a generic local pose
                        # change. Tilting around X/Y makes a centered grounded
                        # proxy penetrate the floor even though no translation
                        # was authored.
                        (math.cos(angle), 0.0, 0.0, math.sin(angle)),
                    )
                    if "rotation" in phase.local_components
                    else baseline_mid.rotation_quaternion_wxyz
                )
                changed_scale = (
                    tuple(
                        value
                        * (
                            1.0
                            if candidate.entities[
                                subject_id
                            ].ground_interaction.ground_entity_id
                            is not None
                            and candidate.entities[
                                subject_id
                            ].ground_interaction.mode
                            in {"must_touch", "must_be_above"}
                            and axis == 2
                            else 1.08
                        )
                        for axis, value in enumerate(baseline_mid.scale)
                    )
                    if "scale" in phase.local_components
                    else baseline_mid.scale
                )
                keyframes.extend(
                    [
                        TrackKeyframe(
                            time_seconds=start,
                            value=baseline_start,
                            interpolation="smooth",
                        ),
                        TrackKeyframe(
                            time_seconds=midpoint,
                            value=baseline_mid.model_copy(
                                update={
                                    "rotation_quaternion_wxyz": changed_rotation,
                                    "scale": changed_scale,
                                }
                            ),
                            interpolation="smooth",
                        ),
                        TrackKeyframe(
                            time_seconds=sample_end,
                            value=baseline_end,
                            interpolation="smooth",
                        ),
                    ]
                )
            tracks[track_id] = TrackSpec(
                track_id=track_id,
                target_entity_id=subject_id,
                type="transform",
                time_range_seconds=(
                    min(item.time_seconds for item in keyframes),
                    duration,
                ),
                keyframes=_deduplicate_keyframes(keyframes),
                interpolation="smooth",
                source_ref=(
                    existing.source_ref
                    if existing is not None
                    else local_phases[0].source_ref
                ),
                source_refs=sorted(
                    {
                        source_ref
                        for phase in local_phases
                        for source_ref in (phase.source_ref, phase.speed_source_ref)
                        if source_ref is not None
                    }
                    - {
                        existing.source_ref
                        if existing is not None
                        else local_phases[0].source_ref
                    }
                ),
            )

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
                        _state_transition_time(candidate, transition_time),
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
                source_refs=sorted({item[2] for item in transitions} - {source_ref}),
            )
            tracks[track.track_id] = track
    return tracks


def _phase_target_follows_subject(
    skeleton: SceneSkeleton,
    phase: SkeletonMotionPhase,
) -> bool:
    """Detect the degenerate case where a target is carried by the moving actor."""

    return any(
        item.kind == "carried"
        and item.subject_id == phase.target_id
        and item.carrier_id == phase.subject_id
        and item.timeline_event_id == phase.timeline_event_id
        for item in skeleton.motion_phases
    )


def _motion_group_build_order(
    objective: ObjectivePlanningBrief,
    grouped: dict[str, list[SkeletonMotionPhase]],
    duration: float,
) -> list[tuple[str, list[SkeletonMotionPhase]]]:
    """Build moving targets before phases whose endpoint samples those targets."""

    base_order = sorted(
        grouped,
        key=lambda subject_id: min(
            (
                _phase_range(objective, phase, duration)[0]
                for phase in grouped[subject_id]
                if phase.kind == "path_move" and not _is_orbit_phase(phase)
            ),
            default=math.inf,
        ),
    )
    dependencies = {
        subject_id: {
            phase.target_id
            for phase in phases
            if phase.kind == "path_move"
            and not _is_orbit_phase(phase)
            and phase.direction_mode in {"toward_target", "away_from_target"}
            and phase.target_id in grouped
            and phase.target_id != subject_id
        }
        for subject_id, phases in grouped.items()
    }
    remaining = set(base_order)
    result: list[str] = []
    while remaining:
        ready = [
            subject_id
            for subject_id in base_order
            if subject_id in remaining and not (dependencies[subject_id] & remaining)
        ]
        if not ready:
            # A mutual-target cycle has no unique deterministic build order.
            # Preserve stable ordering and let geometric validation expose it.
            result.extend(
                subject_id for subject_id in base_order if subject_id in remaining
            )
            break
        result.extend(ready)
        remaining.difference_update(ready)
    return [(subject_id, grouped[subject_id]) for subject_id in result]


def _build_camera(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
    strategy: str,
) -> CameraCandidate:
    parameters = (objective.translation_parameters or {}).get("camera", {})
    explicit_focal = _explicit_camera_focal_length(objective)
    focal = (
        explicit_focal
        if explicit_focal is not None
        else float(parameters.get("focal_length_mm") or profile.default_focal_length_mm)
    )
    movement = skeleton.camera_intent.movement
    start_distance = float(
        parameters.get("start_distance_m") or profile.default_camera_distance_m
    )
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
    explicit_height = _explicit_camera_height(objective)
    configured_height = parameters.get("height_m")
    height = (
        explicit_height
        if explicit_height is not None
        else float(1.5 if configured_height is None else configured_height)
    )
    has_orbit = any(_is_orbit_phase(item) for item in skeleton.motion_phases)
    focus_point, focus_height = _camera_focus(candidate, skeleton)
    explicit_pitch = _explicit_camera_pitch_degrees(objective)
    if explicit_pitch is not None:
        height = (
            focus_point[2] + math.tan(math.radians(explicit_pitch)) * start_distance
        )
    if has_orbit and not _has_explicit_camera_elevation(objective):
        # 缺省轨道镜头提高俯视夹角，避免圆轨道投影成直线往返。
        height = focus_height + start_distance * 0.75
    view = skeleton.camera_intent.view_relation_to_motion
    motion_direction_xy = _primary_linear_motion_direction_xy(candidate, skeleton)
    yaw = _camera_yaw_degrees(
        view,
        motion_direction_xy=motion_direction_xy,
        subject_motion_readability_applicable=any(
            item.kind == "path_move" and not _is_orbit_phase(item)
            for item in skeleton.motion_phases
        ),
        minimum_motion_obliqueness_degrees=(
            profile.minimum_view_subject_motion_obliqueness_degrees
        ),
    )
    start = _camera_position(focus_point, start_distance, height, yaw)
    end = _camera_position(focus_point, end_distance, height, yaw)
    duration = candidate.timeline.duration_seconds
    motion_start, motion_end = _camera_motion_range(objective, duration)
    active_motion_duration = motion_end - motion_start
    tracks: dict[str, TrackSpec] = {}
    camera_motion_source_refs = sorted(
        _implemented_camera_motion_source_refs(objective, skeleton)
        - {skeleton.camera_intent.source_ref}
    )
    if movement in {"push_in", "pull_out", "lateral"}:
        if movement == "lateral":
            travel = max(
                2.0,
                _camera_speed_mps(
                    skeleton.camera_intent,
                    profile,
                    configured=parameters.get("speed_mps"),
                )
                * active_motion_duration,
            )
            angle = math.radians(yaw + 90.0)
            end = (
                start[0] + math.sin(angle) * travel,
                start[1] - math.cos(angle) * travel,
                start[2],
            )
        track = TrackSpec(
            track_id="design_camera_transform",
            type="transform",
            time_range_seconds=(motion_start, motion_end),
            keyframes=[
                TrackKeyframe(
                    time_seconds=motion_start,
                    value=TransformValue(
                        translation_m=start,
                        rotation_quaternion_wxyz=look_at_camera_quaternion(
                            start, focus_point
                        ),
                    ),
                    interpolation="smooth",
                ),
                TrackKeyframe(
                    time_seconds=_track_end_time(
                        candidate,
                        motion_end,
                        motion_start,
                    ),
                    value=TransformValue(
                        translation_m=end,
                        rotation_quaternion_wxyz=look_at_camera_quaternion(
                            end, focus_point
                        ),
                    ),
                    interpolation="smooth",
                ),
            ],
            interpolation="smooth",
            source_ref=skeleton.camera_intent.source_ref,
            source_refs=camera_motion_source_refs,
        )
        tracks[track.track_id] = track
    elif movement in {"follow", "orbit"}:
        target_id = skeleton.camera_intent.movement_target_id
        if target_id is None:
            raise ValueError(f"{movement} 摄影机必须提供 movement_target_id")
        target_position = _design_entity_transform_at(
            candidate,
            candidate.motion_tracks,
            target_id,
            motion_start,
        ).translation_m
        offset = subtract(start, target_position)
        if movement == "follow":
            path = {
                "representation": "polyline",
                "space": "target_relative",
                "target_id": target_id,
                "closed": False,
                "control_points": [offset, offset],
            }
        else:
            horizontal_radius = max(0.1, math.hypot(offset[0], offset[1]))
            path = {
                "representation": "circle",
                "space": "target_relative",
                "target_id": target_id,
                "closed": True,
                "cycle_count": min(
                    1.0,
                    max(
                        0.02,
                        _camera_speed_mps(
                            skeleton.camera_intent,
                            profile,
                            configured=parameters.get("speed_mps"),
                        )
                        * active_motion_duration
                        / (2.0 * math.pi * horizontal_radius),
                    ),
                ),
                "radius_m": horizontal_radius,
                "center_offset_m": [0.0, 0.0, offset[2]],
                "initial_phase_degrees": math.degrees(math.atan2(offset[1], offset[0])),
            }
        tracks["design_camera_path"] = TrackSpec.model_validate(
            {
                "track_id": "design_camera_path",
                "type": "path_follow",
                "time_range_seconds": [motion_start, motion_end],
                "path": path,
                "interpolation": "smooth",
                "source_ref": skeleton.camera_intent.source_ref,
                "source_refs": camera_motion_source_refs,
            }
        )
    look_target_id = skeleton.camera_intent.movement_target_id
    if movement in {"follow", "orbit", "pan"} and look_target_id:
        tracks["design_camera_look_at"] = TrackSpec(
            track_id="design_camera_look_at",
            type="look_at",
            time_range_seconds=(motion_start, motion_end),
            target_id=look_target_id,
            interpolation="smooth" if movement == "pan" else "linear",
            source_ref=skeleton.camera_intent.source_ref,
            source_refs=camera_motion_source_refs,
        )
    source_refs = sorted(
        _implemented_camera_static_source_refs(objective, skeleton)
        | ({skeleton.camera_intent.source_ref} if movement == "static" else set())
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
    projected_sizes = _composition_projected_sizes(
        objective,
        candidate,
        parameters,
    )
    projected_ids: set[str] = set()
    for index, (subject_id, minimum, maximum, source_status, source_ref) in enumerate(
        projected_sizes
    ):
        projected_ids.add(subject_id)
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": (
                    "design_subject_projected_size"
                    if index == 0
                    else f"design_subject_projected_size_{subject_id}_{index}"
                ),
                "type": "projected_size",
                "strength": "hard" if source_status == "explicit" else "soft",
                "subjects": [subject_id],
                "time_range_seconds": [0.0, duration],
                "parameters": {
                    "entity_id": subject_id,
                    "measurement": "height",
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                },
                "source_status": source_status,
                "source_ref": source_ref,
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint

    for index, placement in enumerate(
        objective.composition.get("screen_placements", [])
    ):
        if not isinstance(placement, dict):
            continue
        entity_id = placement.get("subject_id")
        if entity_id not in candidate.entities:
            continue
        horizontal = placement.get("horizontal")
        vertical = placement.get("vertical")
        horizontal_region = _screen_axis_region(
            horizontal.get("value") if isinstance(horizontal, dict) else None,
            axis="horizontal",
        )
        vertical_region = _screen_axis_region(
            vertical.get("value") if isinstance(vertical, dict) else None,
            axis="vertical",
        )
        for axis, node, region in (
            ("horizontal", horizontal, horizontal_region),
            ("vertical", vertical, vertical_region),
        ):
            if region is None:
                continue
            explicit = (
                isinstance(node, dict) and node.get("source_status") == "explicit"
            )
            left, right = region if axis == "horizontal" else (0.0, 1.0)
            top, bottom = region if axis == "vertical" else (0.0, 1.0)
            constraint = ConstraintSpec.model_validate(
                {
                    "constraint_id": f"design_screen_placement_{index + 1:02d}_{axis}",
                    "type": "screen_region",
                    "strength": "hard" if explicit else "soft",
                    "subjects": [entity_id],
                    "time_range_seconds": [0.0, duration],
                    "parameters": {
                        "entity_id": entity_id,
                        "region": [left, top, right, bottom],
                    },
                    "source_status": "explicit" if explicit else "inferred",
                    "source_ref": (
                        f"content.composition.screen_placements[{index}].{axis}"
                    ),
                }
            )
            candidate.constraints[constraint.constraint_id] = constraint

    major_ratio = parameters.get("major_object_frame_ratio")
    major_id = _major_composition_entity_id(objective, skeleton, candidate)
    if (
        major_id is not None
        and major_id not in projected_ids
        and isinstance(major_ratio, (list, tuple))
        and len(major_ratio) == 2
    ):
        source_status = parameters.get("source_status", "inferred")
        if source_status not in {"explicit", "inferred", "default"}:
            source_status = "inferred"
        if source_status == "explicit":
            # This value comes from the emotion table, which currently exposes
            # only a coarse composition-level source. A truly explicit numeric
            # major-object ratio is already handled by visual_scales above.
            source_status = "inferred"
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": "design_major_object_projected_size",
                "type": "projected_size",
                "strength": "soft",
                "subjects": [major_id],
                "time_range_seconds": [0.0, duration],
                "parameters": {
                    "entity_id": major_id,
                    "measurement": "diameter",
                    "minimum": float(major_ratio[0]),
                    "maximum": float(major_ratio[1]),
                },
                "source_status": source_status,
                "source_ref": "translation_parameters.composition.major_object_frame_ratio",
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint
        projected_ids.add(major_id)

    # A composition target is meaningless if it remains outside the frame for
    # its complete narrative window. Require at least one sampled frame of actual
    # presence; stronger user-authored visibility requirements are handled below.
    for entity_id in sorted(projected_ids):
        presence_range = _entity_composition_time_range(
            objective,
            entity_id,
            duration,
        )
        frame_step = (
            candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
        )
        sample_count = max(
            1,
            int(math.ceil((presence_range[1] - presence_range[0]) / frame_step)),
        )
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": f"design_composition_presence_{entity_id}",
                "type": "visibility_fraction",
                "strength": "soft",
                "subjects": [entity_id],
                "time_range_seconds": presence_range,
                "parameters": {
                    "entity_id": entity_id,
                    "minimum_time_fraction": 1.0 / sample_count,
                    "minimum_inside_fraction": 0.01,
                },
                "source_status": "inferred",
                "source_ref": "content.composition",
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint

    negative_space = parameters.get("negative_space_ratio")
    composed_ids = [
        entity.entity_id
        for entity in candidate.entities.values()
        if entity.proxy.type != "plane" and "environment" not in entity.tags
    ]
    if (
        composed_ids
        and isinstance(negative_space, (list, tuple))
        and len(negative_space) == 2
    ):
        source_status = parameters.get("source_status", "inferred")
        if source_status not in {"explicit", "inferred", "default"}:
            source_status = "inferred"
        constraint = ConstraintSpec.model_validate(
            {
                "constraint_id": "design_negative_space",
                "type": "negative_space",
                "strength": "soft",
                "subjects": composed_ids,
                "time_range_seconds": [0.0, duration],
                "parameters": {
                    "entity_ids": composed_ids,
                    "minimum_fraction": float(negative_space[0]),
                    "maximum_fraction": float(negative_space[1]),
                },
                "source_status": source_status,
                "source_ref": "translation_parameters.composition.negative_space_ratio",
            }
        )
        candidate.constraints[constraint.constraint_id] = constraint

    for index, item in enumerate(
        objective.composition.get("visibility_requirements", [])
    ):
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


def _entity_composition_time_range(
    objective: ObjectivePlanningBrief,
    entity_id: str,
    duration: float,
) -> tuple[float, float]:
    ranges = []
    for motion in objective.subject_motion:
        if not isinstance(motion, dict) or motion.get("subject_id") != entity_id:
            continue
        start = motion.get("start_time_seconds")
        end = motion.get("end_time_seconds")
        if (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
            and 0.0 <= float(start) < float(end) <= duration
        ):
            ranges.append((float(start), float(end)))
    if not ranges:
        return (0.0, duration)
    return (min(item[0] for item in ranges), max(item[1] for item in ranges))


def _add_speed_constraints(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> None:
    """把符号速度档位映射为冻结 Profile 范围。"""

    duration = candidate.timeline.duration_seconds
    for phase in skeleton.motion_phases:
        if phase.kind in {"local_transform", "visibility"}:
            # These phases may have a qualitative tempo, but speed_range is a
            # world-translation m/s constraint.  Their source is carried by the
            # transform/visibility track instead of a contradictory speed gate.
            continue
        generic_explicit_motion = (
            phase.kind == "path_move"
            and not _is_orbit_phase(phase)
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
    if camera.speed_intent == "match_subject":
        return
    if camera.movement == "pan":
        # Pan 的速度是角速度/节奏，现有 speed_range 是世界线速度。
        # 把两者强行绑定会要求原地摇摄的摄影机必须平移。
        return
    configured_camera_speed = (
        (objective.translation_parameters or {}).get("camera", {}).get("speed_mps")
    )
    if camera.speed_intent == "unspecified":
        if camera.speed_source_status != "explicit" or not isinstance(
            configured_camera_speed, (int, float)
        ):
            return
        numeric_speed = float(configured_camera_speed)
        minimum, maximum = (
            max(0.0, numeric_speed * 0.95),
            max(profile.numeric_tolerance, numeric_speed * 1.05),
        )
    else:
        minimum, maximum = _speed_range(camera.speed_intent, profile)
    constraint = ConstraintSpec.model_validate(
        {
            "constraint_id": "skeleton_camera_speed",
            "type": "speed_range",
            "strength": "hard" if camera.speed_source_status == "explicit" else "soft",
            "subjects": [],
            "time_range_seconds": list(_camera_motion_range(objective, duration)),
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


def _fit_static_camera_to_composition(
    objective: ObjectivePlanningBrief,
    candidate: CandidateState,
    skeleton: SceneSkeleton,
    profile: PlanningProfile,
) -> None:
    """Move a static-scene camera rig back without changing its push distance."""

    if candidate.camera is None:
        return
    projected = [
        item
        for item in candidate.constraints.values()
        if item.type == "projected_size" and item.source_status == "explicit"
    ]
    focus_point, _ = _camera_focus(candidate, skeleton)
    transform_track = next(
        (item for item in candidate.camera.tracks.values() if item.type == "transform"),
        None,
    )

    def camera_samples() -> list[TransformValue]:
        if transform_track is None:
            return [candidate.camera.solved_transform]
        return [
            item.value
            for item in transform_track.keyframes
            if isinstance(item.value, TransformValue)
        ]

    # Projection is approximately inverse-depth. Repeated passes absorb pitch,
    # proxy bounds and the re-aimed camera while preserving the dolly span.
    for _ in range(3):
        required_offset = 0.0
        for camera_transform in camera_samples():
            if camera_transform.translation_m is None:
                continue
            camera_forward = normalize(
                subtract(focus_point, camera_transform.translation_m)
            )
            for entity in candidate.entities.values():
                if entity.proxy.type == "plane" or "environment" in entity.tags:
                    continue
                entity_position = entity.solved_transform.translation_m or (
                    0.0,
                    0.0,
                    0.0,
                )
                center_depth = dot(
                    subtract(entity_position, camera_transform.translation_m),
                    camera_forward,
                )
                near_surface_depth = center_depth - directional_support_extent(
                    entity.proxy,
                    entity.solved_transform,
                    camera_forward,
                )
                if near_surface_depth <= profile.default_depth_gap_m:
                    required_offset = max(
                        required_offset,
                        profile.default_depth_gap_m - near_surface_depth,
                    )
            for constraint in projected:
                entity_id = constraint.parameters.entity_id
                entity = candidate.entities.get(entity_id)
                if entity is None:
                    continue
                bounds = project_geometry_bounds(
                    entity.proxy,
                    entity.solved_transform,
                    camera_transform.translation_m,
                    camera_transform.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
                    candidate.camera.static.focal_length_mm
                    or profile.default_focal_length_mm,
                    candidate.camera.static.sensor_width_mm,
                    profile.resolution_x / profile.resolution_y,
                )
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                measurement = constraint.parameters.measurement
                size = {
                    "width": width,
                    "height": height,
                    "diameter": max(width, height),
                }[measurement]
                maximum = constraint.parameters.maximum
                if bounds[4] > 0.0 and size > maximum:
                    required_offset = max(
                        required_offset,
                        bounds[4] * (size / maximum - 1.0),
                    )
        if required_offset <= profile.numeric_tolerance:
            break

        def shifted(
            value: TransformValue,
            offset: float = required_offset,
        ) -> TransformValue:
            position = value.translation_m
            if position is None:
                return value
            backward = normalize(
                (position[0] - focus_point[0], position[1] - focus_point[1], 0.0)
            )
            if abs(backward[0]) + abs(backward[1]) <= profile.numeric_tolerance:
                backward = (0.0, -1.0, 0.0)
            # A default camera height is an absolute height above the scene's
            # ground, not an angle relative to a possibly elevated focus point.
            # Only an explicit view angle authorizes changing Z while fitting
            # distance; otherwise a large/tall background object could pull the
            # averaged focus upward and drive the camera below the ground.
            preserve_pitch = _explicit_camera_pitch_degrees(objective) is not None
            moved = (
                position[0] + backward[0] * offset,
                position[1] + backward[1] * offset,
                position[2]
                if not preserve_pitch
                else position[2]
                + offset
                * (position[2] - focus_point[2])
                / max(
                    math.hypot(
                        position[0] - focus_point[0],
                        position[1] - focus_point[1],
                    ),
                    profile.numeric_tolerance,
                ),
            )
            return value.model_copy(
                update={
                    "translation_m": moved,
                    "rotation_quaternion_wxyz": look_at_camera_quaternion(
                        moved,
                        focus_point,
                    ),
                }
            )

        candidate.camera.solved_transform = shifted(candidate.camera.solved_transform)
        if transform_track is not None:
            transform_track.keyframes = [
                item.model_copy(update={"value": shifted(item.value)})
                if isinstance(item.value, TransformValue)
                else item
                for item in transform_track.keyframes
            ]


def _fit_open_environment_ground_to_camera(
    objective: ObjectivePlanningBrief,
    candidate: CandidateState,
    profile: PlanningProfile,
) -> None:
    """Grow open preview ground around the camera path without changing scene scale."""

    if candidate.camera is None:
        return
    scene = (objective.translation_parameters or {}).get("scene", {})
    if scene.get("asset_key") == "interior":
        return
    transform_track = next(
        (item for item in candidate.camera.tracks.values() if item.type == "transform"),
        None,
    )
    camera_positions = [candidate.camera.solved_transform.translation_m]
    if transform_track is not None:
        camera_positions.extend(
            item.value.translation_m
            for item in transform_track.keyframes
            if isinstance(item.value, TransformValue)
        )
    positions = [item for item in camera_positions if item is not None]
    if not positions:
        return
    semantic_dimensions = scene.get("dimensions_m")
    semantic_extent = (
        max(float(item) for item in semantic_dimensions)
        if isinstance(semantic_dimensions, (list, tuple))
        and semantic_dimensions
        and all(isinstance(item, (int, float)) for item in semantic_dimensions)
        else 0.0
    )
    # An open-environment plane is a cheap rendering backdrop, not the semantic
    # coordinate system.  Keep a generous floor so shallow camera pitches do
    # not reveal the finite edge as a void; the web viewer excludes this plane
    # from content framing, so the larger proxy does not hide the real scene.
    render_extent_floor = max(1000.0, semantic_extent * 5.0)

    for entity_id, entity in list(candidate.entities.items()):
        if (
            entity.proxy.type != "plane"
            or "environment" not in entity.tags
            or "ground" not in entity.tags
        ):
            continue
        center = entity.solved_transform.translation_m or (0.0, 0.0, 0.0)
        current_x, current_y = entity.proxy.size_xy_m
        maximum_height = max(abs(item[2] - center[2]) for item in positions)
        margin = max(
            10.0,
            maximum_height * 8.0,
            profile.default_depth_gap_m * 4.0,
        )
        required_x = 2.0 * max(
            current_x / 2.0,
            render_extent_floor / 2.0,
            *(abs(item[0] - center[0]) + margin for item in positions),
        )
        required_y = 2.0 * max(
            current_y / 2.0,
            render_extent_floor / 2.0,
            *(abs(item[1] - center[1]) + margin for item in positions),
        )
        if (
            required_x <= current_x + profile.numeric_tolerance
            and required_y <= current_y + profile.numeric_tolerance
        ):
            continue
        candidate.entities[entity_id] = entity.model_copy(
            update={
                "proxy": entity.proxy.model_copy(
                    update={"size_xy_m": (required_x, required_y)}
                )
            }
        )


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
    focus_point, _ = _camera_focus(candidate, skeleton)
    distances = [
        math.dist(item, focus_point) for item in camera_positions if item is not None
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
    collision_clearances = [
        {
            "entity_ids": list(item.parameters.entity_ids),
            "minimum_meters": item.parameters.minimum_meters,
            "space": item.parameters.space,
        }
        for item in candidate.constraints.values()
        if item.type == "collision_clearance"
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
        "scene_reference_clearances_m": collision_clearances,
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
    has_subject_translation = any(
        item.kind in {"path_move", "carried"} for item in skeleton.motion_phases
    )
    assumptions = [
        f"数值策略采用 {strategy}",
        (
            "存在主体空间运动；未明确机位只采用 Validator 冻结的最小可读斜角"
            if has_subject_translation
            else "不存在主体空间运动；未明确机位沿规范场景纵深轴观察"
        ),
        "所有范围都保留 explicit > inferred > default 的来源优先级",
    ]
    if any(_is_orbit_phase(item) for item in skeleton.motion_phases):
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


def _camera_focus(
    candidate: CandidateState,
    skeleton: SceneSkeleton,
) -> tuple[tuple[float, float, float], float]:
    """Resolve an initial framing point without creating an implicit tracking target."""

    target_id = skeleton.camera_intent.focus_target_id
    if target_id is not None:
        target = candidate.entities[target_id]
        point = target.solved_transform.translation_m or (0.0, 0.0, 0.0)
        return point, _proxy_half_height(target)

    dependent_ids = {
        phase.subject_id
        for phase in skeleton.motion_phases
        if phase.direction_mode == "relative_to_target"
        and phase.path_family in {"circle", "ellipse"}
    }
    anchors = [
        entity
        for entity in candidate.entities.values()
        if entity.entity_id not in dependent_ids and entity.proxy.type != "plane"
    ]
    if not anchors:
        anchors = [
            entity
            for entity in candidate.entities.values()
            if entity.proxy.type != "plane"
        ]
    points = [
        entity.solved_transform.translation_m
        for entity in anchors
        if entity.solved_transform.translation_m is not None
    ]
    if not points:
        return (0.0, 0.0, 0.0), 0.0
    point = tuple(sum(item[axis] for item in points) / len(points) for axis in range(3))
    return point, point[2]


def _composition_projected_sizes(
    objective: ObjectivePlanningBrief,
    candidate: CandidateState,
    parameters: dict[str, Any],
) -> list[tuple[str, float, float, SourceStatus, str]]:
    """Keep a visual-scale range attached to the subject named by the Brief."""

    resolved: list[tuple[str, float, float, SourceStatus, str]] = []
    for index, item in enumerate(objective.composition.get("visual_scales", [])):
        if not isinstance(item, dict):
            continue
        subject_id = item.get("subject_id")
        scale = item.get("scale")
        if (
            not isinstance(subject_id, str)
            or subject_id not in candidate.entities
            or not isinstance(scale, dict)
        ):
            continue
        value_range = _percentage_range(scale.get("value"))
        source_status = scale.get("source_status")
        if value_range is None or source_status not in {
            "explicit",
            "inferred",
            "default",
        }:
            continue
        resolved.append(
            (
                subject_id,
                value_range[0],
                value_range[1],
                source_status,
                f"content.composition.visual_scales[{index}].scale",
            )
        )
    if resolved:
        return resolved

    ratio = parameters.get("subject_frame_ratio")
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        return []
    subject_id = next(
        (
            str(subject.get("id"))
            for subject in objective.subjects
            if isinstance(subject, dict)
            and isinstance(subject.get("id"), str)
            and subject["id"] in candidate.entities
        ),
        None,
    )
    if subject_id is None:
        return []
    source_status = parameters.get("source_status", "inferred")
    if source_status not in {"explicit", "inferred", "default"}:
        source_status = "inferred"
    if source_status == "explicit" and not any(
        requirement.path.startswith("content.composition.shot_size")
        for requirement in objective.explicit_requirements
    ):
        # Do not let an unrelated explicit composition field promote every
        # emotion-table ratio to an explicit camera instruction.
        source_status = "inferred"
    return [
        (
            subject_id,
            float(ratio[0]),
            float(ratio[1]),
            source_status,
            "translation_parameters.composition.subject_frame_ratio",
        )
    ]


def _major_composition_entity_id(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
) -> str | None:
    """Resolve the Brief's major visual object without relying on entity order."""

    for item in objective.composition.get("visual_scales", []):
        if not isinstance(item, dict):
            continue
        entity_id = item.get("subject_id")
        scale = item.get("scale")
        value = scale.get("value") if isinstance(scale, dict) else None
        if (
            isinstance(entity_id, str)
            and entity_id in candidate.entities
            and isinstance(value, str)
            and any(
                marker in value.lower() for marker in ("巨大", "巨物", "giant", "huge")
            )
        ):
            return entity_id

    for relation in skeleton.relations:
        if relation.kind == "scale_dominance":
            return relation.subject_id

    primary_id = next(
        (
            str(item.get("id"))
            for item in objective.subjects
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ),
        None,
    )
    ranks = {"tiny": 0, "small": 1, "human": 2, "unspecified": 2, "large": 3, "huge": 4}
    eligible = [
        item
        for item in skeleton.entities
        if item.entity_id in candidate.entities
        and candidate.entities[item.entity_id].proxy.type != "plane"
        and (item.entity_id != primary_id or len(candidate.entities) == 1)
    ]
    if not eligible:
        return primary_id
    return max(
        eligible,
        key=lambda item: (
            ranks[item.scale_intent],
            _proxy_bounding_radius(candidate.entities[item.entity_id]),
        ),
    ).entity_id


def _percentage_range(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    percentages = [
        float(item) / 100.0 for item in re.findall(r"(\d+(?:\.\d+)?)\s*%", value)
    ]
    if not percentages:
        return None
    minimum = percentages[0]
    maximum = percentages[1] if len(percentages) > 1 else minimum
    if not 0.0 <= minimum <= maximum <= 1.0:
        return None
    return minimum, maximum


def _screen_axis_region(
    value: Any,
    *,
    axis: Literal["horizontal", "vertical"],
) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    regions = (
        (
            (("左", "left"), (0.0, 0.45)),
            (("居中", "中央", "center", "centre"), (0.3, 0.7)),
            (("右", "right"), (0.55, 1.0)),
        )
        if axis == "horizontal"
        else (
            (("上", "top", "upper"), (0.0, 0.45)),
            (("居中", "中央", "center", "middle"), (0.3, 0.7)),
            (("下", "bottom", "lower"), (0.55, 1.0)),
        )
    )
    return next(
        (
            region
            for markers, region in regions
            if any(marker in lowered for marker in markers)
        ),
        None,
    )


def _event_range(
    objective: ObjectivePlanningBrief,
    event_id: str | None,
    duration: float,
) -> tuple[float, float]:
    if event_id is None:
        return (0.0, duration)
    for event in objective.timeline.get("events", []):
        if not isinstance(event, dict) or event.get("id") != event_id:
            continue
        start_value = event.get("start_time_seconds")
        end_value = event.get("end_time_seconds")
        if (
            not isinstance(start_value, (int, float))
            or isinstance(start_value, bool)
            or not isinstance(end_value, (int, float))
            or isinstance(end_value, bool)
        ):
            raise ValueError(f"timeline event 缺少有效时间范围：{event_id}")
        start = float(start_value)
        end = float(end_value)
        if not 0.0 <= start < end <= duration:
            raise ValueError(f"timeline event 时间范围无效：{event_id}")
        return (start, end)
    raise ValueError(f"引用了未知 timeline event：{event_id}")


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
    if phase.source_ref.startswith(prefix):
        try:
            index = int(phase.source_ref[len(prefix) :].split("]", 1)[0])
            motion = objective.subject_motion[index]
        except (IndexError, TypeError, ValueError, AttributeError) as error:
            raise ValueError(
                f"Motion Phase source_ref 无法解析：{phase.phase_id}"
            ) from error
        start_value = motion.get("start_time_seconds")
        end_value = motion.get("end_time_seconds")
        if not isinstance(start_value, (int, float)) or isinstance(start_value, bool):
            raise ValueError(f"Motion Phase 缺少有效开始时间：{phase.phase_id}")
        if not isinstance(end_value, (int, float)) or isinstance(end_value, bool):
            raise ValueError(f"Motion Phase 缺少有效结束时间：{phase.phase_id}")
        start = float(start_value)
        end = float(end_value)
        if not 0.0 <= start < end <= duration:
            raise ValueError(f"Motion Phase 引用了无效动作时间范围：{phase.phase_id}")
        return (start, end)
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


def _track_end_time(
    candidate: CandidateState,
    end: float,
    start: float,
) -> float:
    """Return the final render sample inside a half-open motion interval."""

    frame_step = candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    return max(
        start,
        min(
            end - frame_step,
            candidate.timeline.duration_seconds - frame_step,
        ),
    )


def _state_transition_time(candidate: CandidateState, time_seconds: float) -> float:
    """Apply boundary state changes at the boundary, except beyond the last frame."""

    frame_step = candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    return min(time_seconds, candidate.timeline.duration_seconds - frame_step)


def _camera_motion_range(
    objective: ObjectivePlanningBrief,
    duration: float,
) -> tuple[float, float]:
    movement = objective.camera.get("movement", {})
    start = movement.get("start_time_seconds") if isinstance(movement, dict) else None
    end = movement.get("end_time_seconds") if isinstance(movement, dict) else None
    if start is None and end is None:
        return (0.0, duration)
    if not isinstance(start, (int, float)) or isinstance(start, bool):
        raise ValueError("camera.movement 缺少有效开始时间")
    if not isinstance(end, (int, float)) or isinstance(end, bool):
        raise ValueError("camera.movement 缺少有效结束时间")
    resolved = (float(start), float(end))
    if not 0.0 <= resolved[0] < resolved[1] <= duration:
        raise ValueError("camera.movement 时间范围超出冻结镜头")
    return resolved


def _reject_unrepresentable_reference_frame_switch(
    objective: ObjectivePlanningBrief,
    phases: list[SkeletonMotionPhase],
    duration: float,
) -> None:
    """Reject visible carried-to-world motion that the current IR cannot compose.

    A carried phase is a target-relative path channel.  A later linear phase is
    a world transform channel, but path evaluation intentionally keeps its final
    value after the phase.  Silently emitting both would therefore mask the
    later motion.  Hidden carried subjects do not need that path and are safe.
    """

    for carried in (item for item in phases if item.kind == "carried"):
        carried_start, _ = _phase_range(objective, carried, duration)
        if _subject_hidden_at(phases, objective, duration, carried_start):
            continue
        later_world_phases = [
            item.phase_id
            for item in phases
            if item.kind == "path_move"
            and not _is_orbit_phase(item)
            and _phase_range(objective, item, duration)[1] > carried_start
        ]
        if later_world_phases:
            raise ValueError(
                "可见主体从 carried 参考系切回世界运动尚不能可靠编译："
                + ", ".join(later_world_phases)
            )


def _design_entity_transform_at(
    candidate: CandidateState,
    tracks: dict[str, TrackSpec],
    entity_id: str,
    time_seconds: float,
    resolving: set[str] | None = None,
) -> TransformValue:
    """Resolve deterministic-design tracks using the same frame semantics as IR."""

    resolving = set() if resolving is None else resolving
    if entity_id in resolving:
        raise ValueError(f"设计阶段参考系形成循环：{entity_id}")
    resolving.add(entity_id)
    try:
        entity = candidate.entities[entity_id]
        entity_tracks = [
            item for item in tracks.values() if item.target_entity_id == entity_id
        ]
        transform_track = next(
            (item for item in entity_tracks if item.type == "transform"),
            None,
        )
        path_track = next(
            (item for item in entity_tracks if item.type == "path_follow"),
            None,
        )
        raw = sample_transform_track(
            transform_track,
            time_seconds,
            entity.solved_transform,
        )
        if path_track is not None:
            raw = sample_path_track(path_track, time_seconds, raw)
        if raw.space == "world":
            return raw.model_copy(update={"target_id": None})
        if raw.space == "target_relative":
            if raw.target_id not in candidate.entities:
                raise ValueError(
                    f"设计阶段 target_relative 目标不存在：{raw.target_id}"
                )
            reference = _design_entity_transform_at(
                candidate,
                tracks,
                raw.target_id,
                time_seconds,
                resolving,
            )
        elif raw.space == "local":
            if entity.parent_id is None:
                reference = TransformValue(
                    translation_m=(0.0, 0.0, 0.0),
                    rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    space="world",
                )
            else:
                reference = _design_entity_transform_at(
                    candidate,
                    tracks,
                    entity.parent_id,
                    time_seconds,
                    resolving,
                )
        else:
            raise ValueError("设计阶段实体轨道不支持 camera 参考系")
        return _compose_design_transform(reference, raw)
    finally:
        resolving.remove(entity_id)


def _compose_design_transform(
    reference: TransformValue,
    relative: TransformValue,
) -> TransformValue:
    reference_translation = reference.translation_m or (0.0, 0.0, 0.0)
    reference_rotation = reference.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0)
    reference_scale = reference.scale or (1.0, 1.0, 1.0)
    relative_translation = relative.translation_m or (0.0, 0.0, 0.0)
    relative_rotation = relative.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0)
    relative_scale = relative.scale or (1.0, 1.0, 1.0)
    scaled_offset = tuple(
        relative_translation[index] * reference_scale[index] for index in range(3)
    )
    return TransformValue(
        translation_m=add(
            reference_translation,
            rotate_vector(reference_rotation, scaled_offset),
        ),
        rotation_quaternion_wxyz=quaternion_multiply(
            reference_rotation,
            relative_rotation,
        ),
        scale=tuple(
            reference_scale[index] * relative_scale[index] for index in range(3)
        ),
        space="world",
    )


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
    if (
        semantic_direction_mode == "world_forward"
        or phase.direction_mode == "world_forward"
    ):
        endpoint[1] -= 8.0
    elif phase.direction_mode == "toward_target" and phase.target_id:
        target_entity = candidate.entities[phase.target_id]
        target = _design_entity_transform_at(
            candidate,
            tracks,
            phase.target_id,
            sample_time,
        ).translation_m
        if target is not None:
            delta_x = target[0] - position[0]
            delta_y = target[1] - position[1]
            distance = math.hypot(delta_x, delta_y)
            if distance > 1e-9:
                clearance = (
                    0.0
                    if hidden_at_end
                    else (
                        _proxy_horizontal_radius(candidate.entities[phase.subject_id])
                        + _proxy_horizontal_radius(target_entity)
                    )
                )
                endpoint[0] = target[0] - delta_x / distance * clearance
                endpoint[1] = target[1] - delta_y / distance * clearance
    elif phase.direction_mode == "away_from_target" and phase.target_id:
        target = _design_entity_transform_at(
            candidate,
            tracks,
            phase.target_id,
            sample_time,
        ).translation_m
        if target is not None:
            delta_x = position[0] - target[0]
            delta_y = position[1] - target[1]
            distance = math.hypot(delta_x, delta_y)
            if distance <= 1e-9:
                delta_x, delta_y = fallback_direction
                distance = math.hypot(delta_x, delta_y)
            endpoint[0] += delta_x / distance * 8.0
            endpoint[1] += delta_y / distance * 8.0
    elif phase.direction_mode == "world_left":
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
        if phase.direction_mode in {"world_forward", "world_left", "world_right"}
    }
    if len(selected) > 1:
        raise ValueError("同一主体的未指定方向运动不能选择相反的局部路线")
    if selected == {"world_forward"}:
        return (0.0, -1.0)
    if selected == {"world_left"}:
        return (-1.0, 0.0)
    return (1.0, 0.0)


def _route_anchor_schedules(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
    candidate: CandidateState,
) -> dict[str, list[tuple[float, SkeletonRouteAnchor]]]:
    """Order each subject's symbolic waypoints by their semantic event time."""

    relations = {item.relation_id: item for item in skeleton.relations}
    phases = {item.phase_id: item for item in skeleton.motion_phases}
    frame_step = (
        candidate.timeline.fps_denominator / candidate.timeline.fps_numerator
    )
    schedules: dict[str, list[tuple[float, SkeletonRouteAnchor]]] = {}
    for route in skeleton.route_intents:
        schedules[route.subject_id] = sorted(
            (
                (
                    route_anchor_time_seconds(
                        objective,
                        relations[anchor.relation_id],
                        candidate.timeline.duration_seconds,
                        frame_step,
                        phase=phases[anchor.phase_id],
                    ),
                    anchor,
                )
                for anchor in route.anchors
            ),
            key=lambda item: (item[0], item[1].anchor_id),
        )
    return schedules


def _joint_proximity_route_positions(
    skeleton: SceneSkeleton,
    candidate: CandidateState,
    schedules: dict[str, list[tuple[float, SkeletonRouteAnchor]]],
    route_directions: dict[str, tuple[float, float]],
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Solve both sides of one moving proximity event without build-order bias."""

    anchors_by_relation: dict[
        str,
        list[tuple[str, float, SkeletonRouteAnchor]],
    ] = {}
    for subject_id, anchors in schedules.items():
        for time_seconds, anchor in anchors:
            anchors_by_relation.setdefault(anchor.relation_id, []).append(
                (subject_id, time_seconds, anchor)
            )

    solved: dict[tuple[str, str], tuple[float, float, float]] = {}
    relations = {item.relation_id: item for item in skeleton.relations}
    for relation_id, bindings in anchors_by_relation.items():
        relation = relations[relation_id]
        if relation.kind != "proximity":
            continue
        by_subject = {
            subject_id: (time_seconds, anchor)
            for subject_id, time_seconds, anchor in bindings
        }
        participants = {relation.subject_id, relation.reference_id}
        if set(by_subject) != participants:
            continue

        subject = candidate.entities[relation.subject_id]
        reference = candidate.entities[relation.reference_id]
        subject_position = subject.solved_transform.translation_m or (
            0.0,
            0.0,
            _proxy_half_height(subject),
        )
        reference_position = reference.solved_transform.translation_m or (
            0.0,
            0.0,
            _proxy_half_height(reference),
        )
        center_x = (subject_position[0] + reference_position[0]) * 0.5
        center_y = (subject_position[1] + reference_position[1]) * 0.5
        route_direction = route_directions.get(relation.subject_id, (1.0, 0.0))
        event_time = sum(item[0] for item in by_subject.values()) / len(by_subject)
        route_progress = 8.0 * event_time / candidate.timeline.duration_seconds
        center_x += route_direction[0] * route_progress
        center_y += route_direction[1] * route_progress
        lateral = (-route_direction[1], route_direction[0])
        clearance = max(
            0.75,
            _proxy_lateral_radius(subject, route_direction)
            + _proxy_lateral_radius(reference, route_direction)
            + 0.25,
        )
        half_clearance = clearance * 0.5
        subject_anchor = by_subject[relation.subject_id][1]
        reference_anchor = by_subject[relation.reference_id][1]
        solved[(relation.subject_id, subject_anchor.anchor_id)] = (
            center_x - lateral[0] * half_clearance,
            center_y - lateral[1] * half_clearance,
            subject_position[2],
        )
        solved[(relation.reference_id, reference_anchor.anchor_id)] = (
            center_x + lateral[0] * half_clearance,
            center_y + lateral[1] * half_clearance,
            reference_position[2],
        )
    return solved


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
    tracks: dict[str, TrackSpec],
    subject_id: str,
    anchor: SkeletonRouteAnchor,
    route_direction: tuple[float, float],
    profile: PlanningProfile,
    time_seconds: float,
    joint_positions: dict[
        tuple[str, str],
        tuple[float, float, float],
    ],
) -> tuple[float, float, float]:
    """Resolve one symbolic route anchor into a validated world-space waypoint."""

    joint_position = joint_positions.get((subject_id, anchor.anchor_id))
    if joint_position is not None:
        return joint_position

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
    other_position = _design_entity_transform_at(
        candidate,
        tracks,
        other_id,
        time_seconds,
    ).translation_m
    if relation.kind == "proximity":
        clearance = max(
            0.75,
            _proxy_lateral_radius(subject, route_direction)
            + _proxy_lateral_radius(other, route_direction)
            + 0.25,
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
    raise ValueError(f"Route Anchor {anchor.anchor_id} 的空间关系无法求解为路径点")


def _objective_motion_direction(
    objective: ObjectivePlanningBrief,
    phase: SkeletonMotionPhase,
) -> str | None:
    """Resolve one motion's direction without borrowing a sibling phase."""

    motions = [
        motion
        for motion in objective.subject_motion
        if isinstance(motion, dict) and motion.get("subject_id") == phase.subject_id
    ]
    if phase.motion_id is not None:
        exact = [
            motion for motion in motions if motion.get("motion_id") == phase.motion_id
        ]
        if len(exact) == 1:
            return _motion_direction_value(exact[0])

    prefix = "content.subject_motion["
    if phase.source_ref.startswith(prefix):
        try:
            index = int(phase.source_ref[len(prefix) :].split("]", 1)[0])
            source = objective.subject_motion[index]
        except (IndexError, TypeError, ValueError, AttributeError):
            source = None
        if isinstance(source, dict) and source.get("subject_id") == phase.subject_id:
            return _motion_direction_value(source)

    same_event = [
        motion
        for motion in motions
        if isinstance(motion.get("motion_semantics"), dict)
        and motion["motion_semantics"].get("timeline_event_id")
        == phase.timeline_event_id
    ]
    if len(same_event) == 1:
        return _motion_direction_value(same_event[0])
    return None


def _motion_direction_value(motion: dict[str, Any]) -> str | None:
    semantics = motion.get("motion_semantics")
    direction = semantics.get("direction_mode") if isinstance(semantics, dict) else None
    return direction if isinstance(direction, str) else None


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


def _camera_speed_mps(
    intent: SkeletonCameraIntent,
    profile: PlanningProfile,
    *,
    configured: Any = None,
) -> float:
    if intent.speed_intent == "match_subject":
        return profile.medium_speed_range_mps[0]
    if isinstance(configured, (int, float)) and not isinstance(configured, bool):
        return max(0.0, float(configured))
    if intent.speed_intent in {"slow", "medium", "fast"}:
        minimum, maximum = _speed_range(intent.speed_intent, profile)
        return (minimum + maximum) * 0.5
    return profile.slow_speed_range_mps[0]


def _explicit_camera_value(
    objective: ObjectivePlanningBrief,
    field: str,
) -> Any:
    node = objective.camera.get(field)
    if isinstance(node, dict) and node.get("source_status") == "explicit":
        return node.get("value")
    return None


def _explicit_camera_height(objective: ObjectivePlanningBrief) -> float | None:
    value = _explicit_camera_value(objective, "camera_height")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\b|米)", value.lower())
    return float(match.group(1)) if match else None


def _explicit_camera_focal_length(objective: ObjectivePlanningBrief) -> float | None:
    value = _explicit_camera_value(objective, "lens_intent")
    return camera_lens_focal_length(value)


def _explicit_camera_pitch_degrees(objective: ObjectivePlanningBrief) -> float | None:
    value = _explicit_camera_value(objective, "view_angle")
    return {
        "top_down": 80.0,
        "high_angle": 30.0,
        "low_angle": -10.0,
        "eye_level": 0.0,
    }.get(classify_camera_view_angle(value))


def _implemented_camera_static_source_refs(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
) -> set[str]:
    refs: set[str] = set()
    if _explicit_camera_pitch_degrees(objective) is not None:
        refs.add("content.camera.view_angle")
    if _explicit_camera_height(objective) is not None:
        refs.add("content.camera.camera_height")
    if _explicit_camera_focal_length(objective) is not None:
        refs.add("content.camera.lens_intent")
    focus = objective.camera.get("focus_target_id")
    if (
        isinstance(focus, dict)
        and focus.get("source_status") == "explicit"
        and skeleton.camera_intent.focus_target_id is not None
    ):
        refs.add("content.camera.focus_target_id")
    view = objective.camera.get("view_relation_to_motion")
    if (
        isinstance(view, dict)
        and view.get("source_status") == "explicit"
        and skeleton.camera_intent.view_relation_to_motion != "unspecified"
    ):
        refs.add("content.camera.view_relation_to_motion")
    return refs


def _implemented_camera_motion_source_refs(
    objective: ObjectivePlanningBrief,
    skeleton: SceneSkeleton,
) -> set[str]:
    """Return camera-motion facts represented by the generated track geometry."""

    refs = {skeleton.camera_intent.source_ref}
    if skeleton.camera_intent.speed_source_ref is not None:
        refs.add(skeleton.camera_intent.speed_source_ref)
    for field in ("direction", "trajectory"):
        node = objective.camera.get("movement", {}).get(field)
        if isinstance(node, dict) and node.get("source_status") == "explicit":
            refs.add(f"content.camera.movement.{field}")
    return refs


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
