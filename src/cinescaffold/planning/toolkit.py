from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
from typing import Any

from pydantic import ValidationError

from cinescaffold.planning.domain import (
    CameraCandidate,
    CameraStatic,
    CandidateState,
    ConstraintSpec,
    EntitySpec,
    DurationResolution,
    PlanningProfile,
    TimelineSpec,
    TrackSpec,
    TransformValue,
    ValidationReport,
    Violation,
)
from cinescaffold.planning.geometry import (
    add,
    cross,
    dot,
    geometry_bounding_radius,
    geometry_local_bounds_points,
    length,
    look_at_camera_quaternion,
    normalize,
    project_point,
    project_geometry_bounds,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    sample_path_track,
    sample_scalar_track,
    sample_transform_track,
    subtract,
)
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.planning.store import CandidateStore, MutationResult, canonical_hash


TOOLKIT_VERSION = "0.13"
CONSTRAINT_CATALOG_VERSION = "0.1"
SUPPORTED_CONSTRAINTS = {
    "relative_position",
    "distance_range",
    "depth_order",
    "screen_region",
    "projected_size",
    "projected_scale_ratio",
    "keep_in_frame",
    "look_at",
    "camera_distance",
    "focal_length_range",
    "camera_motion_direction",
    "speed_range",
    "position_at_time",
    "motion_direction",
    "hold",
}
FULL_VALIDATION_CHECKS = [
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
INSPECT_VIEWS = [
    "summary",
    "entities",
    "camera",
    "constraints",
    "violations",
    "timeline",
    "diff",
    "full_ir",
]
CAMERA_SINGLETON_TRACK_TYPES = {"transform", "path_follow", "look_at", "focal_length"}
ENTITY_SINGLETON_TRACK_TYPES = {"transform", "path_follow", "visibility", "look_at"}


def _commit_ready(report: ValidationReport | None, profile: PlanningProfile) -> bool:
    return bool(
        report
        and report.hard_pass
        and report.soft_score >= profile.minimum_soft_score
    )


def _duplicate_camera_track_ids(tracks: dict[str, TrackSpec]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for track in tracks.values():
        if track.type in CAMERA_SINGLETON_TRACK_TYPES:
            grouped.setdefault(track.type, []).append(track.track_id)
    return {
        track_type: sorted(track_ids)
        for track_type, track_ids in grouped.items()
        if len(track_ids) > 1
    }


def _validate_singleton_camera_tracks(tracks: dict[str, TrackSpec]) -> None:
    duplicates = _duplicate_camera_track_ids(tracks)
    if not duplicates:
        return
    details = "; ".join(
        f"{track_type}={','.join(track_ids)}"
        for track_type, track_ids in sorted(duplicates.items())
    )
    raise ValueError(
        f"Camera 单一通道存在重叠轨道：{details}；请用 remove_track_ids 删除旧轨道"
    )


def _duplicate_entity_track_ids(
    tracks: dict[str, TrackSpec],
) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for track in tracks.values():
        if track.target_entity_id and track.type in ENTITY_SINGLETON_TRACK_TYPES:
            grouped.setdefault((track.target_entity_id, track.type), []).append(track.track_id)
    return {
        key: sorted(track_ids)
        for key, track_ids in grouped.items()
        if len(track_ids) > 1
    }


def _validate_singleton_entity_tracks(tracks: dict[str, TrackSpec]) -> None:
    duplicates = _duplicate_entity_track_ids(tracks)
    if not duplicates:
        return
    details = "; ".join(
        f"{entity_id}.{track_type}={','.join(track_ids)}"
        for (entity_id, track_type), track_ids in sorted(duplicates.items())
    )
    raise ValueError(
        f"Entity 单一通道存在重叠轨道：{details}；请用 remove_ids 删除旧轨道"
    )


class ScenePlanningToolkit:
    def __init__(
        self,
        objective_brief: ObjectivePlanningBrief,
        profile: PlanningProfile | None = None,
        scene_id: str | None = None,
        initial_candidate: CandidateState | None = None,
    ) -> None:
        self.objective_brief = objective_brief.model_copy(deep=True)
        self.profile = profile or PlanningProfile()
        expected = _initial_candidate(self.objective_brief, self.profile, scene_id)
        if initial_candidate is not None:
            _validate_initial_candidate(initial_candidate, expected)
            expected = initial_candidate.model_copy(deep=True)
        self.store = CandidateStore(expected)

    def get_capabilities(self, sections: list[str] | None = None) -> dict[str, Any]:
        state = self.store.get()
        requested = sections or [
            "entities",
            "constraints",
            "tracks",
            "camera",
            "validators",
            "limits",
        ]
        data: dict[str, Any] = {
            "versions": {
                "toolkit": TOOLKIT_VERSION,
                "constraint_catalog": CONSTRAINT_CATALOG_VERSION,
                "scene_ir": "0.1",
                "profile": self.profile.profile_id,
            },
            "coordinate_system": {
                "linear_unit": "meter",
                "handedness": "right",
                "up_axis": "+Z",
                "world_semantic_axes": {
                    "right": "+X",
                    "left": "-X",
                    "front": "-Y",
                    "behind": "+Y",
                    "up": "+Z",
                    "down": "-Z",
                },
                "default_horizontal_plane": "XY",
                "default_path_plane_normal": [0.0, 0.0, 1.0],
                "default_path_axis_direction": [1.0, 0.0, 0.0],
                "path_direction_viewpoint": "从 +plane_normal 一侧朝路径中心观察",
                "rotation_representation": "quaternion_wxyz",
                "angle_unit": "radian",
                "initial_phase_unit": "degree",
                "camera_local_forward_axis": "-Z",
                "camera_local_up_axis": "+Y",
                "screen_coordinates": {
                    "range": [0.0, 1.0],
                    "origin": "top_left",
                    "x_direction": "right",
                    "y_direction": "down",
                },
            },
            "sections": requested,
            "supported_geometry": ["box", "sphere", "capsule", "cylinder", "cone", "plane"],
            "supported_ground_interactions": [
                "must_be_above",
                "must_touch",
                "may_intersect",
                "embedded",
                "unconstrained",
            ],
            "ground_interaction_guidance": {
                "must_be_above": "缺省；允许贴地或悬空，不允许穿入",
                "must_touch": "最低点必须在 tolerance_m 内接触地面",
                "may_intersect": "允许不超过 maximum_penetration_m 的部分穿入",
                "embedded": "穿入深度必须位于最小/最大范围，保留完整代理几何",
                "unconstrained": "仅用于 Brief 明确的地下或地面不适用语义",
                "exception_provenance": (
                    "may_intersect/embedded/unconstrained 必须引用 Brief explicit requirement"
                ),
                "without_ground_entity": (
                    "场景没有环境地面平面时不执行地面相交检查；空中或太空实体保持缺省即可"
                ),
            },
            "supported_tracks": ["transform", "path_follow", "visibility", "look_at", "focal_length"],
            "supported_path_representations": [
                "polyline",
                "sampled",
                "circle",
                "ellipse",
                "catmull_rom",
                "lemniscate",
            ],
            "reference_frames": {
                "world": "坐标直接位于规范世界空间",
                "local": "坐标相对实体 parent_id；无父级时等同 world",
                "target_relative": "坐标相对 target_id 当前帧的完整 Transform，可递归嵌套",
                "camera": "实体坐标相对当前活动摄影机；摄影机自身不得使用",
                "closed_path": "closed=true 会确定性补上末段到首点",
                "cycle_count": "闭合路径在 Track 时间段内的循环次数，可为正小数",
                "nested_orbit_readability": (
                    "嵌套 orbit_around 不得同相锁定；未指定周期时应让子轨道具有可辨识节奏"
                ),
                "trajectory_selection": (
                    "普通 orbit_around 使用 circle/ellipse；S 形使用 catmull_rom；"
                    "∞ 形使用 lemniscate；polyline 只用于明确折线路径"
                ),
                "analytic_path_defaults": {
                    "plane_normal": [0.0, 0.0, 1.0],
                    "axis_direction": [1.0, 0.0, 0.0],
                    "initial_phase_degrees": 0.0,
                    "direction": "counterclockwise",
                    "direction_viewpoint": "从 +plane_normal 一侧朝路径中心观察",
                },
            },
            "semantic_distinctions": {
                "relative_position_front_behind": "规范世界 -Y/+Y；不表示摄影机深度",
                "camera_depth_order": "使用 depth_order；深度沿摄影机 -Z 前向取正值",
                "keep_in_frame": "只验证自身投影包围盒入框比例，不验证被其他实体遮挡的比例",
                "look_at_precedence": "生效的 look_at Track 覆盖摄影机 Transform Track 的旋转",
                "path_follow_precedence": (
                    "生效的 path_follow 生成位置并覆盖同实体静态求解位置；"
                    "target_relative 由 Resolver 逐帧递归合成"
                ),
                "unset_optional_values": "未指定的可选实现参数应省略以采用版本化默认值，不要猜其他软件惯例",
            },
            "inspect_views": INSPECT_VIEWS,
            "acceptance": {
                "minimum_soft_score": self.profile.minimum_soft_score,
                "requires_hard_pass": True,
                "commit_ready": _commit_ready(state.validation, self.profile),
                "minimum_orbit_plane_view_alignment": (
                    self.profile.minimum_orbit_plane_view_alignment
                ),
            },
            "supported_constraints": sorted(SUPPORTED_CONSTRAINTS),
            "constraint_parameter_schemas": {
                "relative_position": {
                    "required": ["subject_id", "reference_id", "relation"],
                    "optional": ["space", "minimum_gap", "maximum_gap"],
                    "allowed_values": {
                        "relation": ["left", "right", "front", "behind", "below", "above"],
                        "space": ["world"],
                    },
                    "defaults": {"space": "world"},
                },
                "distance_range": {
                    "required": ["entity_ids", "minimum_meters", "maximum_meters"],
                },
                "depth_order": {
                    "required": ["near_entity_id", "far_entity_id"],
                    "optional": ["camera_id", "minimum_depth_gap_meters"],
                },
                "screen_region": {
                    "required": ["entity_id", "region"],
                    "format": {"region": "[left,top,right,bottom]，归一化左上原点坐标"},
                },
                "projected_size": {
                    "required": ["entity_id", "minimum", "maximum"],
                    "optional": ["measurement"],
                    "allowed_values": {"measurement": ["height", "width", "diameter"]},
                    "defaults": {"measurement": "height"},
                },
                "projected_scale_ratio": {
                    "required": [
                        "numerator_entity_id",
                        "denominator_entity_id",
                        "minimum_ratio",
                        "maximum_ratio",
                    ],
                    "optional": ["measurement"],
                    "allowed_values": {"measurement": ["height", "width", "diameter"]},
                    "defaults": {"measurement": "height"},
                },
                "keep_in_frame": {
                    "required": ["entity_id", "minimum_inside_fraction"],
                },
                "look_at": {
                    "required": ["observer_id", "target_id"],
                    "optional": ["maximum_angle_error_degrees"],
                },
                "camera_distance": {
                    "required": ["target_id", "minimum_meters", "maximum_meters"],
                    "optional": ["camera_id"],
                },
                "focal_length_range": {
                    "required": ["minimum_mm", "maximum_mm"],
                    "optional": ["camera_id"],
                },
                "motion_direction": {
                    "required": ["target_id", "direction"],
                    "optional": ["space", "minimum_displacement_m"],
                    "allowed_values": {
                        "direction": ["left", "right", "forward", "backward", "up", "down"],
                        "space": ["world"],
                    },
                    "defaults": {"space": "world", "minimum_displacement_m": 0.01},
                },
                "camera_motion_direction": {
                    "required": ["direction"],
                    "optional": ["camera_id", "target_id", "space", "minimum_displacement_m"],
                    "allowed_values": {
                        "direction": [
                            "left", "right", "forward", "backward", "up", "down", "push_in", "pull_out"
                        ],
                        "space": ["world", "camera"],
                    },
                    "defaults": {"camera_id": "camera_main", "space": "world", "minimum_displacement_m": 0.01},
                },
                "speed_range": {
                    "required": ["target_id", "maximum_mps"],
                    "optional": ["minimum_mps", "space"],
                    "allowed_values": {"space": ["world"]},
                    "defaults": {"minimum_mps": 0.0, "space": "world"},
                },
                "position_at_time": {
                    "required": ["target_id", "position_m"],
                    "optional": ["space", "tolerance_m"],
                    "allowed_values": {"space": ["world"]},
                    "defaults": {"space": "world", "tolerance_m": 0.01},
                },
                "hold": {
                    "required": ["target_id", "components"],
                    "optional": [
                        "tolerance_m",
                        "rotation_tolerance_degrees",
                        "scale_tolerance",
                    ],
                    "allowed_values": {
                        "components": ["translation", "rotation", "scale", "visibility"]
                    },
                    "defaults": {
                        "tolerance_m": 0.0001,
                        "rotation_tolerance_degrees": 0.01,
                        "scale_tolerance": 0.0001,
                    },
                },
            },
            "constraint_guidance": {
                "camera_motion_direction": (
                    "push_in/pull_out 按摄影机到 target_id（缺省为 focus target）的距离变化验证；"
                    "不得按固定世界轴解释"
                ),
                "speed_range": (
                    "用 target_id=camera_main 表达摄影机移动速度；缓慢/快速不能改写成摄影机距离"
                ),
                "keep_in_frame": "按旋转后代理体的投影包围盒面积比例验证，并使用数值容差",
                "hold": (
                    "逐帧对照约束起点检查所选 components；translation、rotation、scale、visibility "
                    "分别使用自己的容差或布尔一致性"
                ),
            },
            "validators": FULL_VALIDATION_CHECKS,
            "timeline": {
                "fps_numerator": state.timeline.fps_numerator,
                "fps_denominator": state.timeline.fps_denominator,
                "frame_count": state.timeline.frame_count,
                "duration_seconds": state.timeline.duration_seconds,
                "time_domain": "half_open",
                "time_range_seconds": [0.0, state.timeline.duration_seconds],
                "last_frame_time_seconds": (
                    (state.timeline.frame_count - 1)
                    * state.timeline.fps_denominator
                    / state.timeline.fps_numerator
                ),
            },
            "current_counts": {
                "entities": len(state.entities),
                "tracks": len(state.motion_tracks),
                "constraints": len(state.constraints),
                "revision": state.revision,
                "camera_track_ids": sorted(state.camera.tracks) if state.camera else [],
            },
        }
        gaps = [
            "compound_proxy_geometry",
            "occlusion_fraction_validator",
            "collision_clearance_solver",
            "negative_space_validator",
            "event_synchronization_solver",
        ]
        return _envelope(state.revision, state.revision, data=data, capability_gaps=gaps)

    def inspect_candidate(
        self,
        revision: int | None = None,
        view: str = "summary",
        entity_ids: list[str] | None = None,
        camera_ids: list[str] | None = None,
        constraint_ids: list[str] | None = None,
        time_range_seconds: tuple[float, float] | None = None,
        compare_to_revision: int | None = None,
    ) -> dict[str, Any]:
        del camera_ids, time_range_seconds
        state = self.store.get(revision)
        if view == "summary":
            data: Any = {
                "scene_id": state.scene_id,
                "revision": state.revision,
                "entity_count": len(state.entities),
                "motion_track_count": len(state.motion_tracks),
                "constraint_count": len(state.constraints),
                "camera_ready": state.camera is not None,
                "validation": state.validation.model_dump(mode="json") if state.validation else None,
                "candidate_hash": canonical_hash(state),
            }
        elif view == "entities":
            selected = entity_ids or sorted(state.entities)
            data = {
                name: state.entities[name].model_dump(mode="json")
                for name in selected
                if name in state.entities
            }
        elif view == "camera":
            data = state.camera.model_dump(mode="json") if state.camera else None
        elif view == "constraints":
            selected = constraint_ids or sorted(state.constraints)
            data = {
                name: state.constraints[name].model_dump(mode="json")
                for name in selected
                if name in state.constraints
            }
        elif view == "violations":
            data = state.validation.model_dump(mode="json") if state.validation else None
        elif view == "timeline":
            data = state.timeline.model_dump(mode="json") | {"frame_end": state.timeline.frame_end}
        elif view == "diff":
            if compare_to_revision is None:
                return _rejected(state.revision, "diff 视图必须提供 compare_to_revision")
            compared = self.store.get(compare_to_revision)
            data = {
                "from_revision": compared.revision,
                "to_revision": state.revision,
                "from_hash": canonical_hash(compared),
                "to_hash": canonical_hash(state),
            }
        elif view == "full_ir":
            data = state.model_dump(mode="json")
        else:
            return _rejected(state.revision, f"未知 inspect view：{view}")
        return _envelope(state.revision, state.revision, data=data)

    def apply_entity_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [EntitySpec.model_validate(item) for item in upserts]
            upsert_ids = {item.entity_id for item in parsed}
            remove_only_ids = set(remove_ids) - upsert_ids

            def mutate(state: CandidateState):
                referenced = _referenced_entity_ids(state)
                blocked = sorted(remove_only_ids & referenced)
                if blocked:
                    raise ValueError(f"Entity 仍被引用，不能删除：{', '.join(blocked)}")
                changes: list[dict[str, Any]] = []
                for entity_id in remove_only_ids:
                    if state.entities.pop(entity_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"entities.{entity_id}"})
                for entity in parsed:
                    existing = state.entities.get(entity.entity_id)
                    if existing and _proxy_topology(existing.proxy) != _proxy_topology(entity.proxy):
                        raise ValueError(
                            f"Entity 建立后不得更换代理拓扑：{entity.entity_id}；"
                            "请调整尺寸、Transform 或约束，不得通过换形状迎合投影指标"
                        )
                    operation = "replace" if entity.entity_id in state.entities else "add"
                    state.entities[entity.entity_id] = entity
                    changes.append({"operation": operation, "path": f"entities.{entity.entity_id}"})
                _validate_parent_references(state)
                _validate_ground_interaction_references(state, self.objective_brief)
                _validate_reference_frame_graph(state, self.profile)
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))

    def apply_constraint_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [ConstraintSpec.model_validate(item) for item in upserts]
            invalid_hard = sorted(
                item.constraint_id
                for item in parsed
                if item.strength == "hard" and item.source_status != "explicit"
            )
            if invalid_hard:
                return _rejected(
                    self.store.current_revision,
                    "只有 explicit requirement 可成为 hard constraint；"
                    f"请将以下约束改为 soft：{', '.join(invalid_hard)}",
                )
            unsupported = sorted({item.type for item in parsed} - SUPPORTED_CONSTRAINTS)
            if unsupported:
                return _envelope(
                    self.store.current_revision,
                    self.store.current_revision,
                    status="unsupported",
                    capability_gaps=[f"constraint:{name}" for name in unsupported],
                    next_actions=[
                        "不得用语义无关约束替代；若 explicit requirement 没有等价能力，返回 UnsupportedResult"
                    ],
                )
            incompatible_hard = sorted(
                item.constraint_id
                for item in parsed
                if item.strength == "hard"
                and (
                    item.source_ref not in self.store.get().required_source_refs
                    or not _source_mapping_is_compatible(
                        item.source_ref,
                        {f"constraint:{item.type}"},
                    )
                )
            )
            if incompatible_hard:
                return _rejected(
                    self.store.current_revision,
                    "hard constraint 必须直接对应兼容的 explicit requirement；"
                    f"以下约束来源不兼容：{', '.join(incompatible_hard)}",
                )
            upsert_ids = {item.constraint_id for item in parsed}
            remove_only_ids = set(remove_ids) - upsert_ids

            def mutate(state: CandidateState):
                changes: list[dict[str, Any]] = []
                for constraint_id in remove_only_ids:
                    existing = state.constraints.get(constraint_id)
                    if existing and existing.strength == "hard" and existing.source_status == "explicit":
                        raise ValueError("不得删除 explicit hard constraint")
                    if state.constraints.pop(constraint_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"constraints.{constraint_id}"})
                for constraint in parsed:
                    existing = state.constraints.get(constraint.constraint_id)
                    if existing and existing.strength == "hard" and existing.source_status == "explicit":
                        if constraint.strength != "hard" or constraint.source_ref != existing.source_ref:
                            raise ValueError("不得降级或改写 explicit hard constraint 来源")
                    _validate_constraint_time(constraint, state.timeline.duration_seconds)
                    state.constraints[constraint.constraint_id] = constraint
                    changes.append({
                        "operation": "replace" if existing else "add",
                        "path": f"constraints.{constraint.constraint_id}",
                    })
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))

    def apply_motion_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [TrackSpec.model_validate(item) for item in upserts]
            unsupported = sorted(
                track.track_id
                for track in parsed
                if track.type not in ENTITY_SINGLETON_TRACK_TYPES
            )
            if unsupported:
                return _rejected(
                    self.store.current_revision,
                    f"Entity Track 类型不受支持：{', '.join(unsupported)}",
                )
            upsert_ids = {item.track_id for item in parsed}
            remove_only_ids = set(remove_ids) - upsert_ids

            def mutate(state: CandidateState):
                changes: list[dict[str, Any]] = []
                for track_id in remove_only_ids:
                    if state.motion_tracks.pop(track_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"motion_tracks.{track_id}"})
                for track in parsed:
                    if not track.target_entity_id or track.target_entity_id not in state.entities:
                        raise ValueError(f"Track 目标 Entity 不存在：{track.target_entity_id}")
                    _validate_track_time(track, state.timeline.duration_seconds)
                    existing = track.track_id in state.motion_tracks
                    state.motion_tracks[track.track_id] = track
                    changes.append({
                        "operation": "replace" if existing else "add",
                        "path": f"motion_tracks.{track.track_id}",
                    })
                _validate_singleton_entity_tracks(state.motion_tracks)
                _validate_reference_frame_graph(state, self.profile)
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))

    def apply_camera_patch(
        self,
        camera_id: str,
        projection: str,
        active: bool,
        static: dict[str, Any],
        tracks: list[dict[str, Any]],
        remove_track_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_track_ids = remove_track_ids or []
        try:
            if projection != "perspective":
                return _envelope(
                    self.store.current_revision,
                    self.store.current_revision,
                    status="unsupported",
                    capability_gaps=[f"camera_projection:{projection}"],
                )
            parsed_static = CameraStatic.model_validate(static)
            parsed_tracks = [TrackSpec.model_validate(item) for item in tracks]
            unsupported = sorted(
                track.track_id
                for track in parsed_tracks
                if track.type not in CAMERA_SINGLETON_TRACK_TYPES
            )
            if unsupported:
                raise ValueError(f"Camera Track 类型不受支持：{', '.join(unsupported)}")

            def mutate(state: CandidateState):
                current_tracks = deepcopy(state.camera.tracks) if state.camera else {}
                for track_id in remove_track_ids:
                    current_tracks.pop(track_id, None)
                for track in parsed_tracks:
                    if track.target_entity_id is not None:
                        raise ValueError("Camera Track 不应填写 target_entity_id")
                    _validate_track_time(track, state.timeline.duration_seconds)
                    current_tracks[track.track_id] = track
                _validate_singleton_camera_tracks(current_tracks)
                previous = state.camera
                state.camera = CameraCandidate(
                    camera_id=camera_id,
                    projection="perspective",
                    active=active,
                    static=parsed_static,
                    tracks=current_tracks,
                    solved_transform=(
                        previous.solved_transform if previous else TransformValue()
                    ),
                )
                _validate_reference_frame_graph(state, self.profile)
                return ([{"operation": "replace" if previous else "add", "path": "camera"}], [])

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))

    def solve_candidate(
        self,
        scope: str = "all",
        constraint_ids: list[str] | None = None,
        allowed_variables: list[str] | None = None,
        locked_variables: list[str] | None = None,
        profile: str = "research_default",
        strategy: str = "auto",
    ) -> dict[str, Any]:
        del allowed_variables, profile
        if scope not in {"layout", "camera", "motion", "all"}:
            return _rejected(self.store.current_revision, f"未知 solve scope：{scope}")
        if strategy not in {"auto", "heuristic", "numeric", "hybrid"}:
            return _rejected(self.store.current_revision, f"未知 solve strategy：{strategy}")
        selected_ids = set(constraint_ids or [])
        locked = set(locked_variables or [])

        def mutate(state: CandidateState):
            changes: list[dict[str, Any]] = []
            if scope in {"layout", "motion", "all"}:
                for index, entity in enumerate(state.entities.values()):
                    if f"{entity.entity_id}.translation" in locked:
                        continue
                    current = entity.solved_transform
                    if current.translation_m is None:
                        radius = _geometry_half_height(entity.proxy)
                        entity.solved_transform = TransformValue(
                            translation_m=(float(index * 2), float(index * 5), radius),
                            rotation_quaternion_wxyz=current.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
                            scale=current.scale or (1.0, 1.0, 1.0),
                            space="world",
                        )
                        changes.append({"operation": "solve", "path": f"entities.{entity.entity_id}.solved_transform"})
                    else:
                        entity.solved_transform = _complete_transform(entity.solved_transform)

                for constraint in state.constraints.values():
                    if selected_ids and constraint.constraint_id not in selected_ids:
                        continue
                    if constraint.type in {"relative_position", "depth_order", "distance_range"}:
                        if _apply_layout_constraint(state, constraint, self.profile):
                            changes.append({"operation": "solve", "path": f"constraints.{constraint.constraint_id}"})

            if scope in {"camera", "all"}:
                if state.camera is None:
                    focus = next(iter(state.entities), None)
                    state.camera = CameraCandidate(static=CameraStatic(focus_target_id=focus))
                    changes.append({"operation": "solve", "path": "camera"})
                camera = state.camera
                if camera.solved_transform.translation_m is None:
                    camera.solved_transform = TransformValue(
                        translation_m=(0.0, -self.profile.default_camera_distance_m, 2.0),
                        rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                        scale=(1.0, 1.0, 1.0),
                        space="world",
                    )
                    changes.append({"operation": "solve", "path": "camera.solved_transform"})
                else:
                    camera.solved_transform = _complete_transform(camera.solved_transform)
                if camera.static.focal_length_mm is None:
                    camera.static.focal_length_mm = self.profile.default_focal_length_mm
                    changes.append({"operation": "solve", "path": "camera.static.focal_length_mm"})

            return changes, []

        try:
            mutation = self.store.apply(mutate)
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))
        report = self._validate(self.store.get(), FULL_VALIDATION_CHECKS)
        self.store.save_validation(report)
        data = {
            "solve_status": "solved" if report.hard_pass else "partial",
            "solver": "deterministic_heuristic_v0.1",
            "strategy_requested": strategy,
            "random_seed": self.profile.random_seed,
            "hard_pass": report.hard_pass,
            "soft_score": report.soft_score,
            "minimum_soft_score": self.profile.minimum_soft_score,
            "commit_ready": _commit_ready(report, self.profile),
        }
        return _mutation_envelope(
            mutation,
            data=data,
            violations=[item.model_dump(mode="json") for item in report.violations],
            capability_gaps=report.capability_gaps,
        )

    def validate_candidate(
        self,
        revision: int | None = None,
        checks: list[str] | None = None,
        sampling_profile: str = "research_default",
    ) -> dict[str, Any]:
        del sampling_profile
        state = self.store.get(revision)
        selected = checks or FULL_VALIDATION_CHECKS
        unknown = sorted(set(selected) - set(FULL_VALIDATION_CHECKS))
        if unknown:
            return _rejected(state.revision, f"未知 Validator：{', '.join(unknown)}")
        report = self._validate(state, selected)
        if revision is None or revision == self.store.current_revision:
            self.store.save_validation(report)
        return _envelope(
            state.revision,
            state.revision,
            data=report.model_dump(mode="json")
            | {
                "minimum_soft_score": self.profile.minimum_soft_score,
                "commit_ready": _commit_ready(report, self.profile),
            },
            violations=[item.model_dump(mode="json") for item in report.violations],
            capability_gaps=report.capability_gaps,
        )

    def restore_candidate(self, source_revision: int, reason: str) -> dict[str, Any]:
        if not reason.strip():
            return _rejected(self.store.current_revision, "restore 必须记录 reason")
        try:
            result = self.store.restore(source_revision)
            return _mutation_envelope(result, data={"reason": reason})
        except ValueError as error:
            return _rejected(self.store.current_revision, str(error))

    def _validate(self, state: CandidateState, checks: list[str]) -> ValidationReport:
        violations: list[Violation] = []
        gaps: list[str] = []
        if "references" in checks or "hierarchy" in checks:
            violations.extend(_reference_violations(state, self.profile))
        if "timeline" in checks:
            violations.extend(_timeline_violations(state))
        if "transforms" in checks or "rebuildability" in checks:
            violations.extend(_transform_violations(state, self.profile))
        if "camera" in checks or "rebuildability" in checks:
            violations.extend(_camera_violations(state))
        if "projection" in checks:
            violations.extend(_projection_violations(state, self.profile))
        if "motion" in checks:
            violations.extend(
                _orbit_trajectory_violations(
                    state,
                    self.objective_brief,
                )
            )
            violations.extend(
                _motion_readability_violations(
                    state,
                    self.objective_brief,
                    self.profile,
                )
            )
            violations.extend(
                _orbit_projection_readability_violations(
                    state,
                    self.objective_brief,
                    self.profile,
                )
            )
        if "hard_semantics" in checks:
            violations.extend(_hard_semantic_violations(state, self.objective_brief))

        soft_total = 0.0
        soft_passed = 0.0
        for constraint in state.constraints.values():
            if constraint.type not in SUPPORTED_CONSTRAINTS:
                gaps.append(f"constraint:{constraint.type}")
                continue
            result = _constraint_violation(state, constraint, self.profile)
            if constraint.strength == "soft":
                soft_total += constraint.weight
                if result is None:
                    soft_passed += constraint.weight
            if result is not None:
                violations.append(result)
        soft_score = soft_passed / soft_total if soft_total else 1.0
        hard_pass = not any(item.severity == "hard" for item in violations) and not gaps
        return ValidationReport(
            revision=state.revision,
            hard_pass=hard_pass,
            soft_score=soft_score,
            checks=checks,
            violations=violations,
            capability_gaps=sorted(set(gaps)),
        )


def _initial_candidate(
    brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
    scene_id: str | None,
) -> CandidateState:
    resolution = DurationResolution.model_validate(brief.timeline.get("duration_resolution"))
    frame_count = resolution.frame_count
    suffix = brief.source_brief_sha256.removeprefix("sha256:")[:12]
    return CandidateState(
        scene_id=scene_id or f"scene_{suffix}",
        timeline=TimelineSpec(
            fps_numerator=profile.fps_numerator,
            fps_denominator=profile.fps_denominator,
            frame_count=frame_count,
            duration_seconds=resolution.resolved_duration_seconds,
            duration_resolution=resolution,
        ),
        required_source_refs=[item.path for item in brief.explicit_requirements],
        runner_mapped_source_refs=[
            item.path
            for item in brief.explicit_requirements
            if item.path in {
                "content.timeline.duration_seconds",
                "content.timeline.duration_range_seconds",
            }
        ],
    )


def _validate_initial_candidate(actual: CandidateState, expected: CandidateState) -> None:
    if actual.schema_version != expected.schema_version:
        raise ValueError("Checkpoint Candidate Schema 版本不兼容")
    if actual.timeline != expected.timeline:
        raise ValueError("Checkpoint timeline 与当前 Brief/Profile 不一致")
    if actual.required_source_refs != expected.required_source_refs:
        raise ValueError("Checkpoint explicit requirement 索引与当前 Brief 不一致")
    if actual.runner_mapped_source_refs != expected.runner_mapped_source_refs:
        raise ValueError("Checkpoint Runner 映射与当前 Brief 不一致")


def _referenced_entity_ids(state: CandidateState) -> set[str]:
    result = {entity.parent_id for entity in state.entities.values() if entity.parent_id}
    result.update(
        entity.ground_interaction.ground_entity_id
        for entity in state.entities.values()
        if entity.ground_interaction.ground_entity_id
    )
    result.update(
        track.target_entity_id for track in state.motion_tracks.values() if track.target_entity_id
    )
    for entity in state.entities.values():
        if entity.solved_transform.target_id:
            result.add(entity.solved_transform.target_id)
    for track in list(state.motion_tracks.values()) + (
        list(state.camera.tracks.values()) if state.camera else []
    ):
        if track.path and track.path.target_id:
            result.add(track.path.target_id)
        for keyframe in track.keyframes:
            if isinstance(keyframe.value, TransformValue) and keyframe.value.target_id:
                result.add(keyframe.value.target_id)
    for constraint in state.constraints.values():
        result.update(constraint.subjects)
        for key, value in _parameters(constraint).items():
            if key.endswith("_id") and isinstance(value, str) and value != "camera_main":
                result.add(value)
    return result


def _validate_parent_references(state: CandidateState) -> None:
    for entity in state.entities.values():
        if entity.parent_id and entity.parent_id not in state.entities:
            raise ValueError(f"Entity parent 不存在：{entity.parent_id}")
    for entity_id in state.entities:
        seen: set[str] = set()
        current: str | None = entity_id
        while current:
            if current in seen:
                raise ValueError("Entity 父子层级存在循环")
            seen.add(current)
            parent = state.entities.get(current)
            current = parent.parent_id if parent else None


def _validate_ground_interaction_references(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
) -> None:
    explicit_refs = {item.path for item in objective_brief.explicit_requirements}
    exception_modes = {"may_intersect", "embedded", "unconstrained"}
    for entity in state.entities.values():
        interaction = entity.ground_interaction
        if interaction.ground_entity_id and interaction.ground_entity_id not in state.entities:
            raise ValueError(
                f"Entity ground_entity_id 不存在：{entity.entity_id} -> "
                f"{interaction.ground_entity_id}"
            )
        if interaction.ground_entity_id:
            ground = state.entities[interaction.ground_entity_id]
            if ground.proxy.type != "plane" or not _is_environment_entity(ground):
                raise ValueError(
                    f"ground_entity_id 必须指向环境地面平面：{entity.entity_id} -> "
                    f"{interaction.ground_entity_id}"
                )
        if interaction.mode in exception_modes and (
            interaction.source_status != "explicit"
            or interaction.source_ref not in explicit_refs
        ):
            raise ValueError(
                f"{interaction.mode} 只能来自 Brief 的明确地面交互要求：{entity.entity_id}"
            )


def _validate_constraint_time(constraint: ConstraintSpec, duration: float) -> None:
    if constraint.time_range_seconds[1] > duration + 1e-9:
        raise ValueError(f"Constraint 超出 timeline：{constraint.constraint_id}")


def _validate_track_time(track: TrackSpec, duration: float) -> None:
    if track.time_range_seconds[1] > duration + 1e-9:
        raise ValueError(f"Track 超出 timeline：{track.track_id}")
    for keyframe in track.keyframes:
        if not 0 <= keyframe.time_seconds < duration:
            raise ValueError(f"Track keyframe 超出半开 timeline：{track.track_id}")


def _complete_transform(value: TransformValue) -> TransformValue:
    return TransformValue(
        translation_m=value.translation_m or (0.0, 0.0, 0.0),
        rotation_quaternion_wxyz=value.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
        scale=value.scale or (1.0, 1.0, 1.0),
        space="world",
    )


def _geometry_half_height(geometry) -> float:
    if geometry.type == "box":
        return geometry.size_xyz_m[2] / 2.0
    if geometry.type == "sphere":
        return geometry.radius_m
    if geometry.type == "capsule":
        return geometry.radius_m + geometry.segment_length_m / 2.0
    if geometry.type in {"cylinder", "cone"}:
        return geometry.depth_m / 2.0
    return 0.01


def _proxy_topology(geometry) -> tuple[str, str | None]:
    # 尺寸可调整，但基本体类型和自身轴向一经建立即保持稳定。
    return geometry.type, getattr(geometry, "axis", None)


def _apply_layout_constraint(
    state: CandidateState,
    constraint: ConstraintSpec,
    profile: PlanningProfile,
) -> bool:
    params = _parameters(constraint)
    if constraint.type == "depth_order":
        near_id = params.get("near_entity_id")
        far_id = params.get("far_entity_id")
        if near_id not in state.entities or far_id not in state.entities:
            return False
        near = _complete_transform(state.entities[near_id].solved_transform)
        far = _complete_transform(state.entities[far_id].solved_transform)
        gap = float(params.get("minimum_depth_gap_meters") or profile.default_depth_gap_m)
        volume_margin = (
            geometry_bounding_radius(state.entities[near_id].proxy)
            + geometry_bounding_radius(state.entities[far_id].proxy)
        )
        # 使用体积余量，避免只满足中心点关系却让巨型代理穿过摄影机。
        state.entities[far_id].solved_transform = far.model_copy(
            update={
                "translation_m": (
                    far.translation_m[0],
                    near.translation_m[1] + gap + volume_margin,
                    far.translation_m[2],
                )
            }
        )
        return True
    if constraint.type == "relative_position":
        subject_id = params.get("subject_id")
        reference_id = params.get("reference_id")
        if subject_id not in state.entities or reference_id not in state.entities:
            return False
        subject = _complete_transform(state.entities[subject_id].solved_transform)
        reference = _complete_transform(state.entities[reference_id].solved_transform)
        gap = float(params.get("minimum_gap") or 2.0)
        position = list(subject.translation_m)
        relation = params.get("relation")
        axis_sign = {
            "left": (0, -1), "right": (0, 1),
            "front": (1, -1), "behind": (1, 1),
            "below": (2, -1), "above": (2, 1),
        }.get(relation)
        if axis_sign is None:
            return False
        axis, sign = axis_sign
        position[axis] = reference.translation_m[axis] + sign * gap
        state.entities[subject_id].solved_transform = subject.model_copy(
            update={"translation_m": tuple(position)}
        )
        return True
    if constraint.type == "distance_range":
        ids = params.get("entity_ids") or constraint.subjects
        if not isinstance(ids, (list, tuple)) or len(ids) != 2 or any(item not in state.entities for item in ids):
            return False
        first = _complete_transform(state.entities[ids[0]].solved_transform)
        second = _complete_transform(state.entities[ids[1]].solved_transform)
        minimum = float(params.get("minimum_meters", 0.0))
        maximum = float(params.get("maximum_meters", max(minimum, profile.default_depth_gap_m)))
        distance = (minimum + maximum) / 2.0
        state.entities[ids[1]].solved_transform = second.model_copy(
            update={"translation_m": (first.translation_m[0], first.translation_m[1] + distance, second.translation_m[2])}
        )
        return True
    return False


def _reference_violations(
    state: CandidateState,
    profile: PlanningProfile,
) -> list[Violation]:
    violations: list[Violation] = []
    try:
        _validate_parent_references(state)
    except ValueError as error:
        violations.append(_violation("HIERARCHY_INVALID", str(error)))
    try:
        _validate_reference_frame_graph(state, profile)
    except ValueError as error:
        violations.append(_violation("REFERENCE_FRAME_INVALID", str(error)))
    for track in state.motion_tracks.values():
        if track.target_entity_id not in state.entities:
            violations.append(_violation("TRACK_TARGET_MISSING", f"Track 目标不存在：{track.track_id}"))
    for (entity_id, track_type), track_ids in _duplicate_entity_track_ids(
        state.motion_tracks
    ).items():
        violations.append(
            _violation(
                "AMBIGUOUS_ENTITY_TRACKS",
                f"Entity {entity_id} 的 {track_type} 通道存在重叠轨道：{', '.join(track_ids)}",
                entity_ids=[entity_id],
                expected={"track_type": track_type, "maximum_count": 1},
                actual={"track_ids": track_ids},
                adjustable_variables=["entity tracks", "remove_ids"],
            )
        )
    for constraint in state.constraints.values():
        camera_id = state.camera.camera_id if state.camera else "camera_main"
        missing = sorted(
            item
            for item in constraint.subjects
            if item not in state.entities and item != camera_id
        )
        if missing:
            violations.append(_constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", {"missing": missing}))
    return violations


def _timeline_violations(state: CandidateState) -> list[Violation]:
    violations: list[Violation] = []
    for track in list(state.motion_tracks.values()) + (
        list(state.camera.tracks.values()) if state.camera else []
    ):
        try:
            _validate_track_time(track, state.timeline.duration_seconds)
        except ValueError as error:
            violations.append(_violation("TRACK_TIME_INVALID", str(error)))
    return violations


def _motion_readability_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
) -> list[Violation]:
    """拒绝会把嵌套公转看成刚性编队的同相轨迹。"""
    orbit_pairs = _objective_orbit_pairs(objective_brief)
    path_tracks = {
        (track.target_entity_id, track.path.target_id): track
        for track in state.motion_tracks.values()
        if track.type == "path_follow"
        and track.path is not None
        and track.path.closed
        and track.path.space == "target_relative"
        and track.target_entity_id
        and track.path.target_id
    }
    violations: list[Violation] = []
    for child_id, parent_id in sorted(orbit_pairs):
        child_track = path_tracks.get((child_id, parent_id))
        if child_track is None:
            continue
        for orbiting_id, grandparent_id in sorted(orbit_pairs):
            if orbiting_id != parent_id:
                continue
            parent_track = path_tracks.get((parent_id, grandparent_id))
            if parent_track is None:
                continue
            if _nested_orbits_are_phase_locked(
                state,
                child_id,
                parent_id,
                grandparent_id,
                child_track,
                parent_track,
                profile,
            ):
                start = max(
                    child_track.time_range_seconds[0],
                    parent_track.time_range_seconds[0],
                )
                end = min(
                    child_track.time_range_seconds[1],
                    parent_track.time_range_seconds[1],
                )
                violations.append(
                    _violation(
                        "NESTED_ORBIT_PHASE_LOCKED",
                        (
                            f"嵌套公转在画面控制上退化为同相编队：{child_id} -> "
                            f"{parent_id} -> {grandparent_id}；请调整子轨道 cycle_count 或节奏"
                        ),
                        entity_ids=[child_id, parent_id, grandparent_id],
                        time_range_seconds=(start, end),
                        expected={
                            "relative_phase": "随时间显著变化",
                            "purpose": "让控制白模中的嵌套运动可辨识",
                        },
                        actual={
                            "child_track_id": child_track.track_id,
                            "parent_track_id": parent_track.track_id,
                            "child_cycle_count": child_track.path.cycle_count,
                            "parent_cycle_count": parent_track.path.cycle_count,
                            "relative_phase": "近似恒定",
                        },
                        adjustable_variables=[
                            f"motion_tracks.{child_track.track_id}.path.cycle_count",
                            f"motion_tracks.{child_track.track_id}.path.control_points",
                        ],
                    )
                )
    return violations


def _orbit_projection_readability_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
) -> list[Violation]:
    """防止未指定机位时把解析轨道长期拍成近似直线。"""

    if state.camera is None:
        return []
    path_tracks = {
        (track.target_entity_id, track.path.target_id): track
        for track in state.motion_tracks.values()
        if track.type == "path_follow"
        and track.path is not None
        and track.path.representation in {"circle", "ellipse"}
        and track.path.space == "target_relative"
        and track.target_entity_id
        and track.path.target_id
    }
    explicit_view = _has_explicit_camera_view(objective_brief)
    violations: list[Violation] = []
    for subject_id, reference_id in sorted(_objective_orbit_pairs(objective_brief)):
        track = path_tracks.get((subject_id, reference_id))
        if track is None or track.path is None:
            continue
        start, end = track.time_range_seconds
        frame_step = state.timeline.fps_denominator / state.timeline.fps_numerator
        last_frame_time = (
            state.timeline.duration_seconds
            - frame_step
        )
        sample_end = min(end - frame_step, last_frame_time)
        if sample_end < start:
            sample_end = start
        sample_times = [
            start + (sample_end - start) * index / 16.0
            for index in range(17)
        ]
        alignments: list[float] = []
        for time_seconds in sample_times:
            resolver = _WorldTransformResolver(state, time_seconds, profile)
            camera_state = resolver.camera()
            if camera_state is None:
                continue
            reference = resolver.entity(reference_id)
            camera_transform, _ = camera_state
            scaled_center_offset = tuple(
                track.path.center_offset_m[index] * reference.scale[index]
                for index in range(3)
            )
            orbit_center = add(
                reference.translation_m,
                rotate_vector(
                    reference.rotation_quaternion_wxyz,
                    scaled_center_offset,
                ),
            )
            view_vector = subtract(
                camera_transform.translation_m,
                orbit_center,
            )
            if length(view_vector) <= profile.numeric_tolerance:
                continue
            world_normal = rotate_vector(
                reference.rotation_quaternion_wxyz,
                normalize(track.path.plane_normal),
            )
            alignments.append(abs(dot(normalize(world_normal), normalize(view_vector))))
        if not alignments:
            continue
        median_alignment = median(alignments)
        if median_alignment + profile.numeric_tolerance >= (
            profile.minimum_orbit_plane_view_alignment
        ):
            continue
        severity = "warning" if explicit_view else "hard"
        message = (
            f"解析轨道在当前摄影机下长期接近侧视：{subject_id} -> {reference_id}；"
            "请调整摄影机或轨道平面，使白模能够辨识闭合运动"
        )
        if explicit_view:
            message += "；Brief 已明确机位，因此仅记录警告而不覆盖用户要求"
        violation = _violation(
            "ORBIT_PLANE_NEAR_EDGE_ON",
            message,
            entity_ids=[subject_id, reference_id],
            time_range_seconds=(start, end),
            expected={
                "minimum_median_absolute_view_normal_dot": (
                    profile.minimum_orbit_plane_view_alignment
                ),
                "purpose": "让控制白模中的闭合轨道保持可辨识",
            },
            actual={
                "median_absolute_view_normal_dot": median_alignment,
                "minimum_absolute_view_normal_dot": min(alignments),
                "maximum_absolute_view_normal_dot": max(alignments),
                "plane_normal": list(track.path.plane_normal),
                "sample_count": len(alignments),
            },
            adjustable_variables=[
                f"motion_tracks.{track.track_id}.path.plane_normal",
                "camera transform",
            ],
        )
        violations.append(violation.model_copy(update={"severity": severity}))
    return violations


def _orbit_trajectory_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
) -> list[Violation]:
    """要求普通公转使用解析轨迹，明确的异形轨迹除外。"""
    path_tracks = {
        track.target_entity_id: track
        for track in state.motion_tracks.values()
        if track.type == "path_follow" and track.path is not None and track.target_entity_id
    }
    violations: list[Violation] = []
    for subject_id, reference_id in sorted(_objective_orbit_pairs(objective_brief)):
        track = path_tracks.get(subject_id)
        if track is None:
            violations.append(
                _violation(
                    "ORBIT_PATH_REQUIRED",
                    f"orbit_around 必须使用 path_follow：{subject_id} -> {reference_id}",
                    entity_ids=[subject_id, reference_id],
                    expected={"track_type": "path_follow"},
                    actual={"track_type": None},
                    adjustable_variables=[f"motion_tracks.{subject_id}"],
                )
            )
            continue
        path = track.path
        if path.space != "target_relative" or path.target_id != reference_id:
            violations.append(
                _violation(
                    "ORBIT_REFERENCE_FRAME_INVALID",
                    f"orbit_around 必须相对被环绕主体求值：{subject_id} -> {reference_id}",
                    entity_ids=[subject_id, reference_id],
                    expected={"space": "target_relative", "target_id": reference_id},
                    actual={"space": path.space, "target_id": path.target_id},
                    adjustable_variables=[f"motion_tracks.{track.track_id}.path"],
                )
            )
        if (
            path.representation not in {"circle", "ellipse"}
            and not _has_explicit_custom_trajectory(objective_brief, subject_id)
        ):
            violations.append(
                _violation(
                    "ORBIT_TRAJECTORY_NOT_ANALYTIC",
                    (
                        f"普通 orbit_around 不得用 {path.representation} 近似："
                        f"{subject_id} -> {reference_id}"
                    ),
                    entity_ids=[subject_id, reference_id],
                    expected={"representation": ["circle", "ellipse"]},
                    actual={"representation": path.representation},
                    adjustable_variables=[f"motion_tracks.{track.track_id}.path"],
                )
            )
    return violations


def _objective_orbit_pairs(
    objective_brief: ObjectivePlanningBrief,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in objective_brief.scene_design.get("relationships", []):
        if not isinstance(item, dict) or item.get("type") != "orbit_around":
            continue
        subject_id = item.get("subject_id")
        reference_id = item.get("reference_id")
        if isinstance(subject_id, str) and isinstance(reference_id, str):
            pairs.add((subject_id, reference_id))
    return pairs


def _has_explicit_camera_view(objective_brief: ObjectivePlanningBrief) -> bool:
    return any(
        item.path.startswith("content.camera.view_angle")
        for item in objective_brief.explicit_requirements
    )


def _has_explicit_custom_trajectory(
    objective_brief: ObjectivePlanningBrief,
    subject_id: str,
) -> bool:
    analytic_names = {"circle", "circular", "ellipse", "elliptical", "圆", "圆形", "椭圆"}
    for motion in objective_brief.subject_motion:
        if not isinstance(motion, dict) or motion.get("subject_id") != subject_id:
            continue
        trajectory = motion.get("trajectory")
        if not isinstance(trajectory, dict) or trajectory.get("source_status") != "explicit":
            continue
        value = trajectory.get("value")
        if value is None:
            continue
        normalized = str(value).strip().lower()
        return not any(name in normalized for name in analytic_names)
    return False


def _nested_orbits_are_phase_locked(
    state: CandidateState,
    child_id: str,
    parent_id: str,
    grandparent_id: str,
    child_track: TrackSpec,
    parent_track: TrackSpec,
    profile: PlanningProfile,
) -> bool:
    start = max(child_track.time_range_seconds[0], parent_track.time_range_seconds[0])
    end = min(child_track.time_range_seconds[1], parent_track.time_range_seconds[1])
    if end <= start:
        return False
    child_duration = child_track.time_range_seconds[1] - child_track.time_range_seconds[0]
    parent_duration = parent_track.time_range_seconds[1] - parent_track.time_range_seconds[0]
    child_rate = child_track.path.cycle_count / child_duration
    parent_rate = parent_track.path.cycle_count / parent_duration
    if abs(child_rate - parent_rate) > profile.numeric_tolerance:
        return False

    # 均匀探针比较两层轨道方向；点积与叉积都近似不变即为相位锁定。
    last = min(
        end - profile.numeric_tolerance,
        state.timeline.duration_seconds
        - state.timeline.fps_denominator / state.timeline.fps_numerator,
    )
    if last <= start:
        return False
    phase_samples: list[tuple[float, float, float, float]] = []
    parent_directions: list[tuple[float, float, float]] = []
    child_directions: list[tuple[float, float, float]] = []
    for index in range(17):
        time_seconds = start + (last - start) * index / 16.0
        resolver = _WorldTransformResolver(state, time_seconds, profile)
        child = resolver.entity(child_id).translation_m
        parent = resolver.entity(parent_id).translation_m
        grandparent = resolver.entity(grandparent_id).translation_m
        child_offset = subtract(child, parent)
        parent_offset = subtract(parent, grandparent)
        if (
            length(child_offset) <= profile.numeric_tolerance
            or length(parent_offset) <= profile.numeric_tolerance
        ):
            return False
        child_direction = normalize(child_offset)
        parent_direction = normalize(parent_offset)
        child_directions.append(child_direction)
        parent_directions.append(parent_direction)
        cross_value = cross(parent_direction, child_direction)
        phase_samples.append((*cross_value, dot(parent_direction, child_direction)))

    # 静止偏移不冒充轨道相位锁定；这里只处理两层都实际扫过明显角度的情况。
    if not _direction_sweeps(parent_directions) or not _direction_sweeps(child_directions):
        return False
    return all(
        max(sample[axis] for sample in phase_samples)
        - min(sample[axis] for sample in phase_samples)
        <= 0.05
        for axis in range(4)
    )


def _direction_sweeps(directions: list[tuple[float, float, float]]) -> bool:
    first = directions[0]
    return any(dot(first, item) < 0.5 for item in directions[1:])


def _transform_violations(
    state: CandidateState,
    profile: PlanningProfile,
) -> list[Violation]:
    violations: list[Violation] = []
    for entity in state.entities.values():
        transform = entity.solved_transform
        if transform.translation_m is None or transform.rotation_quaternion_wxyz is None or transform.scale is None:
            violations.append(_violation(
                "UNRESOLVED_ENTITY_TRANSFORM",
                f"Entity 仍含未求解 Transform：{entity.entity_id}",
                entity_ids=[entity.entity_id],
            ))
        if entity.parent_id:
            parent_scale = state.entities[entity.parent_id].solved_transform.scale
            if parent_scale and max(parent_scale) - min(parent_scale) > 1e-8:
                violations.append(_violation(
                    "PARENT_NONUNIFORM_SCALE_UNSUPPORTED",
                    f"父级非均匀缩放无法无损编译：{entity.parent_id}",
                    entity_ids=[entity.parent_id, entity.entity_id],
                ))
    violations.extend(_ground_penetration_violations(state, profile))
    return violations


def _ground_penetration_violations(
    state: CandidateState,
    profile: PlanningProfile,
) -> list[Violation]:
    ground_ids = [
        entity.entity_id
        for entity in state.entities.values()
        if entity.proxy.type == "plane" and _is_environment_entity(entity)
    ]
    if not ground_ids:
        return []
    violations: list[Violation] = []
    for time_seconds in _timeline_probe_times(state.timeline):
        resolver = _WorldTransformResolver(state, time_seconds, profile)
        ground_levels: dict[str, float] = {}
        for ground_id in ground_ids:
            transform = resolver.entity(ground_id)
            normal = rotate_vector(transform.rotation_quaternion_wxyz, (0.0, 0.0, 1.0))
            if abs(normal[2]) >= 0.999:
                ground_levels[ground_id] = transform.translation_m[2]
        if not ground_levels:
            continue
        for entity in state.entities.values():
            if entity.entity_id in ground_ids:
                continue
            interaction = entity.ground_interaction
            if interaction.mode == "unconstrained":
                continue
            if interaction.ground_entity_id:
                ground_z = ground_levels.get(interaction.ground_entity_id)
                if ground_z is None:
                    continue
                active_ground_id = interaction.ground_entity_id
            else:
                active_ground_id, ground_z = max(
                    ground_levels.items(),
                    key=lambda item: item[1],
                )
            transform = resolver.entity(entity.entity_id)
            world_z_values: list[float] = []
            for point in geometry_local_bounds_points(entity.proxy):
                scaled = tuple(
                    point[index] * transform.scale[index]
                    for index in range(3)
                )
                rotated = rotate_vector(transform.rotation_quaternion_wxyz, scaled)
                world_z_values.append(transform.translation_m[2] + rotated[2])
            maximum_z = max(world_z_values)
            minimum_z = min(world_z_values)
            penetration_m = max(0.0, ground_z - minimum_z)
            clearance_m = minimum_z - ground_z
            tolerance_m = interaction.tolerance_m
            if maximum_z < ground_z - tolerance_m:
                violations.append(
                    _violation(
                        "ENTITY_FULLY_BELOW_GROUND",
                        f"Entity 整体位于水平地面以下：{entity.entity_id}",
                        entity_ids=[entity.entity_id],
                        time_range_seconds=(time_seconds, time_seconds),
                        expected={"surface_crossing_or_above_z": ground_z},
                        actual={
                            "maximum_z": maximum_z,
                            "ground_z": ground_z,
                            "ground_entity_id": active_ground_id,
                            "mode": interaction.mode,
                        },
                        adjustable_variables=["entity transforms", "ground_interaction"],
                    )
                )
                continue
            violation = _ground_interaction_violation(
                entity,
                interaction.mode,
                penetration_m,
                clearance_m,
                ground_z,
                active_ground_id,
                time_seconds,
            )
            if violation is not None:
                violations.append(violation)
    return violations


def _ground_interaction_violation(
    entity: EntitySpec,
    mode: str,
    penetration_m: float,
    clearance_m: float,
    ground_z: float,
    ground_entity_id: str,
    time_seconds: float,
) -> Violation | None:
    interaction = entity.ground_interaction
    tolerance_m = interaction.tolerance_m
    if mode == "must_be_above" and penetration_m > tolerance_m:
        code = "ENTITY_INTERSECTS_GROUND"
        expected = {"maximum_penetration_m": tolerance_m}
    elif mode == "must_touch" and (
        penetration_m > tolerance_m or clearance_m > tolerance_m
    ):
        code = "ENTITY_GROUND_CONTACT_VIOLATED"
        expected = {"absolute_surface_distance_m": tolerance_m}
    elif (
        mode == "may_intersect"
        and interaction.maximum_penetration_m is not None
        and penetration_m > interaction.maximum_penetration_m + tolerance_m
    ):
        code = "ENTITY_GROUND_PENETRATION_EXCEEDED"
        expected = {"maximum_penetration_m": interaction.maximum_penetration_m}
    elif (
        mode == "embedded"
        and interaction.minimum_penetration_m is not None
        and interaction.maximum_penetration_m is not None
        and not (
            interaction.minimum_penetration_m - tolerance_m
            <= penetration_m
            <= interaction.maximum_penetration_m + tolerance_m
        )
    ):
        code = "ENTITY_EMBEDDING_RANGE_VIOLATED"
        expected = {
            "minimum_penetration_m": interaction.minimum_penetration_m,
            "maximum_penetration_m": interaction.maximum_penetration_m,
        }
    else:
        return None
    return _violation(
        code,
        f"Entity 地面交互不符合 {mode}：{entity.entity_id}",
        entity_ids=[entity.entity_id, ground_entity_id],
        time_range_seconds=(time_seconds, time_seconds),
        expected=expected,
        actual={
            "penetration_m": penetration_m,
            "clearance_m": clearance_m,
            "ground_z": ground_z,
            "ground_entity_id": ground_entity_id,
            "mode": mode,
        },
        adjustable_variables=["entity transforms", "ground_interaction"],
    )


def _camera_violations(state: CandidateState) -> list[Violation]:
    if state.camera is None:
        return [_violation("CAMERA_MISSING", "缺少活动透视摄影机")]
    transform = state.camera.solved_transform
    violations: list[Violation] = []
    if transform.translation_m is None or transform.rotation_quaternion_wxyz is None:
        violations.append(_violation("UNRESOLVED_CAMERA_TRANSFORM", "摄影机 Transform 尚未求解"))
    if state.camera.static.focal_length_mm is None:
        violations.append(_violation("UNRESOLVED_FOCAL_LENGTH", "摄影机焦距尚未求解"))
    if state.camera.static.focus_target_id and state.camera.static.focus_target_id not in state.entities:
        violations.append(_violation("CAMERA_TARGET_MISSING", "摄影机观察目标不存在"))
    for track_type, track_ids in _duplicate_camera_track_ids(state.camera.tracks).items():
        violations.append(
            _violation(
                "AMBIGUOUS_CAMERA_TRACKS",
                f"摄影机 {track_type} 通道存在重叠轨道：{', '.join(track_ids)}",
                expected={"track_type": track_type, "maximum_count": 1},
                actual={"track_ids": track_ids},
                adjustable_variables=["camera tracks", "remove_track_ids"],
            )
        )
    return violations


def _hard_semantic_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
) -> list[Violation]:
    locations: dict[str, set[str]] = {}
    for source_ref in state.runner_mapped_source_refs:
        locations.setdefault(source_ref, set()).add("runner")
    for entity in state.entities.values():
        for source_ref in entity.source_refs:
            locations.setdefault(source_ref, set()).add("entity")
    for track in state.motion_tracks.values():
        if track.source_ref:
            locations.setdefault(track.source_ref, set()).add("entity_track")
    for constraint in state.constraints.values():
        locations.setdefault(constraint.source_ref, set()).add(f"constraint:{constraint.type}")
    if state.camera:
        for source_ref in state.camera.static.source_refs:
            locations.setdefault(source_ref, set()).add("camera_static")
        for track in state.camera.tracks.values():
            if track.source_ref:
                locations.setdefault(track.source_ref, set()).add("camera_track")
    violations: list[Violation] = []
    for source_ref in state.required_source_refs:
        actual = locations.get(source_ref, set())
        if not actual:
            violations.append(
                _violation(
                    "UNMAPPED_EXPLICIT_REQUIREMENT",
                    f"明确客观要求未映射：{source_ref}",
                    expected={"source_ref": source_ref},
                )
            )
        elif not _source_mapping_is_compatible(source_ref, actual):
            violations.append(
                _violation(
                    "EXPLICIT_REQUIREMENT_MAPPING_INCOMPATIBLE",
                    f"明确客观要求被挂到无关字段：{source_ref}",
                    expected={
                        "source_ref": source_ref,
                        "compatible_locations": sorted(_compatible_locations(source_ref)),
                    },
                    actual={"locations": sorted(actual)},
                )
            )
        required_constraint_types = _required_constraint_types(
            objective_brief,
            source_ref,
        )
        missing_constraint_types = sorted(
            constraint_type
            for constraint_type in required_constraint_types
            if f"constraint:{constraint_type}" not in actual
        )
        if missing_constraint_types:
            violations.append(
                _violation(
                    "EXPLICIT_REQUIREMENT_CONSTRAINT_INCOMPLETE",
                    f"明确要求缺少必要约束类型：{source_ref}",
                    expected={"constraint_types": sorted(required_constraint_types)},
                    actual={"locations": sorted(actual)},
                    adjustable_variables=["constraints"],
                )
            )
    return violations


def _required_constraint_types(
    objective_brief: ObjectivePlanningBrief,
    source_ref: str,
) -> set[str]:
    prefix = "content.scene_design.relationships["
    if not source_ref.startswith(prefix):
        return set()
    try:
        index = int(source_ref[len(prefix):].split("]", 1)[0])
        relationship = objective_brief.scene_design.get("relationships", [])[index]
    except (IndexError, TypeError, ValueError):
        return set()
    relation = str(relationship.get("type", "")).lower()
    if any(marker in relation for marker in ("远", "far", "background", "远景", "后景")):
        # 欧氏距离不足以表达画面中的“远处”，还必须证明其摄影机深度在主体之后。
        return {"depth_order"}
    return set()


def _source_mapping_is_compatible(source_ref: str, actual: set[str]) -> bool:
    allowed = _compatible_locations(source_ref)
    return any(
        location in allowed or any(
            prefix.endswith(":*") and location.startswith(prefix[:-1])
            for prefix in allowed
        )
        for location in actual
    )


def _compatible_locations(source_ref: str) -> set[str]:
    if source_ref == "content.timeline.duration_seconds":
        return {"runner"}
    if source_ref.startswith("content.subjects["):
        return {"entity"}
    if source_ref.startswith("content.subject_motion["):
        return {"entity_track", "constraint:*"}
    if source_ref.startswith("content.scene_design.relationships["):
        return {"constraint:*", "entity_track"}
    if source_ref.startswith("content.scene_design.environment"):
        return {"entity"}
    if source_ref.startswith("content.composition"):
        return {"constraint:*", "camera_static", "camera_track"}
    if source_ref == "content.camera.movement.speed":
        return {"camera_track", "constraint:speed_range"}
    if source_ref == "content.camera.movement.type":
        return {"camera_track", "constraint:camera_motion_direction"}
    if source_ref.startswith("content.camera"):
        return {"camera_static", "camera_track", "constraint:*"}
    return {"entity", "entity_track", "constraint:*", "camera_static", "camera_track", "runner"}


def _projection_violations(
    state: CandidateState,
    profile: PlanningProfile,
) -> list[Violation]:
    if state.camera is None:
        return []
    violations: list[Violation] = []
    for time_seconds in _timeline_probe_times(state.timeline):
        resolver = _WorldTransformResolver(state, time_seconds, profile)
        camera = resolver.camera()
        if camera is None:
            continue
        camera_transform, focal_length = camera
        transforms = {
            entity_id: resolver.entity(entity_id)
            for entity_id in state.entities
        }
        for entity_id, entity in state.entities.items():
            if entity.proxy.type == "plane" or _is_environment_entity(entity):
                continue
            bounds = _projection_bounds(
                state,
                entity_id,
                transforms,
                camera_transform,
                focal_length,
                profile,
            )
            if bounds[4] <= 0 or not all(math.isfinite(item) for item in bounds):
                violations.append(
                    _violation(
                        "ENTITY_NOT_PROJECTABLE",
                        f"Entity 在摄影机后方或跨越摄影机平面：{entity_id}",
                        entity_ids=[entity_id],
                        time_range_seconds=(time_seconds, min(time_seconds + 1e-6, state.timeline.duration_seconds)),
                        actual={"projected_bounds": bounds},
                    )
                )
                break
    return violations


def _constraint_violation(
    state: CandidateState,
    constraint: ConstraintSpec,
    profile: PlanningProfile,
) -> Violation | None:
    params = _parameters(constraint)
    sample_times = _constraint_sample_times(constraint, state.timeline)
    for time_seconds in sample_times:
        resolver = _WorldTransformResolver(state, time_seconds, profile)
        transforms = {
            entity_id: resolver.entity(entity_id)
            for entity_id in state.entities
        }
        if constraint.type == "hold":
            target_id = params.get("target_id")
            if target_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            start_time = constraint.time_range_seconds[0]
            baseline = _WorldTransformResolver(state, start_time, profile).entity(target_id)
            actual = transforms[target_id]
            components = set(params.get("components") or [])
            differences: dict[str, Any] = {}
            if "translation" in components:
                translation_delta = length(
                    subtract(actual.translation_m, baseline.translation_m)
                )
                if translation_delta > float(params.get("tolerance_m", 1e-4)):
                    differences["translation_delta_m"] = translation_delta
            if "rotation" in components:
                rotation_delta = _quaternion_angle_degrees(
                    actual.rotation_quaternion_wxyz,
                    baseline.rotation_quaternion_wxyz,
                )
                if rotation_delta > float(
                    params.get("rotation_tolerance_degrees", 0.01)
                ):
                    differences["rotation_delta_degrees"] = rotation_delta
            if "scale" in components:
                scale_delta = max(
                    abs(current - initial)
                    for current, initial in zip(actual.scale, baseline.scale)
                )
                if scale_delta > float(params.get("scale_tolerance", 1e-4)):
                    differences["scale_delta"] = scale_delta
            if "visibility" in components:
                initial_visibility = _entity_visibility_at(state, target_id, start_time)
                current_visibility = _entity_visibility_at(state, target_id, time_seconds)
                if current_visibility != initial_visibility:
                    differences["visibility"] = {
                        "expected": initial_visibility,
                        "actual": current_visibility,
                    }
            if differences:
                return _constraint_error(constraint, "HOLD_VIOLATED", differences)
            continue
        camera = resolver.camera()
        if camera is None:
            return _constraint_error(constraint, "CAMERA_MISSING", None)
        camera_transform, focal_length = camera

        if constraint.type == "depth_order":
            near_id = params.get("near_entity_id")
            far_id = params.get("far_entity_id")
            if near_id not in transforms or far_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            near_depth = _projection(state, near_id, transforms, camera_transform, focal_length, profile)[2]
            far_depth = _projection(state, far_id, transforms, camera_transform, focal_length, profile)[2]
            gap = float(params.get("minimum_depth_gap_meters") or 0.0)
            if near_depth + gap > far_depth + profile.numeric_tolerance:
                return _constraint_error(constraint, "DEPTH_ORDER_VIOLATED", {"near_depth": near_depth, "far_depth": far_depth})

        elif constraint.type == "relative_position":
            subject_id = params.get("subject_id")
            reference_id = params.get("reference_id")
            if subject_id not in transforms or reference_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            subject = transforms[subject_id].translation_m
            reference = transforms[reference_id].translation_m
            delta = subtract(subject, reference)
            relation = params.get("relation")
            signed_gap = {
                "left": -delta[0],
                "right": delta[0],
                "front": -delta[1],
                "behind": delta[1],
                "below": -delta[2],
                "above": delta[2],
            }.get(relation, -math.inf)
            if not _within_range(
                signed_gap,
                float(params.get("minimum_gap") or 0.0),
                float(params.get("maximum_gap") or math.inf),
                profile.numeric_tolerance,
            ):
                return _constraint_error(
                    constraint,
                    "RELATIVE_POSITION_VIOLATED",
                    {
                        "subject": subject,
                        "reference": reference,
                        "signed_gap_m": signed_gap,
                        "evaluated_space": "world",
                    },
                )

        elif constraint.type == "distance_range":
            ids = params.get("entity_ids") or constraint.subjects
            if not isinstance(ids, (list, tuple)) or len(ids) != 2 or any(item not in transforms for item in ids):
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            actual = length(subtract(transforms[ids[0]].translation_m, transforms[ids[1]].translation_m))
            if not _within_range(
                actual,
                float(params.get("minimum_meters", 0.0)),
                float(params.get("maximum_meters", math.inf)),
                profile.numeric_tolerance,
            ):
                return _constraint_error(constraint, "DISTANCE_RANGE_VIOLATED", {"distance_m": actual})

        elif constraint.type in {"screen_region", "projected_size", "keep_in_frame"}:
            entity_id = params.get("entity_id")
            if entity_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            bounds = _projection_bounds(
                state,
                entity_id,
                transforms,
                camera_transform,
                focal_length,
                profile,
            )
            left, top, right, bottom, depth = bounds
            if depth <= 0 or not all(math.isfinite(item) for item in bounds):
                return _constraint_error(
                    constraint,
                    "ENTITY_NOT_PROJECTABLE",
                    {"projected_bounds": bounds},
                )
            x = (left + right) / 2.0
            y = (top + bottom) / 2.0
            if constraint.type == "screen_region":
                region = params.get("region")
                tolerance = profile.numeric_tolerance
                if not _valid_region(region) or not (
                    region[0] - tolerance <= x <= region[2] + tolerance
                    and region[1] - tolerance <= y <= region[3] + tolerance
                ):
                    return _constraint_error(constraint, "SCREEN_REGION_VIOLATED", {"screen_center": [x, y]})
            elif constraint.type == "projected_size":
                width = right - left
                height = bottom - top
                measurement = params.get("measurement", "height")
                size = {"width": width, "height": height, "diameter": max(width, height)}[measurement]
                if not _within_range(
                    size,
                    float(params.get("minimum", 0.0)),
                    float(params.get("maximum", math.inf)),
                    profile.numeric_tolerance,
                ):
                    return _constraint_error(constraint, "PROJECTED_SIZE_VIOLATED", {"projected_size": size})
            else:
                inside = _bounds_inside_fraction(bounds)
                minimum = float(params.get("minimum_inside_fraction", 1.0))
                if inside + profile.numeric_tolerance < minimum:
                    return _constraint_error(
                        constraint,
                        "ENTITY_OUT_OF_FRAME",
                        {"inside_fraction": inside, "projected_bounds": bounds[:4]},
                    )

        elif constraint.type == "projected_scale_ratio":
            first_id = params.get("numerator_entity_id") or params.get("subject_id")
            second_id = params.get("denominator_entity_id") or params.get("reference_id")
            if first_id not in transforms or second_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            first_bounds = _projection_bounds(
                state, first_id, transforms, camera_transform, focal_length, profile
            )
            second_bounds = _projection_bounds(
                state, second_id, transforms, camera_transform, focal_length, profile
            )
            measurement = params.get("measurement", "height")
            first_size = _projected_measurement(first_bounds, measurement)
            second_size = _projected_measurement(second_bounds, measurement)
            ratio = first_size / second_size if second_size > 0 else math.inf
            if not _within_range(
                ratio,
                float(params.get("minimum_ratio", 0.0)),
                float(params.get("maximum_ratio", math.inf)),
                profile.numeric_tolerance,
            ):
                return _constraint_error(constraint, "PROJECTED_SCALE_RATIO_VIOLATED", {"ratio": ratio})

        elif constraint.type == "camera_distance":
            target_id = params.get("target_id")
            if target_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            actual = length(subtract(camera_transform.translation_m, transforms[target_id].translation_m))
            if not _within_range(
                actual,
                float(params.get("minimum_meters", 0.0)),
                float(params.get("maximum_meters", math.inf)),
                profile.numeric_tolerance,
            ):
                return _constraint_error(constraint, "CAMERA_DISTANCE_VIOLATED", {"distance_m": actual})

        elif constraint.type == "focal_length_range":
            if not _within_range(
                focal_length,
                float(params.get("minimum_mm", 0.0)),
                float(params.get("maximum_mm", math.inf)),
                profile.numeric_tolerance,
            ):
                return _constraint_error(constraint, "FOCAL_LENGTH_RANGE_VIOLATED", {"focal_length_mm": focal_length})

        elif constraint.type == "look_at":
            observer_id = params.get("observer_id")
            target_id = params.get("target_id")
            if target_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            target_position = transforms[target_id].translation_m
            if observer_id == state.camera.camera_id:
                observer_position = camera_transform.translation_m
                forward = rotate_vector(camera_transform.rotation_quaternion_wxyz, (0.0, 0.0, -1.0))
            elif observer_id in transforms:
                observer_position = transforms[observer_id].translation_m
                forward = rotate_vector(transforms[observer_id].rotation_quaternion_wxyz, (0.0, 1.0, 0.0))
            else:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            desired = normalize(subtract(target_position, observer_position))
            cosine = max(-1.0, min(1.0, dot(normalize(forward), desired)))
            angle = math.degrees(math.acos(cosine))
            if angle > float(params.get("maximum_angle_error_degrees", 1.0)):
                return _constraint_error(constraint, "LOOK_AT_VIOLATED", {"angle_error_degrees": angle})

        elif constraint.type == "position_at_time":
            target_id = params.get("target_id")
            expected = params.get("position_m")
            if not isinstance(expected, (list, tuple)) or len(expected) != 3:
                return _constraint_error(constraint, "CONSTRAINT_PARAMETER_INVALID", None)
            tolerance = float(params.get("tolerance_m", 0.01))
            if target_id == state.camera.camera_id:
                actual = camera_transform.translation_m
            elif target_id in transforms:
                actual = transforms[target_id].translation_m
            else:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            if length(subtract(actual, tuple(expected))) > tolerance:
                return _constraint_error(constraint, "POSITION_AT_TIME_VIOLATED", {"position_m": actual})

    if constraint.type in {"camera_motion_direction", "motion_direction", "speed_range"}:
        start, end = constraint.time_range_seconds
        end_sample = min(end - 1e-6, state.timeline.duration_seconds - 1e-6)
        if constraint.type == "camera_motion_direction":
            start_state = _camera_state_at(state, start, profile)
            end_state = _camera_state_at(state, end_sample, profile)
            if start_state is None or end_state is None:
                return _constraint_error(constraint, "CAMERA_MISSING", None)
            direction = params.get("direction")
            minimum = float(params.get("minimum_displacement_m", 0.01))
            if direction in {"push_in", "pull_out"}:
                target_id = params.get("target_id") or state.camera.static.focus_target_id
                if target_id not in state.entities:
                    return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
                start_target = _entity_transform_at(
                    state, target_id, start, profile
                ).translation_m
                end_target = _entity_transform_at(
                    state, target_id, end_sample, profile
                ).translation_m
                start_distance = length(subtract(start_state[0].translation_m, start_target))
                end_distance = length(subtract(end_state[0].translation_m, end_target))
                progress = (
                    start_distance - end_distance
                    if direction == "push_in"
                    else end_distance - start_distance
                )
                if progress + profile.numeric_tolerance < minimum:
                    return _constraint_error(
                        constraint,
                        "CAMERA_TARGET_DISTANCE_DIRECTION_VIOLATED",
                        {
                            "target_id": target_id,
                            "start_distance_m": start_distance,
                            "end_distance_m": end_distance,
                            "distance_change_m": progress,
                        },
                    )
                return None
            delta = subtract(end_state[0].translation_m, start_state[0].translation_m)
            if params.get("space") == "camera":
                delta = rotate_vector(
                    quaternion_conjugate(start_state[0].rotation_quaternion_wxyz),
                    delta,
                )
                space = "camera"
            else:
                space = "world"
            if not _direction_matches(
                delta,
                direction,
                minimum,
                profile.numeric_tolerance,
                space=space,
            ):
                return _constraint_error(
                    constraint,
                    "MOTION_DIRECTION_VIOLATED",
                    {"delta_m": delta, "evaluated_space": space},
                )
            return None
        if constraint.type == "speed_range":
            target_id = params.get("target_id")
            if target_id == "camera_main":
                start_state = _camera_state_at(state, start, profile)
                end_state = _camera_state_at(state, end_sample, profile)
                if start_state is None or end_state is None:
                    return _constraint_error(constraint, "CAMERA_MISSING", None)
                delta = subtract(end_state[0].translation_m, start_state[0].translation_m)
            elif target_id in state.entities:
                start_transform = _entity_transform_at(state, target_id, start, profile)
                end_transform = _entity_transform_at(
                    state, target_id, end_sample, profile
                )
                delta = subtract(end_transform.translation_m, start_transform.translation_m)
            else:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            elapsed = end_sample - start
            speed = length(delta) / elapsed if elapsed > 0 else math.inf
            if not _within_range(
                speed,
                float(params.get("minimum_mps", 0.0)),
                float(params.get("maximum_mps", math.inf)),
                profile.numeric_tolerance,
            ):
                return _constraint_error(
                    constraint,
                    "SPEED_RANGE_VIOLATED",
                    {"average_speed_mps": speed, "elapsed_seconds": elapsed},
                )
            return None
        else:
            target_id = params.get("target_id")
            if target_id not in state.entities:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            start_transform = _entity_transform_at(state, target_id, start, profile)
            end_transform = _entity_transform_at(
                state, target_id, end_sample, profile
            )
            delta = subtract(end_transform.translation_m, start_transform.translation_m)
        direction = params.get("direction")
        minimum = float(params.get("minimum_displacement_m", 0.01))
        if not _direction_matches(
            delta,
            direction,
            minimum,
            profile.numeric_tolerance,
            space=params.get("space", "world"),
        ):
            return _constraint_error(constraint, "MOTION_DIRECTION_VIOLATED", {"delta_m": delta})
    return None


def _constraint_sample_times(constraint: ConstraintSpec, timeline: TimelineSpec) -> list[float]:
    start, end = constraint.time_range_seconds
    if constraint.type in {"distance_range", "hold"}:
        # 持续语义逐帧检查，防止只在稀疏探针或首尾碰巧通过。
        frame_step = timeline.fps_denominator / timeline.fps_numerator
        frame_times = [
            frame * frame_step
            for frame in range(timeline.frame_count)
            if start <= frame * frame_step < end
        ]
        if frame_times:
            return frame_times
    return [start, min((start + end) / 2.0, timeline.duration_seconds - 1e-6), min(end - 1e-6, timeline.duration_seconds - 1e-6)]


def _entity_visibility_at(
    state: CandidateState,
    entity_id: str,
    time_seconds: float,
) -> bool:
    track = next(
        (
            item
            for item in state.motion_tracks.values()
            if item.target_entity_id == entity_id and item.type == "visibility"
        ),
        None,
    )
    return bool(sample_scalar_track(track, time_seconds, 1.0))


def _quaternion_angle_degrees(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    # q 与 -q 表示同一旋转，因此使用内积绝对值。
    product = sum(left * right for left, right in zip(first, second))
    cosine_half_angle = max(-1.0, min(1.0, abs(product)))
    return math.degrees(2.0 * math.acos(cosine_half_angle))


def _validate_reference_frame_graph(
    state: CandidateState,
    profile: PlanningProfile,
) -> None:
    resolver = _WorldTransformResolver(state, 0.0, profile)
    for entity_id in sorted(state.entities):
        resolver.entity(entity_id)
    if state.camera is not None:
        resolver.camera()


class _WorldTransformResolver:
    """递归求值实体与摄影机参考系，并在同一帧缓存结果。"""

    def __init__(
        self,
        state: CandidateState,
        time_seconds: float,
        profile: PlanningProfile,
    ) -> None:
        self.state = state
        self.time_seconds = time_seconds
        self.profile = profile
        self._entities: dict[str, TransformValue] = {}
        self._camera: tuple[TransformValue, float] | None = None
        self._resolving: list[str] = []

    def entity(self, entity_id: str) -> TransformValue:
        if entity_id in self._entities:
            return self._entities[entity_id]
        if entity_id not in self.state.entities:
            raise ValueError(f"参考系目标 Entity 不存在：{entity_id}")
        token = f"entity:{entity_id}"
        self._enter(token)
        try:
            entity = self.state.entities[entity_id]
            tracks = [
                track
                for track in self.state.motion_tracks.values()
                if track.target_entity_id == entity_id
            ]
            transform_track = next(
                (track for track in tracks if track.type == "transform"),
                None,
            )
            path_track = next(
                (track for track in tracks if track.type == "path_follow"),
                None,
            )
            raw = sample_transform_track(
                transform_track,
                self.time_seconds,
                entity.solved_transform,
            )
            if path_track is not None:
                raw = sample_path_track(path_track, self.time_seconds, raw)
            world = self._to_world(raw, entity_id=entity_id)
            self._entities[entity_id] = world
            return world
        finally:
            self._leave(token)

    def camera(self) -> tuple[TransformValue, float] | None:
        if self.state.camera is None:
            return None
        if self._camera is not None:
            return self._camera
        token = f"camera:{self.state.camera.camera_id}"
        self._enter(token)
        try:
            tracks = list(self.state.camera.tracks.values())
            transform_track = next(
                (track for track in tracks if track.type == "transform"),
                None,
            )
            path_track = next(
                (track for track in tracks if track.type == "path_follow"),
                None,
            )
            raw = sample_transform_track(
                transform_track,
                self.time_seconds,
                self.state.camera.solved_transform,
            )
            if path_track is not None:
                raw = sample_path_track(path_track, self.time_seconds, raw)
            transform = self._to_world(raw, camera=True)
            focus_id = self.state.camera.static.focus_target_id
            look_track = next(
                (track for track in tracks if track.type == "look_at"),
                None,
            )
            if look_track and look_track.target_id:
                focus_id = look_track.target_id
            if focus_id in self.state.entities:
                target = self.entity(focus_id).translation_m
                transform = transform.model_copy(
                    update={
                        "rotation_quaternion_wxyz": look_at_camera_quaternion(
                            transform.translation_m,
                            target,
                        )
                    }
                )
            focal_track = next(
                (track for track in tracks if track.type == "focal_length"),
                None,
            )
            focal = sample_scalar_track(
                focal_track,
                self.time_seconds,
                self.state.camera.static.focal_length_mm
                or self.profile.default_focal_length_mm,
            )
            self._camera = (transform, focal)
            return self._camera
        finally:
            self._leave(token)

    def _to_world(
        self,
        raw: TransformValue,
        *,
        entity_id: str | None = None,
        camera: bool = False,
    ) -> TransformValue:
        completed = _complete_preserving_frame(raw)
        if completed.space == "world":
            return completed.model_copy(update={"target_id": None})
        if completed.space == "local":
            parent_id = (
                self.state.entities[entity_id].parent_id
                if entity_id is not None
                else None
            )
            reference = self.entity(parent_id) if parent_id else _identity_transform()
        elif completed.space == "target_relative":
            if not completed.target_id:
                raise ValueError("target_relative Transform 缺少 target_id")
            reference = self.entity(completed.target_id)
        elif completed.space == "camera":
            if camera:
                raise ValueError("摄影机不能使用自身 camera 参考系")
            camera_state = self.camera()
            if camera_state is None:
                raise ValueError("camera 参考系要求活动摄影机")
            reference = camera_state[0]
        else:  # pragma: no cover - Schema 已关闭其他值
            raise ValueError(f"未知 Transform 参考系：{completed.space}")
        return _compose_transform(reference, completed)

    def _enter(self, token: str) -> None:
        if token in self._resolving:
            start = self._resolving.index(token)
            cycle = self._resolving[start:] + [token]
            raise ValueError(f"参考系依赖存在循环：{' -> '.join(cycle)}")
        self._resolving.append(token)

    def _leave(self, token: str) -> None:
        if self._resolving and self._resolving[-1] == token:
            self._resolving.pop()


def _complete_preserving_frame(value: TransformValue) -> TransformValue:
    return TransformValue(
        translation_m=value.translation_m or (0.0, 0.0, 0.0),
        rotation_quaternion_wxyz=(
            value.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0)
        ),
        scale=value.scale or (1.0, 1.0, 1.0),
        space=value.space,
        target_id=value.target_id,
    )


def _identity_transform() -> TransformValue:
    return TransformValue(
        translation_m=(0.0, 0.0, 0.0),
        rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        space="world",
    )


def _compose_transform(
    reference: TransformValue,
    relative: TransformValue,
) -> TransformValue:
    scaled_offset = tuple(
        relative.translation_m[index] * reference.scale[index]
        for index in range(3)
    )
    translation = add(
        reference.translation_m,
        rotate_vector(reference.rotation_quaternion_wxyz, scaled_offset),
    )
    return TransformValue(
        translation_m=translation,
        rotation_quaternion_wxyz=quaternion_multiply(
            reference.rotation_quaternion_wxyz,
            relative.rotation_quaternion_wxyz,
        ),
        scale=tuple(
            reference.scale[index] * relative.scale[index]
            for index in range(3)
        ),
        space="world",
    )


def _entity_transform_at(
    state: CandidateState,
    entity_id: str,
    time_seconds: float,
    profile: PlanningProfile | None = None,
) -> TransformValue:
    resolver = _WorldTransformResolver(
        state,
        time_seconds,
        profile or PlanningProfile(),
    )
    return resolver.entity(entity_id)


def _camera_state_at(
    state: CandidateState,
    time_seconds: float,
    profile: PlanningProfile,
) -> tuple[TransformValue, float] | None:
    return _WorldTransformResolver(state, time_seconds, profile).camera()


def _projection(state, entity_id, transforms, camera_transform, focal_length, profile):
    return project_point(
        transforms[entity_id].translation_m,
        camera_transform.translation_m,
        camera_transform.rotation_quaternion_wxyz,
        focal_length,
        state.camera.static.sensor_width_mm,
        profile.resolution_x / profile.resolution_y,
    )


def _projection_bounds(state, entity_id, transforms, camera_transform, focal_length, profile):
    return project_geometry_bounds(
        state.entities[entity_id].proxy,
        transforms[entity_id],
        camera_transform.translation_m,
        camera_transform.rotation_quaternion_wxyz,
        focal_length,
        state.camera.static.sensor_width_mm,
        profile.resolution_x / profile.resolution_y,
    )


def _valid_region(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 4 and value[0] <= value[2] and value[1] <= value[3]


def _bounds_inside_fraction(bounds: tuple[float, float, float, float, float]) -> float:
    left, top, right, bottom, _ = bounds
    if not all(math.isfinite(item) for item in bounds):
        return 0.0
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return 0.0
    inside_width = max(0.0, min(1.0, right) - max(0.0, left))
    inside_height = max(0.0, min(1.0, bottom) - max(0.0, top))
    return min(1.0, inside_width * inside_height / (width * height))


def _projected_measurement(bounds, measurement: str) -> float:
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return {"width": width, "height": height, "diameter": max(width, height)}[measurement]


def _direction_matches(delta, direction, minimum, tolerance, *, space: str):
    if space == "camera":
        mapping = {
            "left": -delta[0],
            "right": delta[0],
            "forward": -delta[2],
            "backward": delta[2],
            "up": delta[1],
            "down": -delta[1],
        }
    else:
        mapping = {
            "left": -delta[0],
            "right": delta[0],
            "forward": -delta[1],
            "backward": delta[1],
            "up": delta[2],
            "down": -delta[2],
        }
    return mapping.get(direction, -math.inf) + tolerance >= minimum


def _within_range(value: float, minimum: float, maximum: float, tolerance: float) -> bool:
    return minimum - tolerance <= value <= maximum + tolerance


def _timeline_probe_times(timeline: TimelineSpec) -> list[float]:
    last = timeline.duration_seconds - timeline.fps_denominator / timeline.fps_numerator
    return [0.0, max(0.0, last / 2.0), max(0.0, last)]


def _is_environment_entity(entity: EntitySpec) -> bool:
    values = {entity.role.lower(), *(item.lower() for item in entity.tags)}
    return bool(values & {"environment", "ground", "terrain", "background_surface"})


def _constraint_error(constraint: ConstraintSpec, code: str, actual: Any) -> Violation:
    return Violation(
        id=f"violation_{constraint.constraint_id}_{code.lower()}",
        code=code,
        severity=constraint.strength,
        constraint_id=constraint.constraint_id,
        entity_ids=constraint.subjects,
        time_range_seconds=constraint.time_range_seconds,
        expected=_parameters(constraint),
        actual=actual,
        adjustable_variables=["entity transforms", "camera transform", "camera focal length"],
        message=f"约束 {constraint.constraint_id} 未满足：{code}",
    )


def _violation(code: str, message: str, **kwargs) -> Violation:
    return Violation(
        id=f"violation_{code.lower()}_{canonical_hash([message, kwargs])[-8:]}",
        code=code,
        severity="hard",
        message=message,
        **kwargs,
    )


def _parameters(constraint: ConstraintSpec) -> dict[str, Any]:
    return constraint.parameters.model_dump(mode="json", exclude_none=True)


def _envelope(
    before: int,
    after: int,
    *,
    status: str = "ok",
    changes: list[dict[str, Any]] | None = None,
    data: Any = None,
    violations: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    capability_gaps: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tool_version": TOOLKIT_VERSION,
        "status": status,
        "revision_before": before,
        "revision_after": after,
        "changes": changes or [],
        "data": data if data is not None else {},
        "violations": violations or [],
        "warnings": warnings or [],
        "capability_gaps": capability_gaps or [],
        "next_actions": next_actions or [],
    }


def _mutation_envelope(
    result: MutationResult,
    *,
    data: Any = None,
    violations: list[dict[str, Any]] | None = None,
    capability_gaps: list[str] | None = None,
) -> dict[str, Any]:
    return _envelope(
        result.revision_before,
        result.revision_after,
        status=result.status,
        changes=result.changes,
        data=data,
        violations=violations,
        warnings=result.warnings,
        capability_gaps=capability_gaps,
    )


def _error_message(error: ValidationError | ValueError) -> str:
    if not isinstance(error, ValidationError):
        return str(error)
    details = error.errors(include_url=False, include_context=False, include_input=False)
    direct: list[dict[str, Any]] = []
    branches: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        location = detail.get("loc", ())
        if len(location) >= 3 and location[0] in {"parameters", "path", "proxy"}:
            branches.setdefault(str(location[1]), []).append(detail)
        else:
            direct.append(detail)
    selected = direct[:4]
    if branches and len(selected) < 4:
        # 非判别联合只报告最接近成功的分支，避免几十条无关候选错误污染上下文。
        closest = min(branches.values(), key=len)
        selected.extend(closest[: 4 - len(selected)])
    messages: list[str] = []
    for detail in selected or details[:4]:
        location = list(detail.get("loc", ()))
        if len(location) >= 3 and location[0] in {"parameters", "path", "proxy"}:
            location.pop(1)
        path = ".".join(str(item) for item in location) or "input"
        messages.append(f"{path}: {detail.get('msg', 'invalid value')}")
    remaining = max(0, len(details) - len(selected))
    suffix = f"；另省略 {remaining} 条候选分支错误" if remaining else ""
    return f"输入校验失败：{'; '.join(messages)}{suffix}"


def _rejected(revision: int, message: str) -> dict[str, Any]:
    return _envelope(
        revision,
        revision,
        status="rejected",
        warnings=[message],
        next_actions=["修正参数后重试"],
    )
