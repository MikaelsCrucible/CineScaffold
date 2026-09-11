from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace
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
from cinescaffold.planning.design import (
    DesignOption,
    EntitySizeRequest,
    SceneSkeleton,
    build_design_candidate,
    design_option_id,
    skeleton_hash,
    task_capability_slice,
    validate_scene_skeleton,
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


TOOLKIT_VERSION = "0.23"
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
REPAIRABLE_VIOLATION_CODES = {
    "CAMERA_MOTION_NEAR_COLLINEAR",
    "ENTITY_OUT_OF_FRAME",
    "PROJECTED_MOTION_UNREADABLE",
    "PROJECTED_SIZE_VIOLATED",
}
REPAIR_PREFERENCES = {
    "balanced",
    "maximize_motion_readability",
    "minimize_change",
    "preserve_composition",
}
PER_FRAME_CONSTRAINTS = {
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
    "position_at_time",
    "hold",
}


@dataclass(frozen=True)
class _CameraRepair:
    suggestion_id: str
    base_revision: int
    base_candidate_hash: str
    focus_entity_id: str
    focus_point_m: tuple[float, float, float]
    yaw_degrees: float
    distance_scale: float
    focal_scale: float
    strategy: str
    target_codes: tuple[str, ...]
    predicted_report: ValidationReport | None
    predicted_candidate_hash: str | None
    camera_anchor_before_m: tuple[float, float, float]
    camera_anchor_after_m: tuple[float, float, float]
    focal_length_before_mm: float
    focal_length_after_mm: float
    change_cost: float
    preserves_explicit_requirements: bool


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
        self._repair_suggestions: dict[str, _CameraRepair] = {}
        self._scene_skeleton: SceneSkeleton | None = None
        self._design_options: dict[str, DesignOption] = {}
        self._last_design_request_hash: str | None = None

    @property
    def has_scene_skeleton(self) -> bool:
        return self._scene_skeleton is not None

    @property
    def has_design_options(self) -> bool:
        return bool(self._design_options)

    @property
    def design_option_applied(self) -> bool:
        return bool(self.store.get().entities)

    @property
    def has_repairable_violations(self) -> bool:
        validation = self.store.get().validation
        return bool(
            validation
            and any(
                item.code in REPAIRABLE_VIOLATION_CODES
                and item.severity != "warning"
                for item in validation.violations
            )
        )

    @property
    def has_current_repair_suggestions(self) -> bool:
        current = self.store.get()
        current_hash = _candidate_hash(current)
        return any(
            item.base_revision == current.revision
            and item.base_candidate_hash == current_hash
            for item in self._repair_suggestions.values()
        )

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
                "projected_motion_readability": (
                    "moving 主体在每个语义动作阶段必须产生足够的屏幕轨迹范围或投影尺度变化；"
                    "不能用沿镜头纵深的微小尺寸变化冒充对白模可读的运动"
                ),
                "default_camera_motion_obliqueness": (
                    "未明确摄影机方向时，线性主体运动不得采用近似迎面或背后共线机位；"
                    "明确机位保留用户要求并降为 warning"
                ),
            },
            "repair_suggestions": {
                "supported_violation_codes": sorted(REPAIRABLE_VIOLATION_CODES),
                "preferences": sorted(REPAIR_PREFERENCES),
                "contract": (
                    "Agent 选择整体策略；Toolkit 确定性搜索具体摄影机数值，"
                    "suggest_repairs 不修改 Candidate，apply_repair 原子应用已验证建议"
                ),
            },
            "inspect_views": INSPECT_VIEWS,
            "acceptance": {
                "minimum_soft_score": self.profile.minimum_soft_score,
                "soft_score_role": "advisory_quality_metric",
                "requires_hard_pass": True,
                "commit_ready": _commit_ready(state.validation, self.profile),
                "minimum_orbit_plane_view_alignment": (
                    self.profile.minimum_orbit_plane_view_alignment
                ),
                "minimum_projected_motion_extent": (
                    self.profile.minimum_projected_motion_extent
                ),
                "minimum_projected_motion_scale_ratio": (
                    self.profile.minimum_projected_motion_scale_ratio
                ),
                "minimum_camera_motion_obliqueness_degrees": (
                    self.profile.minimum_camera_motion_obliqueness_degrees
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

    def submit_scene_skeleton(self, skeleton: dict[str, Any]) -> dict[str, Any]:
        """保存不含数值坐标的符号骨架，Candidate revision 保持不变。"""

        revision = self.store.current_revision
        if self.store.get().entities:
            return _rejected(revision, "Candidate 已开始物化，不得重新提交 Scene Skeleton")
        try:
            parsed = SceneSkeleton.model_validate(skeleton)
            validate_scene_skeleton(self.objective_brief, parsed)
        except (ValidationError, ValueError) as error:
            return _rejected(revision, _error_message(error))
        self._scene_skeleton = parsed
        self._design_options.clear()
        self._last_design_request_hash = None
        return _envelope(
            revision,
            revision,
            data={
                "skeleton_hash": skeleton_hash(parsed),
                "entity_count": len(parsed.entities),
                "relation_count": len(parsed.relations),
                "motion_phase_count": len(parsed.motion_phases),
                "symbolic_only": True,
                "next_tool": "request_design_options",
            },
        )

    def request_design_options(
        self,
        preference: str = "balanced",
        max_options: int = 3,
        custom_size_requests: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """依据骨架与可选尺寸范围生成少量候选，并用同一 Validator 预测。"""

        revision = self.store.current_revision
        if self._scene_skeleton is None:
            return _rejected(revision, "必须先提交 Scene Skeleton")
        if self.store.get().entities:
            return _rejected(revision, "Candidate 已物化；后续请使用 Validator 修复接口")
        allowed = {
            "balanced",
            "preserve_composition",
            "maximize_motion_readability",
        }
        if preference not in allowed:
            return _rejected(revision, f"未知 design preference：{preference}")
        if not 1 <= max_options <= 3:
            return _rejected(revision, "max_options 必须位于 1 到 3")
        try:
            parsed_size_requests = [
                EntitySizeRequest.model_validate(item)
                for item in (custom_size_requests or [])
            ]
        except ValidationError as error:
            return _rejected(revision, _error_message(error))
        known_entities = {item.entity_id for item in self._scene_skeleton.entities}
        request_ids = [item.entity_id for item in parsed_size_requests]
        unknown = sorted(set(request_ids) - known_entities)
        if unknown:
            return _rejected(
                revision,
                f"自定义尺寸引用未知 Scene Skeleton Entity：{', '.join(unknown)}",
            )
        if len(request_ids) != len(set(request_ids)):
            return _rejected(revision, "同一实体只能提交一个自定义尺寸范围")
        size_requests = {item.entity_id: item for item in parsed_size_requests}
        request_hash = canonical_hash(
            {
                "preference": preference,
                "max_options": max_options,
                "custom_size_requests": [
                    item.model_dump(mode="json") for item in parsed_size_requests
                ],
            }
        )
        if request_hash == self._last_design_request_hash:
            return _rejected(
                revision,
                "不得重复完全相同的 Design Options 请求；请选择现有 option 或调整尺寸范围",
            )
        self._last_design_request_hash = request_hash
        self._design_options.clear()
        order = [preference, *sorted(allowed - {preference})][:max_options]
        base = self.store.get()
        skeleton_sha256 = skeleton_hash(self._scene_skeleton)
        options: list[DesignOption] = []
        option_errors: list[str] = []
        for strategy in order:
            try:
                candidate, envelopes, assumptions = build_design_candidate(
                    self.objective_brief,
                    self._scene_skeleton,
                    base,
                    self.profile,
                    strategy,
                    size_requests,
                )
                report = self._validate(candidate, FULL_VALIDATION_CHECKS)
            except (ValidationError, ValueError) as error:
                option_errors.append(f"{strategy}: {_error_message(error)}")
                continue
            option_id = design_option_id(
                skeleton_sha256=skeleton_sha256,
                base_revision=revision,
                strategy=strategy,
                candidate=candidate,
            )
            option = DesignOption(
                option_id=option_id,
                base_revision=revision,
                skeleton_hash=skeleton_sha256,
                strategy=strategy,
                candidate=candidate,
                numeric_envelopes=envelopes,
                assumptions=assumptions,
                relevant_capabilities=task_capability_slice(
                    self._scene_skeleton,
                    self.profile,
                ),
                predicted=_design_report_summary(report, self.profile),
            )
            options.append(option)
            self._design_options[option_id] = option
        return _envelope(
            revision,
            revision,
            status="ok" if options else "no_change",
            data={
                "skeleton_hash": skeleton_sha256,
                "options": [_design_option_payload(item) for item in options],
                "selection_rule": (
                    "先保留 explicit，再比较 hard violation、soft score 与策略偏好"
                ),
                "custom_size_request_count": len(size_requests),
                "next_tool": "apply_design_option" if options else None,
            },
            warnings=(
                []
                if options
                else [
                    "未能从当前符号骨架生成合法数值候选",
                    *sorted(set(option_errors)),
                ]
            ),
        )

    def apply_design_option(
        self,
        base_revision: int,
        option_id: str,
    ) -> dict[str, Any]:
        """原子物化完整选项，避免模型抄写建议数值。"""

        revision = self.store.current_revision
        option = self._design_options.get(option_id)
        if option is None:
            return _rejected(revision, f"Design Option 不存在或已失效：{option_id}")
        if base_revision != option.base_revision or revision != option.base_revision:
            return _rejected(
                revision,
                f"Design Option 基于 revision {option.base_revision}；请重新请求选项",
            )
        if (
            self._scene_skeleton is None
            or skeleton_hash(self._scene_skeleton) != option.skeleton_hash
        ):
            return _rejected(revision, "Scene Skeleton 已变化；请重新请求选项")

        def mutate(state: CandidateState):
            state.entities = deepcopy(option.candidate.entities)
            state.motion_tracks = deepcopy(option.candidate.motion_tracks)
            state.constraints = deepcopy(option.candidate.constraints)
            state.camera = deepcopy(option.candidate.camera)
            return (
                [
                    {
                        "operation": "materialize_design",
                        "path": "candidate",
                        "option_id": option_id,
                        "strategy": option.strategy,
                    }
                ],
                [],
            )

        try:
            mutation = self.store.apply(mutate)
        except (ValidationError, ValueError) as error:
            return _rejected(revision, _error_message(error))
        report = self._validate(self.store.get(), FULL_VALIDATION_CHECKS)
        self.store.save_validation(report)
        self._design_options.clear()
        actual = _design_report_summary(report, self.profile)
        return _mutation_envelope(
            mutation,
            data={
                "option_id": option_id,
                "strategy": option.strategy,
                "predicted": option.predicted,
                "actual": actual,
                "prediction_matched": option.predicted == actual,
                "commit_ready": _commit_ready(report, self.profile),
                "next_tool": (
                    "commit_request"
                    if _commit_ready(report, self.profile)
                    else "solve_or_repair_candidate"
                ),
            },
            violations=[item.model_dump(mode="json") for item in report.violations],
            capability_gaps=report.capability_gaps,
        )

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

    def suggest_repairs(
        self,
        revision: int | None = None,
        violation_ids: list[str] | None = None,
        preference: str = "balanced",
        max_options: int = 3,
    ) -> dict[str, Any]:
        """只读搜索少量摄影机修复方案，具体数值由 Validator 复验。"""

        selected_revision = self.store.current_revision if revision is None else revision
        if selected_revision != self.store.current_revision:
            return _rejected(
                self.store.current_revision,
                "修复建议只能针对当前 revision；历史 revision 请先 restore_candidate",
            )
        if preference not in REPAIR_PREFERENCES:
            return _rejected(
                selected_revision,
                f"未知 repair preference：{preference}",
            )
        if not 1 <= max_options <= 3:
            return _rejected(selected_revision, "max_options 必须位于 1 到 3")

        state = self.store.get(selected_revision)
        report = self._validate(state, FULL_VALIDATION_CHECKS)
        requested_ids = set(violation_ids or [])
        known_ids = {item.id for item in report.violations}
        unknown_ids = sorted(requested_ids - known_ids)
        if unknown_ids:
            return _rejected(
                selected_revision,
                f"Violation 不存在于当前 revision：{', '.join(unknown_ids)}",
            )
        selected = [
            item
            for item in report.violations
            if item.code in REPAIRABLE_VIOLATION_CODES
            and item.severity != "warning"
            and (not requested_ids or item.id in requested_ids)
        ]
        if not selected:
            return _envelope(
                selected_revision,
                selected_revision,
                status="no_change",
                data={
                    "baseline": _repair_report_summary(report, ()),
                    "options": [],
                    "evaluated_candidates": 0,
                },
                warnings=["当前 revision 没有可由建议接口处理的非 warning violation"],
            )

        gap = _camera_repair_capability_gap(state)
        if gap is not None:
            return _envelope(
                selected_revision,
                selected_revision,
                status="unsupported",
                data={
                    "baseline": _repair_report_summary(
                        report,
                        tuple(sorted({item.code for item in selected})),
                    ),
                    "options": [],
                    "evaluated_candidates": 0,
                },
                capability_gaps=[gap],
            )

        target_codes = tuple(sorted({item.code for item in selected}))
        repairs, evaluated = self._search_camera_repairs(
            state,
            report,
            selected,
            preference,
            max_options,
        )
        for repair in repairs:
            self._repair_suggestions[repair.suggestion_id] = repair
        return _envelope(
            selected_revision,
            selected_revision,
            status="ok" if repairs else "no_change",
            data={
                "baseline": _repair_report_summary(report, target_codes),
                "options": [_repair_option_payload(item, self.profile) for item in repairs],
                "evaluated_candidates": evaluated,
                "selection_rule": (
                    "先减少目标 hard violation，再避免新增非目标 hard violation，"
                    "最后按 preference 与最小改动排序"
                ),
            },
            warnings=(
                []
                if repairs
                else ["冻结搜索空间内没有找到比当前 Candidate 更好的摄影机方案"]
            ),
        )

    def apply_repair(
        self,
        base_revision: int,
        suggestion_id: str,
    ) -> dict[str, Any]:
        """原子应用建议；revision 或内容变化后建议立即失效。"""

        repair = self._repair_suggestions.get(suggestion_id)
        current_revision = self.store.current_revision
        if repair is None:
            return _rejected(current_revision, f"修复建议不存在或已失效：{suggestion_id}")
        if base_revision != repair.base_revision or current_revision != repair.base_revision:
            return _rejected(
                current_revision,
                (
                    f"修复建议基于 revision {repair.base_revision}，"
                    f"当前为 {current_revision}；请重新 suggest_repairs"
                ),
            )
        current = self.store.get()
        if _candidate_hash(current) != repair.base_candidate_hash:
            return _rejected(current_revision, "Candidate 内容已变化；请重新 suggest_repairs")

        preview = current.model_copy(deep=True)
        _apply_camera_repair(preview, repair)
        if _candidate_hash(preview) != repair.predicted_candidate_hash:
            return _rejected(current_revision, "建议重放结果与预测不一致；拒绝写入")
        before_report = self._validate(current, FULL_VALIDATION_CHECKS)

        def mutate(state: CandidateState):
            _apply_camera_repair(state, repair)
            return (
                [
                    {
                        "operation": "repair",
                        "path": "camera",
                        "suggestion_id": suggestion_id,
                        "strategy": repair.strategy,
                    }
                ],
                [],
            )

        try:
            mutation = self.store.apply(mutate)
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, _error_message(error))
        after = self.store.get()
        after_report = self._validate(after, FULL_VALIDATION_CHECKS)
        self.store.save_validation(after_report)
        self._repair_suggestions.clear()
        return _mutation_envelope(
            mutation,
            data={
                "suggestion_id": suggestion_id,
                "strategy": repair.strategy,
                "before": _repair_report_summary(before_report, repair.target_codes),
                "after": _repair_report_summary(after_report, repair.target_codes),
                "prediction_matched": (
                    _repair_report_summary(after_report, repair.target_codes)
                    == _repair_report_summary(
                        repair.predicted_report,
                        repair.target_codes,
                    )
                ),
                "commit_ready": _commit_ready(after_report, self.profile),
            },
            violations=[
                item.model_dump(mode="json") for item in after_report.violations
            ],
            capability_gaps=after_report.capability_gaps,
        )

    def _search_camera_repairs(
        self,
        state: CandidateState,
        baseline_report: ValidationReport,
        selected: list[Violation],
        preference: str,
        max_options: int,
    ) -> tuple[list[_CameraRepair], int]:
        target_codes = tuple(sorted({item.code for item in selected}))
        focus_ids = _repair_focus_entity_ids(state, selected)
        baseline_target_count = _target_violation_count(
            baseline_report,
            target_codes,
        )
        baseline_non_target_hard = _non_target_hard_count(
            baseline_report,
            target_codes,
        )
        base_hash = _candidate_hash(state)
        candidates: list[_CameraRepair] = []
        evaluated = 0
        for focus_id in focus_ids:
            focus_time = _repair_focus_time(state, selected, focus_id)
            resolver = _WorldTransformResolver(state, focus_time, self.profile)
            if resolver.camera() is None:
                continue
            focus_point = resolver.entity(focus_id).translation_m
            for yaw_degrees, distance_scale, focal_scale in (
                _camera_repair_parameter_grid(target_codes)
            ):
                preview = state.model_copy(deep=True)
                try:
                    repair = _build_camera_repair(
                        preview,
                        base_hash=base_hash,
                        focus_entity_id=focus_id,
                        focus_point_m=focus_point,
                        yaw_degrees=yaw_degrees,
                        distance_scale=distance_scale,
                        focal_scale=focal_scale,
                        target_codes=target_codes,
                        profile=self.profile,
                    )
                    _apply_camera_repair(preview, repair)
                    _validate_reference_frame_graph(preview, self.profile)
                except (ValidationError, ValueError, ZeroDivisionError):
                    continue
                evaluated += 1
                predicted = self._validate(preview, FULL_VALIDATION_CHECKS)
                if _target_violation_count(predicted, target_codes) >= baseline_target_count:
                    continue
                if _non_target_hard_count(predicted, target_codes) > baseline_non_target_hard:
                    continue
                candidates.append(
                    _finish_camera_repair(
                        repair,
                        preview,
                        predicted,
                        baseline_report,
                    )
                )

        ranked = sorted(
            candidates,
            key=lambda item: _repair_sort_key(item, preference, self.profile),
        )
        distinct: list[_CameraRepair] = []
        seen_strategies: set[tuple[str, str]] = set()
        for item in ranked:
            key = (item.strategy, item.focus_entity_id)
            if key in seen_strategies:
                continue
            seen_strategies.add(key)
            distinct.append(item)
            if len(distinct) >= max_options:
                break
        return distinct, evaluated

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
                _typed_motion_semantic_violations(
                    state,
                    self.objective_brief,
                    self.profile,
                )
            )
            violations.extend(
                _orbit_trajectory_violations(
                    state,
                    self.objective_brief,
                )
            )
            violations.extend(
                _nested_orbit_readability_violations(
                    state,
                    self.objective_brief,
                    self.profile,
                )
            )
            violations.extend(
                _projected_motion_readability_violations(
                    state,
                    self.objective_brief,
                    self.profile,
                )
            )
            violations.extend(
                _camera_motion_collinearity_violations(
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


def _candidate_hash(state: CandidateState) -> str:
    return canonical_hash(
        state.model_dump(mode="json", exclude={"revision", "validation"})
    )


def _camera_repair_capability_gap(state: CandidateState) -> str | None:
    if state.camera is None:
        return "repair:camera_missing"
    path_tracks = [
        track for track in state.camera.tracks.values() if track.type == "path_follow"
    ]
    if path_tracks:
        return "repair:camera_path_follow_not_supported_v0.1"
    transform_tracks = [
        track for track in state.camera.tracks.values() if track.type == "transform"
    ]
    for track in transform_tracks:
        for keyframe in track.keyframes:
            value = keyframe.value
            if (
                not isinstance(value, TransformValue)
                or value.space != "world"
                or value.translation_m is None
            ):
                return "repair:camera_transform_requires_world_translation"
    return None


def _repair_focus_entity_ids(
    state: CandidateState,
    violations: list[Violation],
) -> list[str]:
    ordered: list[str] = []
    for violation in violations:
        for entity_id in violation.entity_ids:
            entity = state.entities.get(entity_id)
            if entity is not None and not _is_environment_entity(entity):
                if entity_id not in ordered:
                    ordered.append(entity_id)
    if (
        state.camera is not None
        and state.camera.static.focus_target_id in state.entities
        and state.camera.static.focus_target_id not in ordered
    ):
        ordered.append(state.camera.static.focus_target_id)
    if not ordered:
        ordered.extend(
            entity_id
            for entity_id, entity in state.entities.items()
            if not _is_environment_entity(entity)
        )
    return ordered[:3]


def _repair_focus_time(
    state: CandidateState,
    violations: list[Violation],
    focus_entity_id: str,
) -> float:
    ranges = [
        item.time_range_seconds
        for item in violations
        if focus_entity_id in item.entity_ids and item.time_range_seconds is not None
    ]
    if not ranges:
        return max(0.0, state.timeline.duration_seconds / 2.0)
    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    frame_step = state.timeline.fps_denominator / state.timeline.fps_numerator
    return min((start + end) / 2.0, state.timeline.duration_seconds - frame_step)


def _build_camera_repair(
    state: CandidateState,
    *,
    base_hash: str,
    focus_entity_id: str,
    focus_point_m: tuple[float, float, float],
    yaw_degrees: float,
    distance_scale: float,
    focal_scale: float,
    target_codes: tuple[str, ...],
    profile: PlanningProfile,
) -> _CameraRepair:
    if state.camera is None:
        raise ValueError("摄影机不存在")
    camera_state = _WorldTransformResolver(state, 0.0, profile).camera()
    if camera_state is None:
        raise ValueError("摄影机不可求值")
    anchor_before, focal_before = camera_state
    minimum_focal, maximum_focal = _camera_focal_limits(state)
    focal_after = min(maximum_focal, max(minimum_focal, focal_before * focal_scale))
    actual_focal_scale = focal_after / focal_before
    anchor_after = _repair_camera_position(
        anchor_before.translation_m,
        focus_point_m,
        yaw_degrees,
        distance_scale,
    )
    change_cost = (
        length(subtract(anchor_after, anchor_before.translation_m))
        / max(profile.default_camera_distance_m, 1.0)
        + abs(math.log(max(actual_focal_scale, 1e-9)))
    )
    strategy = _camera_repair_strategy(
        yaw_degrees,
        distance_scale,
        actual_focal_scale,
    )
    identity = {
        "base_candidate_hash": base_hash,
        "focus_entity_id": focus_entity_id,
        "focus_point_m": focus_point_m,
        "yaw_degrees": yaw_degrees,
        "distance_scale": distance_scale,
        "focal_scale": actual_focal_scale,
        "target_codes": target_codes,
    }
    suggestion_id = f"repair_{canonical_hash(identity).removeprefix('sha256:')[:16]}"
    return _CameraRepair(
        suggestion_id=suggestion_id,
        base_revision=state.revision,
        base_candidate_hash=base_hash,
        focus_entity_id=focus_entity_id,
        focus_point_m=focus_point_m,
        yaw_degrees=yaw_degrees,
        distance_scale=distance_scale,
        focal_scale=actual_focal_scale,
        strategy=strategy,
        target_codes=target_codes,
        predicted_report=None,
        predicted_candidate_hash=None,
        camera_anchor_before_m=anchor_before.translation_m,
        camera_anchor_after_m=anchor_after,
        focal_length_before_mm=focal_before,
        focal_length_after_mm=focal_after,
        change_cost=change_cost,
        preserves_explicit_requirements=True,
    )


def _finish_camera_repair(
    repair: _CameraRepair,
    preview: CandidateState,
    predicted: ValidationReport,
    baseline: ValidationReport,
) -> _CameraRepair:
    return replace(
        repair,
        predicted_report=predicted,
        predicted_candidate_hash=_candidate_hash(preview),
        preserves_explicit_requirements=(
            _explicit_hard_signatures(predicted)
            <= _explicit_hard_signatures(baseline)
        ),
    )


def _apply_camera_repair(state: CandidateState, repair: _CameraRepair) -> None:
    if state.camera is None:
        raise ValueError("摄影机不存在")
    camera = state.camera
    solved = _complete_transform(camera.solved_transform)
    transform_tracks = [
        track for track in camera.tracks.values() if track.type == "transform"
    ]
    if not transform_tracks or camera.solved_transform.translation_m is not None:
        solved_position = _repair_camera_position(
            solved.translation_m,
            repair.focus_point_m,
            repair.yaw_degrees,
            repair.distance_scale,
        )
        solved = solved.model_copy(
            update={
                "translation_m": solved_position,
                "rotation_quaternion_wxyz": look_at_camera_quaternion(
                    solved_position,
                    repair.focus_point_m,
                ),
                "space": "world",
                "target_id": None,
            }
        )

    repaired_tracks: dict[str, TrackSpec] = {}
    for track_id, track in camera.tracks.items():
        if track.type == "transform":
            keyframes = []
            for keyframe in track.keyframes:
                value = keyframe.value
                if not isinstance(value, TransformValue) or value.translation_m is None:
                    raise ValueError("摄影机修复要求 Transform 关键帧包含 world translation")
                position = _repair_camera_position(
                    value.translation_m,
                    repair.focus_point_m,
                    repair.yaw_degrees,
                    repair.distance_scale,
                )
                repaired_value = value.model_copy(
                    update={
                        "translation_m": position,
                        "rotation_quaternion_wxyz": look_at_camera_quaternion(
                            position,
                            repair.focus_point_m,
                        ),
                    }
                )
                keyframes.append(keyframe.model_copy(update={"value": repaired_value}))
            track = track.model_copy(update={"keyframes": keyframes})
        elif track.type == "focal_length":
            keyframes = [
                keyframe.model_copy(
                    update={"value": float(keyframe.value) * repair.focal_scale}
                )
                for keyframe in track.keyframes
            ]
            track = track.model_copy(update={"keyframes": keyframes})
        repaired_tracks[track_id] = track

    static_focal = (
        camera.static.focal_length_mm
        or repair.focal_length_before_mm
    ) * repair.focal_scale
    static = camera.static.model_copy(
        update={
            "focal_length_mm": static_focal,
            # 只把主体当搜索锚点，不擅自把固定机位改成动态跟拍。
            "focus_target_id": camera.static.focus_target_id,
        }
    )
    state.camera = camera.model_copy(
        update={
            "static": static,
            "tracks": repaired_tracks,
            "solved_transform": solved,
        }
    )


def _repair_camera_position(
    position: tuple[float, float, float],
    focus: tuple[float, float, float],
    yaw_degrees: float,
    distance_scale: float,
) -> tuple[float, float, float]:
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    delta = subtract(position, focus)
    return (
        focus[0] + distance_scale * (cosine * delta[0] - sine * delta[1]),
        focus[1] + distance_scale * (sine * delta[0] + cosine * delta[1]),
        focus[2] + distance_scale * delta[2],
    )


def _camera_focal_limits(state: CandidateState) -> tuple[float, float]:
    minimum = 18.0
    maximum = 120.0
    for constraint in state.constraints.values():
        if constraint.type != "focal_length_range":
            continue
        params = _parameters(constraint)
        minimum = max(minimum, float(params.get("minimum_mm", minimum)))
        maximum = min(maximum, float(params.get("maximum_mm", maximum)))
    if minimum > maximum:
        raise ValueError("摄影机焦距约束无可行交集")
    return minimum, maximum


def _camera_repair_strategy(
    yaw_degrees: float,
    distance_scale: float,
    focal_scale: float,
) -> str:
    if yaw_degrees <= -75.0:
        return "side_view_left"
    if yaw_degrees >= 75.0:
        return "side_view_right"
    if yaw_degrees < -1e-6:
        return "three_quarter_left"
    if yaw_degrees > 1e-6:
        return "three_quarter_right"
    if distance_scale > 1.0 or focal_scale < 1.0:
        return "wider_framing"
    return "tighter_framing"


def _camera_repair_parameter_grid(
    target_codes: tuple[str, ...],
) -> list[tuple[float, float, float]]:
    """按问题族冻结小型模板，避免对无关自由度做笛卡尔穷举。"""

    codes = set(target_codes)
    values: list[tuple[float, float, float]] = []
    if "CAMERA_MOTION_NEAR_COLLINEAR" in codes:
        values.extend(
            (yaw, distance, 1.0)
            for yaw in (-30.0, 30.0, -45.0, 45.0, -60.0, 60.0, -90.0, 90.0)
            for distance in (0.85, 1.0, 1.2)
        )
    if "PROJECTED_MOTION_UNREADABLE" in codes:
        values.extend(
            (yaw, distance, focal)
            for yaw in (-30.0, 30.0, -45.0, 45.0, -60.0, 60.0, -90.0, 90.0)
            for distance, focal in (
                (1.0, 1.0),
                (0.85, 1.0),
                (0.7, 1.15),
                (0.5, 1.3),
                (0.5, 1.6),
            )
        )
    if codes & {"ENTITY_OUT_OF_FRAME", "PROJECTED_SIZE_VIOLATED"}:
        framing_pairs = (
            (1.4, 0.7),
            (1.2, 0.85),
            (1.0, 0.7),
            (1.0, 0.85),
            (0.85, 1.15),
            (0.7, 1.3),
            (0.5, 1.6),
        )
        values.extend(
            (yaw, distance, focal)
            for yaw in (0.0, -30.0, 30.0)
            for distance, focal in framing_pairs
        )
    # 保持固定顺序去重，使相同输入始终产生相同 suggestion_id 排序。
    return list(dict.fromkeys(values))


def _target_violation_count(
    report: ValidationReport,
    target_codes: tuple[str, ...],
) -> int:
    return sum(
        item.code in target_codes and item.severity != "warning"
        for item in report.violations
    )


def _non_target_hard_count(
    report: ValidationReport,
    target_codes: tuple[str, ...],
) -> int:
    return sum(
        item.severity == "hard" and item.code not in target_codes
        for item in report.violations
    )


def _explicit_hard_signatures(report: ValidationReport) -> set[tuple[str, str | None]]:
    return {
        (item.code, item.constraint_id)
        for item in report.violations
        if item.severity == "hard" and item.code.startswith("EXPLICIT_REQUIREMENT")
    }


def _repair_report_summary(
    report: ValidationReport,
    target_codes: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "hard_pass": report.hard_pass,
        "soft_score": round(report.soft_score, 6),
        "hard_violation_count": sum(
            item.severity == "hard" for item in report.violations
        ),
        "target_violation_count": _target_violation_count(report, target_codes),
        "target_codes": list(target_codes),
        "capability_gaps": report.capability_gaps,
    }


def _repair_sort_key(
    repair: _CameraRepair,
    preference: str,
    profile: PlanningProfile,
) -> tuple[Any, ...]:
    report = repair.predicted_report
    if report is None:
        return (math.inf,)
    hard_count = sum(item.severity == "hard" for item in report.violations)
    target_count = _target_violation_count(report, repair.target_codes)
    commit_penalty = 0 if _commit_ready(report, profile) else 1
    if preference == "minimize_change":
        return (target_count, hard_count, repair.change_cost, -report.soft_score)
    if preference == "preserve_composition":
        return (target_count, hard_count, -report.soft_score, repair.change_cost)
    if preference == "maximize_motion_readability":
        return (
            target_count,
            hard_count,
            -abs(repair.yaw_degrees),
            -report.soft_score,
            repair.change_cost,
        )
    return (
        commit_penalty,
        target_count,
        hard_count,
        -report.soft_score,
        repair.change_cost,
    )


def _repair_option_payload(
    repair: _CameraRepair,
    profile: PlanningProfile,
) -> dict[str, Any]:
    report = repair.predicted_report
    if report is None:
        raise ValueError("修复建议缺少预测报告")
    tradeoffs: list[str] = []
    if abs(repair.yaw_degrees) >= 75.0:
        tradeoffs.append("侧向位移最清楚，但会显著改变原机位方位")
    elif abs(repair.yaw_degrees) > 1e-6:
        tradeoffs.append("斜侧机位兼顾纵深与横向运动可读性")
    if repair.distance_scale > 1.0:
        tradeoffs.append("摄影机后移，主体投影可能变小")
    elif repair.distance_scale < 1.0:
        tradeoffs.append("摄影机靠近，主体投影可能变大")
    if repair.focal_scale > 1.0:
        tradeoffs.append("焦距增加，视野收窄")
    elif repair.focal_scale < 1.0:
        tradeoffs.append("焦距减小，视野扩大")
    return {
        "suggestion_id": repair.suggestion_id,
        "base_revision": repair.base_revision,
        "strategy": repair.strategy,
        "summary": (
            f"以 {repair.focus_entity_id} 为观察目标，将摄影机方位旋转 "
            f"{repair.yaw_degrees:g}°，距离缩放 {repair.distance_scale:g} 倍"
        ),
        "exact_changes": {
            "search_anchor_entity_id": repair.focus_entity_id,
            "camera_anchor_translation_m": list(repair.camera_anchor_after_m),
            "yaw_degrees": repair.yaw_degrees,
            "distance_scale": repair.distance_scale,
            "focal_length_mm": repair.focal_length_after_mm,
        },
        "predicted": _repair_report_summary(report, repair.target_codes)
        | {"commit_ready": _commit_ready(report, profile)},
        "preserves_explicit_requirements": repair.preserves_explicit_requirements,
        "change_cost": round(repair.change_cost, 6),
        "tradeoffs": tradeoffs,
    }


def _design_report_summary(
    report: ValidationReport,
    profile: PlanningProfile,
) -> dict[str, Any]:
    return {
        "hard_pass": report.hard_pass,
        "soft_score": round(report.soft_score, 6),
        "hard_violation_count": sum(
            item.severity == "hard" for item in report.violations
        ),
        "soft_violation_count": sum(
            item.severity == "soft" for item in report.violations
        ),
        "violation_codes": sorted({item.code for item in report.violations}),
        "commit_ready": _commit_ready(report, profile),
        "capability_gaps": report.capability_gaps,
    }


def _design_option_payload(option: DesignOption) -> dict[str, Any]:
    return {
        "option_id": option.option_id,
        "base_revision": option.base_revision,
        "strategy": option.strategy,
        "numeric_envelopes": option.numeric_envelopes,
        "assumptions": list(option.assumptions),
        "relevant_capabilities": option.relevant_capabilities,
        "predicted": option.predicted,
        "tradeoffs": {
            "balanced": ["折中保持构图与屏幕运动可读性"],
            "preserve_composition": ["摄影机倾向后移，运动幅度可能较弱"],
            "maximize_motion_readability": ["斜侧角更大，可能偏离缺省构图"],
        }[option.strategy],
    }


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


def _nested_orbit_readability_violations(
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
                    ).model_copy(update={"severity": "warning"})
                )
    return violations


def _typed_motion_semantic_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
) -> list[Violation]:
    """逐阶段复验类型化运动语义，不依赖动作文本或场景身份。"""
    if objective_brief.schema_version not in {"0.3", "0.4", "0.5"}:
        return []
    frame_step = state.timeline.fps_denominator / state.timeline.fps_numerator
    last_frame_time = state.timeline.duration_seconds - frame_step
    violations: list[Violation] = []
    for motion_index, motion in enumerate(objective_brief.subject_motion):
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        subject_id = motion.get("subject_id")
        if not isinstance(subject_id, str) or subject_id not in state.entities:
            continue
        start = float(motion.get("start_time_seconds") or 0.0)
        end = float(motion.get("end_time_seconds") or state.timeline.duration_seconds)
        sample_times = _timeline_frame_times(
            state.timeline,
            start=max(0.0, start),
            end=min(end, state.timeline.duration_seconds),
        )
        if not sample_times:
            continue
        positions = [
            _WorldTransformResolver(state, time_seconds, profile).entity(
                subject_id
            ).translation_m
            for time_seconds in sample_times
        ]
        baseline = positions[0]
        movement_extent = max(length(subtract(item, baseline)) for item in positions)
        motion_mode = semantics.get("motion_mode")
        action_kind = semantics.get("action_kind")
        stationary_tolerance = 1e-4
        displacement_action = action_kind in {"board", "disembark"}
        if motion_mode == "stationary" or (
            motion_mode == "local_interaction" and not displacement_action
        ):
            if movement_extent > stationary_tolerance:
                violations.append(
                    _violation(
                        "MOTION_MODE_STATIONARY_VIOLATED",
                        f"无整体位移阶段发生了世界空间移动：{subject_id}",
                        entity_ids=[subject_id],
                        time_range_seconds=(start, end),
                        expected={"maximum_displacement_m": stationary_tolerance},
                        actual={"displacement_extent_m": movement_extent},
                        adjustable_variables=[f"motion_tracks.{subject_id}"],
                    )
                )
        elif motion_mode == "self_propelled" or displacement_action:
            if movement_extent <= stationary_tolerance:
                violations.append(
                    _violation(
                        "SELF_PROPELLED_MOTION_MISSING",
                        f"自主运动阶段没有产生实体位移：{subject_id}",
                        entity_ids=[subject_id],
                        time_range_seconds=(start, end),
                        expected={"minimum_displacement_m": stationary_tolerance},
                        actual={"displacement_extent_m": movement_extent},
                        adjustable_variables=[f"motion_tracks.{subject_id}"],
                    )
                )
            violations.extend(
                _typed_direction_violations(
                    state,
                    semantics,
                    subject_id,
                    positions,
                    sample_times,
                    start,
                    end,
                    profile,
                )
            )

        violations.extend(
            _typed_path_violations(
                state,
                semantics,
                subject_id,
                start,
                end,
                motion_index,
            )
        )
        postconditions = semantics.get("postconditions")
        visibility_after = (
            postconditions.get("external_visibility")
            if isinstance(postconditions, dict)
            else "unchanged"
        )
        if visibility_after in {"hidden", "visible"}:
            sample_time = min(max(end, 0.0), last_frame_time)
            actual_visible = _entity_visibility_at(state, subject_id, sample_time)
            expected_visible = visibility_after == "visible"
            if actual_visible != expected_visible:
                violations.append(
                    _violation(
                        "MOTION_POSTCONDITION_VISIBILITY_UNMET",
                        f"动作阶段结束后的外部可见性不符合语义：{subject_id}",
                        entity_ids=[subject_id],
                        time_range_seconds=(start, end),
                        expected={
                            "external_visibility": visibility_after,
                            "at_seconds": sample_time,
                        },
                        actual={
                            "external_visibility": (
                                "visible" if actual_visible else "hidden"
                            )
                        },
                        adjustable_variables=[f"motion_tracks.{subject_id}.visibility"],
                    )
                )

        contained_by_id = (
            postconditions.get("contained_by_id")
            if isinstance(postconditions, dict)
            else None
        )
        if isinstance(contained_by_id, str) and contained_by_id in state.entities:
            sample_time = min(max(end - frame_step, start), last_frame_time)
            subject_position = _WorldTransformResolver(
                state,
                sample_time,
                profile,
            ).entity(subject_id).translation_m
            container_position = _WorldTransformResolver(
                state,
                sample_time,
                profile,
            ).entity(contained_by_id)
            center_is_contained, local_center = _point_within_proxy_bounds(
                subject_position,
                container_position,
                state.entities[contained_by_id],
                margin_m=0.25,
            )
            endpoint_distance = length(
                subtract(subject_position, container_position.translation_m)
            )
            if not center_is_contained and not _has_carrier_binding(
                state,
                subject_id,
                contained_by_id,
                start,
                end,
            ):
                violations.append(
                    _violation(
                        "MOTION_POSTCONDITION_CONTAINMENT_UNMET",
                        f"动作阶段结束时主体既未接近也未绑定目标：{subject_id}",
                        entity_ids=[subject_id, contained_by_id],
                        time_range_seconds=(start, end),
                        expected={
                            "contained_by_id": contained_by_id,
                            "subject_center": "inside_target_proxy_bounds",
                        },
                        actual={
                            "endpoint_distance_m": endpoint_distance,
                            "subject_center_in_target_local_m": local_center,
                        },
                        adjustable_variables=[
                            f"motion_tracks.{subject_id}",
                            f"entities.{subject_id}.parent_id",
                        ],
                    )
                )

        if motion_mode != "carried":
            continue
        carrier_id = semantics.get("carrier_id")
        if not isinstance(carrier_id, str) or carrier_id not in state.entities:
            continue
        sample_time = min(max(start, 0.0), last_frame_time)
        if not _entity_visibility_at(state, subject_id, sample_time):
            continue
        if not _has_carrier_binding(
            state,
            subject_id,
            carrier_id,
            start,
            end,
        ):
            violations.append(
                _violation(
                    "CARRIED_SUBJECT_UNBOUND",
                    f"可见的 carried 主体未绑定到载体：{subject_id} -> {carrier_id}",
                    entity_ids=[subject_id, carrier_id],
                    time_range_seconds=(start, end),
                    expected={"carrier_id": carrier_id, "space": "target_relative_or_parent"},
                    actual={"binding": None},
                    adjustable_variables=[
                        f"entities.{subject_id}.parent_id",
                        f"motion_tracks.{subject_id}",
                        f"motion_tracks.{subject_id}.visibility",
                    ],
                )
            )
    return violations


def _point_within_proxy_bounds(
    world_point: tuple[float, float, float],
    proxy_transform: TransformValue,
    proxy_entity: EntitySpec,
    *,
    margin_m: float,
) -> tuple[bool, tuple[float, float, float]]:
    """把世界点还原到代理局部空间并检查局部包围范围。"""
    unrotated = rotate_vector(
        quaternion_conjugate(proxy_transform.rotation_quaternion_wxyz),
        subtract(world_point, proxy_transform.translation_m),
    )
    local_point = tuple(
        unrotated[index] / proxy_transform.scale[index]
        for index in range(3)
    )
    bounds = geometry_local_bounds_points(proxy_entity.proxy)
    minimums = tuple(min(point[index] for point in bounds) for index in range(3))
    maximums = tuple(max(point[index] for point in bounds) for index in range(3))
    local_margin = tuple(
        margin_m / proxy_transform.scale[index]
        for index in range(3)
    )
    contained = all(
        minimums[index] - local_margin[index]
        <= local_point[index]
        <= maximums[index] + local_margin[index]
        for index in range(3)
    )
    return contained, local_point


def _typed_direction_violations(
    state: CandidateState,
    semantics: dict[str, Any],
    subject_id: str,
    positions: list[tuple[float, float, float]],
    sample_times: list[float],
    start: float,
    end: float,
    profile: PlanningProfile,
) -> list[Violation]:
    mode = semantics.get("direction_mode")
    target_id = semantics.get("target_id")
    tolerance = 1e-4
    actual: dict[str, Any]
    if mode in {"toward_target", "away_from_target"}:
        if not isinstance(target_id, str) or target_id not in state.entities:
            return []
        target_positions = [
            _WorldTransformResolver(state, time_seconds, profile).entity(
                target_id
            ).translation_m
            for time_seconds in sample_times
        ]
        start_distance = length(subtract(positions[0], target_positions[0]))
        end_distance = length(subtract(positions[-1], target_positions[-1]))
        progress = (
            start_distance - end_distance
            if mode == "toward_target"
            else end_distance - start_distance
        )
        if progress > tolerance:
            return []
        actual = {
            "start_distance_m": start_distance,
            "end_distance_m": end_distance,
            "directed_progress_m": progress,
        }
    elif mode == "world_forward":
        delta = subtract(positions[-1], positions[0])
        progress = -delta[1]
        if progress > tolerance:
            return []
        actual = {"world_displacement_m": delta, "forward_progress_m": progress}
    elif mode == "relative_to_target":
        if (
            isinstance(target_id, str)
            and target_id in state.entities
            and _has_carrier_binding(state, subject_id, target_id, start, end)
        ):
            return []
        actual = {"target_id": target_id, "target_relative_binding": False}
    else:
        return []
    return [
        _violation(
            "MOTION_DIRECTION_SEMANTICS_UNMET",
            f"实体运动方向不符合类型化语义：{subject_id} / {mode}",
            entity_ids=[
                subject_id,
                *([target_id] if isinstance(target_id, str) else []),
            ],
            time_range_seconds=(start, end),
            expected={"direction_mode": mode, "minimum_progress_m": tolerance},
            actual=actual,
            adjustable_variables=[f"motion_tracks.{subject_id}"],
        )
    ]


def _typed_path_violations(
    state: CandidateState,
    semantics: dict[str, Any],
    subject_id: str,
    start: float,
    end: float,
    motion_index: int,
) -> list[Violation]:
    path_type = semantics.get("path_type")
    if path_type in {None, "unspecified", "stationary", "linear"}:
        return []
    expected_representations = {
        "circular": {"circle"},
        "elliptical": {"ellipse"},
        "s_curve": {"catmull_rom"},
        "figure_eight": {"lemniscate"},
    }.get(path_type)
    if expected_representations is None:
        return [
            _violation(
                "MOTION_PATH_FAMILY_UNSUPPORTED",
                f"类型化路径尚无可验证实现：{subject_id} / {path_type}",
                entity_ids=[subject_id],
                time_range_seconds=(start, end),
                expected={"path_type": path_type},
                actual={"supported": False, "motion_index": motion_index},
                adjustable_variables=[f"motion_tracks.{subject_id}"],
            )
        ]
    matching_tracks = [
        track
        for track in state.motion_tracks.values()
        if track.target_entity_id == subject_id
        and track.type == "path_follow"
        and track.path is not None
        and track.time_range_seconds[0] < end
        and track.time_range_seconds[1] > start
    ]
    actual_representations = {
        track.path.representation
        for track in matching_tracks
    }
    representation_matches = bool(actual_representations & expected_representations)
    geometry_matches = (
        any(_catmull_rom_has_inflection(track) for track in matching_tracks)
        if path_type == "s_curve"
        else representation_matches
    )
    if representation_matches and geometry_matches:
        return []
    return [
        _violation(
            "MOTION_PATH_FAMILY_UNMET",
            f"实体路径实现不符合类型化语义：{subject_id} / {path_type}",
            entity_ids=[subject_id],
            time_range_seconds=(start, end),
            expected={"representations": sorted(expected_representations)},
            actual={
                "representations": sorted(actual_representations),
                "geometry_matches": geometry_matches,
            },
            adjustable_variables=[f"motion_tracks.{subject_id}"],
        )
    ]


def _catmull_rom_has_inflection(track: TrackSpec) -> bool:
    """用三维转向法线换向排除直线或单向弧线冒充 S 曲线。"""
    if track.path is None or track.path.representation != "catmull_rom":
        return False
    directions = [
        normalize(subtract(right, left))
        for left, right in zip(
            track.path.control_points,
            track.path.control_points[1:],
        )
        if length(subtract(right, left)) > 1e-6
    ]
    turns = [
        normalize(cross(left, right))
        for left, right in zip(directions, directions[1:])
        if length(cross(left, right)) > 1e-4
    ]
    return any(
        dot(left, right) < -0.25
        for index, left in enumerate(turns)
        for right in turns[index + 1:]
    )


def _has_carrier_binding(
    state: CandidateState,
    subject_id: str,
    carrier_id: str,
    start: float,
    end: float,
) -> bool:
    entity = state.entities[subject_id]
    if entity.parent_id == carrier_id:
        return True
    for track in state.motion_tracks.values():
        if track.target_entity_id != subject_id:
            continue
        track_start, track_end = track.time_range_seconds
        if track_end <= start or track_start >= end:
            continue
        if (
            track.path is not None
            and track.path.space == "target_relative"
            and track.path.target_id == carrier_id
        ):
            return True
        if any(
            isinstance(keyframe.value, TransformValue)
            and keyframe.value.space == "target_relative"
            and keyframe.value.target_id == carrier_id
            for keyframe in track.keyframes
        ):
            return True
    return False


def _projected_motion_readability_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
) -> list[Violation]:
    """确保语义移动在控制白模的屏幕投影中可辨识。"""

    if state.camera is None:
        return []
    parameters = objective_brief.translation_parameters or {}
    motions = parameters.get("motions")
    if not isinstance(motions, list):
        return []

    explicit_camera = _has_explicit_camera_staging(objective_brief)
    frame_step = state.timeline.fps_denominator / state.timeline.fps_numerator
    last_frame_time = state.timeline.duration_seconds - frame_step
    violations: list[Violation] = []
    for motion in motions:
        if not isinstance(motion, dict):
            continue
        motion_type = str(motion.get("motion_type", ""))
        # carried 表示相对载体静止；可读性由载体运动和显隐转场承担。
        if motion_type in {"", "static", "interactive", "carried"}:
            continue
        subject_id = motion.get("subject_id")
        if not isinstance(subject_id, str) or subject_id not in state.entities:
            continue
        start = float(motion.get("start_time_seconds", 0.0))
        end = float(motion.get("end_time_seconds", state.timeline.duration_seconds))
        sample_end = min(end - frame_step, last_frame_time)
        if sample_end <= start:
            continue
        sample_times = [
            start + (sample_end - start) * index / 8.0
            for index in range(9)
        ]
        centers: list[tuple[float, float]] = []
        sizes: list[float] = []
        for time_seconds in sample_times:
            resolver = _WorldTransformResolver(state, time_seconds, profile)
            camera_state = resolver.camera()
            if camera_state is None:
                continue
            camera_transform, focal_length = camera_state
            transforms = {
                entity_id: resolver.entity(entity_id)
                for entity_id in state.entities
            }
            center = _projection(
                state,
                subject_id,
                transforms,
                camera_transform,
                focal_length,
                profile,
            )
            bounds = _projection_bounds(
                state,
                subject_id,
                transforms,
                camera_transform,
                focal_length,
                profile,
            )
            if not all(math.isfinite(item) for item in (*center, *bounds)):
                continue
            centers.append((center[0], center[1]))
            sizes.append(_projected_measurement(bounds, "diameter"))
        if len(centers) < 2 or len(sizes) < 2:
            continue

        projected_extent = math.hypot(
            max(item[0] for item in centers) - min(item[0] for item in centers),
            max(item[1] for item in centers) - min(item[1] for item in centers),
        )
        positive_sizes = [item for item in sizes if item > profile.numeric_tolerance]
        scale_ratio = (
            max(positive_sizes) / min(positive_sizes)
            if positive_sizes
            else math.inf
        )
        if (
            projected_extent + profile.numeric_tolerance
            >= profile.minimum_projected_motion_extent
            or scale_ratio + profile.numeric_tolerance
            >= profile.minimum_projected_motion_scale_ratio
        ):
            continue

        message = (
            f"主体 {subject_id} 的语义移动在屏幕投影中近似静止；"
            "请调整运动方向、摄影机距离或摄影机方位，使白模能够辨识该动作"
        )
        if explicit_camera:
            message += "；Brief 已明确摄影机设计，因此仅记录警告而不覆盖用户要求"
        violation = _violation(
            "PROJECTED_MOTION_UNREADABLE",
            message,
            entity_ids=[subject_id],
            time_range_seconds=(start, end),
            expected={
                "minimum_projected_centroid_extent": (
                    profile.minimum_projected_motion_extent
                ),
                "minimum_projected_scale_ratio": (
                    profile.minimum_projected_motion_scale_ratio
                ),
                "acceptance": "满足任意一项",
                "purpose": "让控制白模中的主体运动保持可辨识",
            },
            actual={
                "projected_centroid_extent": projected_extent,
                "projected_scale_ratio": scale_ratio,
                "sample_count": len(centers),
                "motion_index": motion.get("motion_index"),
            },
            adjustable_variables=[
                f"motion_tracks.{subject_id}",
                "camera transform",
                "camera focal length",
            ],
        )
        violations.append(
            violation.model_copy(update={"severity": "warning"})
            if explicit_camera
            else violation
        )
    return violations


def _camera_motion_collinearity_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
) -> list[Violation]:
    """防止缺省机位借迎面或背面运动轻易通过可读性门禁。"""

    if state.camera is None:
        return []
    motions = (objective_brief.translation_parameters or {}).get("motions")
    if not isinstance(motions, list):
        return []
    explicit_direction = _has_explicit_camera_direction(objective_brief)
    frame_step = state.timeline.fps_denominator / state.timeline.fps_numerator
    last_frame_time = state.timeline.duration_seconds - frame_step
    violations: list[Violation] = []
    for motion in motions:
        if not isinstance(motion, dict):
            continue
        if motion.get("motion_type") in {None, "", "static", "interactive", "carried"}:
            continue
        if motion.get("path_type") not in {None, "linear", "unspecified"}:
            continue
        subject_id = motion.get("subject_id")
        if not isinstance(subject_id, str) or subject_id not in state.entities:
            continue
        start = float(motion.get("start_time_seconds", 0.0))
        end = float(motion.get("end_time_seconds", state.timeline.duration_seconds))
        sample_end = min(end - frame_step, last_frame_time)
        if sample_end <= start:
            continue
        start_transform = _WorldTransformResolver(state, start, profile).entity(subject_id)
        end_transform = _WorldTransformResolver(state, sample_end, profile).entity(subject_id)
        displacement = subtract(
            end_transform.translation_m,
            start_transform.translation_m,
        )
        if length(displacement) <= profile.numeric_tolerance:
            continue
        motion_direction = normalize(displacement)
        sample_times = [
            start + (sample_end - start) * index / 8.0
            for index in range(9)
        ]
        obliqueness_degrees: list[float] = []
        for time_seconds in sample_times:
            resolver = _WorldTransformResolver(state, time_seconds, profile)
            camera_state = resolver.camera()
            if camera_state is None:
                continue
            subject = resolver.entity(subject_id)
            camera_transform, _ = camera_state
            subject_to_camera = subtract(
                camera_transform.translation_m,
                subject.translation_m,
            )
            if length(subject_to_camera) <= profile.numeric_tolerance:
                continue
            absolute_alignment = min(
                1.0,
                abs(dot(motion_direction, normalize(subject_to_camera))),
            )
            obliqueness_degrees.append(
                math.degrees(math.acos(absolute_alignment))
            )
        if not obliqueness_degrees:
            continue
        median_obliqueness = median(obliqueness_degrees)
        if median_obliqueness + profile.numeric_tolerance >= (
            profile.minimum_camera_motion_obliqueness_degrees
        ):
            continue
        message = (
            f"线性运动主体 {subject_id} 与摄影机视线长期近似共线；"
            "未明确要求迎面或背面跟拍时，请改用斜侧机位"
        )
        severity = "warning" if explicit_direction else "hard"
        if explicit_direction:
            message += "；Brief 已明确摄影机方向，因此仅记录警告"
        violation = _violation(
            "CAMERA_MOTION_NEAR_COLLINEAR",
            message,
            entity_ids=[subject_id],
            time_range_seconds=(start, end),
            expected={
                "minimum_median_obliqueness_degrees": (
                    profile.minimum_camera_motion_obliqueness_degrees
                ),
                "purpose": "避免缺省摄影机采用迎面或背面共线机位",
            },
            actual={
                "median_obliqueness_degrees": median_obliqueness,
                "minimum_obliqueness_degrees": min(obliqueness_degrees),
                "maximum_obliqueness_degrees": max(obliqueness_degrees),
                "sample_count": len(obliqueness_degrees),
                "motion_index": motion.get("motion_index"),
            },
            adjustable_variables=[
                "camera transform",
                "camera focal length",
                f"motion_tracks.{subject_id}",
            ],
        )
        violations.append(violation.model_copy(update={"severity": severity}))
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
    for motion in objective_brief.subject_motion:
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict) or semantics.get("action_kind") != "orbit":
            continue
        subject_id = motion.get("subject_id")
        target_id = semantics.get("target_id")
        if isinstance(subject_id, str) and isinstance(target_id, str):
            pairs.add((subject_id, target_id))
    return pairs


def _has_explicit_camera_view(objective_brief: ObjectivePlanningBrief) -> bool:
    return any(
        item.path.startswith("content.camera.view_angle")
        for item in objective_brief.explicit_requirements
    )


def _has_explicit_camera_staging(
    objective_brief: ObjectivePlanningBrief,
) -> bool:
    return any(
        item.path.startswith("content.camera")
        for item in objective_brief.explicit_requirements
    )


def _has_explicit_camera_direction(
    objective_brief: ObjectivePlanningBrief,
) -> bool:
    return any(
        item.path == "content.camera.view_relation_to_motion"
        and item.value in {"front", "rear"}
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
        semantics = motion.get("motion_semantics")
        if isinstance(semantics, dict):
            return semantics.get("path_type") not in {
                "circular",
                "elliptical",
                "unspecified",
            }
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
    unsupported_ground_ids: set[str] = set()
    for time_seconds in _timeline_frame_times(state.timeline):
        resolver = _WorldTransformResolver(state, time_seconds, profile)
        ground_levels: dict[str, float] = {}
        for ground_id in ground_ids:
            transform = resolver.entity(ground_id)
            normal = rotate_vector(transform.rotation_quaternion_wxyz, (0.0, 0.0, 1.0))
            if abs(normal[2]) >= 0.999:
                ground_levels[ground_id] = transform.translation_m[2]
            else:
                unsupported_ground_ids.add(ground_id)
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
    for ground_id in sorted(unsupported_ground_ids):
        violations.append(
            _violation(
                "GROUND_ORIENTATION_UNSUPPORTED",
                f"当前地面交互 Validator 仅支持水平环境平面：{ground_id}",
                entity_ids=[ground_id],
                expected={"ground_normal": "parallel_to_world_z"},
                actual={"ground_orientation": "non_horizontal"},
                adjustable_variables=["ground transform", "ground_interaction"],
            )
        )
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
        if (
            entity.ground_interaction is not None
            and entity.ground_interaction.source_ref
        ):
            locations.setdefault(
                entity.ground_interaction.source_ref,
                set(),
            ).add("ground_interaction")
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
    # 同一客观要求可能在 Brief 的多个字段中重复表达，统一继承映射证据。
    for equivalent_refs in _equivalent_explicit_source_refs(objective_brief):
        equivalent_locations = set().union(
            *(locations.get(source_ref, set()) for source_ref in equivalent_refs)
        )
        for source_ref in equivalent_refs:
            locations.setdefault(source_ref, set()).update(equivalent_locations)
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
    violations.extend(
        _explicit_requirement_binding_violations(
            state,
            objective_brief,
        )
    )
    return violations


def _explicit_requirement_binding_violations(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
) -> list[Violation]:
    """确认来源映射指向 Brief 声明的实体，而不只检查字段类别。"""
    equivalent_groups = _equivalent_explicit_source_refs(objective_brief)
    violations: list[Violation] = []
    for source_ref in state.required_source_refs:
        expected_ids = _expected_source_entity_ids(objective_brief, source_ref)
        if not expected_ids:
            continue
        equivalent_refs = next(
            (group for group in equivalent_groups if source_ref in group),
            {source_ref},
        )
        evidence = _source_entity_evidence(
            state,
            objective_brief,
            equivalent_refs,
        )
        if not evidence or any(expected_ids <= item for item in evidence):
            continue
        violations.append(
            _violation(
                "EXPLICIT_REQUIREMENT_ENTITY_BINDING_MISMATCH",
                f"明确要求映射到了错误实体：{source_ref}",
                entity_ids=sorted(expected_ids),
                expected={"entity_ids": sorted(expected_ids)},
                actual={"evidence_entity_ids": [sorted(item) for item in evidence]},
                adjustable_variables=["entities", "motion_tracks", "constraints"],
            )
        )
    return violations


def _expected_source_entity_ids(
    objective_brief: ObjectivePlanningBrief,
    source_ref: str,
) -> set[str]:
    subject_prefix = "content.subjects["
    motion_prefix = "content.subject_motion["
    relation_prefix = "content.scene_design.relationships["
    layer_prefix = "content.scene_design.spatial_layers["
    try:
        if source_ref.startswith(subject_prefix):
            index = int(source_ref[len(subject_prefix):].split("]", 1)[0])
            entity_id = objective_brief.subjects[index].get("id")
            return {entity_id} if isinstance(entity_id, str) else set()
        if source_ref.startswith(motion_prefix):
            index = int(source_ref[len(motion_prefix):].split("]", 1)[0])
            entity_id = objective_brief.subject_motion[index].get("subject_id")
            return {entity_id} if isinstance(entity_id, str) else set()
        if source_ref.startswith(relation_prefix):
            index = int(source_ref[len(relation_prefix):].split("]", 1)[0])
            relationship = objective_brief.scene_design.get("relationships", [])[index]
            return {
                entity_id
                for entity_id in (
                    relationship.get("subject_id"),
                    relationship.get("reference_id"),
                )
                if isinstance(entity_id, str)
            }
        if source_ref.startswith(layer_prefix):
            index = int(source_ref[len(layer_prefix):].split("]", 1)[0])
            layer = objective_brief.scene_design.get("spatial_layers", [])[index]
            return {
                entity_id
                for entity_id in layer.get("content_ids", [])
                if isinstance(entity_id, str)
            }
    except (IndexError, TypeError, ValueError, AttributeError):
        return set()
    return set()


def _source_entity_evidence(
    state: CandidateState,
    objective_brief: ObjectivePlanningBrief,
    source_refs: set[str],
) -> list[set[str]]:
    evidence: list[set[str]] = []
    for entity in state.entities.values():
        if source_refs.intersection(entity.source_refs):
            evidence.append({entity.entity_id})
        interaction = entity.ground_interaction
        if interaction.source_ref in source_refs:
            ids = {entity.entity_id}
            if interaction.ground_entity_id:
                ids.add(interaction.ground_entity_id)
            evidence.append(ids)
    for track in state.motion_tracks.values():
        if track.source_ref not in source_refs:
            continue
        ids = {
            entity_id
            for entity_id in (track.target_entity_id, track.target_id)
            if isinstance(entity_id, str)
        }
        if track.path is not None and track.path.target_id:
            ids.add(track.path.target_id)
        evidence.append(ids)
    for constraint in state.constraints.values():
        if constraint.source_ref not in source_refs:
            continue
        parameters = constraint.parameters.model_dump(mode="python", exclude_none=True)
        ids = set(constraint.subjects)
        ids.update(_known_entity_ids(parameters, set(state.entities)))
        evidence.append(ids)

    # 类型化运动由专门 Validator 复验；这里补充其主体与目标的绑定证据。
    for motion_index, motion in enumerate(objective_brief.subject_motion):
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        motion_refs = {
            f"content.subject_motion[{motion_index}].action",
            f"content.subject_motion[{motion_index}].motion_semantics",
        }
        if not source_refs.intersection(motion_refs):
            continue
        subject_id = motion.get("subject_id")
        if not isinstance(subject_id, str):
            continue
        has_motion_evidence = any(
            track.target_entity_id == subject_id
            and track.type in {"transform", "path_follow"}
            and track.source_ref in source_refs
            for track in state.motion_tracks.values()
        )
        if has_motion_evidence:
            ids = {subject_id}
            target_id = semantics.get("target_id")
            if isinstance(target_id, str):
                ids.add(target_id)
            evidence.append(ids)
    return evidence


def _known_entity_ids(value: Any, known_ids: set[str]) -> set[str]:
    if isinstance(value, str):
        return {value} if value in known_ids else set()
    if isinstance(value, dict):
        return set().union(
            *(_known_entity_ids(item, known_ids) for item in value.values())
        )
    if isinstance(value, (list, tuple)):
        return set().union(*(_known_entity_ids(item, known_ids) for item in value))
    return set()


def _equivalent_motion_source_refs(
    objective_brief: ObjectivePlanningBrief,
) -> list[set[str]]:
    """识别同一客观运动在 Brief 中的等价来源路径。"""
    explicit_refs = {item.path for item in objective_brief.explicit_requirements}
    relationships = objective_brief.scene_design.get("relationships", [])
    events = objective_brief.timeline.get("events", [])
    groups: list[set[str]] = []
    for motion_index, motion in enumerate(objective_brief.subject_motion):
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        subject_id = motion.get("subject_id")
        target_id = semantics.get("target_id")
        action_kind = semantics.get("action_kind")
        refs = {
            ref
            for ref in (
                f"content.subject_motion[{motion_index}].action",
                f"content.subject_motion[{motion_index}].motion_semantics",
                f"content.subject_motion[{motion_index}].direction",
                f"content.subject_motion[{motion_index}].trajectory",
            )
            if ref in explicit_refs
        }
        event_id = semantics.get("timeline_event_id")
        if isinstance(event_id, str):
            for event_index, event in enumerate(events):
                if isinstance(event, dict) and event.get("id") == event_id:
                    ref = f"content.timeline.events[{event_index}]"
                    if ref in explicit_refs:
                        refs.add(ref)
        if action_kind == "orbit" and isinstance(subject_id, str) and isinstance(target_id, str):
            for relation_index, relationship in enumerate(relationships):
                if not isinstance(relationship, dict):
                    continue
                relation_type = str(relationship.get("type", "")).strip().lower()
                is_orbit = any(
                    marker in relation_type
                    for marker in ("orbit", "公转", "环绕", "绕")
                )
                if (
                    is_orbit
                    and relationship.get("subject_id") == subject_id
                    and relationship.get("reference_id") == target_id
                ):
                    ref = f"content.scene_design.relationships[{relation_index}]"
                    if ref in explicit_refs:
                        refs.add(ref)
        if len(refs) > 1:
            groups.append(refs)
    return groups


def _equivalent_explicit_source_refs(
    objective_brief: ObjectivePlanningBrief,
) -> list[set[str]]:
    return [
        *_equivalent_motion_source_refs(objective_brief),
        *_equivalent_spatial_layer_source_refs(objective_brief),
    ]


def _equivalent_spatial_layer_source_refs(
    objective_brief: ObjectivePlanningBrief,
) -> list[set[str]]:
    """Pair explicit foreground/background layers with matching distance relations."""

    explicit_refs = {item.path for item in objective_brief.explicit_requirements}
    layers = objective_brief.scene_design.get("spatial_layers", [])
    relationships = objective_brief.scene_design.get("relationships", [])
    groups: list[set[str]] = []
    for layer_index, layer in enumerate(layers):
        layer_ref = f"content.scene_design.spatial_layers[{layer_index}]"
        if not isinstance(layer, dict) or layer_ref not in explicit_refs:
            continue
        content_ids = {
            item for item in layer.get("content_ids", []) if isinstance(item, str)
        }
        layer_class = _depth_semantic_class(layer.get("layer"))
        if not content_ids or layer_class is None:
            continue
        refs = {layer_ref}
        for relation_index, relationship in enumerate(relationships):
            relation_ref = f"content.scene_design.relationships[{relation_index}]"
            if not isinstance(relationship, dict) or relation_ref not in explicit_refs:
                continue
            relation_ids = {
                item
                for item in (
                    relationship.get("subject_id"),
                    relationship.get("reference_id"),
                )
                if isinstance(item, str)
            }
            relation_class = _depth_semantic_class(
                " ".join(
                    str(relationship.get(name) or "")
                    for name in ("type", "strength")
                )
            )
            if content_ids & relation_ids and relation_class == layer_class:
                refs.add(relation_ref)
        if len(refs) > 1:
            groups.append(refs)
    return groups


def _depth_semantic_class(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if any(marker in normalized for marker in ("远", "far", "background", "后景")):
        return "far"
    if any(marker in normalized for marker in ("近", "near", "foreground", "前景")):
        return "near"
    return None


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
    has_entity_pair = all(
        isinstance(relationship.get(name), str)
        for name in ("subject_id", "reference_id")
    )
    if any(
        marker in relation
        for marker in ("远", "far", "background", "远景", "后景")
    ):
        # 欧氏距离不足以表达画面中的“远处”，还必须证明其摄影机深度在主体之后。
        return {"depth_order"} if has_entity_pair else set()
    if any(
        marker in relation
        for marker in ("近", "靠近", "旁边", "身边", "near", "beside", "adjacent")
    ):
        return {"distance_range"} if has_entity_pair else set()
    if any(
        marker in relation
        for marker in (
            "左", "右", "上方", "下方", "前方", "后方",
            "left", "right", "above", "below", "front", "behind",
        )
    ):
        return {"relative_position"} if has_entity_pair else set()
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
        return {"constraint:*", "entity_track", "ground_interaction"}
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
    for time_seconds in _timeline_frame_times(state.timeline):
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
            if not _entity_visibility_at(state, entity_id, time_seconds):
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
    if constraint.type in PER_FRAME_CONSTRAINTS:
        # 点式空间语义在整个离散时间域逐帧成立，不能靠三点探针抽查。
        frame_times = _timeline_frame_times(timeline, start=start, end=end)
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


def _timeline_frame_times(
    timeline: TimelineSpec,
    *,
    start: float = 0.0,
    end: float | None = None,
) -> list[float]:
    frame_step = timeline.fps_denominator / timeline.fps_numerator
    upper = timeline.duration_seconds if end is None else end
    return [
        frame * frame_step
        for frame in range(timeline.frame_count)
        if start <= frame * frame_step < upper
    ]


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
