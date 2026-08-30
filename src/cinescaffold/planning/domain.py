from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
TimeRange = tuple[float, float]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoxGeometry(StrictModel):
    type: Literal["box"]
    size_xyz_m: Vec3

    @field_validator("size_xyz_m")
    @classmethod
    def validate_size(cls, value: Vec3) -> Vec3:
        return _positive_vector(value, "box.size_xyz_m")


class SphereGeometry(StrictModel):
    type: Literal["sphere"]
    radius_m: float = Field(gt=0)


class CapsuleGeometry(StrictModel):
    type: Literal["capsule"]
    radius_m: float = Field(gt=0)
    segment_length_m: float = Field(gt=0)
    axis: Literal["+X", "+Y", "+Z"] = "+Z"


class CylinderGeometry(StrictModel):
    type: Literal["cylinder"]
    radius_m: float = Field(gt=0)
    depth_m: float = Field(gt=0)
    axis: Literal["+X", "+Y", "+Z"] = "+Z"


class ConeGeometry(StrictModel):
    type: Literal["cone"]
    radius_bottom_m: float = Field(gt=0)
    radius_top_m: float = Field(ge=0)
    depth_m: float = Field(gt=0)
    axis: Literal["+X", "+Y", "+Z"] = "+Z"


class PlaneGeometry(StrictModel):
    type: Literal["plane"]
    size_xy_m: Vec2

    @field_validator("size_xy_m")
    @classmethod
    def validate_size(cls, value: Vec2) -> Vec2:
        if not all(math.isfinite(item) and item > 0 for item in value):
            raise ValueError("plane.size_xy_m 必须为有限正数")
        return value


ProxyGeometry = Annotated[
    BoxGeometry
    | SphereGeometry
    | CapsuleGeometry
    | CylinderGeometry
    | ConeGeometry
    | PlaneGeometry,
    Field(discriminator="type"),
]


class TransformValue(StrictModel):
    translation_m: Vec3 | None = None
    rotation_quaternion_wxyz: Quaternion | None = None
    scale: Vec3 | None = None
    space: Literal["world", "local", "camera", "target_relative"] = "world"
    target_id: str | None = None

    @field_validator("translation_m")
    @classmethod
    def validate_translation(cls, value: Vec3 | None) -> Vec3 | None:
        if value is not None and not all(math.isfinite(item) for item in value):
            raise ValueError("translation_m 必须为有限数")
        return value

    @field_validator("rotation_quaternion_wxyz")
    @classmethod
    def validate_rotation(cls, value: Quaternion | None) -> Quaternion | None:
        if value is None:
            return None
        norm = math.sqrt(sum(item * item for item in value))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
            raise ValueError("rotation_quaternion_wxyz 必须归一化")
        return value

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: Vec3 | None) -> Vec3 | None:
        return None if value is None else _positive_vector(value, "scale")

    @model_validator(mode="after")
    def validate_space_target(self) -> TransformValue:
        if self.space == "target_relative" and not self.target_id:
            raise ValueError("target_relative Transform 必须提供 target_id")
        return self


class PathSpec(StrictModel):
    representation: Literal["polyline", "sampled"] = "polyline"
    space: Literal["world", "local", "camera", "target_relative"] = "world"
    target_id: str | None = None
    control_points: list[Vec3] = Field(min_length=2)
    closed: bool = False
    parameterization: Literal["normalized_time", "arc_length"] = "normalized_time"
    orientation_mode: Literal["keep", "tangent", "look_at", "keyframed"] = "keep"


class TrackKeyframe(StrictModel):
    time_seconds: float = Field(ge=0)
    value: TransformValue | float | bool
    interpolation: Literal["step", "linear", "smooth"] = "linear"


TrackType = Literal["transform", "path_follow", "visibility", "look_at", "focal_length"]


class TrackSpec(StrictModel):
    track_id: str = Field(min_length=1)
    target_entity_id: str | None = None
    type: TrackType
    time_range_seconds: TimeRange
    keyframes: list[TrackKeyframe] = Field(default_factory=list)
    path: PathSpec | None = None
    target_id: str | None = None
    interpolation: Literal["step", "linear", "smooth"] = "linear"
    locked_components: list[str] = Field(default_factory=list)
    source_ref: str | None = None

    @model_validator(mode="after")
    def validate_track_shape(self) -> TrackSpec:
        start, end = self.time_range_seconds
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
            raise ValueError("Track time_range_seconds 必须为合法半开区间")
        if self.type == "path_follow" and self.path is None:
            raise ValueError("path_follow Track 必须提供 path")
        if self.type == "look_at" and not self.target_id:
            raise ValueError("look_at Track 必须提供 target_id")
        return self


class EntitySpec(StrictModel):
    entity_id: str = Field(min_length=1)
    label: str | None = None
    role: str
    proxy: ProxyGeometry
    parent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    locked_fields: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    solved_transform: TransformValue = Field(default_factory=TransformValue)


CONSTRAINT_KINDS = (
    "relative_position",
    "distance_range",
    "depth_order",
    "orientation_relation",
    "contact",
    "collision_clearance",
    "parent_attachment",
    "screen_region",
    "projected_size",
    "projected_scale_ratio",
    "keep_in_frame",
    "visibility_fraction",
    "occlusion_order",
    "negative_space",
    "framing",
    "look_at",
    "camera_distance",
    "view_angle",
    "focal_length_range",
    "camera_motion_direction",
    "camera_motion_smoothness",
    "position_at_time",
    "orientation_at_time",
    "motion_direction",
    "speed_range",
    "hold",
    "event_order",
    "synchronization",
    "path_adherence",
    "motion_smoothness",
)
ConstraintKind = Literal[
    "relative_position",
    "distance_range",
    "depth_order",
    "orientation_relation",
    "contact",
    "collision_clearance",
    "parent_attachment",
    "screen_region",
    "projected_size",
    "projected_scale_ratio",
    "keep_in_frame",
    "visibility_fraction",
    "occlusion_order",
    "negative_space",
    "framing",
    "look_at",
    "camera_distance",
    "view_angle",
    "focal_length_range",
    "camera_motion_direction",
    "camera_motion_smoothness",
    "position_at_time",
    "orientation_at_time",
    "motion_direction",
    "speed_range",
    "hold",
    "event_order",
    "synchronization",
    "path_adherence",
    "motion_smoothness",
]


class RelativePositionParameters(StrictModel):
    subject_id: str
    reference_id: str
    relation: Literal["left", "right", "front", "behind", "below", "above"]
    space: Literal["world", "camera", "target_relative"] = "world"
    minimum_gap: float | None = Field(default=None, ge=0)
    maximum_gap: float | None = Field(default=None, ge=0)


class DistanceRangeParameters(StrictModel):
    entity_ids: tuple[str, str]
    minimum_meters: float = Field(ge=0)
    maximum_meters: float = Field(ge=0)


class DepthOrderParameters(StrictModel):
    near_entity_id: str
    far_entity_id: str
    camera_id: str = "camera_main"
    minimum_depth_gap_meters: float | None = Field(default=None, ge=0)


class ScreenRegionParameters(StrictModel):
    entity_id: str
    region: tuple[float, float, float, float]


class ProjectedSizeParameters(StrictModel):
    entity_id: str
    measurement: Literal["height", "width", "diameter"] = "height"
    minimum: float = Field(ge=0)
    maximum: float = Field(gt=0)


class ProjectedScaleRatioParameters(StrictModel):
    numerator_entity_id: str
    denominator_entity_id: str
    measurement: Literal["height", "width", "diameter"] = "height"
    minimum_ratio: float = Field(ge=0)
    maximum_ratio: float = Field(gt=0)


class KeepInFrameParameters(StrictModel):
    entity_id: str
    minimum_inside_fraction: float = Field(ge=0, le=1)


class LookAtParameters(StrictModel):
    observer_id: str
    target_id: str
    maximum_angle_error_degrees: float = Field(default=1.0, ge=0)


class CameraDistanceParameters(StrictModel):
    camera_id: str = "camera_main"
    target_id: str
    minimum_meters: float = Field(ge=0)
    maximum_meters: float = Field(gt=0)


class FocalLengthRangeParameters(StrictModel):
    camera_id: str = "camera_main"
    minimum_mm: float = Field(gt=0)
    maximum_mm: float = Field(gt=0)


class MotionDirectionParameters(StrictModel):
    target_id: str
    direction: Literal["left", "right", "forward", "backward", "up", "down"]
    space: Literal["world", "camera", "target_relative"] = "world"
    minimum_displacement_m: float = Field(default=0.01, ge=0)


class CameraMotionDirectionParameters(StrictModel):
    camera_id: str = "camera_main"
    target_id: str | None = None
    direction: Literal[
        "left", "right", "forward", "backward", "up", "down", "push_in", "pull_out"
    ]
    space: Literal["world", "camera", "target_relative"] = "world"
    minimum_displacement_m: float = Field(default=0.01, ge=0)


class SpeedRangeParameters(StrictModel):
    target_id: str
    minimum_mps: float = Field(default=0.0, ge=0)
    maximum_mps: float = Field(gt=0)
    space: Literal["world", "camera", "target_relative"] = "world"


class PositionAtTimeParameters(StrictModel):
    target_id: str
    position_m: Vec3
    space: Literal["world", "camera", "target_relative"] = "world"
    tolerance_m: float = Field(default=0.01, ge=0)


class HoldParameters(StrictModel):
    target_id: str
    components: list[Literal["translation", "rotation", "scale", "visibility"]]
    tolerance_m: float = Field(default=1e-4, ge=0)


class UnsupportedConstraintParameters(StrictModel):
    pass


ConstraintParameters = (
    RelativePositionParameters
    | DistanceRangeParameters
    | DepthOrderParameters
    | ScreenRegionParameters
    | ProjectedSizeParameters
    | ProjectedScaleRatioParameters
    | KeepInFrameParameters
    | LookAtParameters
    | CameraDistanceParameters
    | FocalLengthRangeParameters
    | MotionDirectionParameters
    | CameraMotionDirectionParameters
    | SpeedRangeParameters
    | PositionAtTimeParameters
    | HoldParameters
    | UnsupportedConstraintParameters
)


class ConstraintSpec(StrictModel):
    constraint_id: str = Field(min_length=1)
    type: ConstraintKind
    strength: Literal["hard", "soft"]
    weight: float = Field(default=1.0, gt=0)
    subjects: list[str] = Field(default_factory=list)
    time_range_seconds: TimeRange
    parameters: ConstraintParameters
    source_status: Literal["explicit", "inferred", "default", "agent_selected", "unknown"]
    source_ref: str

    @model_validator(mode="after")
    def validate_time_range(self) -> ConstraintSpec:
        start, end = self.time_range_seconds
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
            raise ValueError("Constraint time_range_seconds 必须为合法半开区间")
        expected = {
            "relative_position": RelativePositionParameters,
            "distance_range": DistanceRangeParameters,
            "depth_order": DepthOrderParameters,
            "screen_region": ScreenRegionParameters,
            "projected_size": ProjectedSizeParameters,
            "projected_scale_ratio": ProjectedScaleRatioParameters,
            "keep_in_frame": KeepInFrameParameters,
            "look_at": LookAtParameters,
            "camera_distance": CameraDistanceParameters,
            "focal_length_range": FocalLengthRangeParameters,
            "camera_motion_direction": CameraMotionDirectionParameters,
            "speed_range": SpeedRangeParameters,
            "position_at_time": PositionAtTimeParameters,
            "motion_direction": MotionDirectionParameters,
            "hold": HoldParameters,
        }.get(self.type, UnsupportedConstraintParameters)
        if not isinstance(self.parameters, expected):
            raise ValueError(f"Constraint {self.type} 的 parameters 类型不匹配")
        return self


class CameraStatic(StrictModel):
    focal_length_mm: float | None = Field(default=None, gt=0)
    sensor_width_mm: float = Field(default=36.0, gt=0)
    focus_target_id: str | None = None
    source_refs: list[str] = Field(default_factory=list)


class CameraCandidate(StrictModel):
    camera_id: str = "camera_main"
    projection: Literal["perspective"] = "perspective"
    active: bool = True
    static: CameraStatic = Field(default_factory=CameraStatic)
    tracks: dict[str, TrackSpec] = Field(default_factory=dict)
    solved_transform: TransformValue = Field(default_factory=TransformValue)


class TimelineSpec(StrictModel):
    fps_numerator: int = Field(default=24, gt=0)
    fps_denominator: int = Field(default=1, gt=0)
    frame_start: int = 1
    frame_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)

    @property
    def frame_end(self) -> int:
        return self.frame_start + self.frame_count - 1


class Violation(StrictModel):
    id: str
    code: str
    severity: Literal["hard", "soft", "warning"]
    constraint_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    time_range_seconds: TimeRange | None = None
    expected: Any = None
    actual: Any = None
    adjustable_variables: list[str] = Field(default_factory=list)
    message: str


class ValidationReport(StrictModel):
    revision: int
    hard_pass: bool
    soft_score: float = Field(ge=0, le=1)
    checks: list[str]
    violations: list[Violation] = Field(default_factory=list)
    capability_gaps: list[str] = Field(default_factory=list)


class CandidateState(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    scene_id: str
    revision: int = 0
    timeline: TimelineSpec
    entities: dict[str, EntitySpec] = Field(default_factory=dict)
    motion_tracks: dict[str, TrackSpec] = Field(default_factory=dict)
    constraints: dict[str, ConstraintSpec] = Field(default_factory=dict)
    camera: CameraCandidate | None = None
    required_source_refs: list[str] = Field(default_factory=list)
    runner_mapped_source_refs: list[str] = Field(default_factory=list)
    validation: ValidationReport | None = None


class PlanningProfile(StrictModel):
    profile_id: str = "research_default_v0.1"
    fps_numerator: int = 24
    fps_denominator: int = 1
    default_duration_seconds: float = 6.0
    resolution_x: int = 1280
    resolution_y: int = 720
    minimum_soft_score: float = 0.75
    default_focal_length_mm: float = 35.0
    default_camera_distance_m: float = 12.0
    default_depth_gap_m: float = 12.0
    numeric_tolerance: float = Field(default=1e-8, gt=0)
    minimum_proxy_axis_projection_ratio: float = Field(default=0.30, gt=0, lt=1)
    minimum_readability_projected_extent: float = Field(default=0.01, gt=0)
    random_seed: int = 0


class CommitRequest(StrictModel):
    type: Literal["commit_request"]
    candidate_revision: int = Field(ge=0)
    summary: str


class UnsupportedResult(StrictModel):
    type: Literal["unsupported"]
    brief_paths: list[str]
    missing_capabilities: list[str]
    evidence: str


class InfeasibleResult(StrictModel):
    type: Literal["infeasible"]
    conflicting_constraint_ids: list[str]
    evidence: list[str]
    attempted_revisions: list[int]


AgentTerminal = Annotated[
    CommitRequest | UnsupportedResult | InfeasibleResult,
    Field(discriminator="type"),
]


def _positive_vector(value: tuple[float, ...], name: str):
    if not all(math.isfinite(item) and item > 0 for item in value):
        raise ValueError(f"{name} 必须为有限正数")
    return value
