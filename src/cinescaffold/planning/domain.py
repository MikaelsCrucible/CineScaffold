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
    size_xyz_m: Vec3 = Field(
        description="沿规范 X/Y/Z 三轴的完整边长，单位米；不是半尺寸",
    )

    @field_validator("size_xyz_m")
    @classmethod
    def validate_size(cls, value: Vec3) -> Vec3:
        return _positive_vector(value, "box.size_xyz_m")


class SphereGeometry(StrictModel):
    type: Literal["sphere"]
    radius_m: float = Field(gt=0, description="球半径，单位米")


class CapsuleGeometry(StrictModel):
    type: Literal["capsule"]
    radius_m: float = Field(gt=0, description="两端半球半径，单位米")
    segment_length_m: float = Field(
        gt=0,
        description="不含两端半球的中间线段长度，单位米",
    )
    axis: Literal["+X", "+Y", "+Z"] = Field(
        default="+Z",
        description="胶囊中轴在自身局部坐标中的正轴方向",
    )


class CylinderGeometry(StrictModel):
    type: Literal["cylinder"]
    radius_m: float = Field(gt=0)
    depth_m: float = Field(gt=0, description="沿 axis 的完整高度，单位米")
    axis: Literal["+X", "+Y", "+Z"] = Field(
        default="+Z",
        description="圆柱中轴在自身局部坐标中的正轴方向",
    )


class ConeGeometry(StrictModel):
    type: Literal["cone"]
    radius_bottom_m: float = Field(gt=0)
    radius_top_m: float = Field(ge=0)
    depth_m: float = Field(gt=0, description="沿 axis 的完整高度，单位米")
    axis: Literal["+X", "+Y", "+Z"] = Field(
        default="+Z",
        description="圆锥中轴在自身局部坐标中的正轴方向",
    )


class PlaneGeometry(StrictModel):
    type: Literal["plane"]
    size_xy_m: Vec2 = Field(
        description="自身局部 XY 平面的完整宽高，单位米",
    )

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
    translation_m: Vec3 | None = Field(
        default=None,
        description="米制 [x,y,z]；规范世界为右手 +Z-up，null 表示交给 Solver",
    )
    rotation_quaternion_wxyz: Quaternion | None = Field(
        default=None,
        description="归一化 [w,x,y,z]；null 表示交给 Solver",
    )
    scale: Vec3 | None = Field(
        default=None,
        description="三轴正比例；null 表示交给 Solver",
    )
    space: Literal["world", "local", "camera", "target_relative"] = Field(
        default="world",
        description="world=规范世界；local=父实体；camera=当前摄影机；target_relative=target_id 当前 Transform",
    )
    target_id: str | None = Field(
        default=None,
        description="仅 target_relative 必填；不要同时手算世界坐标",
    )

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
        if self.space != "target_relative" and self.target_id is not None:
            raise ValueError(f"{self.space} Transform 不接受 target_id")
        return self


class BasePathSpec(StrictModel):
    space: Literal["world", "local", "camera", "target_relative"] = "world"
    target_id: str | None = None
    closed: bool = False
    cycle_count: float = Field(default=1.0, gt=0)
    parameterization: Literal["normalized_time", "arc_length"] = "normalized_time"
    orientation_mode: Literal["keep"] = "keep"

    @model_validator(mode="after")
    def validate_reference_frame(self) -> BasePathSpec:
        if self.space == "target_relative" and not self.target_id:
            raise ValueError("target_relative Path 必须提供 target_id")
        if self.space != "target_relative" and self.target_id is not None:
            raise ValueError(f"{self.space} Path 不接受 target_id")
        if not math.isfinite(self.cycle_count) or self.cycle_count <= 0:
            raise ValueError("Path cycle_count 必须为有限正数")
        if not self.closed and abs(self.cycle_count - 1.0) > 1e-9:
            raise ValueError("只有闭合 Path 可以使用非 1 的 cycle_count")
        return self


class PolylinePathSpec(BasePathSpec):
    representation: Literal["polyline"] = "polyline"
    control_points: list[Vec3] = Field(min_length=2)


class SampledPathSpec(BasePathSpec):
    representation: Literal["sampled"]
    control_points: list[Vec3] = Field(min_length=2)


class CatmullRomPathSpec(BasePathSpec):
    representation: Literal["catmull_rom"]
    control_points: list[Vec3] = Field(min_length=4)


class AnalyticPathSpec(BasePathSpec):
    center_offset_m: Vec3 = (0.0, 0.0, 0.0)
    plane_normal: Vec3 = (0.0, 0.0, 1.0)
    axis_direction: Vec3 = (1.0, 0.0, 0.0)
    initial_phase_degrees: float = 0.0
    direction: Literal["counterclockwise", "clockwise"] = "counterclockwise"
    closed: Literal[True] = True

    @model_validator(mode="after")
    def validate_basis(self) -> AnalyticPathSpec:
        vectors = {
            "center_offset_m": self.center_offset_m,
            "plane_normal": self.plane_normal,
            "axis_direction": self.axis_direction,
        }
        if not all(math.isfinite(item) for value in vectors.values() for item in value):
            raise ValueError("解析 Path 的向量必须为有限数")
        normal_length = math.sqrt(sum(item * item for item in self.plane_normal))
        axis_length = math.sqrt(sum(item * item for item in self.axis_direction))
        if normal_length <= 1e-9 or axis_length <= 1e-9:
            raise ValueError("plane_normal 与 axis_direction 不得为零向量")
        cosine = abs(
            sum(
                self.plane_normal[index] * self.axis_direction[index]
                for index in range(3)
            )
            / (normal_length * axis_length)
        )
        if cosine >= 1.0 - 1e-6:
            raise ValueError("axis_direction 不得与 plane_normal 平行")
        if not math.isfinite(self.initial_phase_degrees):
            raise ValueError("initial_phase_degrees 必须为有限数")
        return self


class CirclePathSpec(AnalyticPathSpec):
    representation: Literal["circle"]
    radius_m: float = Field(gt=0)


class EllipsePathSpec(AnalyticPathSpec):
    representation: Literal["ellipse"]
    semi_major_axis_m: float = Field(gt=0)
    semi_minor_axis_m: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_axes(self) -> EllipsePathSpec:
        if self.semi_major_axis_m < self.semi_minor_axis_m:
            raise ValueError("ellipse semi_major_axis_m 不得小于 semi_minor_axis_m")
        return self


class LemniscatePathSpec(AnalyticPathSpec):
    representation: Literal["lemniscate"]
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)


PathSpec = Annotated[
    PolylinePathSpec
    | SampledPathSpec
    | CatmullRomPathSpec
    | CirclePathSpec
    | EllipsePathSpec
    | LemniscatePathSpec,
    Field(discriminator="representation"),
]


class TrackKeyframe(StrictModel):
    time_seconds: float = Field(
        ge=0,
        description="镜头起点后的秒数，必须位于冻结的半开时间域内",
    )
    value: TransformValue | float | bool = Field(
        description="值类型由 Track type 决定；Transform 坐标约定见 TransformValue",
    )
    interpolation: Literal["step", "linear", "smooth"] = Field(
        default="linear",
        description="从本关键帧到下一关键帧的插值方式",
    )


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

    @field_validator("path", mode="before")
    @classmethod
    def preserve_legacy_polyline_default(cls, value: Any) -> Any:
        if isinstance(value, dict) and "representation" not in value:
            # v0.10 以前的 Path 省略类型时固定解释为 polyline。
            return value | {"representation": "polyline"}
        return value

    @model_validator(mode="after")
    def validate_track_shape(self) -> TrackSpec:
        start, end = self.time_range_seconds
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
            raise ValueError("Track time_range_seconds 必须为合法半开区间")
        if self.type == "path_follow" and self.path is None:
            raise ValueError("path_follow Track 必须提供 path")
        if self.type == "look_at" and not self.target_id:
            raise ValueError("look_at Track 必须提供 target_id")
        if self.type != "path_follow" and self.path is not None:
            raise ValueError(f"{self.type} Track 不接受 path")
        if self.type != "look_at" and self.target_id is not None:
            raise ValueError(f"{self.type} Track 不接受 target_id")
        if self.type in {"path_follow", "look_at"} and self.keyframes:
            raise ValueError(f"{self.type} Track 不接受未使用的 keyframes")
        keyframe_times = [item.time_seconds for item in self.keyframes]
        if len(keyframe_times) != len(set(keyframe_times)):
            raise ValueError("Track 不接受时间相同的歧义关键帧")
        if self.type == "transform":
            values = [item.value for item in self.keyframes]
            if any(not isinstance(item, TransformValue) for item in values):
                raise ValueError("transform Track 的关键帧必须使用 TransformValue")
            frames = {
                (item.space, item.target_id)
                for item in values
                if isinstance(item, TransformValue)
            }
            if len(frames) > 1:
                raise ValueError("同一 transform Track 的关键帧必须使用相同参考系")
        elif self.type == "visibility" and any(
            not isinstance(item.value, bool) for item in self.keyframes
        ):
            raise ValueError("visibility Track 的关键帧值必须为 bool")
        elif self.type == "focal_length" and any(
            isinstance(item.value, bool) or not isinstance(item.value, (int, float))
            for item in self.keyframes
        ):
            raise ValueError("focal_length Track 的关键帧值必须为数值")
        return self


class GroundInteractionSpec(StrictModel):
    mode: Literal[
        "must_be_above",
        "must_touch",
        "may_intersect",
        "embedded",
        "unconstrained",
    ] = Field(
        default="must_be_above",
        description="仅存在环境地面平面时参与验证；无地面场景保持缺省即可",
    )
    ground_entity_id: str | None = None
    tolerance_m: float = Field(default=1e-3, ge=0)
    minimum_penetration_m: float | None = Field(default=None, ge=0)
    maximum_penetration_m: float | None = Field(default=None, ge=0)
    source_status: Literal["explicit", "inferred", "default", "agent_selected"] = "default"
    source_ref: str | None = None

    @model_validator(mode="after")
    def validate_penetration_range(self) -> GroundInteractionSpec:
        if self.mode == "may_intersect":
            if self.maximum_penetration_m is None or self.maximum_penetration_m <= 0:
                raise ValueError("may_intersect 必须提供正数 maximum_penetration_m")
            if self.minimum_penetration_m is not None:
                raise ValueError("may_intersect 不接受 minimum_penetration_m")
        elif self.mode == "embedded":
            if self.minimum_penetration_m is None or self.minimum_penetration_m <= 0:
                raise ValueError("embedded 必须提供正数 minimum_penetration_m")
            if self.maximum_penetration_m is None:
                raise ValueError("embedded 必须提供 maximum_penetration_m")
            if self.maximum_penetration_m < self.minimum_penetration_m:
                raise ValueError("embedded 穿入范围上下界颠倒")
        elif self.minimum_penetration_m is not None or self.maximum_penetration_m is not None:
            raise ValueError(f"{self.mode} 不接受穿入深度范围")
        if self.source_status == "explicit" and not self.source_ref:
            raise ValueError("explicit 地面交互必须提供 source_ref")
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
    ground_interaction: GroundInteractionSpec = Field(default_factory=GroundInteractionSpec)
    solved_transform: TransformValue = Field(default_factory=TransformValue)


CONSTRAINT_KINDS = (
    "relative_position",
    "distance_range",
    "surface_clearance_range",
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
    "surface_clearance_range",
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
    space: Literal["world"] = "world"
    minimum_gap: float | None = Field(default=None, ge=0)
    maximum_gap: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_gap_range(self) -> RelativePositionParameters:
        if (
            self.minimum_gap is not None
            and self.maximum_gap is not None
            and self.maximum_gap < self.minimum_gap
        ):
            raise ValueError("relative_position 间距上下界颠倒")
        return self


class DistanceRangeParameters(StrictModel):
    entity_ids: tuple[str, str]
    minimum_meters: float = Field(ge=0)
    maximum_meters: float = Field(ge=0)


class SurfaceClearanceRangeParameters(StrictModel):
    entity_ids: tuple[str, str]
    minimum_ratio: float = Field(ge=0)
    preferred_ratio: float | None = Field(default=None, ge=0)
    maximum_ratio: float = Field(ge=0)
    scale_basis: Literal["larger_directional_extent"] = (
        "larger_directional_extent"
    )
    space: Literal["world", "ground_plane"] = "ground_plane"

    @model_validator(mode="after")
    def validate_ratio_range(self) -> SurfaceClearanceRangeParameters:
        values = [self.minimum_ratio, self.maximum_ratio]
        if self.preferred_ratio is not None:
            values.append(self.preferred_ratio)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("surface_clearance_range 比例必须是有限数")
        if self.maximum_ratio < self.minimum_ratio:
            raise ValueError("surface_clearance_range 比例上下界颠倒")
        if self.preferred_ratio is not None and not (
            self.minimum_ratio <= self.preferred_ratio <= self.maximum_ratio
        ):
            raise ValueError("surface_clearance_range 偏好比例越界")
        return self


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
    space: Literal["world"] = "world"
    minimum_displacement_m: float = Field(default=0.01, ge=0)


class CameraMotionDirectionParameters(StrictModel):
    camera_id: str = "camera_main"
    target_id: str | None = None
    direction: Literal[
        "left", "right", "forward", "backward", "up", "down", "push_in", "pull_out"
    ]
    space: Literal["world", "camera"] = "world"
    minimum_displacement_m: float = Field(default=0.01, ge=0)


class SpeedRangeParameters(StrictModel):
    target_id: str
    minimum_mps: float = Field(default=0.0, ge=0)
    maximum_mps: float = Field(gt=0)
    space: Literal["world"] = "world"


class PositionAtTimeParameters(StrictModel):
    target_id: str
    position_m: Vec3
    space: Literal["world"] = "world"
    tolerance_m: float = Field(default=0.01, ge=0)


class HoldParameters(StrictModel):
    target_id: str
    components: list[
        Literal["translation", "rotation", "scale", "visibility"]
    ] = Field(min_length=1)
    tolerance_m: float = Field(default=1e-4, ge=0)
    rotation_tolerance_degrees: float = Field(default=0.01, ge=0)
    scale_tolerance: float = Field(default=1e-4, ge=0)


class UnsupportedConstraintParameters(StrictModel):
    pass


ConstraintParameters = (
    RelativePositionParameters
    | DistanceRangeParameters
    | SurfaceClearanceRangeParameters
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
            "surface_clearance_range": SurfaceClearanceRangeParameters,
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
    focal_length_mm: float | None = Field(
        default=None,
        gt=0,
        description="毫米焦距；null 表示交给 Solver",
    )
    sensor_width_mm: float = Field(
        default=36.0,
        gt=0,
        description="摄影机传感器宽度，单位毫米",
    )
    focus_target_id: str | None = Field(
        default=None,
        description="缺省观察目标；look_at Track 在其生效区间内覆盖它",
    )
    source_refs: list[str] = Field(
        default_factory=list,
        description="该摄影机静态选择映射的 Brief 字段路径",
    )


class CameraCandidate(StrictModel):
    camera_id: str = "camera_main"
    projection: Literal["perspective"] = "perspective"
    active: bool = True
    static: CameraStatic = Field(default_factory=CameraStatic)
    tracks: dict[str, TrackSpec] = Field(default_factory=dict)
    solved_transform: TransformValue = Field(default_factory=TransformValue)


class ExactDurationRequest(StrictModel):
    mode: Literal["exact"]
    seconds: float = Field(gt=0)


class RangeDurationRequest(StrictModel):
    mode: Literal["range"]
    minimum_seconds: float = Field(gt=0)
    maximum_seconds: float = Field(gt=0)


class InferredDurationRequest(StrictModel):
    mode: Literal["inferred"]


class LegacyDurationRequest(StrictModel):
    mode: Literal["legacy_frozen"]


class BriefDurationRequest(StrictModel):
    mode: Literal["cinematic_brief"]
    source_status: Literal["explicit", "inferred", "default"]
    seconds: float = Field(gt=0)


DurationRequest = Annotated[
    ExactDurationRequest
    | RangeDurationRequest
    | InferredDurationRequest
    | LegacyDurationRequest
    | BriefDurationRequest,
    Field(discriminator="mode"),
]


class DurationResolution(StrictModel):
    request: DurationRequest
    resolution_method: Literal[
        "user_exact",
        "agent_within_user_range",
        "agent_inferred",
        "legacy_frozen",
        "brief_explicit",
        "brief_inferred",
        "brief_default",
    ]
    proposed_duration_seconds: float = Field(gt=0)
    resolved_duration_seconds: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    reason: str


class TimelineSpec(StrictModel):
    fps_numerator: int = Field(default=24, gt=0)
    fps_denominator: int = Field(default=1, gt=0)
    frame_start: int = 1
    frame_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    duration_resolution: DurationResolution

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
    profile_id: str = "research_default_v0.7"
    fps_numerator: int = 24
    fps_denominator: int = 1
    resolution_x: int = 1280
    resolution_y: int = 720
    minimum_soft_score: float = 0.0
    default_focal_length_mm: float = 35.0
    default_camera_distance_m: float = 12.0
    default_depth_gap_m: float = 12.0
    far_clearance_ratio_range: tuple[float, float] = (0.5, 2.0)
    far_clearance_preferred_ratio: float = 1.0
    stationary_speed_max_mps: float = 0.0001
    slow_speed_range_mps: tuple[float, float] = (0.01, 2.0)
    medium_speed_range_mps: tuple[float, float] = (0.5, 4.0)
    fast_speed_range_mps: tuple[float, float] = (2.0, 12.0)
    minimum_orbit_plane_view_alignment: float = Field(default=0.35, gt=0, le=1)
    minimum_projected_motion_extent: float = Field(default=0.08, gt=0, le=1)
    minimum_projected_motion_scale_ratio: float = Field(default=1.2, gt=1)
    minimum_camera_motion_obliqueness_degrees: float = Field(
        default=20.0,
        gt=0,
        lt=90,
    )
    numeric_tolerance: float = Field(default=1e-8, gt=0)
    random_seed: int = 0

    @model_validator(mode="after")
    def validate_far_clearance_profile(self) -> PlanningProfile:
        minimum, maximum = self.far_clearance_ratio_range
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0
            or maximum < minimum
        ):
            raise ValueError("far_clearance_ratio_range 必须是合法有限范围")
        if not minimum <= self.far_clearance_preferred_ratio <= maximum:
            raise ValueError("far_clearance_preferred_ratio 必须位于范围内")
        return self


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
