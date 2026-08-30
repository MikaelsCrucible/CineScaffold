from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from cinescaffold.planning.domain import (
    DurationResolution,
    GroundInteractionSpec,
    ProxyGeometry,
    Quaternion,
    StrictModel,
    Vec3,
)


class ScreenCoordinates(StrictModel):
    range: tuple[float, float] = (0.0, 1.0)
    origin: Literal["top_left"] = "top_left"
    x_direction: Literal["right"] = "right"
    y_direction: Literal["down"] = "down"


class CoordinateSystem(StrictModel):
    linear_unit: Literal["meter"] = "meter"
    angle_unit: Literal["radian"] = "radian"
    handedness: Literal["right"] = "right"
    up_axis: Literal["+Z"] = "+Z"
    rotation_representation: Literal["quaternion_wxyz"] = "quaternion_wxyz"
    transform_space: Literal["parent_local_trs"] = "parent_local_trs"
    camera_local_forward_axis: Literal["-Z"] = "-Z"
    camera_local_up_axis: Literal["+Y"] = "+Y"
    screen_coordinates: ScreenCoordinates = Field(default_factory=ScreenCoordinates)
    camera_depth: Literal["positive_forward_meters"] = "positive_forward_meters"


class IRTimeline(StrictModel):
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)
    frame_start: int
    frame_count: int = Field(gt=0)
    frame_end: int
    duration_seconds: float = Field(gt=0)
    time_domain: Literal["half_open"] = "half_open"
    motion_blur: Literal[False] = False
    duration_resolution: DurationResolution

    @model_validator(mode="after")
    def validate_frames(self) -> IRTimeline:
        if self.frame_end != self.frame_start + self.frame_count - 1:
            raise ValueError("frame_end 与 frame_count 不一致")
        expected = self.frame_count * self.fps_denominator / self.fps_numerator
        if abs(self.duration_seconds - expected) > 1e-9:
            raise ValueError("duration_seconds 与 FPS/frame_count 不一致")
        return self


class EntityState(StrictModel):
    translation_m: Vec3
    rotation_quaternion_wxyz: Quaternion
    scale: Vec3
    render_visible: bool


class CameraState(StrictModel):
    translation_m: Vec3
    rotation_quaternion_wxyz: Quaternion
    focal_length_mm: float = Field(gt=0)


class EntityFrameSample(StrictModel):
    frame: int
    value: EntityState


class CameraFrameSample(StrictModel):
    frame: int
    value: CameraState


class ConstantEntityTrack(StrictModel):
    mode: Literal["constant"]
    value: EntityState


class PerFrameEntityTrack(StrictModel):
    mode: Literal["per_frame"]
    samples: list[EntityFrameSample]


class ConstantCameraTrack(StrictModel):
    mode: Literal["constant"]
    value: CameraState


class PerFrameCameraTrack(StrictModel):
    mode: Literal["per_frame"]
    samples: list[CameraFrameSample]


EntityStateTrack = Annotated[
    ConstantEntityTrack | PerFrameEntityTrack,
    Field(discriminator="mode"),
]
CameraStateTrack = Annotated[
    ConstantCameraTrack | PerFrameCameraTrack,
    Field(discriminator="mode"),
]


class EntityIR(StrictModel):
    entity_id: str
    label: str
    role: str
    geometry: ProxyGeometry
    parent_id: str | None
    local_state_track: EntityStateTrack
    material_id: str
    object_index: int = Field(ge=1, le=32767)
    semantic_forward_axis: Literal["+Y"] = "+Y"
    tags: list[str]
    ground_interaction: GroundInteractionSpec = Field(default_factory=GroundInteractionSpec)


class CameraIntrinsics(StrictModel):
    sensor_fit: Literal["HORIZONTAL"] = "HORIZONTAL"
    sensor_width_mm: float = Field(gt=0)
    sensor_height_mm: float = Field(gt=0)
    shift_x: float = 0.0
    shift_y: float = 0.0
    clip_start_m: float = Field(default=0.1, gt=0)
    clip_end_m: float = Field(default=2000.0, gt=0)
    depth_of_field: Literal[False] = False


class CameraIR(StrictModel):
    camera_id: str
    projection: Literal["perspective"]
    intrinsics: CameraIntrinsics
    state_track: CameraStateTrack


class MaterialIR(StrictModel):
    material_id: str
    model: Literal["principled_clay"] = "principled_clay"
    base_color_linear_rgba: tuple[float, float, float, float]
    roughness: float = Field(ge=0, le=1)
    metallic: float = Field(ge=0, le=1)


class LightIR(StrictModel):
    light_id: str
    type: Literal["SUN", "AREA", "POINT"]
    translation_m: Vec3
    rotation_quaternion_wxyz: Quaternion
    color_linear_rgb: Vec3
    energy: float = Field(gt=0)
    size_m: float | None = Field(default=None, gt=0)


class LightingIR(StrictModel):
    purpose: Literal["technical_preview", "semantic"] = "semantic"
    mode: Literal["neutral_camera_rig", "explicit_world"] = "explicit_world"
    rig_id: Literal["neutral_camera_rig_v0.1"] | None = None
    cast_shadows: bool = True
    world_color_linear_rgb: Vec3
    world_strength: float = Field(ge=0)
    lights: list[LightIR]

    @model_validator(mode="after")
    def validate_lighting_mode(self) -> LightingIR:
        if self.mode == "neutral_camera_rig":
            if self.purpose != "technical_preview":
                raise ValueError("neutral_camera_rig 只能用于 technical_preview")
            if self.rig_id != "neutral_camera_rig_v0.1":
                raise ValueError("neutral_camera_rig 必须冻结 rig_id")
            if self.cast_shadows:
                raise ValueError("neutral_camera_rig 不得生成方向性投影")
            if self.lights:
                raise ValueError("neutral_camera_rig 不得混入世界空间显式灯光")
        elif self.rig_id is not None:
            raise ValueError("explicit_world 不得声明中性预览 rig_id")
        return self


class ColorManagement(StrictModel):
    display_device: Literal["sRGB"] = "sRGB"
    view_transform: Literal["Standard"] = "Standard"
    look: Literal["NONE"] = "NONE"
    exposure: float = 0.0
    gamma: float = 1.0


class RenderOutput(StrictModel):
    output_id: str
    type: Literal[
        "rgb_png_sequence",
        "depth_openexr_sequence",
        "object_index_openexr_sequence",
        "h264_preview",
    ]
    relative_directory: str | None = None
    relative_path: str | None = None

    @field_validator("relative_directory", "relative_path")
    @classmethod
    def validate_relative_path(cls, value: str | None) -> str | None:
        if value is not None and (value.startswith("/") or ".." in value.split("/")):
            raise ValueError("输出路径必须是安全相对路径")
        return value


class RenderIR(StrictModel):
    engine: Literal["BLENDER_EEVEE_NEXT"] = "BLENDER_EEVEE_NEXT"
    resolution_x: int = Field(gt=0)
    resolution_y: int = Field(gt=0)
    resolution_percentage: int = Field(default=100, ge=1, le=100)
    pixel_aspect_x: float = Field(default=1.0, gt=0)
    pixel_aspect_y: float = Field(default=1.0, gt=0)
    transparent_background: bool = False
    motion_blur: Literal[False] = False
    color_management: ColorManagement = Field(default_factory=ColorManagement)
    outputs: list[RenderOutput]


class AcceptanceIR(StrictModel):
    constraint_plan_hash: str
    constraints: list[dict[str, Any]]
    required_validators: list[str]
    sampling_profile_id: str
    minimum_soft_score: float = Field(ge=0, le=1)


class ProvenanceIR(StrictModel):
    cinematic_brief_schema_version: Literal["0.1"] = "0.1"
    cinematic_brief_hash: str
    constraint_plan_schema_version: Literal["0.1"] = "0.1"
    constraint_plan_hash: str
    candidate_revision: int = Field(ge=0)
    candidate_hash: str
    commit_gate_version: str
    planning_toolkit_version: str
    constraint_catalog_version: str
    solver: str
    sampling_profile_id: str
    scene_ir_compiler_version: str
    random_seed: int
    target_runtime_profile: str
    executor_api_version: str
    expected_blender: str
    agent_run_id: str
    trace_ref: str


class SceneIR(StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    scene_id: str
    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)
    timeline: IRTimeline
    render: RenderIR
    materials: list[MaterialIR]
    lighting: LightingIR
    entities: list[EntityIR]
    camera: CameraIR
    acceptance: AcceptanceIR
    provenance: ProvenanceIR
