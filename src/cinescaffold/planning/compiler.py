from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cinescaffold.planning.domain import CommitRequest, PlanningProfile, TransformValue
from cinescaffold.planning.geometry import (
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    sample_scalar_track,
)
from cinescaffold.planning.ir import (
    AcceptanceIR,
    CameraFrameSample,
    CameraIR,
    CameraIntrinsics,
    CameraState,
    EntityFrameSample,
    EntityIR,
    EntityState,
    IRTimeline,
    LightingIR,
    MaterialIR,
    PerFrameCameraTrack,
    PerFrameEntityTrack,
    ProvenanceIR,
    RenderIR,
    RenderOutput,
    SceneIR,
)
from cinescaffold.planning.store import canonical_hash
from cinescaffold.planning.toolkit import (
    CONSTRAINT_CATALOG_VERSION,
    FULL_VALIDATION_CHECKS,
    TOOLKIT_VERSION,
    ScenePlanningToolkit,
    _camera_state_at,
    _entity_transform_at,
)


COMPILER_VERSION = "0.5"
COMMIT_GATE_VERSION = "0.5"


@dataclass(frozen=True)
class CommitGateResult:
    status: str
    scene_ir: SceneIR | None
    scene_ir_hash: str | None
    constraint_plan: dict[str, Any]
    validation: dict[str, Any]
    violations: list[dict[str, Any]]


class SceneIRCommitGate:
    def __init__(self, toolkit: ScenePlanningToolkit) -> None:
        self.toolkit = toolkit

    def commit(
        self,
        request: CommitRequest,
        *,
        agent_run_id: str,
        trace_ref: str,
    ) -> CommitGateResult:
        state = self.toolkit.store.get(request.candidate_revision)
        validation_envelope = self.toolkit.validate_candidate(
            revision=request.candidate_revision,
            checks=FULL_VALIDATION_CHECKS,
        )
        report = validation_envelope["data"]
        constraint_plan = state.model_dump(mode="json")
        if not report["hard_pass"] or report["soft_score"] < self.toolkit.profile.minimum_soft_score:
            return CommitGateResult(
                status="rejected",
                scene_ir=None,
                scene_ir_hash=None,
                constraint_plan=constraint_plan,
                validation=report,
                violations=validation_envelope["violations"],
            )

        scene_ir = compile_scene_ir(
            self.toolkit,
            state,
            agent_run_id=agent_run_id,
            trace_ref=trace_ref,
        )
        _validate_compiled_scene_ir(scene_ir)
        scene_ir_hash = canonical_hash(scene_ir)
        self.toolkit.store.commit(request.candidate_revision)
        return CommitGateResult(
            status="success",
            scene_ir=scene_ir,
            scene_ir_hash=scene_ir_hash,
            constraint_plan=constraint_plan,
            validation=report,
            violations=[],
        )


def compile_scene_ir(
    toolkit: ScenePlanningToolkit,
    state,
    *,
    agent_run_id: str,
    trace_ref: str,
) -> SceneIR:
    profile = toolkit.profile
    timeline = IRTimeline(
        fps_numerator=state.timeline.fps_numerator,
        fps_denominator=state.timeline.fps_denominator,
        frame_start=state.timeline.frame_start,
        frame_count=state.timeline.frame_count,
        frame_end=state.timeline.frame_end,
        duration_seconds=state.timeline.duration_seconds,
    )
    frames = range(timeline.frame_start, timeline.frame_end + 1)
    world_samples: dict[str, list[TransformValue]] = {name: [] for name in state.entities}
    for frame in frames:
        time_seconds = _frame_time(frame, timeline)
        for entity_id in state.entities:
            world_samples[entity_id].append(_entity_transform_at(state, entity_id, time_seconds))

    entities: list[EntityIR] = []
    for object_index, entity_id in enumerate(sorted(state.entities), start=1):
        entity = state.entities[entity_id]
        samples: list[EntityFrameSample] = []
        for offset, frame in enumerate(range(timeline.frame_start, timeline.frame_end + 1)):
            world = world_samples[entity_id][offset]
            local = world
            if entity.parent_id:
                local = _world_to_parent_local(world, world_samples[entity.parent_id][offset])
            visible = _visibility_at(state, entity_id, _frame_time(frame, timeline))
            samples.append(
                EntityFrameSample(
                    frame=frame,
                    value=EntityState(
                        translation_m=local.translation_m,
                        rotation_quaternion_wxyz=local.rotation_quaternion_wxyz,
                        scale=local.scale,
                        render_visible=visible,
                    ),
                )
            )
        entities.append(
            EntityIR(
                entity_id=entity.entity_id,
                label=entity.label or entity.entity_id,
                role=entity.role,
                geometry=entity.proxy,
                parent_id=entity.parent_id,
                local_state_track=PerFrameEntityTrack(mode="per_frame", samples=samples),
                material_id="clay_default",
                object_index=object_index,
                tags=entity.tags,
            )
        )

    assert state.camera is not None
    camera_samples: list[CameraFrameSample] = []
    previous_quaternion = None
    for frame in range(timeline.frame_start, timeline.frame_end + 1):
        time_seconds = _frame_time(frame, timeline)
        camera_state = _camera_state_at(state, time_seconds, profile)
        assert camera_state is not None
        transform, focal_length = camera_state
        quaternion = _continuous_quaternion(
            transform.rotation_quaternion_wxyz,
            previous_quaternion,
        )
        previous_quaternion = quaternion
        camera_samples.append(
            CameraFrameSample(
                frame=frame,
                value=CameraState(
                    translation_m=transform.translation_m,
                    rotation_quaternion_wxyz=quaternion,
                    focal_length_mm=focal_length,
                ),
            )
        )
    aspect = profile.resolution_x / profile.resolution_y
    camera = CameraIR(
        camera_id=state.camera.camera_id,
        projection="perspective",
        intrinsics=CameraIntrinsics(
            sensor_width_mm=state.camera.static.sensor_width_mm,
            sensor_height_mm=state.camera.static.sensor_width_mm / aspect,
        ),
        state_track=PerFrameCameraTrack(mode="per_frame", samples=camera_samples),
    )

    constraint_plan_hash = canonical_hash(state)
    return SceneIR(
        scene_id=state.scene_id,
        timeline=timeline,
        render=RenderIR(
            resolution_x=profile.resolution_x,
            resolution_y=profile.resolution_y,
            outputs=[
                RenderOutput(output_id="clay_rgb", type="rgb_png_sequence", relative_directory="rgb"),
                RenderOutput(output_id="depth_raw", type="depth_openexr_sequence", relative_directory="depth_raw"),
                RenderOutput(output_id="object_index_raw", type="object_index_openexr_sequence", relative_directory="object_id_raw"),
                RenderOutput(output_id="clay_preview", type="h264_preview", relative_path="clay_preview.mp4"),
            ],
        ),
        materials=[
            MaterialIR(
                material_id="clay_default",
                base_color_linear_rgba=(0.65, 0.65, 0.65, 1.0),
                roughness=0.8,
                metallic=0.0,
            )
        ],
        lighting=LightingIR(
            purpose="technical_preview",
            mode="neutral_camera_rig",
            rig_id="neutral_camera_rig_v0.1",
            cast_shadows=False,
            world_color_linear_rgb=(0.08, 0.08, 0.08),
            world_strength=0.2,
            lights=[],
        ),
        entities=entities,
        camera=camera,
        acceptance=AcceptanceIR(
            constraint_plan_hash=constraint_plan_hash,
            constraints=[
                item.model_dump(mode="json", exclude_none=True)
                for item in state.constraints.values()
            ],
            required_validators=FULL_VALIDATION_CHECKS,
            sampling_profile_id=profile.profile_id,
            minimum_soft_score=profile.minimum_soft_score,
        ),
        provenance=ProvenanceIR(
            cinematic_brief_hash=toolkit.objective_brief.source_brief_sha256,
            constraint_plan_hash=constraint_plan_hash,
            candidate_revision=state.revision,
            candidate_hash=canonical_hash(state),
            commit_gate_version=COMMIT_GATE_VERSION,
            planning_toolkit_version=TOOLKIT_VERSION,
            constraint_catalog_version=CONSTRAINT_CATALOG_VERSION,
            solver="deterministic_heuristic_v0.1",
            sampling_profile_id=profile.profile_id,
            scene_ir_compiler_version=COMPILER_VERSION,
            random_seed=profile.random_seed,
            target_runtime_profile="blender_5_2_lts_v0.1",
            executor_api_version="0.1",
            expected_blender=">=5.2,<5.3",
            agent_run_id=agent_run_id,
            trace_ref=trace_ref,
        ),
    )


def _visibility_at(state, entity_id: str, time_seconds: float) -> bool:
    track = next(
        (
            item
            for item in state.motion_tracks.values()
            if item.target_entity_id == entity_id and item.type == "visibility"
        ),
        None,
    )
    if track is None:
        return True
    return bool(sample_scalar_track(track, time_seconds, 1.0))


def _world_to_parent_local(child: TransformValue, parent: TransformValue) -> TransformValue:
    parent_scale = parent.scale
    if max(parent_scale) - min(parent_scale) > 1e-8:
        raise ValueError("v0.1 不支持非均匀缩放父级的无损 TRS 分解")
    inverse_rotation = quaternion_conjugate(parent.rotation_quaternion_wxyz)
    relative = rotate_vector(
        inverse_rotation,
        tuple(a - b for a, b in zip(child.translation_m, parent.translation_m)),
    )
    scale = tuple(a / b for a, b in zip(child.scale, parent.scale))
    local_rotation = quaternion_multiply(inverse_rotation, child.rotation_quaternion_wxyz)
    return TransformValue(
        translation_m=tuple(item / parent_scale[0] for item in relative),
        rotation_quaternion_wxyz=local_rotation,
        scale=scale,
        space="local",
    )


def _frame_time(frame: int, timeline: IRTimeline) -> float:
    return (frame - timeline.frame_start) * timeline.fps_denominator / timeline.fps_numerator


def _continuous_quaternion(value, previous):
    if previous is not None and sum(a * b for a, b in zip(value, previous)) < 0:
        return tuple(-item for item in value)
    return value


def _validate_compiled_scene_ir(scene_ir: SceneIR) -> None:
    expected_frames = list(range(scene_ir.timeline.frame_start, scene_ir.timeline.frame_end + 1))
    for entity in scene_ir.entities:
        if entity.local_state_track.mode == "per_frame":
            actual = [item.frame for item in entity.local_state_track.samples]
            if actual != expected_frames:
                raise ValueError(f"Entity 逐帧轨道不完整：{entity.entity_id}")
    if scene_ir.camera.state_track.mode == "per_frame":
        actual = [item.frame for item in scene_ir.camera.state_track.samples]
        if actual != expected_frames:
            raise ValueError("Camera 逐帧轨道不完整")
    values = scene_ir.model_dump(mode="json")
    _reject_null_or_non_finite(values, "$")


def _reject_null_or_non_finite(value: Any, path: str) -> None:
    if value is None:
        # 仅允许协议中明确可空的父级和输出路径。
        if not path.endswith(".parent_id") and not path.endswith(".relative_directory") and not path.endswith(".relative_path") and not path.endswith(".size_m"):
            raise ValueError(f"Scene IR 含未解析 null：{path}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Scene IR 含非有限数：{path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_null_or_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_null_or_non_finite(item, f"{path}[{index}]")
