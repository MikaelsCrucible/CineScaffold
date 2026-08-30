from __future__ import annotations

import math
from typing import Any

from cinescaffold.planning.ir import SceneIR


def validate_scene_ir_for_execution(scene_ir: SceneIR) -> None:
    """在修改 Blender 前检查跨字段与逐帧执行不变量。"""
    entity_ids = [item.entity_id for item in scene_ir.entities]
    _require_unique(entity_ids, "entity_id")
    _require_unique([item.object_index for item in scene_ir.entities], "object_index")
    material_ids = [item.material_id for item in scene_ir.materials]
    _require_unique(material_ids, "material_id")
    _require_unique([item.light_id for item in scene_ir.lighting.lights], "light_id")

    entity_set = set(entity_ids)
    material_set = set(material_ids)
    parent_map: dict[str, str | None] = {}
    for entity in scene_ir.entities:
        if entity.parent_id is not None and entity.parent_id not in entity_set:
            raise ValueError(f"Entity 父级不存在：{entity.entity_id} -> {entity.parent_id}")
        if entity.material_id not in material_set:
            raise ValueError(f"Entity 材质不存在：{entity.entity_id} -> {entity.material_id}")
        parent_map[entity.entity_id] = entity.parent_id
        _validate_entity_track(scene_ir, entity.entity_id, entity.local_state_track)
    _validate_parent_cycles(parent_map)
    _validate_camera_track(scene_ir)

    preview_outputs = [item for item in scene_ir.render.outputs if item.type == "h264_preview"]
    if len(preview_outputs) != 1 or not preview_outputs[0].relative_path:
        raise ValueError("Scene IR 必须声明唯一且带 relative_path 的 H.264 preview")
    if scene_ir.provenance.executor_api_version != "0.1":
        raise ValueError(
            f"Executor API 不兼容：{scene_ir.provenance.executor_api_version}，当前为 0.1"
        )
    _reject_non_finite(scene_ir.model_dump(mode="json"), "$")


def _validate_entity_track(scene_ir: SceneIR, entity_id: str, track) -> None:
    if track.mode == "per_frame":
        _validate_frames(scene_ir, [item.frame for item in track.samples], f"Entity {entity_id}")
        states = [item.value for item in track.samples]
    else:
        states = [track.value]
    for state in states:
        _validate_quaternion(state.rotation_quaternion_wxyz, f"Entity {entity_id}")
        if not all(value > 0 for value in state.scale):
            raise ValueError(f"Entity {entity_id} scale 必须全部为正数")


def _validate_camera_track(scene_ir: SceneIR) -> None:
    track = scene_ir.camera.state_track
    if track.mode == "per_frame":
        _validate_frames(scene_ir, [item.frame for item in track.samples], "Camera")
        states = [item.value for item in track.samples]
    else:
        states = [track.value]
    for state in states:
        _validate_quaternion(state.rotation_quaternion_wxyz, "Camera")


def _validate_frames(scene_ir: SceneIR, actual: list[int], label: str) -> None:
    expected = list(range(scene_ir.timeline.frame_start, scene_ir.timeline.frame_end + 1))
    if actual != expected:
        raise ValueError(f"{label} 逐帧轨道没有完整覆盖 timeline")


def _validate_quaternion(value: tuple[float, float, float, float], label: str) -> None:
    norm = math.sqrt(sum(item * item for item in value))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{label} 四元数没有归一化")


def _validate_parent_cycles(parent_map: dict[str, str | None]) -> None:
    for entity_id in parent_map:
        seen: set[str] = set()
        current: str | None = entity_id
        while current is not None:
            if current in seen:
                raise ValueError(f"Entity 父级形成环：{entity_id}")
            seen.add(current)
            current = parent_map[current]


def _require_unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Scene IR 中 {label} 必须唯一")


def _reject_non_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Scene IR 含非有限数：{path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")
