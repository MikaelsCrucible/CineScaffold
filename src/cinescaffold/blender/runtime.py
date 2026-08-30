from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


EXECUTOR_VERSION = "0.3"
FLOAT_TOLERANCE = 1e-5
BLENDER_ENGINE_MAP = {"BLENDER_EEVEE_NEXT": "BLENDER_EEVEE"}
NEUTRAL_CAMERA_RIG_SPECS = (
    ("neutral_center", (1.0, 0.0, 0.0, 0.0), 0.45),
    ("neutral_left", (0.95371695, 0.0, 0.3007058, 0.0), 0.55),
    ("neutral_right", (0.95371695, 0.0, -0.3007058, 0.0), 0.55),
    ("neutral_top", (0.95371695, 0.3007058, 0.0, 0.0), 0.55),
    ("neutral_bottom", (0.95371695, -0.3007058, 0.0, 0.0), 0.55),
)


def apply_scene_ir(
    scene_ir: dict[str, Any],
    output_dir: str,
    scene_ir_hash: str,
) -> dict[str, Any]:
    """在 Blender 内从空场景确定性构建、验证并保存 Scene IR。"""
    bpy, bmesh, matrix_type = _blender_modules()
    _validate_payload_shape(scene_ir)
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    _clear_scene(bpy)
    scene = bpy.context.scene
    _apply_scene_settings(scene, scene_ir)
    materials = _create_materials(bpy, scene_ir["materials"])
    entities = _create_entities(bpy, bmesh, matrix_type, scene_ir["entities"], materials)
    _apply_parenting(entities, scene_ir["entities"])
    _apply_entity_tracks(scene, entities, scene_ir["entities"])
    camera = _create_camera(bpy, scene, scene_ir["camera"])
    _create_lighting(bpy, scene, scene_ir["lighting"], camera)

    preview_path = _preview_path(scene_ir, target_dir)
    scene["cinescaffold_scene_ir_hash"] = scene_ir_hash
    scene["cinescaffold_executor_version"] = EXECUTOR_VERSION
    scene["cinescaffold_preview_path"] = str(preview_path)
    scene["cinescaffold_render_fps"] = scene.render.fps
    scene["cinescaffold_render_fps_base"] = scene.render.fps_base
    scene["cinescaffold_render_engine"] = scene.render.engine
    scene["cinescaffold_render_resolution_x"] = scene.render.resolution_x
    scene["cinescaffold_render_resolution_y"] = scene.render.resolution_y
    scene.frame_set(scene_ir["timeline"]["frame_start"])

    snapshot = _runtime_snapshot(scene, scene_ir, entities, camera)
    validation = _validate_runtime(scene_ir, snapshot)
    snapshot_path = target_dir / "runtime_snapshot.json"
    validation_path = target_dir / "runtime_validation.json"
    blend_path = target_dir / "scene.blend"
    _write_json(snapshot_path, snapshot)
    _write_json(validation_path, validation)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)

    return {
        "status": "ok" if validation["passed"] else "runtime_mismatch",
        "executor_version": EXECUTOR_VERSION,
        "scene_ir_hash": scene_ir_hash,
        "blender_version": bpy.app.version_string,
        "validation_passed": validation["passed"],
        "violation_count": len(validation["violations"]),
        "artifacts": {
            "blend": str(blend_path),
            "runtime_snapshot": str(snapshot_path),
            "runtime_validation": str(validation_path),
        },
    }


def render_clay_video(render_profile: str = "preview") -> dict[str, Any]:
    """按固定档位渲染诊断预览或正式控制视频。"""
    bpy, _, _ = _blender_modules()
    scene = bpy.context.scene
    preview_value = scene.get("cinescaffold_preview_path")
    if not isinstance(preview_value, str) or not preview_value:
        raise ValueError("当前场景缺少 cinescaffold_preview_path")
    control_path = Path(preview_value).resolve()
    source_fps = int(scene.get("cinescaffold_render_fps", scene.render.fps))
    source_fps_base = float(scene.get("cinescaffold_render_fps_base", scene.render.fps_base))
    source_engine = str(scene.get("cinescaffold_render_engine", scene.render.engine))
    source_resolution_x = int(
        scene.get("cinescaffold_render_resolution_x", scene.render.resolution_x)
    )
    source_resolution_y = int(
        scene.get("cinescaffold_render_resolution_y", scene.render.resolution_y)
    )
    plan = _render_profile_plan(
        render_profile,
        frame_start=scene.frame_start,
        frame_end=scene.frame_end,
        fps=source_fps,
        fps_base=source_fps_base,
        render_engine=source_engine,
        resolution_x=source_resolution_x,
        resolution_y=source_resolution_y,
    )
    output_path = (
        control_path.parent / "diagnostic_preview.mp4"
        if render_profile == "preview"
        else control_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene.frame_step = plan["frame_step"]
    scene.render.engine = plan["render_engine"]
    if render_profile == "preview":
        _configure_workbench_preview(scene)
    scene.render.fps = source_fps
    scene.render.fps_base = plan["fps_base"]
    scene.render.resolution_x = plan["resolution_x"]
    scene.render.resolution_y = plan["resolution_y"]
    scene.render.resolution_percentage = 100

    scene.render.filepath = str(output_path)
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "NONE"
    bpy.ops.render.render(animation=True)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"白模视频没有生成：{output_path}")
    return {
        "status": "ok",
        "render_profile": render_profile,
        "render_engine": scene.render.engine,
        "scene_ir_hash": scene.get("cinescaffold_scene_ir_hash", ""),
        "artifact": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "frame_step": scene.frame_step,
        "rendered_frame_count": plan["rendered_frame_count"],
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
    }


def render_clay_preview() -> dict[str, Any]:
    """兼容旧入口；旧入口保持完整控制视频语义。"""
    return render_clay_video("control")


def _render_profile_plan(
    render_profile: str,
    *,
    frame_start: int,
    frame_end: int,
    fps: int,
    fps_base: float,
    render_engine: str,
    resolution_x: int,
    resolution_y: int,
) -> dict[str, Any]:
    if render_profile not in {"preview", "control"}:
        raise ValueError(f"未知渲染档位：{render_profile}")
    source_frame_count = frame_end - frame_start + 1
    if render_profile == "control":
        return {
            "frame_step": 1,
            "rendered_frame_count": source_frame_count,
            "fps_base": fps_base,
            "render_engine": render_engine,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
        }

    frame_step = 2
    rendered_frame_count = (source_frame_count + frame_step - 1) // frame_step
    source_duration = source_frame_count * fps_base / fps
    output_fps = rendered_frame_count / source_duration
    return {
        "frame_step": frame_step,
        "rendered_frame_count": rendered_frame_count,
        "fps_base": fps / output_fps,
        "render_engine": "BLENDER_WORKBENCH",
        "resolution_x": max(1, round(resolution_x / 2)),
        "resolution_y": max(1, round(resolution_y / 2)),
    }


def _configure_workbench_preview(scene) -> None:
    """使用固定的中性建模视图，避免引入创作性照明。"""
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "MATERIAL"
    shading.show_shadows = True
    shading.show_cavity = True
    shading.cavity_type = "WORLD"
    shading.show_specular_highlight = False
    shading.background_type = "WORLD"


def _blender_modules():
    try:
        import bmesh  # type: ignore[import-not-found]
        import bpy  # type: ignore[import-not-found]
        from mathutils import Matrix  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - 只在 Blender 内运行
        raise RuntimeError("该入口必须在 Blender Python 中运行") from error
    return bpy, bmesh, Matrix


def _validate_payload_shape(scene_ir: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "timeline",
        "render",
        "materials",
        "lighting",
        "entities",
        "camera",
    }
    missing = sorted(required - set(scene_ir))
    if missing:
        raise ValueError(f"Scene IR 缺少字段：{', '.join(missing)}")
    if scene_ir["schema_version"] != "0.1":
        raise ValueError(f"不支持的 Scene IR 版本：{scene_ir['schema_version']}")


def _clear_scene(bpy) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for datablock in list(collection):
            collection.remove(datablock)


def _apply_scene_settings(scene, scene_ir: dict[str, Any]) -> None:
    timeline = scene_ir["timeline"]
    render = scene_ir["render"]
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.frame_start = timeline["frame_start"]
    scene.frame_end = timeline["frame_end"]
    scene.render.fps = timeline["fps_numerator"]
    scene.render.fps_base = float(timeline["fps_denominator"])
    scene.render.engine = _blender_render_engine(render["engine"])
    scene.render.resolution_x = render["resolution_x"]
    scene.render.resolution_y = render["resolution_y"]
    scene.render.resolution_percentage = render["resolution_percentage"]
    scene.render.pixel_aspect_x = render["pixel_aspect_x"]
    scene.render.pixel_aspect_y = render["pixel_aspect_y"]
    scene.render.film_transparent = render["transparent_background"]
    scene.render.use_file_extension = True

    color = render["color_management"]
    scene.display_settings.display_device = color["display_device"]
    scene.view_settings.view_transform = color["view_transform"]
    scene.view_settings.look = "None" if color["look"] == "NONE" else color["look"]
    scene.view_settings.exposure = color["exposure"]
    scene.view_settings.gamma = color["gamma"]


def _create_materials(bpy, material_specs: list[dict[str, Any]]) -> dict[str, Any]:
    materials: dict[str, Any] = {}
    for spec in material_specs:
        material = bpy.data.materials.new(name=f"CS_MAT_{spec['material_id']}")
        material.use_nodes = True
        material["cinescaffold_id"] = spec["material_id"]
        node = material.node_tree.nodes.get("Principled BSDF")
        if node is None:
            raise RuntimeError("Principled BSDF 节点不存在")
        node.inputs["Base Color"].default_value = spec["base_color_linear_rgba"]
        node.inputs["Roughness"].default_value = spec["roughness"]
        node.inputs["Metallic"].default_value = spec["metallic"]
        materials[spec["material_id"]] = material
    return materials


def _create_entities(bpy, bmesh, matrix_type, specs, materials) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    for spec in specs:
        entity_id = spec["entity_id"]
        mesh = _create_proxy_mesh(bpy, bmesh, matrix_type, entity_id, spec["geometry"])
        obj = bpy.data.objects.new(name=f"CS_ENTITY_{entity_id}", object_data=mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.rotation_mode = "QUATERNION"
        obj.pass_index = spec["object_index"]
        obj["cinescaffold_id"] = entity_id
        obj["cinescaffold_kind"] = "entity"
        obj["cinescaffold_geometry_json"] = json.dumps(spec["geometry"], sort_keys=True)
        obj["cinescaffold_role"] = spec["role"]
        material_id = spec["material_id"]
        if material_id not in materials:
            raise ValueError(f"Entity 引用不存在的材质：{material_id}")
        obj.data.materials.append(materials[material_id])
        entities[entity_id] = obj
    return entities


def _create_proxy_mesh(bpy, bmesh, matrix_type, entity_id: str, geometry: dict[str, Any]):
    mesh = bpy.data.meshes.new(name=f"CS_MESH_{entity_id}")
    bm = bmesh.new()
    geometry_type = geometry["type"]
    if geometry_type == "box":
        sx, sy, sz = geometry["size_xyz_m"]
        bmesh.ops.create_cube(
            bm,
            size=1.0,
            matrix=matrix_type.Diagonal((sx, sy, sz, 1.0)),
        )
    elif geometry_type == "sphere":
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=32,
            v_segments=16,
            radius=geometry["radius_m"],
        )
    elif geometry_type == "capsule":
        _create_capsule(bmesh, bm, matrix_type, geometry)
    elif geometry_type in {"cylinder", "cone"}:
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=False,
            segments=32,
            radius1=geometry.get("radius_m", geometry.get("radius_bottom_m")),
            radius2=geometry.get("radius_m", geometry.get("radius_top_m")),
            depth=geometry["depth_m"],
            matrix=_axis_matrix(matrix_type, geometry["axis"]),
        )
    elif geometry_type == "plane":
        sx, sy = geometry["size_xy_m"]
        bmesh.ops.create_grid(
            bm,
            x_segments=1,
            y_segments=1,
            size=1.0,
            matrix=matrix_type.Diagonal((sx / 2.0, sy / 2.0, 1.0, 1.0)),
        )
    else:
        raise ValueError(f"不支持的 ProxyGeometry：{geometry_type}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh


def _create_capsule(bmesh, bm, matrix_type, geometry: dict[str, Any]) -> None:
    radius = geometry["radius_m"]
    segment = geometry["segment_length_m"]
    axis_matrix = _axis_matrix(matrix_type, geometry["axis"])
    bmesh.ops.create_cone(
        bm,
        cap_ends=True,
        cap_tris=False,
        segments=32,
        radius1=radius,
        radius2=radius,
        depth=segment,
        matrix=axis_matrix,
    )
    for offset in (-segment / 2.0, segment / 2.0):
        bmesh.ops.create_uvsphere(
            bm,
            u_segments=32,
            v_segments=16,
            radius=radius,
            matrix=axis_matrix @ matrix_type.Translation((0.0, 0.0, offset)),
        )


def _axis_matrix(matrix_type, axis: str):
    if axis == "+Z":
        return matrix_type.Identity(4)
    if axis == "+X":
        return matrix_type.Rotation(math.pi / 2.0, 4, "Y")
    if axis == "+Y":
        return matrix_type.Rotation(-math.pi / 2.0, 4, "X")
    raise ValueError(f"不支持的几何主轴：{axis}")


def _apply_parenting(entities: dict[str, Any], specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        parent_id = spec["parent_id"]
        if parent_id is None:
            continue
        if parent_id not in entities:
            raise ValueError(f"Entity 引用不存在的父级：{parent_id}")
        obj = entities[spec["entity_id"]]
        obj.parent = entities[parent_id]
        obj.matrix_parent_inverse.identity()


def _apply_entity_tracks(scene, entities: dict[str, Any], specs: list[dict[str, Any]]) -> None:
    frames = range(scene.frame_start, scene.frame_end + 1)
    for spec in specs:
        obj = entities[spec["entity_id"]]
        track = spec["local_state_track"]
        for frame in frames:
            state = _track_value(track, frame)
            _set_object_state(obj, state)
            if track["mode"] == "per_frame":
                obj.keyframe_insert("location", frame=frame)
                obj.keyframe_insert("rotation_quaternion", frame=frame)
                obj.keyframe_insert("scale", frame=frame)
                obj.keyframe_insert("hide_render", frame=frame)


def _set_object_state(obj, state: dict[str, Any]) -> None:
    obj.location = state["translation_m"]
    obj.rotation_quaternion = state["rotation_quaternion_wxyz"]
    obj.scale = state["scale"]
    obj.hide_render = not state["render_visible"]


def _create_camera(bpy, scene, spec: dict[str, Any]):
    data = bpy.data.cameras.new(name=f"CS_CAMERA_DATA_{spec['camera_id']}")
    obj = bpy.data.objects.new(name=f"CS_CAMERA_{spec['camera_id']}", object_data=data)
    scene.collection.objects.link(obj)
    obj.rotation_mode = "QUATERNION"
    obj["cinescaffold_id"] = spec["camera_id"]
    obj["cinescaffold_kind"] = "camera"
    data.type = "PERSP"
    intrinsics = spec["intrinsics"]
    data.sensor_fit = intrinsics["sensor_fit"]
    data.sensor_width = intrinsics["sensor_width_mm"]
    data.sensor_height = intrinsics["sensor_height_mm"]
    data.shift_x = intrinsics["shift_x"]
    data.shift_y = intrinsics["shift_y"]
    data.clip_start = intrinsics["clip_start_m"]
    data.clip_end = intrinsics["clip_end_m"]
    data.dof.use_dof = intrinsics["depth_of_field"]

    track = spec["state_track"]
    for frame in range(scene.frame_start, scene.frame_end + 1):
        state = _track_value(track, frame)
        obj.location = state["translation_m"]
        obj.rotation_quaternion = state["rotation_quaternion_wxyz"]
        data.lens = state["focal_length_mm"]
        if track["mode"] == "per_frame":
            obj.keyframe_insert("location", frame=frame)
            obj.keyframe_insert("rotation_quaternion", frame=frame)
            data.keyframe_insert("lens", frame=frame)
    scene.camera = obj
    return obj


def _create_lighting(bpy, scene, spec: dict[str, Any], camera) -> None:
    world = bpy.data.worlds.new(name="CS_WORLD")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("World Background 节点不存在")
    background.inputs["Color"].default_value = (*spec["world_color_linear_rgb"], 1.0)
    background.inputs["Strength"].default_value = spec["world_strength"]
    scene.world = world
    scene["cinescaffold_lighting_purpose"] = spec["purpose"]
    scene["cinescaffold_lighting_mode"] = spec["mode"]
    scene["cinescaffold_lighting_rig_id"] = spec["rig_id"] or ""
    scene["cinescaffold_cast_shadows"] = spec["cast_shadows"]
    if spec["mode"] == "neutral_camera_rig":
        _create_neutral_camera_rig(bpy, scene, camera)
        return
    for light_spec in spec["lights"]:
        data = bpy.data.lights.new(
            name=f"CS_LIGHT_DATA_{light_spec['light_id']}",
            type=light_spec["type"],
        )
        data.color = light_spec["color_linear_rgb"]
        data.energy = light_spec["energy"]
        data.use_shadow = spec["cast_shadows"]
        if light_spec["size_m"] is not None and light_spec["type"] == "AREA":
            data.shape = "DISK"
            data.size = light_spec["size_m"]
        obj = bpy.data.objects.new(name=f"CS_LIGHT_{light_spec['light_id']}", object_data=data)
        scene.collection.objects.link(obj)
        obj.rotation_mode = "QUATERNION"
        obj.location = light_spec["translation_m"]
        obj.rotation_quaternion = light_spec["rotation_quaternion_wxyz"]
        obj["cinescaffold_id"] = light_spec["light_id"]
        obj["cinescaffold_kind"] = "light"


def _create_neutral_camera_rig(bpy, scene, camera) -> None:
    """创建随摄影机移动的对称无影技术灯组。"""
    for light_id, local_rotation, energy in NEUTRAL_CAMERA_RIG_SPECS:
        data = bpy.data.lights.new(name=f"CS_LIGHT_DATA_{light_id}", type="SUN")
        data.color = (1.0, 1.0, 1.0)
        data.energy = energy
        data.use_shadow = False
        obj = bpy.data.objects.new(name=f"CS_LIGHT_{light_id}", object_data=data)
        scene.collection.objects.link(obj)
        obj.parent = camera
        obj.matrix_parent_inverse.identity()
        obj.rotation_mode = "QUATERNION"
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_quaternion = local_rotation
        obj["cinescaffold_id"] = light_id
        obj["cinescaffold_kind"] = "light"
        obj["cinescaffold_lighting_purpose"] = "technical_preview"


def _track_value(track: dict[str, Any], frame: int) -> dict[str, Any]:
    if track["mode"] == "constant":
        return track["value"]
    offset = frame - track["samples"][0]["frame"]
    sample = track["samples"][offset]
    if sample["frame"] != frame:
        raise ValueError(f"逐帧轨道在第 {frame} 帧不连续")
    return sample["value"]


def _preview_path(scene_ir: dict[str, Any], output_dir: Path) -> Path:
    outputs = [item for item in scene_ir["render"]["outputs"] if item["type"] == "h264_preview"]
    if len(outputs) != 1 or not outputs[0].get("relative_path"):
        raise ValueError("Scene IR 必须声明唯一 H.264 preview 输出")
    candidate = (output_dir / outputs[0]["relative_path"]).resolve()
    if output_dir != candidate and output_dir not in candidate.parents:
        raise ValueError("H.264 preview 输出越过运行目录")
    return candidate


def _runtime_snapshot(scene, scene_ir, entities, camera) -> dict[str, Any]:
    entity_frames: dict[str, list[dict[str, Any]]] = {entity_id: [] for entity_id in entities}
    camera_frames: list[dict[str, Any]] = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        for entity_id, obj in entities.items():
            entity_frames[entity_id].append(
                {
                    "frame": frame,
                    "translation_m": list(obj.location),
                    "rotation_quaternion_wxyz": list(obj.rotation_quaternion),
                    "scale": list(obj.scale),
                    "render_visible": not obj.hide_render,
                }
            )
        camera_frames.append(
            {
                "frame": frame,
                "translation_m": list(camera.location),
                "rotation_quaternion_wxyz": list(camera.rotation_quaternion),
                "focal_length_mm": camera.data.lens,
            }
        )
    bpy, _, _ = _blender_modules()
    scene_entity_ids = sorted(
        str(obj.get("cinescaffold_id"))
        for obj in scene.objects
        if obj.get("cinescaffold_kind") == "entity"
    )
    unexpected_renderable_objects = sorted(
        obj.name
        for obj in scene.objects
        if obj.type == "MESH" and obj.get("cinescaffold_kind") != "entity" and not obj.hide_render
    )
    return {
        "snapshot_version": "0.2",
        "scene_ir_hash": scene.get("cinescaffold_scene_ir_hash", ""),
        "blender_version": bpy.app.version_string,
        "timeline": {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "fps_numerator": scene.render.fps,
            "fps_denominator": scene.render.fps_base,
        },
        "render": {
            "engine": scene.render.engine,
            "resolution_x": scene.render.resolution_x,
            "resolution_y": scene.render.resolution_y,
            "resolution_percentage": scene.render.resolution_percentage,
            "pixel_aspect_x": scene.render.pixel_aspect_x,
            "pixel_aspect_y": scene.render.pixel_aspect_y,
        },
        "lighting": _lighting_snapshot(scene),
        "scene_entity_ids": scene_entity_ids,
        "unexpected_renderable_objects": unexpected_renderable_objects,
        "entities": [
            {
                "entity_id": spec["entity_id"],
                "parent_id": spec["parent_id"],
                "object_index": entities[spec["entity_id"]].pass_index,
                "geometry": json.loads(entities[spec["entity_id"]]["cinescaffold_geometry_json"]),
                "mesh": _mesh_snapshot(entities[spec["entity_id"]]),
                "frames": entity_frames[spec["entity_id"]],
            }
            for spec in scene_ir["entities"]
        ],
        "camera": {
            "camera_id": scene_ir["camera"]["camera_id"],
            "intrinsics": {
                "sensor_fit": camera.data.sensor_fit,
                "sensor_width_mm": camera.data.sensor_width,
                "sensor_height_mm": camera.data.sensor_height,
                "shift_x": camera.data.shift_x,
                "shift_y": camera.data.shift_y,
                "clip_start_m": camera.data.clip_start,
                "clip_end_m": camera.data.clip_end,
            },
            "frames": camera_frames,
        },
    }


def _validate_runtime(scene_ir: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    expected_entities = {item["entity_id"]: item for item in scene_ir["entities"]}
    actual_entities = {item["entity_id"]: item for item in snapshot["entities"]}
    actual_scene_ids = set(snapshot["scene_entity_ids"])
    if set(expected_entities) != actual_scene_ids:
        violations.append(_violation("identity", sorted(expected_entities), sorted(actual_scene_ids)))
    if snapshot["unexpected_renderable_objects"]:
        violations.append(
            _violation("unexpected_renderable_objects", [], snapshot["unexpected_renderable_objects"])
        )

    for entity_id in sorted(set(expected_entities) & set(actual_entities)):
        expected = expected_entities[entity_id]
        actual = actual_entities[entity_id]
        if expected["parent_id"] != actual["parent_id"]:
            violations.append(
                _violation(f"entities.{entity_id}.parent_id", expected["parent_id"], actual["parent_id"])
            )
        if expected["object_index"] != actual["object_index"]:
            violations.append(
                _violation(
                    f"entities.{entity_id}.object_index",
                    expected["object_index"],
                    actual["object_index"],
                )
            )
        if expected["geometry"] != actual["geometry"]:
            violations.append(
                _violation(f"entities.{entity_id}.geometry", expected["geometry"], actual["geometry"])
            )
        _validate_mesh_geometry(
            expected["geometry"],
            actual["mesh"],
            f"entities.{entity_id}.mesh",
            violations,
        )
        for frame_state in actual["frames"]:
            frame = frame_state["frame"]
            expected_state = _track_value(expected["local_state_track"], frame)
            _compare_state(
                violations,
                f"entities.{entity_id}.frames.{frame}",
                expected_state,
                frame_state,
            )

    timeline = scene_ir["timeline"]
    actual_timeline = snapshot["timeline"]
    for key in ("frame_start", "frame_end", "fps_numerator", "fps_denominator"):
        if not _close(timeline[key], actual_timeline[key]):
            violations.append(_violation(f"timeline.{key}", timeline[key], actual_timeline[key]))

    expected_render = scene_ir["render"]
    actual_render = snapshot["render"]
    expected_engine = _blender_render_engine(expected_render["engine"])
    if expected_engine != actual_render["engine"]:
        violations.append(_violation("render.engine", expected_engine, actual_render["engine"]))
    for key in (
        "resolution_x",
        "resolution_y",
        "resolution_percentage",
        "pixel_aspect_x",
        "pixel_aspect_y",
    ):
        if not _close(expected_render[key], actual_render[key]):
            violations.append(_violation(f"render.{key}", expected_render[key], actual_render[key]))

    _validate_lighting(
        scene_ir["lighting"],
        snapshot["lighting"],
        violations,
        camera_id=scene_ir["camera"]["camera_id"],
    )

    camera_spec = scene_ir["camera"]
    for frame_state in snapshot["camera"]["frames"]:
        frame = frame_state["frame"]
        expected_state = _track_value(camera_spec["state_track"], frame)
        _compare_state(violations, f"camera.frames.{frame}", expected_state, frame_state)
    for key, expected in camera_spec["intrinsics"].items():
        if key == "depth_of_field":
            continue
        actual = snapshot["camera"]["intrinsics"][key]
        equal = _close(expected, actual) if isinstance(expected, (int, float)) else expected == actual
        if not equal:
            violations.append(_violation(f"camera.intrinsics.{key}", expected, actual))

    return {
        "validation_version": "0.2",
        "passed": not violations,
        "float_tolerance": FLOAT_TOLERANCE,
        "violations": violations,
    }


def _mesh_snapshot(obj) -> dict[str, Any]:
    coordinates = [tuple(vertex.co) for vertex in obj.data.vertices]
    if coordinates:
        bounds = [
            max(point[axis] for point in coordinates) - min(point[axis] for point in coordinates)
            for axis in range(3)
        ]
    else:
        bounds = [0.0, 0.0, 0.0]
    return {
        "vertex_count": len(obj.data.vertices),
        "polygon_count": len(obj.data.polygons),
        "local_bounds_size_m": bounds,
    }


def _validate_mesh_geometry(
    geometry: dict[str, Any],
    mesh: dict[str, Any],
    path: str,
    violations: list[dict[str, Any]],
) -> None:
    expected_bounds = _expected_local_bounds(geometry)
    actual_bounds = mesh.get("local_bounds_size_m", [])
    if mesh.get("vertex_count", 0) <= 0 or mesh.get("polygon_count", 0) <= 0:
        violations.append(
            _violation(
                f"{path}.topology",
                {"minimum_vertex_count": 1, "minimum_polygon_count": 1},
                {
                    "vertex_count": mesh.get("vertex_count"),
                    "polygon_count": mesh.get("polygon_count"),
                },
            )
        )
    if len(actual_bounds) != 3 or any(
        not _close(expected, actual)
        for expected, actual in zip(expected_bounds, actual_bounds)
    ):
        violations.append(
            _violation(f"{path}.local_bounds_size_m", expected_bounds, actual_bounds)
        )


def _expected_local_bounds(geometry: dict[str, Any]) -> list[float]:
    geometry_type = geometry["type"]
    if geometry_type == "box":
        return list(geometry["size_xyz_m"])
    if geometry_type == "sphere":
        diameter = 2.0 * geometry["radius_m"]
        return [diameter, diameter, diameter]
    if geometry_type == "plane":
        return [*geometry["size_xy_m"], 0.0]
    if geometry_type == "capsule":
        radius = geometry["radius_m"]
        values = [2.0 * radius, 2.0 * radius, geometry["segment_length_m"] + 2.0 * radius]
        return _align_bounds(values, geometry["axis"])
    radius = geometry.get(
        "radius_m",
        max(geometry.get("radius_bottom_m", 0.0), geometry.get("radius_top_m", 0.0)),
    )
    values = [2.0 * radius, 2.0 * radius, geometry["depth_m"]]
    return _align_bounds(values, geometry["axis"])


def _align_bounds(values: list[float], axis: str) -> list[float]:
    if axis == "+Z":
        return values
    if axis == "+X":
        return [values[2], values[1], values[0]]
    if axis == "+Y":
        return [values[0], values[2], values[1]]
    raise ValueError(f"不支持的几何主轴：{axis}")


def _lighting_snapshot(scene) -> dict[str, Any]:
    background = scene.world.node_tree.nodes.get("Background") if scene.world else None
    lights = []
    for obj in sorted(
        (item for item in scene.objects if item.get("cinescaffold_kind") == "light"),
        key=lambda item: str(item.get("cinescaffold_id")),
    ):
        lights.append(
            {
                "light_id": str(obj.get("cinescaffold_id")),
                "type": obj.data.type,
                "parent_id": (
                    str(obj.parent.get("cinescaffold_id"))
                    if obj.parent is not None
                    else None
                ),
                "translation_m": list(obj.location),
                "local_rotation_quaternion_wxyz": list(obj.rotation_quaternion),
                "energy": obj.data.energy,
                "color_linear_rgb": list(obj.data.color),
                "cast_shadows": bool(obj.data.use_shadow),
                "size_m": obj.data.size if obj.data.type == "AREA" else None,
            }
        )
    return {
        "purpose": scene.get("cinescaffold_lighting_purpose", ""),
        "mode": scene.get("cinescaffold_lighting_mode", ""),
        "rig_id": scene.get("cinescaffold_lighting_rig_id", "") or None,
        "cast_shadows": bool(scene.get("cinescaffold_cast_shadows", True)),
        "world_color_linear_rgb": (
            list(background.inputs["Color"].default_value[:3]) if background else None
        ),
        "world_strength": background.inputs["Strength"].default_value if background else None,
        "lights": lights,
    }


def _validate_lighting(
    expected: dict[str, Any],
    actual: dict[str, Any],
    violations: list[dict[str, Any]],
    *,
    camera_id: str,
) -> None:
    for key in ("purpose", "mode", "rig_id", "cast_shadows"):
        if expected[key] != actual[key]:
            violations.append(_violation(f"lighting.{key}", expected[key], actual[key]))
    for key in ("world_color_linear_rgb", "world_strength"):
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, list):
            equal = all(_close(left, right) for left, right in zip(expected_value, actual_value))
        else:
            equal = _close(expected_value, actual_value)
        if not equal:
            violations.append(_violation(f"lighting.{key}", expected_value, actual_value))

    actual_lights = {item["light_id"]: item for item in actual["lights"]}
    if expected["mode"] == "neutral_camera_rig":
        expected_ids = {item[0] for item in NEUTRAL_CAMERA_RIG_SPECS}
        if set(actual_lights) != expected_ids:
            violations.append(
                _violation("lighting.neutral_rig_ids", sorted(expected_ids), sorted(actual_lights))
            )
        for light_id, rotation, energy in NEUTRAL_CAMERA_RIG_SPECS:
            light = actual_lights.get(light_id)
            if light is None:
                continue
            if light["type"] != "SUN" or light["parent_id"] != camera_id:
                violations.append(
                    _violation(
                        f"lighting.lights.{light_id}.attachment",
                        {"type": "SUN", "parent_id": camera_id},
                        {"type": light["type"], "parent_id": light["parent_id"]},
                    )
                )
            if light["cast_shadows"]:
                violations.append(
                    _violation(f"lighting.lights.{light_id}.cast_shadows", False, True)
                )
            if not _close(light["energy"], energy) or not all(
                _close(left, right)
                for left, right in zip(light["local_rotation_quaternion_wxyz"], rotation)
            ):
                violations.append(
                    _violation(
                        f"lighting.lights.{light_id}.rig_values",
                        {"rotation": rotation, "energy": energy},
                        {
                            "rotation": light["local_rotation_quaternion_wxyz"],
                            "energy": light["energy"],
                        },
                    )
                )
        return

    expected_lights = {item["light_id"]: item for item in expected["lights"]}
    if set(actual_lights) != set(expected_lights):
        violations.append(
            _violation(
                "lighting.explicit_light_ids",
                sorted(expected_lights),
                sorted(actual_lights),
            )
        )
    for light_id in sorted(set(expected_lights) & set(actual_lights)):
        expected_light = expected_lights[light_id]
        actual_light = actual_lights[light_id]
        scalar_fields = ("energy", "size_m")
        vector_fields = (
            "translation_m",
            "rotation_quaternion_wxyz",
            "color_linear_rgb",
        )
        if expected_light["type"] != actual_light["type"]:
            violations.append(
                _violation(
                    f"lighting.lights.{light_id}.type",
                    expected_light["type"],
                    actual_light["type"],
                )
            )
        for key in scalar_fields:
            expected_value = expected_light[key]
            actual_value = actual_light[key]
            if expected_value is None and actual_value is None:
                continue
            if expected_value is None or actual_value is None or not _close(
                expected_value, actual_value
            ):
                violations.append(
                    _violation(
                        f"lighting.lights.{light_id}.{key}",
                        expected_value,
                        actual_value,
                    )
                )
        for key in vector_fields:
            actual_key = (
                "local_rotation_quaternion_wxyz"
                if key == "rotation_quaternion_wxyz"
                else key
            )
            if not all(
                _close(left, right)
                for left, right in zip(expected_light[key], actual_light[actual_key])
            ):
                violations.append(
                    _violation(
                        f"lighting.lights.{light_id}.{key}",
                        expected_light[key],
                        actual_light[actual_key],
                    )
                )
        if actual_light["cast_shadows"] != expected["cast_shadows"]:
            violations.append(
                _violation(
                    f"lighting.lights.{light_id}.cast_shadows",
                    expected["cast_shadows"],
                    actual_light["cast_shadows"],
                )
            )


def _compare_state(
    violations: list[dict[str, Any]],
    path: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> None:
    for key in ("translation_m", "scale"):
        if key in expected and not all(
            _close(left, right) for left, right in zip(expected[key], actual[key])
        ):
            violations.append(_violation(f"{path}.{key}", expected[key], actual[key]))
    quaternion_key = "rotation_quaternion_wxyz"
    if quaternion_key in expected:
        dot = abs(
            sum(
                left * right
                for left, right in zip(expected[quaternion_key], actual[quaternion_key])
            )
        )
        if abs(1.0 - dot) > FLOAT_TOLERANCE:
            violations.append(
                _violation(f"{path}.{quaternion_key}", expected[quaternion_key], actual[quaternion_key])
            )
    for key in ("render_visible", "focal_length_mm"):
        if key not in expected:
            continue
        if isinstance(expected[key], bool):
            equal = expected[key] == actual[key]
        else:
            equal = _close(expected[key], actual[key])
        if not equal:
            violations.append(_violation(f"{path}.{key}", expected[key], actual[key]))


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    )


def _blender_render_engine(scene_ir_engine: str) -> str:
    """把稳定 IR 枚举映射到 Blender 5.2 的实际枚举。"""
    if scene_ir_engine not in BLENDER_ENGINE_MAP:
        raise ValueError(f"不支持的 Scene IR render engine：{scene_ir_engine}")
    return BLENDER_ENGINE_MAP[scene_ir_engine]


def _violation(path: str, expected: Any, actual: Any) -> dict[str, Any]:
    return {
        "code": "runtime_value_mismatch",
        "severity": "error",
        "path": path,
        "expected": expected,
        "actual": actual,
        "layer": "blender",
        "retryability": "rebuild_required",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
