from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any


VIEWER_MANIFEST_SCHEMA_VERSION = "0.1"
GLB_EXPORT_RECIPE_VERSION = "glb_v0.1"


def inspect_glb(path: Path) -> dict[str, Any]:
    """读取并验证 GLB 2.0 容器，返回浏览器预览所需的结构摘要。"""
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("GLB 文件过短")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError("GLB magic 无效")
    if version != 2:
        raise ValueError(f"不支持的 GLB 版本：{version}")
    if declared_length != len(data):
        raise ValueError("GLB 声明长度与文件长度不一致")

    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != b"JSON":
        raise ValueError("GLB 首个 chunk 不是 JSON")
    json_end = 20 + json_length
    if json_end > len(data):
        raise ValueError("GLB JSON chunk 越界")
    try:
        document = json.loads(data[20:json_end].rstrip(b" \x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GLB JSON chunk 无效") from error
    if not isinstance(document, dict):
        raise ValueError("GLB JSON 根节点不是对象")

    nodes = document.get("nodes", [])
    animations = document.get("animations", [])
    if not isinstance(nodes, list) or not isinstance(animations, list):
        raise ValueError("GLB nodes/animations 结构无效")
    return {
        "format_version": version,
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
        "document": document,
        "node_count": len(nodes),
        "animation_count": len(animations),
    }


def build_viewer_manifest(
    *,
    scene_ir: dict[str, Any],
    scene_ir_hash: str,
    glb_path: Path,
    blender_version: str,
    runtime_validation: dict[str, Any],
) -> dict[str, Any]:
    """把 Scene IR 与已验证 GLB 绑定成稳定、无服务器路径的查看器清单。"""
    inspection = inspect_glb(glb_path)
    document = inspection["document"]
    nodes = document.get("nodes", [])
    entity_nodes = _mapped_nodes(nodes, kind="entity")
    camera_nodes = _mapped_nodes(nodes, kind="camera")

    entity_ids = [item["entity_id"] for item in scene_ir["entities"]]
    missing_entities = sorted(set(entity_ids) - set(entity_nodes))
    unexpected_entities = sorted(set(entity_nodes) - set(entity_ids))
    if missing_entities or unexpected_entities:
        raise ValueError(
            "GLB Entity 映射不一致："
            f"missing={missing_entities}, unexpected={unexpected_entities}"
        )

    camera_id = scene_ir["camera"]["camera_id"]
    if camera_id not in camera_nodes:
        raise ValueError(f"GLB 缺少默认摄影机节点：{camera_id}")
    camera_node_index = camera_nodes[camera_id]["node_index"]
    camera_node = nodes[camera_node_index]
    if "camera" not in camera_node:
        raise ValueError(f"GLB 节点没有摄影机数据：{camera_id}")

    timeline = scene_ir["timeline"]
    visibility_tracks = [
        {
            "entity_id": item["entity_id"],
            "ranges": _visibility_ranges(
                item["local_state_track"],
                frame_start=timeline["frame_start"],
                frame_end=timeline["frame_end"],
            ),
        }
        for item in scene_ir["entities"]
    ]
    animated_entity_ids = sorted(
        item["entity_id"]
        for item in scene_ir["entities"]
        if _entity_track_has_trs_changes(item["local_state_track"])
    )
    camera_animated = _camera_track_has_trs_changes(scene_ir["camera"]["state_track"])
    animated_node_indices = _animated_node_indices(document)
    expected_animated_node_indices = {
        entity_nodes[entity_id]["node_index"] for entity_id in animated_entity_ids
    }
    if camera_animated:
        expected_animated_node_indices.add(camera_node_index)
    missing_animation_nodes = sorted(expected_animated_node_indices - animated_node_indices)
    if missing_animation_nodes:
        raise ValueError(f"GLB 缺少预期的 TRS 动画节点：{missing_animation_nodes}")

    warnings = _capability_warnings(scene_ir)
    return {
        "schema_version": VIEWER_MANIFEST_SCHEMA_VERSION,
        "scene_ir": {
            "schema_version": scene_ir["schema_version"],
            "scene_id": scene_ir["scene_id"],
            "canonical_hash": scene_ir_hash,
            "artifact_id": "scene_ir",
        },
        "glb": {
            "artifact_id": "glb_preview",
            "file_name": glb_path.name,
            "format_version": inspection["format_version"],
            "sha256": inspection["sha256"],
            "size_bytes": inspection["size_bytes"],
            "node_count": inspection["node_count"],
        },
        "timeline": {
            "frame_start": timeline["frame_start"],
            "frame_end": timeline["frame_end"],
            "frame_count": timeline["frame_count"],
            "fps_numerator": timeline["fps_numerator"],
            "fps_denominator": timeline["fps_denominator"],
            "duration_seconds": timeline["duration_seconds"],
            "time_domain": timeline["time_domain"],
        },
        "default_camera": {
            "camera_id": camera_id,
            **camera_nodes[camera_id],
        },
        "entities": [
            {
                "entity_id": item["entity_id"],
                "parent_id": item["parent_id"],
                **entity_nodes[item["entity_id"]],
            }
            for item in scene_ir["entities"]
        ],
        "visibility_tracks": visibility_tracks,
        "animations": {
            "clip_count": inspection["animation_count"],
            "clip_names": [
                str(item.get("name", ""))
                for item in document.get("animations", [])
                if isinstance(item, dict)
            ],
            "animated_entity_ids": animated_entity_ids,
            "camera_animated": camera_animated,
            "frame_step": 1,
        },
        "runtime_validation": {
            "passed": runtime_validation.get("passed") is True,
            "violation_count": len(runtime_validation.get("violations", [])),
        },
        "capabilities": {
            "entity_trs": "glb_animation",
            "camera_trs": "glb_animation",
            "entity_visibility": "scene_ir_timeline",
            "camera_focal_length": "scene_ir_timeline",
            "materials": "principled_subset",
        },
        "fallback": {
            "preferred_delivery": "glb_plus_scene_ir",
            "on_glb_error": "mp4",
            "video_artifact_ids": ["diagnostic_preview", "clay_preview"],
        },
        "warnings": warnings,
        "exporter": {
            "recipe_version": GLB_EXPORT_RECIPE_VERSION,
            "blender_version": blender_version,
            "compression": "none",
            "coordinate_conversion": "gltf_y_up",
        },
    }


def write_viewer_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mapped_nodes(nodes: list[Any], *, kind: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        extras = node.get("extras")
        if not isinstance(extras, dict) or extras.get("cinescaffold_kind") != kind:
            continue
        stable_id = extras.get("cinescaffold_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(f"GLB {kind} 节点缺少稳定 ID")
        if stable_id in mapped:
            raise ValueError(f"GLB {kind} 稳定 ID 重复：{stable_id}")
        mapped[stable_id] = {
            "node_name": str(node.get("name", "")),
            "node_index": node_index,
        }
    return mapped


def _animated_node_indices(document: dict[str, Any]) -> set[int]:
    animated: set[int] = set()
    for animation in document.get("animations", []):
        if not isinstance(animation, dict):
            continue
        for channel in animation.get("channels", []):
            if not isinstance(channel, dict):
                continue
            target = channel.get("target")
            if not isinstance(target, dict):
                continue
            node_index = target.get("node")
            if isinstance(node_index, int) and target.get("path") in {
                "translation",
                "rotation",
                "scale",
            }:
                animated.add(node_index)
    return animated


def _visibility_ranges(
    track: dict[str, Any],
    *,
    frame_start: int,
    frame_end: int,
) -> list[dict[str, Any]]:
    if track["mode"] == "constant":
        return [
            {
                "frame_start": frame_start,
                "frame_end_exclusive": frame_end + 1,
                "visible": bool(track["value"]["render_visible"]),
            }
        ]

    samples = track["samples"]
    if not samples:
        raise ValueError("逐帧 Entity 轨道不能为空")
    ranges: list[dict[str, Any]] = []
    range_start = samples[0]["frame"]
    visible = bool(samples[0]["value"]["render_visible"])
    previous_frame = range_start
    for sample in samples[1:]:
        frame = sample["frame"]
        next_visible = bool(sample["value"]["render_visible"])
        if frame != previous_frame + 1:
            raise ValueError("逐帧 Entity 轨道不连续")
        if next_visible != visible:
            ranges.append(
                {
                    "frame_start": range_start,
                    "frame_end_exclusive": frame,
                    "visible": visible,
                }
            )
            range_start = frame
            visible = next_visible
        previous_frame = frame
    ranges.append(
        {
            "frame_start": range_start,
            "frame_end_exclusive": previous_frame + 1,
            "visible": visible,
        }
    )
    return ranges


def _capability_warnings(scene_ir: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if any(
        _track_has_visibility_changes(item["local_state_track"])
        for item in scene_ir["entities"]
    ):
        warnings.append(
            {
                "code": "visibility_requires_scene_ir",
                "message": (
                    "Entity 显隐由 Scene IR 时间轴驱动，GLB 本身不包含该语义。"
                ),
            }
        )
    if _camera_track_has_focal_changes(scene_ir["camera"]["state_track"]):
        warnings.append(
            {
                "code": "camera_focal_length_requires_scene_ir",
                "message": "动态焦距由 Scene IR 时间轴驱动，GLB 仅负责摄影机 TRS。",
            }
        )
    return warnings


def _track_has_visibility_changes(track: dict[str, Any]) -> bool:
    if track["mode"] == "constant":
        return not bool(track["value"]["render_visible"])
    values = {bool(sample["value"]["render_visible"]) for sample in track["samples"]}
    return values != {True}


def _camera_track_has_focal_changes(track: dict[str, Any]) -> bool:
    if track["mode"] == "constant":
        return False
    return len({float(sample["value"]["focal_length_mm"]) for sample in track["samples"]}) > 1


def _entity_track_has_trs_changes(track: dict[str, Any]) -> bool:
    if track["mode"] == "constant":
        return False
    values = {
        (
            tuple(sample["value"]["translation_m"]),
            tuple(sample["value"]["rotation_quaternion_wxyz"]),
            tuple(sample["value"]["scale"]),
        )
        for sample in track["samples"]
    }
    return len(values) > 1


def _camera_track_has_trs_changes(track: dict[str, Any]) -> bool:
    if track["mode"] == "constant":
        return False
    values = {
        (
            tuple(sample["value"]["translation_m"]),
            tuple(sample["value"]["rotation_quaternion_wxyz"]),
        )
        for sample in track["samples"]
    }
    return len(values) > 1


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
