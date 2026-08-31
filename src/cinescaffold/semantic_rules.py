from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


TRANSLATION_PARAMETERS_VERSION = "0.2"


def load_translation_rules(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    _validate_rule_table(value)
    return value


def translation_rules_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_translation_rules(
    model_content: dict[str, Any],
    rules: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """补齐缺省语义，并生成可审计的确定性量化快照。"""
    content = deepcopy(model_content)
    _apply_semantic_defaults(content, rules)

    emotion = _classify_emotion(content, rules)
    profile = rules["emotion_classes"][emotion["class_id"]]
    _apply_emotion_semantics(content, profile, emotion)
    subjects = _subject_parameters(content, rules)
    motions = _motion_parameters(content, rules)
    scene = _scene_parameters(content, rules)
    explicit_overrides = _explicit_override_paths(content)

    parameters = {
        "schema_version": TRANSLATION_PARAMETERS_VERSION,
        "rules_version": rules["version"],
        "coordinate_system": deepcopy(rules["coordinate_system"]),
        "input_slots": _input_slots(content),
        "emotion_class": emotion,
        "subjects": subjects,
        "motions": motions,
        "scene": scene,
        "camera": {
            **deepcopy(profile["camera"]),
            "source_status": emotion["derived_status"],
        },
        "composition": {
            **deepcopy(profile["composition"]),
            "source_status": emotion["derived_status"],
        },
        "lighting": {
            **deepcopy(profile["lighting"]),
            "source_status": emotion["derived_status"],
            "application_scope": "final_video_generation_only",
            "applied_to_blender_preview": False,
        },
        "explicit_override_paths": explicit_overrides,
        "precedence": ["explicit", "inferred", "default"],
    }
    return content, parameters


def _apply_emotion_semantics(
    content: dict[str, Any],
    profile: dict[str, Any],
    emotion: dict[str, Any],
) -> None:
    status = emotion["derived_status"]
    source_text = _emotion_source_text(content, emotion.get("matched_keyword"))
    camera_profile = profile["camera"]
    camera = content["camera"]
    _set_unless_explicit(
        camera,
        "camera_height",
        _annotated(f"{camera_profile['height_m']:g} m", status, source_text),
    )
    _set_unless_explicit(
        camera,
        "lens_intent",
        _annotated(f"{camera_profile['focal_length_mm']:g} mm", status, source_text),
    )
    _set_unless_explicit(
        camera,
        "view_angle",
        _annotated(f"俯仰 {camera_profile['pitch_degrees']:+g}°", status, source_text),
    )
    focus_target = _focus_target(content, camera_profile["focus"])
    _set_unless_explicit(
        camera,
        "focus_target_id",
        _annotated(focus_target, status, source_text),
    )
    movement = camera["movement"]
    _set_unless_explicit(
        movement,
        "type",
        _annotated(camera_profile["movement"], status, source_text),
    )
    _set_unless_explicit(
        movement,
        "speed",
        _annotated(_camera_speed_text(camera_profile), status, source_text),
    )
    _set_unless_explicit(
        movement,
        "trajectory",
        _annotated(_camera_trajectory(camera_profile["movement"]), status, source_text),
    )
    if movement.get("start_time_seconds") is None:
        movement["start_time_seconds"] = 0.0
    if movement.get("end_time_seconds") is None:
        movement["end_time_seconds"] = content["timeline"]["duration_seconds"]

    composition_profile = profile["composition"]
    composition = content["composition"]
    composition["patterns"] = _keep_explicit(composition.get("patterns", [])) + [
        _statement(
            f"主体占画幅 {_percent_range(composition_profile['subject_frame_ratio'])}",
            status,
            source_text,
        ),
        _statement(
            f"负空间占比 {_percent_range(composition_profile['negative_space_ratio'])}",
            status,
            source_text,
        ),
        _statement(
            f"地平线距画幅底部 {composition_profile['horizon_from_bottom_ratio'] * 100:g}%",
            status,
            source_text,
        ),
        _statement(
            f"主要物体占画幅 {_percent_range(composition_profile['major_object_frame_ratio'])}",
            status,
            source_text,
        ),
    ]
    primary_id = _primary_subject_id(content)
    if primary_id:
        composition["screen_placements"] = _keep_explicit(
            composition.get("screen_placements", [])
        )
        if not any(
            item.get("subject_id") == primary_id
            for item in composition["screen_placements"]
            if isinstance(item, dict)
        ):
            composition["screen_placements"].append(
                {
                    "subject_id": primary_id,
                    "horizontal": _annotated(
                        composition_profile["subject_horizontal"], status, source_text
                    ),
                    "vertical": _annotated(None, "unknown"),
                }
            )
        composition["visual_scales"] = _keep_explicit(
            composition.get("visual_scales", [])
        )
        if not any(
            item.get("subject_id") == primary_id
            for item in composition["visual_scales"]
            if isinstance(item, dict)
        ):
            composition["visual_scales"].append(
                {
                    "subject_id": primary_id,
                    "scale": _annotated(
                        _percent_range(composition_profile["subject_frame_ratio"]),
                        status,
                        source_text,
                    ),
                }
            )

    lighting = content["mood"].get("lighting_intent", [])
    if not any(_status(item) == "explicit" for item in lighting):
        lighting_profile = profile["lighting"]
        content["mood"]["lighting_intent"] = [
            _statement(
                f"主光垂直角度 {lighting_profile['key_vertical_angle_degrees']:+g}°",
                status,
                source_text,
            ),
            _statement(
                f"环境光强度 {lighting_profile['ambient_intensity']:g}",
                status,
                source_text,
            ),
            _statement(
                f"阴影硬度 {lighting_profile['shadow_hardness']:g}",
                status,
                source_text,
            ),
            _statement(
                "仅供最终视频生成，不应用到 Blender 白模",
                status,
                source_text,
            ),
        ]


def objective_translation_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Agent 1 只接收空间、运动、构图和摄影机量化结果。"""
    objective = {
        key: deepcopy(value)
        for key, value in parameters.items()
        if key not in {"lighting", "emotion_class"}
    }
    slots = objective.get("input_slots")
    if isinstance(slots, dict):
        slots.pop("feeling", None)
    overrides = objective.get("explicit_override_paths")
    if isinstance(overrides, list):
        objective["explicit_override_paths"] = [
            path for path in overrides if not path.startswith("mood.")
        ]
    return objective


def _apply_semantic_defaults(content: dict[str, Any], rules: dict[str, Any]) -> None:
    defaults = rules["defaults"]
    subjects = content.setdefault("subjects", [])
    if not subjects:
        subjects.append(
            {
                "id": "person_01",
                "category": _annotated(defaults["subject_label"], "default"),
                "description": _annotated(None, "unknown"),
                "narrative_role": _annotated(None, "unknown"),
                "attributes": [],
            }
        )
        _add_default_uncertainty(content, "subjects", defaults["subject_label"])

    scene = content.setdefault("scene_design", {})
    environment = scene.get("environment")
    if not isinstance(environment, dict) or not environment.get("value"):
        scene["environment"] = _annotated(defaults["environment_label"], "default")
        _add_default_uncertainty(
            content,
            "scene_design.environment",
            defaults["environment_label"],
        )

    mood = content.setdefault("mood", {})
    tones = mood.setdefault("emotional_tones", [])
    if not tones:
        tones.append(_statement("中性", "default"))
        _add_default_uncertainty(content, "mood.emotional_tones", "中性")

    timeline = content.setdefault("timeline", {})
    duration = timeline.get("duration_seconds")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        timeline["duration_seconds"] = defaults["duration_seconds"]
        timeline["duration_range_seconds"] = None
        timeline["duration_source_status"] = "default"
        _add_default_uncertainty(
            content,
            "timeline.duration_seconds",
            str(defaults["duration_seconds"]),
        )

    motions = content.setdefault("subject_motion", [])
    motion_ids = {item.get("subject_id") for item in motions if isinstance(item, dict)}
    for subject in subjects:
        subject_id = subject.get("id")
        if not isinstance(subject_id, str) or subject_id in motion_ids:
            continue
        motions.append(
            {
                "subject_id": subject_id,
                "action": _annotated(defaults["action_label"], "default"),
                "direction": _annotated(None, "unknown"),
                "speed": _annotated("静止", "default"),
                "trajectory": _annotated(None, "unknown"),
                "start_time_seconds": 0.0,
                "end_time_seconds": timeline["duration_seconds"],
                "secondary_motion": [],
            }
        )
        _add_default_uncertainty(
            content,
            f"subject_motion[{subject_id}]",
            defaults["action_label"],
        )


def _classify_emotion(content: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    tones = content["mood"]["emotional_tones"]
    first_declared_tone: dict[str, Any] | None = None
    for tone in tones:
        if not isinstance(tone, dict):
            continue
        if tone.get("source_status") != "default" and first_declared_tone is None:
            first_declared_tone = tone
        text = str(tone.get("value") or "")
        for class_id, profile in rules["emotion_classes"].items():
            for keyword in profile["keywords"]:
                if keyword in text:
                    return {
                        "class_id": class_id,
                        "label": profile["label"],
                        "matched_keyword": keyword,
                        "source_status": tone.get("source_status", "inferred"),
                        "derived_status": "inferred",
                    }
    default_id = rules["defaults"]["emotion_class"]
    profile = rules["emotion_classes"][default_id]
    if first_declared_tone is not None:
        return {
            "class_id": default_id,
            "label": profile["label"],
            "matched_keyword": None,
            "source_status": first_declared_tone.get("source_status", "explicit"),
            "derived_status": "inferred",
        }
    return {
        "class_id": default_id,
        "label": profile["label"],
        "matched_keyword": None,
        "source_status": "default",
        "derived_status": "default",
    }


def _subject_parameters(content: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for subject in content["subjects"]:
        category = _value(subject.get("category"))
        description = _value(subject.get("description"))
        source_text = _source_text(subject.get("category")) or ""
        search_text = " ".join(item for item in (category, description, source_text) if item)
        height, height_status = _subject_height(search_text, rules)
        quantity, quantity_status = _subject_quantity(search_text)
        result.append(
            {
                "subject_id": subject["id"],
                "quantity_min": quantity,
                "quantity_max": None if "一群" in search_text else quantity,
                "quantity_source_status": quantity_status,
                "reference_height_m": height,
                "height_source_status": height_status,
                "ground_contact_position_m": [0.0, 0.0, 0.0],
                "facing_direction_world": deepcopy(
                    rules["coordinate_system"]["forward_vector"]
                ),
                "placement_source_status": "default",
                "dominant_scale_multiplier_range": (
                    [1.5, 3.0]
                    if any(word in search_text for word in ("巨大", "高耸"))
                    else [1.0, 1.0]
                ),
                **_major_object_parameters(search_text, rules),
            }
        )
    return result


def _motion_parameters(content: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    duration = float(content["timeline"]["duration_seconds"])
    subjects = content["subjects"]
    result: list[dict[str, Any]] = []
    for index, motion in enumerate(content["subject_motion"]):
        action = _value(motion.get("action")) or rules["defaults"]["action_label"]
        motion_type = _motion_type(action, rules)
        speed_range = deepcopy(rules["speed_ranges_mps"].get(
            motion_type,
            rules["defaults"]["moving_speed_mps"],
        ))
        target_id, direction_mode = _motion_target(action, motion, subjects)
        path_type = _path_type(action, _value(motion.get("trajectory")), rules)
        result.append(
            {
                "motion_index": index,
                "subject_id": motion.get("subject_id"),
                "motion_type": motion_type,
                "speed_range_mps": speed_range,
                "direction_mode": direction_mode,
                "direction_vector_world": (
                    deepcopy(rules["coordinate_system"]["forward_vector"])
                    if direction_mode == "world_forward"
                    else None
                ),
                "target_id": target_id,
                "path_type": path_type,
                "start_time_seconds": _number_or(motion.get("start_time_seconds"), 0.0),
                "end_time_seconds": _number_or(motion.get("end_time_seconds"), duration),
                "source_status": "inferred" if _status(motion.get("action")) != "default" else "default",
            }
        )
    return result


def _scene_parameters(content: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    environment = content["scene_design"]["environment"]
    text = _value(environment) or rules["defaults"]["environment_label"]
    asset_key = "blank"
    for candidate, keywords in rules["scene_assets"].items():
        if any(keyword in text for keyword in keywords):
            asset_key = candidate
            break
    status = "inferred" if _status(environment) != "default" else "default"
    return {
        "environment_label": text,
        "asset_key": asset_key,
        "dimensions_m": deepcopy(rules["scene_dimensions_m"][asset_key]),
        "asset_resolution": "proxy_fallback",
        "source_status": status,
    }


def _input_slots(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "who": [_slot(item.get("category")) for item in content["subjects"]],
        "where": _slot(content["scene_design"]["environment"]),
        "doing": [_slot(item.get("action")) for item in content["subject_motion"]],
        "feeling": [_slot(item) for item in content["mood"]["emotional_tones"]],
    }


def _set_unless_explicit(
    container: dict[str, Any],
    key: str,
    replacement: dict[str, Any],
) -> None:
    if _status(container.get(key)) != "explicit":
        container[key] = replacement


def _emotion_source_text(content: dict[str, Any], keyword: str | None) -> str | None:
    tones = content["mood"]["emotional_tones"]
    for tone in tones:
        value = _value(tone) or ""
        if keyword is None or keyword in value:
            return _source_text(tone)
    return None


def _primary_subject_id(content: dict[str, Any]) -> str | None:
    for subject in content["subjects"]:
        subject_id = subject.get("id")
        if isinstance(subject_id, str):
            return subject_id
    return None


def _focus_target(content: dict[str, Any], strategy: str) -> str | None:
    primary_id = _primary_subject_id(content)
    if strategy == "scene_center":
        return None
    if strategy == "ahead_of_primary_subject":
        return None
    if strategy == "largest_object":
        for subject in content["subjects"]:
            text = " ".join(
                item
                for item in (
                    _value(subject.get("category")),
                    _value(subject.get("description")),
                    _source_text(subject.get("category")),
                    _source_text(subject.get("description")),
                )
                if item
            )
            if any(marker in text for marker in ("巨大", "高耸", "巨型")):
                return str(subject.get("id"))
    return primary_id


def _camera_speed_text(camera: dict[str, Any]) -> str:
    speed = camera.get("speed_mps")
    if speed is None:
        return "与主体速度同步"
    return f"{float(speed):g} m/s"


def _camera_trajectory(movement: str) -> str:
    if movement == "orbit":
        return "圆形环绕"
    if movement == "static":
        return "静止"
    return "直线"


def _percent_range(values: list[float]) -> str:
    return f"{values[0] * 100:g}%-{values[1] * 100:g}%"


def _keep_explicit(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [deepcopy(item) for item in values if _contains_explicit(item)]


def _contains_explicit(value: Any) -> bool:
    if isinstance(value, list):
        return any(_contains_explicit(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("source_status") == "explicit":
        return True
    return any(
        _contains_explicit(item)
        for key, item in value.items()
        if key not in {"source_status", "source_text"}
    )


def _explicit_override_paths(content: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for root in ("camera", "composition"):
        _collect_explicit(content.get(root), root, paths)
    _collect_explicit(content.get("mood", {}).get("lighting_intent"), "mood.lighting_intent", paths)
    return paths


def _collect_explicit(value: Any, path: str, output: list[str]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_explicit(item, f"{path}[{index}]", output)
        return
    if not isinstance(value, dict):
        return
    if value.get("source_status") == "explicit":
        output.append(path)
    for key, item in value.items():
        if key not in {"source_status", "source_text"}:
            _collect_explicit(item, f"{path}.{key}", output)


def _subject_height(text: str, rules: dict[str, Any]) -> tuple[float, str]:
    for keyword, height in rules["subject_heights_m"].items():
        if keyword in text:
            return float(height), "inferred"
    return float(rules["defaults"]["reference_height_m"]), "default"


def _major_object_parameters(text: str, rules: dict[str, Any]) -> dict[str, Any]:
    for keyword, parameters in rules["major_object_defaults"].items():
        if keyword in text:
            return deepcopy(parameters)
    return {
        "minimum_footprint_m": None,
        "minimum_height_m": None,
        "default_scene_depth_ratio": None,
    }


def _subject_quantity(text: str) -> tuple[int, str]:
    if "一群" in text:
        return 5, "inferred"
    match = re.search(r"(?:^|\D)(\d+)(?:个|名|位|只|辆|艘|颗)?", text)
    if match:
        return max(1, int(match.group(1))), "inferred"
    chinese = {"一个": 1, "一名": 1, "一位": 1, "一辆": 1, "一艘": 1, "两个": 2, "两名": 2, "两位": 2, "三 个": 3, "三个": 3}
    for marker, quantity in chinese.items():
        if marker in text:
            return quantity, "inferred"
    return 1, "default"


def _motion_type(action: str, rules: dict[str, Any]) -> str:
    # 长词优先，避免“飞行”被“飞”提前截断。
    for motion_type in (
        "jumping",
        "running",
        "walking",
        "flying",
        "moving",
        "interactive",
        "static",
    ):
        keywords = sorted(rules["action_classes"][motion_type], key=len, reverse=True)
        if any(keyword in action for keyword in keywords):
            return motion_type
    return "moving"


def _motion_target(
    action: str,
    motion: dict[str, Any],
    subjects: list[dict[str, Any]],
) -> tuple[str | None, str]:
    direction = _value(motion.get("direction")) or ""
    combined = f"{action} {direction}"
    target_id: str | None = None
    for subject in subjects:
        subject_id = subject.get("id")
        if subject_id == motion.get("subject_id"):
            continue
        category = _value(subject.get("category")) or ""
        if category and category in combined:
            target_id = str(subject_id)
            break
    if target_id and any(word in combined for word in ("远离", "背离", "逃离")):
        return target_id, "away_from_target"
    if target_id and any(word in combined for word in ("走向", "跑向", "朝", "接近", "驶到", "进入", "上车", "绕")):
        return target_id, "toward_or_relative_to_target"
    return None, "world_forward"


def _path_type(action: str, trajectory: str | None, rules: dict[str, Any]) -> str:
    text = f"{action} {trajectory or ''}"
    for path_type, keywords in rules["path_keywords"].items():
        if keywords and any(keyword in text for keyword in keywords):
            return path_type
    return "linear"


def _add_default_uncertainty(content: dict[str, Any], field: str, selected: str) -> None:
    uncertainties = content.setdefault("uncertainties", [])
    if any(item.get("field") == field for item in uncertainties if isinstance(item, dict)):
        return
    uncertainties.append(
        {
            "field": field,
            "reason": "用户未明确提供该输入槽位，按转换规则使用缺省值",
            "resolution": "use_default",
            "selected_value": selected,
        }
    )


def _slot(value: Any) -> dict[str, Any]:
    return {
        "value": _value(value),
        "source_status": _status(value),
        "source_text": _source_text(value),
    }


def _annotated(
    value: str | None,
    status: str,
    source_text: str | None = None,
) -> dict[str, Any]:
    return {"value": value, "source_status": status, "source_text": source_text}


def _statement(
    value: str,
    status: str,
    source_text: str | None = None,
) -> dict[str, Any]:
    return {"value": value, "source_status": status, "source_text": source_text}


def _value(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return None


def _status(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("source_status"), str):
        return value["source_status"]
    return "unknown"


def _source_text(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("source_text"), str):
        return value["source_text"]
    return None


def _number_or(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


def _validate_rule_table(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("语义量化表根节点必须是对象")
    required = {
        "version",
        "coordinate_system",
        "defaults",
        "subject_heights_m",
        "major_object_defaults",
        "action_classes",
        "speed_ranges_mps",
        "path_keywords",
        "scene_assets",
        "scene_dimensions_m",
        "emotion_classes",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"语义量化表缺少字段：{', '.join(missing)}")
    if value["coordinate_system"].get("forward_vector") != [0.0, -1.0, 0.0]:
        raise ValueError("语义量化表必须遵守 CineScaffold 的 -Y 世界前方约定")
    if set(value["emotion_classes"]) != {"E1", "E2", "E3", "E4", "E5", "E6"}:
        raise ValueError("语义量化表必须完整定义 E1-E6")
    for asset_key in value["scene_assets"]:
        if asset_key not in value["scene_dimensions_m"]:
            raise ValueError(f"场景资产 {asset_key} 缺少尺寸规则")
