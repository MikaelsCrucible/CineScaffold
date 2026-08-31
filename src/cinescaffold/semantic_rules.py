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
    source_prompt: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """补齐缺省语义，并生成可审计的确定性量化快照。"""
    _validate_rule_table(rules)
    content = deepcopy(model_content)
    _apply_semantic_defaults(content, rules)
    _apply_timeline_rules(content, rules, source_prompt)

    emotion = _classify_emotion(content, rules)
    profile = rules["emotion_classes"][emotion["class_id"]]
    resolved_profile = deepcopy(profile)
    resolved_profile["camera"] = _resolve_camera_profile(
        profile["camera"],
        float(content["timeline"]["duration_seconds"]),
    )
    _apply_emotion_semantics(content, resolved_profile, emotion)
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
            **deepcopy(resolved_profile["camera"]),
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
        resolved_range_duration = _resolve_duration_range(timeline, defaults)
        if resolved_range_duration is not None:
            timeline["duration_seconds"] = resolved_range_duration
            timeline["duration_source_status"] = "inferred"
            _add_inference_uncertainty(
                content,
                "timeline.duration_seconds",
                str(resolved_range_duration),
                "用户只给出时长范围，按转换规则选择范围最大值",
            )
        else:
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


def _apply_timeline_rules(
    content: dict[str, Any],
    rules: dict[str, Any],
    source_prompt: str | None,
) -> None:
    """把明确的先后叙事转换为分段时间，而非让每个动作覆盖全片。"""
    if not source_prompt:
        return
    timeline_rules = rules["timeline"]
    if not _contains_sequential_marker(source_prompt, timeline_rules):
        return

    events = content["timeline"].get("events", [])
    if len(events) < 2:
        raise ValueError("自然语言包含先后关系，但模型没有拆分 timeline.events")

    duration = float(content["timeline"]["duration_seconds"])
    simultaneous = any(
        marker in source_prompt for marker in timeline_rules["simultaneous_markers"]
    )
    if _all_events_cover_full_timeline(events, duration):
        if simultaneous:
            raise ValueError("自然语言同时包含先后与并行关系，但模型没有给出分段时间")
        _partition_events_equally(content, events, duration)

    _align_full_timeline_motions(content, events, duration)


def _contains_sequential_marker(text: str, timeline_rules: dict[str, Any]) -> bool:
    if any(marker in text for marker in timeline_rules["sequential_markers"]):
        return True
    for first, second in timeline_rules["sequential_pairs"]:
        first_index = text.find(first)
        second_index = text.find(second, first_index + len(first))
        if first_index >= 0 and second_index >= 0:
            return True
    return False


def _all_events_cover_full_timeline(
    events: list[dict[str, Any]],
    duration: float,
) -> bool:
    return all(
        _covers_full_timeline(
            event.get("start_time_seconds"),
            event.get("end_time_seconds"),
            duration,
        )
        for event in events
    )


def _covers_full_timeline(start: Any, end: Any, duration: float) -> bool:
    start_value = 0.0 if not _is_number(start) else float(start)
    end_value = duration if not _is_number(end) else float(end)
    return abs(start_value) <= 1e-9 and abs(end_value - duration) <= 1e-9


def _partition_events_equally(
    content: dict[str, Any],
    events: list[dict[str, Any]],
    duration: float,
) -> None:
    count = len(events)
    ranges: list[str] = []
    for index, event in enumerate(events):
        start = duration * index / count
        end = duration * (index + 1) / count
        event["start_time_seconds"] = start
        event["end_time_seconds"] = end
        event["source_status"] = "inferred"
        ranges.append(f"{event.get('id', index)}={start:g}-{end:g}秒")
    _add_inference_uncertainty(
        content,
        "timeline.events",
        "；".join(ranges),
        "用户明确了事件顺序但未给出各阶段时长，按事件数量等分总时长",
    )


def _align_full_timeline_motions(
    content: dict[str, Any],
    events: list[dict[str, Any]],
    duration: float,
) -> None:
    for motion in content["subject_motion"]:
        if not _covers_full_timeline(
            motion.get("start_time_seconds"),
            motion.get("end_time_seconds"),
            duration,
        ):
            continue
        event = _matching_timeline_event(motion, events)
        if event is None:
            continue
        start = event.get("start_time_seconds")
        end = event.get("end_time_seconds")
        if _is_number(start) and _is_number(end):
            motion["start_time_seconds"] = float(start)
            motion["end_time_seconds"] = float(end)


def _matching_timeline_event(
    motion: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    subject_id = motion.get("subject_id")
    action = _value(motion.get("action")) or ""
    source_text = _source_text(motion.get("action")) or ""
    candidates = [
        event
        for event in events
        if subject_id in event.get("reference_ids", [])
    ]
    scored: list[tuple[int, dict[str, Any]]] = []
    for event in candidates:
        event_text = " ".join(
            text
            for text in (event.get("description"), event.get("source_text"))
            if isinstance(text, str)
        )
        score = 0
        if action and (action in event_text or event_text in action):
            score += 2
        if source_text and (source_text in event_text or event_text in source_text):
            score += 3
        scored.append((score, event))
    positive = [item for item in scored if item[0] > 0]
    if positive:
        positive.sort(key=lambda item: item[0], reverse=True)
        if len(positive) == 1 or positive[0][0] > positive[1][0]:
            return positive[0][1]
    return candidates[0] if len(candidates) == 1 else None


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


def _resolve_camera_profile(
    camera: dict[str, Any],
    duration_seconds: float,
) -> dict[str, Any]:
    """按冻结时长解析摄影机速度，避免距离与速度互相矛盾。"""
    resolved = deepcopy(camera)
    policy = resolved.pop("speed_policy")
    if policy == "derive_from_distance_and_duration":
        start = float(resolved["start_distance_m"])
        end = float(resolved["end_distance_m"])
        resolved["speed_mps"] = abs(end - start) / duration_seconds
    return resolved


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


def _add_inference_uncertainty(
    content: dict[str, Any],
    field: str,
    selected: str,
    reason: str,
) -> None:
    uncertainties = content.setdefault("uncertainties", [])
    if any(item.get("field") == field for item in uncertainties if isinstance(item, dict)):
        return
    uncertainties.append(
        {
            "field": field,
            "reason": reason,
            "resolution": "use_inference",
            "selected_value": selected,
        }
    )


def _resolve_duration_range(
    timeline: dict[str, Any],
    defaults: dict[str, Any],
) -> float | None:
    duration_range = timeline.get("duration_range_seconds")
    if not isinstance(duration_range, dict):
        return None
    minimum = duration_range.get("minimum_seconds")
    maximum = duration_range.get("maximum_seconds")
    if not _is_number(minimum) or not _is_number(maximum):
        raise ValueError("时长范围必须包含数值 minimum_seconds 和 maximum_seconds")
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    if minimum_value < 0 or maximum_value <= 0 or minimum_value > maximum_value:
        raise ValueError("时长范围必须满足 0 <= minimum_seconds <= maximum_seconds")
    if defaults.get("duration_range_resolution") != "maximum_seconds":
        raise ValueError("当前只支持 maximum_seconds 时长范围解析策略")
    return maximum_value


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
        "timeline",
        "emotion_classes",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"语义量化表缺少字段：{', '.join(missing)}")
    if value["coordinate_system"].get("forward_vector") != [0.0, -1.0, 0.0]:
        raise ValueError("语义量化表必须遵守 CineScaffold 的 -Y 世界前方约定")
    if set(value["emotion_classes"]) != {"E1", "E2", "E3", "E4", "E5", "E6"}:
        raise ValueError("语义量化表必须完整定义 E1-E6")
    timeline = value["timeline"]
    if not isinstance(timeline, dict):
        raise ValueError("语义量化表 timeline 必须是对象")
    for key in ("sequential_markers", "simultaneous_markers"):
        markers = timeline.get(key)
        if not isinstance(markers, list) or not all(
            isinstance(marker, str) and marker for marker in markers
        ):
            raise ValueError(f"语义量化表 timeline.{key} 必须是非空字符串数组")
    pairs = timeline.get("sequential_pairs")
    if not isinstance(pairs, list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(marker, str) and marker for marker in pair)
        for pair in pairs
    ):
        raise ValueError("语义量化表 timeline.sequential_pairs 必须是二元字符串数组")
    for class_id, profile in value["emotion_classes"].items():
        camera = profile.get("camera")
        if not isinstance(camera, dict):
            raise ValueError(f"情绪类别 {class_id} 缺少摄影机规则")
        _validate_camera_rule(class_id, camera)
    for asset_key in value["scene_assets"]:
        if asset_key not in value["scene_dimensions_m"]:
            raise ValueError(f"场景资产 {asset_key} 缺少尺寸规则")


def _validate_camera_rule(class_id: str, camera: dict[str, Any]) -> None:
    policy = camera.get("speed_policy")
    supported = {
        "derive_from_distance_and_duration",
        "match_subject",
        "fixed",
        "static",
    }
    if policy not in supported:
        raise ValueError(f"情绪类别 {class_id} 的摄影机速度策略无效")

    start = camera.get("start_distance_m")
    end = camera.get("end_distance_m")
    speed = camera.get("speed_mps")
    if not _is_number(start) or float(start) < 0:
        raise ValueError(f"情绪类别 {class_id} 的摄影机起始距离必须是非负数")

    if policy == "derive_from_distance_and_duration":
        if camera.get("movement") not in {"push_in", "pull_out"}:
            raise ValueError(f"情绪类别 {class_id} 的径向速度策略只适用于推近或后拉")
        if not _is_number(end) or float(end) < 0:
            raise ValueError(f"情绪类别 {class_id} 的径向运镜必须给出非负终止距离")
        if speed is not None:
            raise ValueError(f"情绪类别 {class_id} 的径向运镜不得同时写死速度")
        return

    if policy == "match_subject":
        if speed is not None:
            raise ValueError(f"情绪类别 {class_id} 的跟拍速度应由主体运动决定")
        return

    if not _is_number(speed) or float(speed) < 0:
        raise ValueError(f"情绪类别 {class_id} 的固定摄影机速度必须是非负数")
    if policy == "static":
        if float(speed) != 0 or not _is_number(end) or float(end) != float(start):
            raise ValueError(f"情绪类别 {class_id} 的静止摄影机距离和速度必须保持不变")
