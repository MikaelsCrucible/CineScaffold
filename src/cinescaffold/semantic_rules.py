from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


TRANSLATION_PARAMETERS_VERSION = "0.4"


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
    _normalize_motion_metadata(content)
    _apply_timeline_rules(content, rules, source_prompt)
    _apply_scene_dynamics(content)

    emotion = _classify_emotion(content, rules)
    profile = rules["emotion_classes"][emotion["class_id"]]
    resolved_profile = deepcopy(profile)
    resolved_profile["camera"] = _camera_profile_for_explicit_movement(
        content,
        rules,
        profile["camera"],
    )
    resolved_profile["camera"] = _resolve_camera_profile(
        resolved_profile["camera"],
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
        "scene_dynamics": deepcopy(content["scene_dynamics"]),
        "temporal_relations": deepcopy(content["timeline"].get("relations", [])),
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
    return content, reconcile_translation_parameters(content, parameters)


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


def objective_translation_parameters(
    parameters: dict[str, Any],
    content: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Agent 1 只接收空间、运动、构图和摄影机量化结果。"""
    reconciled = (
        reconcile_translation_parameters(content, parameters)
        if content is not None
        else deepcopy(parameters)
    )
    objective = {
        key: deepcopy(value)
        for key, value in reconciled.items()
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


def reconcile_translation_parameters(
    content: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """让显式摄影机语义覆盖旧 Brief 中与之矛盾的缺省量化值。"""
    reconciled = deepcopy(parameters)
    movement_node = content.get("camera", {}).get("movement", {}).get("type")
    if _status(movement_node) != "explicit":
        return reconciled
    movement = _camera_movement_kind(_value(movement_node))
    camera = reconciled.get("camera")
    if movement is None or not isinstance(camera, dict):
        return reconciled

    duration = _number_or(content.get("timeline", {}).get("duration_seconds"), 15.0)
    duration = max(duration, 1e-6)
    start = _number_or(camera.get("start_distance_m"), 15.0)
    end_value = camera.get("end_distance_m")
    end = float(end_value) if _is_number(end_value) else None

    camera["movement"] = movement
    camera["source_status"] = "explicit"
    if movement == "push_in":
        if end is None or end >= start:
            end = max(start * 0.5, 0.1)
        camera["end_distance_m"] = end
        camera["speed_mps"] = abs(start - end) / duration
    elif movement == "pull_out":
        if end is None or end <= start:
            end = max(start * 1.5, start + 0.1)
        camera["end_distance_m"] = end
        camera["speed_mps"] = abs(end - start) / duration
    elif movement == "static":
        camera["end_distance_m"] = start
        camera["speed_mps"] = 0.0
    return reconciled


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
                "motion_id": f"default_hold_{subject_id}",
                "subject_id": subject_id,
                "action": _annotated(defaults["action_label"], "default"),
                "motion_semantics": {
                    "action_kind": "hold",
                    "motion_type": "static",
                    "motion_mode": "stationary",
                    "direction_mode": "none",
                    "target_id": None,
                    "carrier_id": None,
                    "path_type": "stationary",
                    "timeline_event_id": None,
                    "narrative_required": False,
                    "postconditions": {
                        "contained_by_id": None,
                        "external_visibility": "unchanged",
                    },
                    "source_status": "default",
                    "source_text": None,
                },
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
    """Validate independent entity ranges and typed relations without global partitioning."""

    timeline = content["timeline"]
    timeline.setdefault("relations", [])
    events = timeline.get("events", [])
    duration = float(timeline["duration_seconds"])
    event_ids = [item.get("id") for item in events if isinstance(item, dict)]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("timeline.events 的 id 必须唯一")

    for event in events:
        start = event.get("start_time_seconds")
        end = event.get("end_time_seconds")
        if not _is_number(start) or not _is_number(end):
            raise ValueError("动态事件必须给出关键时间范围")
        if not 0.0 <= float(start) < float(end) <= duration:
            raise ValueError(f"timeline event 时间范围无效：{event.get('id')}")

    sequential = bool(
        source_prompt
        and _contains_sequential_marker(source_prompt, rules["timeline"])
    )
    if sequential:
        if len(events) < 2:
            raise ValueError("自然语言包含先后关系，但模型没有拆分 timeline.events")
        if _all_events_cover_full_timeline(events, duration):
            raise ValueError(
                "顺序事件不能全部覆盖完整镜头；请按主体独立时间线给出范围和 temporal relation"
            )
        if not timeline.get("relations"):
            raise ValueError("自然语言包含先后关系，但模型没有给出 temporal relation")

    _normalize_temporal_relations(timeline.get("relations", []), events)
    _validate_temporal_relations(timeline.get("relations", []), events)
    _align_full_timeline_motions(content, events, duration)
    _validate_motion_ranges(content.get("subject_motion", []), duration)


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


def _validate_temporal_relations(
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    event_map = {str(item["id"]): item for item in events}
    relation_ids: set[str] = set()
    tolerance = 1e-6
    for item in relations:
        relation_id = str(item.get("relation_id") or "")
        if not relation_id or relation_id in relation_ids:
            raise ValueError("timeline.relations 的 relation_id 必须非空且唯一")
        relation_ids.add(relation_id)
        source_id = item.get("source_event_id")
        target_id = item.get("target_event_id")
        if source_id not in event_map or target_id not in event_map:
            raise ValueError(f"temporal relation 引用未知事件：{relation_id}")
        source = event_map[str(source_id)]
        target = event_map[str(target_id)]
        source_start = float(source["start_time_seconds"])
        source_end = float(source["end_time_seconds"])
        target_start = float(target["start_time_seconds"])
        target_end = float(target["end_time_seconds"])
        relation = item.get("relation")
        if relation == "before":
            gap = target_start - source_end
            valid = gap >= -tolerance
        elif relation == "after":
            gap = source_start - target_end
            valid = gap >= -tolerance
        elif relation == "meets":
            gap = target_start - source_end
            valid = abs(gap) <= tolerance
        elif relation == "overlaps":
            gap = 0.0
            valid = max(source_start, target_start) < min(source_end, target_end)
        elif relation == "during":
            gap = 0.0
            valid = target_start - tolerance <= source_start and source_end <= target_end + tolerance
        elif relation == "starts_before":
            gap = target_start - source_start
            valid = gap >= -tolerance
        elif relation == "starts_after":
            gap = source_start - target_start
            valid = gap >= -tolerance
        elif relation == "starts_with":
            gap = 0.0
            valid = abs(source_start - target_start) <= tolerance
        elif relation == "ends_with":
            gap = 0.0
            valid = abs(source_end - target_end) <= tolerance
        else:
            raise ValueError(f"未知 temporal relation：{relation}")
        minimum_gap = item.get("minimum_gap_seconds")
        maximum_gap = item.get("maximum_gap_seconds")
        if relation in {"before", "after", "meets", "starts_before", "starts_after"}:
            if _is_number(minimum_gap):
                valid = valid and gap + tolerance >= float(minimum_gap)
            if _is_number(maximum_gap):
                valid = valid and gap - tolerance <= float(maximum_gap)
        if not valid:
            raise ValueError(f"temporal relation 与事件时间不一致：{relation_id}")


def _normalize_temporal_relations(
    relations: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Repair qualitative relation labels when the numeric ranges are authoritative.

    Semantic models sometimes use ``before`` to mean "starts before" even when
    two independently timed actions overlap.  Rejecting the whole Brief loses a
    valid timeline, so relations without an explicit numeric gap are rewritten
    to the most precise relation supported by their actual intervals.
    """

    event_map = {str(item["id"]): item for item in events}
    for item in relations:
        source = event_map.get(str(item.get("source_event_id")))
        target = event_map.get(str(item.get("target_event_id")))
        if source is None or target is None:
            continue
        minimum_gap = item.get("minimum_gap_seconds")
        maximum_gap = item.get("maximum_gap_seconds")
        has_explicit_gap = item.get("source_status") == "explicit" and (
            _is_number(maximum_gap)
            or (_is_number(minimum_gap) and float(minimum_gap) > 0.0)
        )
        if has_explicit_gap:
            continue
        if _relation_matches_ranges(item, source, target):
            continue
        item["relation"] = _relation_for_ranges(source, target)
        # 推断出的零间隔不是独立用户约束；标签纠正后不能留下矛盾边界。
        item["minimum_gap_seconds"] = None
        item["maximum_gap_seconds"] = None


def _relation_matches_ranges(
    item: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    probe = dict(item)
    try:
        _validate_temporal_relations([probe], [source, target])
    except ValueError:
        return False
    return True


def _relation_for_ranges(
    source: dict[str, Any],
    target: dict[str, Any],
) -> str:
    tolerance = 1e-6
    source_start = float(source["start_time_seconds"])
    source_end = float(source["end_time_seconds"])
    target_start = float(target["start_time_seconds"])
    target_end = float(target["end_time_seconds"])
    if abs(source_end - target_start) <= tolerance:
        return "meets"
    if source_end < target_start:
        return "before"
    if target_end < source_start:
        return "after"
    if abs(source_start - target_start) <= tolerance:
        return "starts_with"
    if abs(source_end - target_end) <= tolerance:
        return "ends_with"
    if target_start <= source_start and source_end <= target_end:
        return "during"
    if source_start < target_start:
        return "starts_before"
    if source_start > target_start:
        return "starts_after"
    return "overlaps"


def _normalize_motion_metadata(content: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, motion in enumerate(content.get("subject_motion", [])):
        motion_id = motion.get("motion_id") or f"motion_{index + 1:02d}"
        if not isinstance(motion_id, str) or not motion_id or motion_id in seen:
            raise ValueError("subject_motion.motion_id 必须非空且唯一")
        motion["motion_id"] = motion_id
        seen.add(motion_id)
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        _normalize_motion_primitives(motion, semantics)
        if "narrative_required" not in semantics:
            action = motion.get("action")
            semantics["narrative_required"] = bool(
                isinstance(action, dict) and action.get("source_status") == "explicit"
            )


def _normalize_motion_primitives(
    motion: dict[str, Any],
    semantics: dict[str, Any],
) -> None:
    """把叙事动词收敛为 Planning 实际使用的运动事实。

    arrive/depart/transport 等词不能证明几何方向或目标；只有明确来源的
    方向才能进入相对几何，容纳和载运继续作为独立状态关系。
    """

    motion_mode = semantics.get("motion_mode")
    semantics["action_kind"] = {
        "stationary": "hold",
        "local_interaction": "interact",
        "self_propelled": "locomotion",
        "carried": "locomotion",
    }.get(motion_mode, "other")

    direction = motion.get("direction")
    direction_is_explicit = (
        isinstance(direction, dict)
        and direction.get("source_status") == "explicit"
        and direction.get("value") not in {None, ""}
    )
    if motion_mode == "carried":
        semantics["direction_mode"] = "none"
        semantics["target_id"] = None
        # 被运载主体在载体坐标系内静止，世界轨迹只继承 carrier_id。
        semantics["path_type"] = "stationary"
        return

    if not direction_is_explicit and semantics.get("direction_mode") in {
        "world_forward",
        "toward_target",
        "away_from_target",
        "relative_to_target",
    }:
        semantics["direction_mode"] = "none"
        # 容纳已有独立后置状态；几何 target_id 不得重复事件参与者或容器。
        semantics["target_id"] = None


def _apply_scene_dynamics(content: dict[str, Any]) -> None:
    dynamic = bool(content.get("scene_design", {}).get("environmental_motion"))
    for motion in content.get("subject_motion", []):
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        postconditions = semantics.get("postconditions")
        changes_state = isinstance(postconditions, dict) and (
            postconditions.get("contained_by_id") is not None
            or postconditions.get("external_visibility") in {"visible", "hidden"}
        )
        if semantics.get("motion_mode") != "stationary" or changes_state:
            dynamic = True
            break
    expected = "dynamic" if dynamic else "static"
    declared = content.get("scene_dynamics")
    if not isinstance(declared, dict):
        content["scene_dynamics"] = {
            "mode": expected,
            "source_status": "inferred",
            "reason": "根据主体运动和状态转换确定",
        }
        return
    if declared.get("mode") != expected:
        declared.update(
            {
                "mode": expected,
                "source_status": "inferred",
                "reason": "模型分类与主体时间线不一致，已按类型化动作确定性纠正",
            }
        )


def _align_full_timeline_motions(
    content: dict[str, Any],
    events: list[dict[str, Any]],
    duration: float,
) -> None:
    subject_counts: dict[str, int] = {}
    for motion in content["subject_motion"]:
        subject_id = str(motion.get("subject_id"))
        subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1
    for motion in content["subject_motion"]:
        if not _covers_full_timeline(
            motion.get("start_time_seconds"),
            motion.get("end_time_seconds"),
            duration,
        ):
            continue
        event = _matching_timeline_event(motion, events)
        if event is None:
            if (
                motion.get("start_time_seconds") is None
                and motion.get("end_time_seconds") is None
                and subject_counts.get(str(motion.get("subject_id"))) == 1
            ):
                motion["start_time_seconds"] = 0.0
                motion["end_time_seconds"] = duration
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
    semantics = motion.get("motion_semantics")
    event_id = semantics.get("timeline_event_id") if isinstance(semantics, dict) else None
    if event_id is None:
        return None
    return next((event for event in events if event.get("id") == event_id), None)


def _validate_motion_ranges(
    motions: list[dict[str, Any]],
    duration: float,
) -> None:
    for index, motion in enumerate(motions):
        start = motion.get("start_time_seconds")
        end = motion.get("end_time_seconds")
        if not _is_number(start) or not _is_number(end):
            raise ValueError(f"subject_motion[{index}] 必须给出关键时间范围")
        if not 0.0 <= float(start) < float(end) <= duration:
            raise ValueError(f"subject_motion[{index}] 的关键时间范围无效")


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
    subject_ids = {
        str(subject["id"])
        for subject in content["subjects"]
        if isinstance(subject, dict) and isinstance(subject.get("id"), str)
    }
    event_ids = {
        str(event["id"])
        for event in content["timeline"].get("events", [])
        if isinstance(event, dict) and isinstance(event.get("id"), str)
    }
    result: list[dict[str, Any]] = []
    for index, motion in enumerate(content["subject_motion"]):
        semantics = _validated_motion_semantics(motion, subject_ids, event_ids, index)
        motion_type = semantics["motion_type"]
        speed_range = deepcopy(rules["speed_ranges_mps"].get(
            motion_type,
            rules["defaults"]["moving_speed_mps"],
        ))
        direction_mode = semantics["direction_mode"]
        result.append(
            {
                "motion_index": index,
                "motion_id": motion["motion_id"],
                "subject_id": motion.get("subject_id"),
                "action_kind": semantics["action_kind"],
                "motion_type": motion_type,
                "motion_mode": semantics["motion_mode"],
                "speed_range_mps": speed_range,
                "direction_mode": direction_mode,
                "direction_vector_world": (
                    deepcopy(rules["coordinate_system"]["forward_vector"])
                    if direction_mode == "world_forward"
                    else None
                ),
                "target_id": semantics["target_id"],
                "carrier_id": semantics["carrier_id"],
                "path_type": semantics["path_type"],
                "timeline_event_id": semantics["timeline_event_id"],
                "narrative_required": bool(semantics["narrative_required"]),
                "postconditions": deepcopy(semantics["postconditions"]),
                "start_time_seconds": _number_or(motion.get("start_time_seconds"), 0.0),
                "end_time_seconds": _number_or(motion.get("end_time_seconds"), duration),
                "source_status": semantics["source_status"],
            }
        )
    return result


def _validated_motion_semantics(
    motion: dict[str, Any],
    subject_ids: set[str],
    event_ids: set[str],
    index: int,
) -> dict[str, Any]:
    """校验模型给出的类型语义；量化层不重新解释动作文本。"""
    semantics = motion.get("motion_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"subject_motion[{index}] 缺少 motion_semantics")

    subject_id = motion.get("subject_id")
    if subject_id not in subject_ids:
        raise ValueError(f"subject_motion[{index}].subject_id 未引用有效主体")

    motion_type = semantics.get("motion_type")
    motion_mode = semantics.get("motion_mode")
    expected_type = {
        "stationary": "static",
        "local_interaction": "interactive",
        "carried": "carried",
    }.get(motion_mode)
    if expected_type is not None and motion_type != expected_type:
        raise ValueError(
            f"subject_motion[{index}] 的 motion_mode={motion_mode} "
            f"必须使用 motion_type={expected_type}"
        )
    if motion_mode == "self_propelled" and motion_type in {
        "static",
        "interactive",
        "carried",
    }:
        raise ValueError(f"subject_motion[{index}] 的自主运动类型不一致")

    target_id = semantics.get("target_id")
    direction_mode = semantics.get("direction_mode")
    if direction_mode in {
        "toward_target",
        "away_from_target",
        "relative_to_target",
    } and target_id not in subject_ids:
        raise ValueError(f"subject_motion[{index}] 的相对方向缺少有效 target_id")
    if target_id == subject_id:
        raise ValueError(f"subject_motion[{index}] 不能以自身作为 target_id")

    carrier_id = semantics.get("carrier_id")
    if motion_mode == "carried":
        if carrier_id not in subject_ids or carrier_id == subject_id:
            raise ValueError(f"subject_motion[{index}] 的 carried 运动缺少有效 carrier_id")
    elif carrier_id is not None:
        raise ValueError(f"subject_motion[{index}] 仅 carried 运动可设置 carrier_id")

    postconditions = semantics.get("postconditions")
    contained_by_id = (
        postconditions.get("contained_by_id")
        if isinstance(postconditions, dict)
        else None
    )
    if contained_by_id is not None and (
        contained_by_id not in subject_ids or contained_by_id == subject_id
    ):
        raise ValueError(f"subject_motion[{index}] 的 contained_by_id 无效")
    if contained_by_id is not None and target_id is not None and target_id != contained_by_id:
        raise ValueError(
            f"subject_motion[{index}] 的几何目标与 contained_by_id 冲突"
        )

    path_type = semantics.get("path_type")
    if motion_mode == "stationary" and path_type != "stationary":
        raise ValueError(f"subject_motion[{index}] 的静止阶段必须使用 stationary 路径")
    if motion_mode == "self_propelled" and path_type == "stationary":
        raise ValueError(f"subject_motion[{index}] 的自主运动不能使用 stationary 路径")
    if path_type in {"circular", "elliptical"} and (
        direction_mode != "relative_to_target" or target_id is None
    ):
        raise ValueError(
            f"subject_motion[{index}] 的相对闭合路径缺少有效几何目标"
        )

    timeline_event_id = semantics.get("timeline_event_id")
    if timeline_event_id is not None and timeline_event_id not in event_ids:
        raise ValueError(f"subject_motion[{index}] 的 timeline_event_id 未引用有效事件")
    if not isinstance(semantics.get("narrative_required"), bool):
        raise ValueError(f"subject_motion[{index}] 缺少 narrative_required")
    return semantics


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


def _camera_profile_for_explicit_movement(
    content: dict[str, Any],
    rules: dict[str, Any],
    base_camera: dict[str, Any],
) -> dict[str, Any]:
    """选择与显式运镜一致的运动模板，同时保留当前情绪的镜头造型。"""
    movement_node = content.get("camera", {}).get("movement", {}).get("type")
    if _status(movement_node) != "explicit":
        return deepcopy(base_camera)
    movement = _camera_movement_kind(_value(movement_node))
    if movement is None:
        return deepcopy(base_camera)

    resolved = deepcopy(base_camera)
    motion_fields = {
        "movement",
        "speed_policy",
        "speed_mps",
        "start_distance_m",
        "end_distance_m",
    }
    for emotion_profile in rules["emotion_classes"].values():
        template = emotion_profile["camera"]
        if template.get("movement") == movement:
            for field in motion_fields:
                resolved[field] = deepcopy(template.get(field))
            return resolved

    resolved["movement"] = movement
    return resolved


def _camera_movement_kind(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = (
        ("push_in", ("push_in", "pushin", "dolly_in", "推近", "推进")),
        ("pull_out", ("pull_out", "pullout", "dolly_out", "拉远", "后拉", "拉开")),
        ("orbit", ("orbit", "环绕", "绕拍")),
        ("follow", ("follow", "跟随", "跟拍")),
        ("lateral", ("lateral", "truck", "横移", "侧移")),
        ("static", ("static", "fixed", "静止", "固定")),
    )
    for kind, markers in aliases:
        if any(marker in normalized for marker in markers):
            return kind
    return None


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
        "speed_ranges_mps",
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
