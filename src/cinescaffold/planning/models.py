from __future__ import annotations

import re
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from cinescaffold.camera_semantics import classify_camera_movement
from cinescaffold.errors import ConfigurationError
from cinescaffold.motion_semantics import planning_motion_shape
from cinescaffold.planning.design import SkeletonRelation, route_anchor_time_seconds
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.relationships import classify_relationship

# PydanticAI 2.36 predates this official alias, so its bundled DeepSeek profile
# would otherwise send tool_choice="required", which Flash thinking rejects.
_DEEPSEEK_FLASH_PROFILE = OpenAIModelProfile(
    supports_thinking=True,
    openai_supports_tool_choice_required=False,
)


def create_planning_model(
    provider: str,
    model_name: str | None,
    objective_brief: ObjectivePlanningBrief,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    disable_request_timeout: bool = False,
) -> Model:
    if provider == "mock":
        return _create_mock_model(objective_brief)
    if not model_name:
        raise ConfigurationError(f"{provider} Agent 需要显式指定 --model")
    if not api_key:
        environment_name = (
            "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        )
        raise ConfigurationError(f"缺少 {environment_name}")
    if provider == "openai":
        if disable_request_timeout:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
                timeout=None,
            )
            return OpenAIResponsesModel(
                model_name,
                provider=OpenAIProvider(openai_client=client),
            )
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )
    if provider == "deepseek":
        if base_url or disable_request_timeout:
            client_options: dict[str, Any] = {
                "api_key": api_key,
                "base_url": base_url or "https://api.deepseek.com",
            }
            if disable_request_timeout:
                client_options["timeout"] = None
            client = AsyncOpenAI(**client_options)
            deepseek_provider = DeepSeekProvider(openai_client=client)
        else:
            deepseek_provider = DeepSeekProvider(api_key=api_key)
        profile = _DEEPSEEK_FLASH_PROFILE if model_name == "deepseek-flash" else None
        return OpenAIChatModel(
            model_name,
            provider=deepseek_provider,
            profile=profile,
        )
    raise ConfigurationError(f"未知 Agent Provider：{provider}")


def _create_mock_model(objective: ObjectivePlanningBrief) -> FunctionModel:
    actions = _mock_design_actions(objective)

    def callback(messages: list, info: AgentInfo) -> ModelResponse:
        returns = _tool_returns(messages)
        action_index = len(returns)
        available_tools = {item.name for item in info.function_tools}
        usage = RequestUsage(
            input_tokens=120 + action_index * 10,
            output_tokens=30,
            details={"mock_estimated": 1},
        )
        commit_ready = any(
            isinstance(item.content, dict)
            and (
                item.content.get("data", {}).get("commit_ready") is True
                or item.content.get("data", {})
                .get("acceptance", {})
                .get("commit_ready")
                is True
            )
            for item in returns
        )
        next_action = next(
            (item for item in actions[action_index:] if item[0] in available_tools),
            None,
        )
        if next_action is not None and not commit_ready:
            name, arguments = next_action
            if name == "apply_design_option":
                options = next(
                    (
                        item.content.get("data", {}).get("options", [])
                        for item in reversed(returns)
                        if isinstance(item.content, dict)
                        and item.tool_name == "request_design_options"
                    ),
                    [],
                )
                if not options:
                    raise ValueError("Mock Agent 未收到 Design Options")
                arguments = {
                    "base_revision": options[0]["base_revision"],
                    "option_id": options[0]["option_id"],
                }
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        name, arguments, tool_call_id=f"mock_call_{action_index:03d}"
                    )
                ],
                usage=usage,
                finish_reason="tool_call",
            )
        revision = max(_latest_revision(returns), _prompt_revision(messages))
        output_tool = next(
            item for item in info.output_tools if item.name.endswith("CommitRequest")
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name,
                    {
                        "type": "commit_request",
                        "candidate_revision": revision,
                        "summary": "Mock Agent 已完成确定性构造与验证。",
                    },
                    tool_call_id="mock_final",
                )
            ],
            usage=usage,
            finish_reason="stop",
        )

    return FunctionModel(callback, model_name="mock-scene-planner-v0.1")


def _mock_design_actions(
    objective: ObjectivePlanningBrief,
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "submit_scene_skeleton",
            {"skeleton": build_deterministic_scene_skeleton(objective)},
        ),
        (
            "request_design_options",
            {"preference": "balanced", "max_options": 3},
        ),
        ("apply_design_option", {}),
        (
            "solve_candidate",
            {
                "scope": "all",
                "constraint_ids": [],
                "locked_variables": [],
                "strategy": "auto",
            },
        ),
    ]


def build_deterministic_scene_skeleton(
    objective: ObjectivePlanningBrief,
) -> dict[str, Any]:
    """Build a conservative symbolic skeleton from an already typed Objective Brief.

    The planner mock and the product fallback share this path so fallback behavior
    remains deterministic, validated, and independent of another Provider call.
    """

    entities: list[dict[str, Any]] = []
    categories: dict[str, str] = {}
    for index, subject in enumerate(objective.subjects):
        entity_id = str(subject.get("id") or f"entity_{index + 1:02d}")
        category = _annotated_value(subject.get("category")) or "object"
        categories[entity_id] = category
        entities.append(
            {
                "entity_id": entity_id,
                "semantic_type": category,
                "role": _annotated_value(subject.get("narrative_role")) or "subject",
                "proxy_family": _mock_proxy_family(category),
                "scale_intent": _mock_scale_intent(subject),
                "source_refs": [
                    item.path
                    for item in objective.explicit_requirements
                    if item.path.startswith(f"content.subjects[{index}]")
                ],
            }
        )
    environment = _annotated_value(objective.scene_design.get("environment"))
    ground_id = None
    if environment and _environment_supports_ground(objective, environment):
        ground_id = "environment_ground"
        suffix = 1
        while ground_id in categories:
            ground_id = f"environment_ground_{suffix:02d}"
            suffix += 1
        entities.insert(
            0,
            {
                "entity_id": ground_id,
                "semantic_type": environment,
                "role": "environment",
                "proxy_family": "ground_plane",
                "scale_intent": "large",
                "source_refs": [
                    item.path
                    for item in objective.explicit_requirements
                    if item.path == "content.scene_design.environment"
                ],
            },
        )

    relations: list[dict[str, Any]] = []
    for index, relationship in enumerate(
        objective.scene_design.get("relationships", [])
    ):
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        if not subject_id or not reference_id:
            continue
        meaning = classify_relationship(relationship)
        if meaning is None:
            raise ValueError(
                f"当前 Cinematic Brief 使用了非规范关系类型："
                f"{relationship.get('type')}"
            )
        kind = {
            "far": "camera_depth_order",
            "proximity": "proximity",
            "relative_position": "relative_position",
            "scale_dominance": "scale_dominance",
        }[meaning.kind]
        relation_payload = {
            "relation_id": f"relationship_{index + 1:02d}",
            "kind": kind,
            "subject_id": subject_id,
            "reference_id": reference_id,
            "source_status": _planning_source_status(
                relationship.get("source_status"),
            ),
            "source_ref": f"content.scene_design.relationships[{index}]",
            "timeline_event_id": relationship.get("timeline_event_id"),
            "temporal_mode": relationship.get("temporal_mode", "throughout"),
        }
        if meaning.direction is not None:
            relation_payload["direction"] = meaning.direction
        relations.append(relation_payload)

    phases: list[dict[str, Any]] = []
    phase_ranges: dict[str, tuple[float, float]] = {}
    for index, motion in enumerate(objective.subject_motion):
        subject_id = motion.get("subject_id")
        if not subject_id:
            continue
        semantics = (
            motion.get("motion_semantics")
            if isinstance(motion.get("motion_semantics"), dict)
            else {}
        )
        motion_type = semantics.get("motion_type")
        event_id = semantics.get("timeline_event_id") or _mock_motion_event_id(
            objective,
            motion,
        )
        speed_intent = _mock_speed_intent(
            _annotated_value(motion.get("speed")),
            motion_type,
        )
        semantics_status = _planning_source_status(
            semantics.get("source_status"),
        )
        action_status = (
            _planning_source_status(
                motion.get("action", {}).get("source_status"),
            )
            if isinstance(motion.get("action"), dict)
            else "inferred"
        )
        if semantics_status == "explicit":
            phase_source_status = semantics_status
            phase_source_ref = f"content.subject_motion[{index}].motion_semantics"
        elif action_status == "explicit":
            # The typed interpretation may be inferred even though the user's
            # action itself is explicit. Keep that user requirement attached to
            # the phase that actually implements it.
            phase_source_status = action_status
            phase_source_ref = f"content.subject_motion[{index}].action"
        else:
            phase_source_status = semantics_status
            phase_source_ref = (
                f"content.subject_motion[{index}].motion_semantics"
                if semantics
                else f"content.subject_motion[{index}].action"
            )
        typed_shape = planning_motion_shape(semantics)
        kind = typed_shape.kind
        path_family = typed_shape.path_family
        direction_mode = typed_shape.direction_mode
        target_id = typed_shape.target_id
        carrier_id = typed_shape.carrier_id
        phase_id = f"motion_{index + 1:02d}"
        phases.append(
            {
                "phase_id": phase_id,
                "motion_id": motion.get("motion_id"),
                "subject_id": subject_id,
                "kind": kind,
                "timeline_event_id": event_id,
                "target_id": target_id,
                "carrier_id": carrier_id,
                "direction_mode": direction_mode,
                "path_family": path_family,
                "local_components": (
                    list(semantics.get("local_components") or [])
                    if kind == "local_transform"
                    else []
                ),
                "speed_intent": speed_intent,
                "speed_source_status": (
                    motion["speed"].get("source_status")
                    if isinstance(motion.get("speed"), dict)
                    and motion["speed"].get("source_status") != "unknown"
                    else None
                ),
                "speed_source_ref": (
                    f"content.subject_motion[{index}].speed"
                    if isinstance(motion.get("speed"), dict)
                    and motion["speed"].get("source_status") != "unknown"
                    else None
                ),
                "source_status": phase_source_status,
                "source_ref": phase_source_ref,
                "narrative_required": bool(semantics.get("narrative_required")),
            }
        )
        start_seconds = motion.get("start_time_seconds")
        end_seconds = motion.get("end_time_seconds")
        if isinstance(start_seconds, (int, float)) and isinstance(
            end_seconds, (int, float)
        ):
            phase_ranges[phase_id] = (float(start_seconds), float(end_seconds))
        postconditions = semantics.get("postconditions")
        contained_by_id = (
            postconditions.get("contained_by_id")
            if isinstance(postconditions, dict)
            else None
        )
        if (
            contained_by_id in categories
            and event_id is not None
            and semantics.get("motion_mode") != "carried"
        ):
            pair = {str(subject_id), str(contained_by_id)}
            has_transition_relation = any(
                item["kind"] == "proximity"
                and {item["subject_id"], item["reference_id"]} == pair
                and item.get("timeline_event_id") == event_id
                for item in relations
            )
            if not has_transition_relation:
                # A containment transition implies a shared spatial boundary,
                # regardless of the narrative verb used to describe it.
                relations.append(
                    {
                        "relation_id": f"containment_transition_{index + 1:02d}",
                        "kind": "proximity",
                        "subject_id": str(contained_by_id),
                        "reference_id": str(subject_id),
                        "timeline_event_id": event_id,
                        "temporal_mode": "at_end",
                        "source_status": "inferred",
                        "source_ref": f"content.subject_motion[{index}].motion_semantics",
                    }
                )
        if isinstance(postconditions, dict) and postconditions.get(
            "external_visibility"
        ) in {"becomes_visible", "becomes_hidden"}:
            visibility_state = {
                "becomes_visible": "visible",
                "becomes_hidden": "hidden",
            }[postconditions["external_visibility"]]
            phases.append(
                {
                    "phase_id": f"motion_{index + 1:02d}_visibility",
                    "motion_id": motion.get("motion_id"),
                    "subject_id": subject_id,
                    "kind": "visibility",
                    "timeline_event_id": event_id,
                    "target_id": None,
                    "carrier_id": None,
                    "direction_mode": "none",
                    "path_family": "stationary",
                    "speed_intent": "unspecified",
                    "speed_source_status": None,
                    "speed_source_ref": None,
                    "visibility_state": visibility_state,
                    "transition_at": "at_end",
                    "source_status": phase_source_status,
                    "source_ref": phase_source_ref,
                    "narrative_required": bool(semantics.get("narrative_required")),
                }
            )

    event_ids = {
        str(item.get("id"))
        for item in objective.timeline.get("events", [])
        if item.get("id") is not None
        and isinstance(item.get("start_time_seconds"), (int, float))
        and isinstance(item.get("end_time_seconds"), (int, float))
    }
    route_anchors: dict[str, list[dict[str, Any]]] = {}
    used_anchor_bindings: set[tuple[str, str]] = set()
    duration_node = objective.timeline.get("duration_resolution")
    frame_count = (
        duration_node.get("frame_count") if isinstance(duration_node, dict) else None
    )
    duration_seconds = (
        duration_node.get("resolved_duration_seconds")
        if isinstance(duration_node, dict)
        else objective.timeline.get("duration_seconds")
    )
    frame_step = (
        float(duration_seconds) / float(frame_count)
        if isinstance(duration_seconds, (int, float))
        and isinstance(frame_count, int)
        and frame_count > 0
        else 1.0 / 24.0
    )
    for relation in relations:
        event_id = relation.get("timeline_event_id")
        if (
            relation.get("kind") != "proximity"
            or event_id not in event_ids
        ):
            continue
        anchor_time = route_anchor_time_seconds(
            objective,
            SkeletonRelation.model_validate(relation),
            float(duration_seconds),
            frame_step,
        )
        participants = {relation["subject_id"], relation["reference_id"]}
        for subject_id in sorted(participants):
            exact_candidates = [
                phase
                for phase in phases
                if phase.get("kind") == "path_move"
                and phase.get("subject_id") == subject_id
                and phase.get("timeline_event_id") == event_id
                and phase["phase_id"] in phase_ranges
                and phase_ranges[phase["phase_id"]][0] - 1e-6
                <= anchor_time
                <= phase_ranges[phase["phase_id"]][1] + 1e-6
            ]
            candidates = exact_candidates or [
                phase
                for phase in phases
                if phase.get("kind") == "path_move"
                and phase.get("subject_id") == subject_id
                and phase["phase_id"] in phase_ranges
                and phase_ranges[phase["phase_id"]][0] - 1e-6
                <= anchor_time
                <= phase_ranges[phase["phase_id"]][1] + 1e-6
            ]
            if not candidates:
                continue
            phase = min(
                candidates,
                key=lambda item: (
                    phase_ranges[item["phase_id"]][1]
                    - phase_ranges[item["phase_id"]][0],
                    phase_ranges[item["phase_id"]][0],
                    item["phase_id"],
                ),
            )
            slot = (phase["phase_id"], relation["relation_id"])
            if slot in used_anchor_bindings:
                continue
            used_anchor_bindings.add(slot)
            route_anchors.setdefault(subject_id, []).append(
                {
                    "anchor_id": (
                        f"route_anchor_{relation['relation_id']}_{subject_id}"
                    ),
                    "phase_id": phase["phase_id"],
                    "relation_id": relation["relation_id"],
                }
            )
    route_intents = [
        {
            "route_id": f"route_{subject_id}",
            "subject_id": subject_id,
            "anchors": anchors,
            # A proximity event establishes a waypoint, not a globally straight
            # route. Direction continuity must come from an explicit route axis
            # or the planner's scene-level judgment; the deterministic fallback
            # must not turn one local meeting point into a no-turn hard rule.
            "continuity": "allow_turns",
        }
        for subject_id, anchors in sorted(route_anchors.items())
    ]

    focus_target = _annotated_value(objective.camera.get("focus_target_id"))
    movement = (
        classify_camera_movement(
            _annotated_value(objective.camera.get("movement", {}).get("type"))
        )
        or "static"
    )
    movement_status = (
        objective.camera.get("movement", {})
        .get("type", {})
        .get("source_status", "inferred")
        if isinstance(objective.camera.get("movement", {}).get("type"), dict)
        else "inferred"
    )
    if movement_status == "unknown":
        movement_status = "default"
    camera_speed = objective.camera.get("movement", {}).get("speed")
    camera_speed_status = (
        camera_speed.get("source_status")
        if isinstance(camera_speed, dict)
        and camera_speed.get("source_status") != "unknown"
        else None
    )
    return {
        "entities": entities,
        "relations": relations,
        "motion_phases": phases,
        "route_intents": route_intents,
        "camera_intent": {
            "movement": movement,
            "focus_target_id": focus_target,
            "movement_target_id": objective.camera.get("movement", {}).get("target_id"),
            "view_relation_to_motion": (
                _annotated_value(objective.camera.get("view_relation_to_motion"))
                or "unspecified"
            ),
            "speed_intent": _mock_speed_intent(
                _annotated_value(camera_speed),
                "static" if movement == "static" else None,
            ),
            "speed_source_status": camera_speed_status,
            "speed_source_ref": (
                "content.camera.movement.speed"
                if camera_speed_status is not None
                else None
            ),
            "source_status": movement_status,
            "source_ref": "content.camera.movement.type",
        },
    }


def _mock_scene_skeleton(objective: ObjectivePlanningBrief) -> dict[str, Any]:
    """Test alias for the deterministic fallback builder."""

    return build_deterministic_scene_skeleton(objective)


def _planning_source_status(
    value: Any,
    *,
    fallback: Any = "inferred",
) -> str:
    """Collapse boundary-only ``unknown`` into a non-hard Planning status."""

    allowed = {"explicit", "inferred", "default", "agent_selected"}
    if value in allowed:
        return str(value)
    return str(fallback) if fallback in allowed else "inferred"


def _mock_proxy_family(category: str) -> str:
    lowered = category.lower()
    if _contains_category_marker(
        lowered,
        (
            "道路",
            "公路",
            "马路",
            "地面",
            "地板",
            "road",
            "street",
            "ground",
            "floor",
        ),
    ):
        return "ground_plane"
    if lowered in {"车", "car", "ship", "vehicle"} or _contains_category_marker(
        lowered,
        (
            "飞船",
            "载具",
            "车辆",
            "汽车",
            "卡车",
            "摩托",
            "无人机",
            "spaceship",
            "spacecraft",
            "ship",
            "vehicle",
            "truck",
            "motorcycle",
            "drone",
        ),
    ):
        return "vehicle_box"
    # Celestial phrases must be classified before the generic Chinese 人 marker:
    # 人造卫星 is a satellite, not a human.  Vehicle phrases were already handled
    # above, so 月球车 and 太阳能车 still remain vehicles.
    if _contains_category_marker(
        lowered,
        (
            "太阳",
            "地球",
            "月亮",
            "月球",
            "卫星",
            "sun",
            "earth",
            "moon",
            "planet",
            "satellite",
            "star",
        ),
    ):
        return "celestial_sphere"
    if _contains_category_marker(
        lowered,
        (
            "人",
            "男",
            "女",
            "乘客",
            "行人",
            "儿童",
            "孩子",
            "man",
            "woman",
            "person",
            "human",
            "passenger",
            "pedestrian",
            "child",
        ),
    ):
        return "human_capsule"
    return "generic_box"


def _contains_category_marker(value: str, markers: tuple[str, ...]) -> bool:
    """Match CJK markers as phrases and Latin markers as complete words.

    Raw substring matching made unrelated English words such as ``friendship``
    inherit the ``ship`` proxy and made mixed Chinese terms such as 人造卫星 hit
    the generic 人 marker.  Category labels can contain spaces or punctuation,
    so word boundaries are a safer deterministic fallback for Latin aliases.
    """

    for marker in markers:
        if re.search(r"[\u3400-\u9fff]", marker):
            if marker in value:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", value):
            return True
    return False


def _environment_supports_ground(
    objective: ObjectivePlanningBrief,
    environment: str,
) -> bool:
    """Do not invent a support plane for space, blank, or volumetric scenes."""

    lowered = environment.lower()
    if _contains_category_marker(
        lowered,
        (
            "荒漠",
            "沙漠",
            "道路",
            "公路",
            "马路",
            "路边",
            "街道",
            "城市",
            "室内",
            "房间",
            "森林",
            "山",
            "雪地",
            "废墟",
            "草地",
            "田野",
            "地面",
            "desert",
            "road",
            "street",
            "city",
            "interior",
            "room",
            "forest",
            "mountain",
            "snow",
            "ruins",
            "field",
            "ground",
        ),
    ):
        # The semantic environment is the primary source. A coarse `blank`
        # asset fallback must not erase an explicit road, room, or terrain.
        return True
    scene = (objective.translation_parameters or {}).get("scene", {})
    asset_key = scene.get("asset_key") if isinstance(scene, dict) else None
    if isinstance(asset_key, str):
        if asset_key in {
            "desert",
            "city",
            "interior",
            "forest",
            "mountain",
            "snowfield",
            "ruins",
        }:
            return True
        if asset_key in {"space", "ocean", "blank"}:
            return False
    return False


def _mock_scale_intent(subject: dict[str, Any]) -> str:
    category = _annotated_value(subject.get("category")) or ""
    scale_evidence: list[str] = [category]
    for attribute in subject.get("attributes", []):
        if isinstance(attribute, dict):
            scale_evidence.extend(
                str(attribute.get(field) or "") for field in ("name", "value")
            )
    rendered = " ".join(scale_evidence).lower()
    if any(marker in rendered for marker in ("巨大", "huge", "giant")):
        return "huge"
    family = _mock_proxy_family(category)
    lowered = category.lower()
    if family == "human_capsule":
        return "human"
    if family == "celestial_sphere":
        if _contains_category_marker(lowered, ("太阳", "sun", "star")):
            return "huge"
        if _contains_category_marker(
            lowered, ("月亮", "月球", "卫星", "moon", "satellite")
        ):
            return "small"
        return "large"
    if family == "vehicle_box":
        return "large"
    return "unspecified"


def _mock_motion_event_id(
    objective: ObjectivePlanningBrief,
    motion: dict[str, Any],
) -> str | None:
    """Mock 仅按已经结构化的时间边界关联事件。"""

    start = motion.get("start_time_seconds")
    end = motion.get("end_time_seconds")
    if start is None or end is None:
        return None
    for event in objective.timeline.get("events", []):
        if (
            event.get("id")
            and event.get("start_time_seconds") == start
            and event.get("end_time_seconds") == end
        ):
            return str(event["id"])
    return None


def _mock_speed_intent(value: str | None, motion_type: str | None) -> str:
    """仅供离线 Mock 夹具把既有类型化速度映射为档位。"""

    if motion_type == "static":
        return "stationary"
    if motion_type in {"walking", "slow"}:
        return "slow"
    if motion_type in {"running", "flying", "fast"}:
        return "fast"
    lowered = (value or "").lower()
    if any(
        marker in lowered for marker in ("与主体速度同步", "同步主体", "match_subject")
    ):
        return "match_subject"
    if lowered in {"静止", "0 m/s", "stationary"}:
        return "stationary"
    if any(marker in lowered for marker in ("缓慢", "慢", "slow")):
        return "slow"
    if any(marker in lowered for marker in ("快速", "快", "fast")):
        return "fast"
    return "unspecified"


def _tool_returns(messages: list) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _latest_revision(returns: list[ToolReturnPart]) -> int:
    for item in reversed(returns):
        if isinstance(item.content, dict) and isinstance(
            item.content.get("revision_after"), int
        ):
            return item.content["revision_after"]
    return 0


def _prompt_revision(messages: list) -> int:
    """恢复测试从首条用户提示读取可信 Candidate revision。"""

    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            content = getattr(part, "content", None)
            if not isinstance(content, str):
                continue
            match = re.search(r"Candidate revision (\d+)", content)
            if match:
                return int(match.group(1))
    return 0


def _annotated_value(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return None
