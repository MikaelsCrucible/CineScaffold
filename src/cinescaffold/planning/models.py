from __future__ import annotations

import json
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

from cinescaffold.errors import ConfigurationError
from cinescaffold.planning.objective import ObjectivePlanningBrief


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
        environment_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
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
            (
                item
                for item in actions[action_index:]
                if item[0] in available_tools
            ),
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
                parts=[ToolCallPart(name, arguments, tool_call_id=f"mock_call_{action_index:03d}")],
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
    if environment:
        ground_id = "environment_ground"
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
    orbit_targets: dict[str, str] = {}
    board_targets: dict[str, str] = {}
    if ground_id:
        for entity_id, category in categories.items():
            if _mock_proxy_family(category) in {"human_capsule", "vehicle_box"}:
                relations.append(
                    {
                        "relation_id": f"ground_{entity_id}",
                        "kind": "ground_support",
                        "subject_id": entity_id,
                        "reference_id": ground_id,
                        "source_status": "inferred",
                        "source_ref": "translation_parameters.scene.asset_key",
                    }
                )
    for index, relationship in enumerate(objective.scene_design.get("relationships", [])):
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        relation_type = str(relationship.get("type") or "").lower()
        if subject_id and not reference_id and ground_id and relation_type == "beside_road":
            relations.append(
                {
                    "relation_id": f"relationship_{index + 1:02d}",
                    "kind": "ground_support",
                    "subject_id": subject_id,
                    "reference_id": ground_id,
                    "source_status": relationship.get("source_status", "inferred"),
                    "source_ref": f"content.scene_design.relationships[{index}]",
                }
            )
            continue
        if not subject_id or not reference_id:
            continue
        relation_text = str(relationship.get("type") or relationship.get("strength") or "").lower()
        if any(marker in relation_text for marker in ("远", "far", "background", "distant")):
            kind = "camera_depth_order"
        elif any(marker in relation_text for marker in ("orbit", "公转", "绕")):
            kind = "orbit_around"
        else:
            kind = "proximity"
        relation_payload = {
            "relation_id": f"relationship_{index + 1:02d}",
            "kind": kind,
            "subject_id": subject_id,
            "reference_id": reference_id,
            "source_status": relationship.get("source_status", "inferred"),
            "source_ref": f"content.scene_design.relationships[{index}]",
        }
        if relation_type == "stop_beside":
            relation_payload.update(
                {"timeline_event_id": "wait_and_arrive", "temporal_mode": "at_end"}
            )
        elif relation_type == "board_into":
            relation_payload.update(
                {"timeline_event_id": "boarding", "temporal_mode": "at_end"}
            )
            board_targets[str(subject_id)] = str(reference_id)
        relations.append(relation_payload)
        if kind == "orbit_around":
            orbit_targets[str(subject_id)] = str(reference_id)

    visual_scales = objective.composition.get("visual_scales", [])
    subject_ids = list(categories)
    for index, visual_scale in enumerate(visual_scales):
        scale = visual_scale.get("scale") if isinstance(visual_scale, dict) else None
        if not isinstance(scale, dict) or scale.get("source_status") != "explicit":
            continue
        subject_id = visual_scale.get("subject_id")
        reference_id = next((item for item in subject_ids if item != subject_id), None)
        if subject_id in categories and reference_id is not None:
            relations.append(
                {
                    "relation_id": f"visual_scale_{index + 1:02d}",
                    "kind": "scale_dominance",
                    "subject_id": subject_id,
                    "reference_id": reference_id,
                    "source_status": "explicit",
                    "source_ref": f"content.composition.visual_scales[{index}].scale",
                }
            )

    phases: list[dict[str, Any]] = []
    phase_ranges: dict[str, tuple[float, float]] = {}
    for index, motion in enumerate(objective.subject_motion):
        subject_id = motion.get("subject_id")
        if not subject_id:
            continue
        semantics = motion.get("motion_semantics") if isinstance(motion.get("motion_semantics"), dict) else {}
        motion_type = semantics.get("motion_type")
        path_type = semantics.get("path_type")
        event_id = semantics.get("timeline_event_id") or _mock_motion_event_id(
            objective,
            motion,
        )
        speed_intent = _mock_speed_intent(
            _annotated_value(motion.get("speed")),
            motion_type,
        )
        if (
            path_type in {"circular", "elliptical", "orbit_around"}
            or subject_id in orbit_targets
        ):
            kind = "orbit"
            path_family = "circle"
            direction_mode = "orbit_around"
            target_id = semantics.get("target_id") or orbit_targets.get(subject_id)
            carrier_id = semantics.get("carrier_id")
        elif (
            semantics.get("motion_mode") == "carried"
            or (subject_id in board_targets and event_id == "departure")
        ):
            kind = "carried"
            path_family = "stationary"
            direction_mode = "none"
            target_id = None
            carrier_id = semantics.get("carrier_id") or board_targets.get(subject_id)
        elif motion_type == "interactive" or semantics.get("motion_mode") == "local_interaction":
            kind = "local_transform"
            path_family = "stationary"
            direction_mode = "none"
            target_id = None
            carrier_id = semantics.get("carrier_id")
        elif (
            motion_type == "static"
            or speed_intent == "stationary"
            or (subject_id in board_targets and event_id == "wait_and_arrive")
            # 进入关系是状态转换，不是把乘员代理驶入载体中心的指令。
            or (subject_id in board_targets and event_id == "boarding")
        ):
            kind = "hold"
            path_family = "stationary"
            direction_mode = "none"
            target_id = None
            carrier_id = semantics.get("carrier_id")
        else:
            kind = "linear_move"
            path_family = (
                "parabolic"
                if path_type == "parabolic"
                else "catmull_rom"
                if path_type == "s_curve"
                else "lemniscate"
                if path_type == "figure_eight"
                else "linear"
            )
            target_id = semantics.get("target_id")
            direction_mode = semantics.get("direction_mode") or "none"
            carrier_id = semantics.get("carrier_id")
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
                    ["rotation", "scale"] if kind == "local_transform" else []
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
                "source_status": semantics.get("source_status") or (
                    motion.get("action", {}).get("source_status", "inferred")
                    if isinstance(motion.get("action"), dict)
                    else "inferred"
                ),
                "source_ref": (
                    f"content.subject_motion[{index}].motion_semantics"
                    if semantics
                    else f"content.subject_motion[{index}].action"
                ),
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
        if (
            isinstance(postconditions, dict)
            and postconditions.get("external_visibility") in {"visible", "hidden"}
        ):
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
                    "visibility_state": postconditions["external_visibility"],
                    "transition_at": "at_end",
                    "source_status": semantics.get("source_status", "inferred"),
                    "source_ref": f"content.subject_motion[{index}].motion_semantics",
                    "narrative_required": bool(semantics.get("narrative_required")),
                }
            )

    event_ends = {
        str(item.get("id")): float(item["end_time_seconds"])
        for item in objective.timeline.get("events", [])
        if item.get("id") is not None
        and isinstance(item.get("end_time_seconds"), (int, float))
    }
    route_anchors: dict[str, list[dict[str, Any]]] = {}
    used_boundaries: set[tuple[str, str]] = set()
    for relation in relations:
        event_id = relation.get("timeline_event_id")
        if (
            relation.get("kind") != "proximity"
            or relation.get("temporal_mode") != "at_end"
            or event_id not in event_ends
        ):
            continue
        anchor_time = event_ends[event_id]
        participants = {relation["subject_id"], relation["reference_id"]}
        candidates = [
            (phase_ranges[phase["phase_id"]][1], phase)
            for phase in phases
            if phase.get("kind") == "linear_move"
            and phase.get("subject_id") in participants
            and phase["phase_id"] in phase_ranges
            and phase_ranges[phase["phase_id"]][1] <= anchor_time
        ]
        if not candidates:
            continue
        _, phase = max(candidates, key=lambda item: item[0])
        boundary = (phase["phase_id"], "at_end")
        if boundary in used_boundaries:
            continue
        used_boundaries.add(boundary)
        subject_id = str(phase["subject_id"])
        route_anchors.setdefault(subject_id, []).append(
            {
                "anchor_id": f"route_anchor_{relation['relation_id']}",
                "phase_id": phase["phase_id"],
                "boundary": "at_end",
                "relation_id": relation["relation_id"],
            }
        )
    route_intents = [
        {
            "route_id": f"route_{subject_id}",
            "subject_id": subject_id,
            "anchors": anchors,
            "continuity": "preserve_direction",
        }
        for subject_id, anchors in sorted(route_anchors.items())
    ]

    focus_target = _annotated_value(objective.camera.get("focus_target_id"))
    movement_value = _annotated_value(objective.camera.get("movement", {}).get("type")) or ""
    movement_text = movement_value.lower()
    movement = next(
        (
            kind
            for kind, markers in (
                ("pan", ("pan", "摇摄", "摇镜", "原地旋转", "固定机位旋转", "不平移")),
                ("push_in", ("推", "push", "dolly_in")),
                ("pull_out", ("拉远", "后拉", "pull", "dolly_out")),
                ("follow", ("跟随", "跟拍", "follow")),
                ("orbit", ("环绕", "绕拍", "orbit")),
                ("lateral", ("横移", "侧移", "lateral", "truck")),
                ("static", ("静止", "固定", "static", "fixed")),
            )
            if any(marker in movement_text for marker in markers)
        ),
        "static",
    )
    movement_status = (
        objective.camera.get("movement", {}).get("type", {}).get("source_status", "inferred")
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
    """Backward-compatible test helper for the deterministic fallback builder."""

    return build_deterministic_scene_skeleton(objective)


def _mock_proxy_family(category: str) -> str:
    lowered = category.lower()
    if any(marker in lowered for marker in ("人", "man", "woman", "person", "human")):
        return "human_capsule"
    if any(marker in lowered for marker in ("太阳", "地球", "月", "sun", "earth", "moon", "planet")):
        return "celestial_sphere"
    if any(marker in lowered for marker in ("车", "飞船", "vehicle", "ship", "car")):
        return "vehicle_box"
    return "generic_box"


def _mock_scale_intent(subject: dict[str, Any]) -> str:
    rendered = json.dumps(subject, ensure_ascii=False).lower()
    if any(marker in rendered for marker in ("巨大", "huge", "giant")):
        return "huge"
    return "human" if _mock_proxy_family(rendered) == "human_capsule" else "unspecified"


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

    if motion_type in {"static", "interactive"}:
        return "stationary"
    if motion_type in {"walking", "slow"}:
        return "slow"
    if motion_type in {"running", "flying", "fast"}:
        return "fast"
    lowered = (value or "").lower()
    if any(marker in lowered for marker in ("与主体速度同步", "同步主体", "match_subject")):
        return "match_subject"
    if lowered in {"静止", "0 m/s", "stationary"}:
        return "stationary"
    if any(marker in lowered for marker in ("缓慢", "慢", "slow")):
        return "slow"
    if any(marker in lowered for marker in ("快速", "快", "fast")):
        return "fast"
    return "unspecified"


def _mock_actions(objective: ObjectivePlanningBrief) -> list[tuple[str, dict[str, Any]]]:
    actions: list[tuple[str, dict[str, Any]]] = [
        (
            "get_capabilities",
            {"sections": ["entities", "constraints", "tracks", "camera", "validators", "limits"]},
        )
    ]
    entities: list[dict[str, Any]] = []
    for index, subject in enumerate(objective.subjects):
        entity_id = str(subject.get("id") or f"entity_{index + 1:02d}")
        category = _annotated_value(subject.get("category")) or "object"
        entities.append(
            {
                "entity_id": entity_id,
                "label": f"{category}代理",
                "role": _annotated_value(subject.get("narrative_role")) or "subject",
                "proxy": _proxy_for_category(category),
                "parent_id": None,
                "tags": [category],
                "source_refs": [
                    item.path
                    for item in objective.explicit_requirements
                    if item.path.startswith(f"content.subjects[{index}]")
                ],
            }
        )
    if entities:
        actions.append(("apply_entity_patch", {"upserts": entities, "remove_ids": []}))

    constraints: list[dict[str, Any]] = []
    for index, relationship in enumerate(objective.scene_design.get("relationships", [])):
        source_ref = f"content.scene_design.relationships[{index}]"
        relation = " ".join(
            str(relationship.get(name, ""))
            for name in ("type", "strength")
        ).lower()
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        if subject_id and reference_id and any(marker in relation for marker in ("远", "far", "background")):
            constraints.extend(
                [
                    {
                        "constraint_id": f"depth_{index + 1:02d}",
                        "type": "depth_order",
                        "strength": (
                            "hard"
                            if relationship.get("source_status") == "explicit"
                            else "soft"
                        ),
                        "weight": 1.0,
                        "subjects": [reference_id, subject_id],
                        "time_range_seconds": [0.0, _duration(objective)],
                        "parameters": {
                            "near_entity_id": reference_id,
                            "far_entity_id": subject_id,
                            "camera_id": "camera_main",
                            "minimum_depth_gap_meters": 20.0,
                        },
                        "source_status": relationship.get(
                            "source_status",
                            "inferred",
                        ),
                        "source_ref": source_ref,
                    },
                    {
                        "constraint_id": f"clearance_{index + 1:02d}",
                        "type": "surface_clearance_range",
                        "strength": (
                            "hard"
                            if relationship.get("source_status") == "explicit"
                            else "soft"
                        ),
                        "weight": 1.0,
                        "subjects": [reference_id, subject_id],
                        "time_range_seconds": [0.0, _duration(objective)],
                        "parameters": {
                            "entity_ids": [reference_id, subject_id],
                            "minimum_ratio": 0.5,
                            "preferred_ratio": 1.0,
                            "maximum_ratio": 2.0,
                            "scale_basis": "larger_directional_extent",
                            "space": "ground_plane",
                        },
                        "source_status": relationship.get(
                            "source_status",
                            "inferred",
                        ),
                        "source_ref": source_ref,
                    },
                ]
            )

    motion_tracks: list[dict[str, Any]] = []
    for index, motion in enumerate(objective.subject_motion):
        semantics = motion.get("motion_semantics")
        typed_semantics = semantics if isinstance(semantics, dict) else None
        source_ref = (
            f"content.subject_motion[{index}].motion_semantics"
            if typed_semantics is not None
            else f"content.subject_motion[{index}].action"
        )
        action = _annotated_value(motion.get("action")) or ""
        if typed_semantics is not None:
            action_status = typed_semantics.get("source_status", "inferred")
        elif isinstance(motion.get("action"), dict):
            action_status = motion["action"].get("source_status", "inferred")
        else:
            action_status = "inferred"
        target_id = motion.get("subject_id")
        if not target_id:
            continue
        if typed_semantics is not None and typed_semantics.get("motion_mode") == "carried":
            # 载运阶段由载体轨迹驱动，Mock 不伪造人物自己的世界轨迹。
            continue
        is_static = (
            typed_semantics.get("motion_type") in {"static", "interactive"}
            if typed_semantics is not None
            else any(
                marker in action.lower()
                for marker in ("静止", "不动", "still", "stationary")
            )
        )
        start_time = float(motion.get("start_time_seconds") or 0.0)
        end_time = float(motion.get("end_time_seconds") or _duration(objective))
        if is_static:
            constraints.append(
                {
                    "constraint_id": f"hold_{target_id}",
                    "type": "hold",
                    "strength": "hard" if action_status == "explicit" else "soft",
                    "weight": 1.0,
                    "subjects": [target_id],
                    "time_range_seconds": [start_time, end_time],
                    "parameters": {
                        "target_id": target_id,
                        "components": ["translation", "rotation", "scale"],
                        "tolerance_m": 0.0001,
                    },
                    "source_status": action_status,
                    "source_ref": source_ref,
                }
            )
        else:
            direction = (
                (0.0, -1.0, 0.0)
                if typed_semantics is not None
                and typed_semantics.get("direction_mode") == "world_forward"
                else _direction(_annotated_value(motion.get("direction")))
            )
            if direction:
                motion_tracks.append(
                    {
                        "track_id": f"motion_{target_id}",
                        "target_entity_id": target_id,
                        "type": "transform",
                        "time_range_seconds": [start_time, end_time],
                        "keyframes": [
                            {"time_seconds": start_time, "value": {"translation_m": [0.0, 0.0, 1.0]}, "interpolation": "smooth"},
                            {"time_seconds": min(end_time, _last_time(objective)), "value": {"translation_m": _direction_endpoint(direction)}, "interpolation": "smooth"},
                        ],
                        "path": None,
                        "target_id": None,
                        "interpolation": "smooth",
                        "source_ref": source_ref,
                    }
                )
    if constraints:
        actions.append(("apply_constraint_patch", {"upserts": constraints, "remove_ids": []}))
    if motion_tracks:
        actions.append(("apply_motion_patch", {"upserts": motion_tracks, "remove_ids": []}))

    camera_refs = [
        item.path
        for item in objective.explicit_requirements
        if item.path.startswith("content.camera")
    ]
    focus_target = _annotated_value(objective.camera.get("focus_target_id"))
    translation_camera = (objective.translation_parameters or {}).get("camera", {})
    semantic_movement = objective.camera.get("movement", {}).get("type")
    movement = _annotated_value(semantic_movement) or str(
        translation_camera.get("movement") or ""
    )
    movement_speed = _annotated_value(objective.camera.get("movement", {}).get("speed")) or ""
    movement_speed_status = (
        objective.camera.get("movement", {}).get("speed", {}).get("source_status")
        if isinstance(objective.camera.get("movement", {}).get("speed"), dict)
        else None
    )
    movement_is_explicit = (
        isinstance(semantic_movement, dict)
        and semantic_movement.get("source_status") == "explicit"
    )
    camera_tracks: list[dict[str, Any]] = []
    if any(marker in movement.lower() for marker in ("推近", "push", "dolly in", "push_in")):
        movement_ref = (
            "content.camera.movement.type"
            if movement_is_explicit
            else "translation_parameters.camera.movement"
        )
        camera_height = float(translation_camera.get("height_m") or 2.0)
        start_distance = float(translation_camera.get("start_distance_m") or 12.0)
        end_distance = float(translation_camera.get("end_distance_m") or 7.0)
        camera_tracks.append(
            {
                "track_id": "camera_move_01",
                "target_entity_id": None,
                "type": "transform",
                "time_range_seconds": [0.0, _duration(objective)],
                "keyframes": [
                    {"time_seconds": 0.0, "value": {"translation_m": [0.0, -start_distance, camera_height]}, "interpolation": "smooth"},
                    {"time_seconds": _last_time(objective), "value": {"translation_m": [0.0, -end_distance, camera_height]}, "interpolation": "smooth"},
                ],
                "path": None,
                "target_id": None,
                "interpolation": "smooth",
                "source_ref": movement_ref,
            }
        )
        constraints.append(
            {
                "constraint_id": "camera_push_in",
                "type": "camera_motion_direction",
                "strength": "hard" if movement_is_explicit else "soft",
                "weight": 1.0,
                "subjects": [],
                "time_range_seconds": [0.0, _duration(objective)],
                "parameters": {
                    "camera_id": "camera_main",
                    "target_id": focus_target,
                    "direction": "push_in",
                    "space": "world",
                    "minimum_displacement_m": (
                        3.0
                        if movement_is_explicit
                        else max(0.1, (start_distance - end_distance) * 0.8)
                    ),
                },
                "source_status": "explicit" if movement_is_explicit else "inferred",
                "source_ref": movement_ref,
            }
        )
        # 摄影机约束必须在先前约束 Patch 之后单独提交。
        actions.append(("apply_constraint_patch", {"upserts": [constraints[-1]], "remove_ids": []}))

    if (
        movement_speed_status == "explicit"
        and any(marker in movement_speed.lower() for marker in ("慢", "slow"))
    ):
        actions.append(
            (
                "apply_constraint_patch",
                {
                    "upserts": [
                        {
                            "constraint_id": "camera_slow_speed",
                            "type": "speed_range",
                            "strength": "hard",
                            "weight": 1.0,
                            "subjects": [],
                            "time_range_seconds": [0.0, _duration(objective)],
                            "parameters": {
                                "target_id": "camera_main",
                                "minimum_mps": 0.01,
                                "maximum_mps": 2.0,
                                "space": "world",
                            },
                            "source_status": "explicit",
                            "source_ref": "content.camera.movement.speed",
                        }
                    ],
                    "remove_ids": [],
                },
            )
        )
    actions.append(
        (
            "apply_camera_patch",
            {
                "camera_id": "camera_main",
                "projection": "perspective",
                "active": True,
                "static": {
                    "focal_length_mm": float(
                        translation_camera.get("focal_length_mm") or 35.0
                    ),
                    "sensor_width_mm": 36.0,
                    "focus_target_id": focus_target,
                    "source_refs": camera_refs,
                },
                "tracks": camera_tracks,
                "remove_track_ids": [],
            },
        )
    )
    actions.append(
        (
            "solve_candidate",
            {
                "scope": "all",
                "constraint_ids": [],
                "locked_variables": [],
                "strategy": "auto",
            },
        )
    )
    return actions


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
        if isinstance(item.content, dict) and isinstance(item.content.get("revision_after"), int):
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


def _proxy_for_category(category: str) -> dict[str, Any]:
    lowered = category.lower()
    if any(marker in lowered for marker in ("人", "man", "woman", "person")):
        return {"type": "capsule", "radius_m": 0.3, "segment_length_m": 1.2, "axis": "+Z"}
    if any(marker in lowered for marker in ("飞船", "ship", "spacecraft")):
        return {"type": "box", "size_xyz_m": [80.0, 30.0, 15.0]}
    return {"type": "box", "size_xyz_m": [2.0, 2.0, 2.0]}


def _annotated_value(value: Any) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return None


def _duration(objective: ObjectivePlanningBrief) -> float:
    resolution = objective.timeline.get("duration_resolution")
    value = resolution.get("resolved_duration_seconds") if isinstance(resolution, dict) else None
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Objective Planning Brief 缺少冻结时长")
    return float(value)


def _last_time(objective: ObjectivePlanningBrief) -> float:
    return max(0.0, _duration(objective) - 0.001)


def _direction(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    mapping = {
        "左": "left", "left": "left", "右": "right", "right": "right",
        "前": "forward", "forward": "forward", "后": "backward", "backward": "backward",
        "上": "up", "up": "up", "下": "down", "down": "down",
    }
    return next((result for marker, result in mapping.items() if marker in lowered), None)


def _direction_endpoint(direction: str) -> list[float]:
    return {
        "left": [-2.0, 0.0, 1.0],
        "right": [2.0, 0.0, 1.0],
        "forward": [0.0, 2.0, 1.0],
        "backward": [0.0, -2.0, 1.0],
        "up": [0.0, 0.0, 3.0],
        "down": [0.0, 0.0, 0.1],
    }[direction]
