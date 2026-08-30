from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from cinescaffold.errors import ConfigurationError
from cinescaffold.planning.objective import ObjectivePlanningBrief


def create_planning_model(
    provider: str,
    model_name: str | None,
    objective_brief: ObjectivePlanningBrief,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Model:
    if provider == "mock":
        return _create_mock_model(objective_brief)
    if not model_name:
        raise ConfigurationError(f"{provider} Agent 需要显式指定 --model")
    if not api_key:
        environment_name = "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"
        raise ConfigurationError(f"缺少 {environment_name}")
    if provider == "openai":
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(base_url=base_url, api_key=api_key),
        )
    if provider == "deepseek":
        if base_url:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            deepseek_provider = DeepSeekProvider(openai_client=client)
        else:
            deepseek_provider = DeepSeekProvider(api_key=api_key)
        return OpenAIChatModel(model_name, provider=deepseek_provider)
    raise ConfigurationError(f"未知 Agent Provider：{provider}")


def _create_mock_model(objective: ObjectivePlanningBrief) -> FunctionModel:
    actions = _mock_actions(objective)

    def callback(messages: list, info: AgentInfo) -> ModelResponse:
        duration_tool = next(
            (
                item
                for item in info.output_tools
                if "duration_seconds" in item.parameters_json_schema.get("properties", {})
            ),
            None,
        )
        if duration_tool is not None:
            minimum, maximum = _mock_duration_bounds(messages)
            duration = (minimum + maximum) / 2.0 if minimum is not None else 6.0
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        duration_tool.name,
                        {
                            "duration_seconds": duration,
                            "reason": "Mock 按单镜头动作与运镜复杂度选择时长。",
                        },
                        tool_call_id="mock_duration",
                    )
                ],
                usage=RequestUsage(input_tokens=90, output_tokens=20, details={"mock_estimated": 1}),
                finish_reason="stop",
            )
        returns = _tool_returns(messages)
        action_index = len(returns)
        usage = RequestUsage(
            input_tokens=120 + action_index * 10,
            output_tokens=30,
            details={"mock_estimated": 1},
        )
        commit_ready = any(
            isinstance(item.content, dict)
            and item.content.get("data", {}).get("acceptance", {}).get("commit_ready") is True
            for item in returns
        )
        if action_index < len(actions) and not commit_ready:
            name, arguments = actions[action_index]
            return ModelResponse(
                parts=[ToolCallPart(name, arguments, tool_call_id=f"mock_call_{action_index:03d}")],
                usage=usage,
                finish_reason="tool_call",
            )
        revision = _latest_revision(returns)
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
                "locked_fields": [],
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
        relation = str(relationship.get("type", "")).lower()
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        if subject_id and reference_id and any(marker in relation for marker in ("远", "far", "background")):
            constraints.append(
                {
                    "constraint_id": f"depth_{index + 1:02d}",
                    "type": "depth_order",
                    "strength": "hard" if relationship.get("source_status") == "explicit" else "soft",
                    "weight": 1.0,
                    "subjects": [reference_id, subject_id],
                    "time_range_seconds": [0.0, _duration(objective)],
                    "parameters": {
                        "near_entity_id": reference_id,
                        "far_entity_id": subject_id,
                        "camera_id": "camera_main",
                        "minimum_depth_gap_meters": 20.0,
                    },
                    "source_status": relationship.get("source_status", "inferred"),
                    "source_ref": source_ref,
                }
            )

    motion_tracks: list[dict[str, Any]] = []
    for index, motion in enumerate(objective.subject_motion):
        source_ref = f"content.subject_motion[{index}].action"
        action = _annotated_value(motion.get("action")) or ""
        target_id = motion.get("subject_id")
        if not target_id:
            continue
        if any(marker in action.lower() for marker in ("静止", "不动", "still", "stationary")):
            constraints.append(
                {
                    "constraint_id": f"hold_{target_id}",
                    "type": "hold",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": [target_id],
                    "time_range_seconds": [0.0, _duration(objective)],
                    "parameters": {
                        "target_id": target_id,
                        "components": ["translation", "rotation", "scale"],
                        "tolerance_m": 0.0001,
                    },
                    "source_status": "explicit",
                    "source_ref": source_ref,
                }
            )
        else:
            direction = _direction(_annotated_value(motion.get("direction")))
            if direction:
                motion_tracks.append(
                    {
                        "track_id": f"motion_{target_id}",
                        "target_entity_id": target_id,
                        "type": "transform",
                        "time_range_seconds": [0.0, _duration(objective)],
                        "keyframes": [
                            {"time_seconds": 0.0, "value": {"translation_m": [0.0, 0.0, 1.0]}, "interpolation": "smooth"},
                            {"time_seconds": _last_time(objective), "value": {"translation_m": _direction_endpoint(direction)}, "interpolation": "smooth"},
                        ],
                        "path": None,
                        "target_id": None,
                        "interpolation": "smooth",
                        "locked_components": [],
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
    if not focus_target and entities:
        focus_target = entities[0]["entity_id"]
    movement = _annotated_value(objective.camera.get("movement", {}).get("type")) or ""
    movement_speed = _annotated_value(objective.camera.get("movement", {}).get("speed")) or ""
    camera_tracks: list[dict[str, Any]] = []
    if any(marker in movement.lower() for marker in ("推近", "push", "dolly in")):
        movement_ref = "content.camera.movement.type"
        camera_tracks.append(
            {
                "track_id": "camera_move_01",
                "target_entity_id": None,
                "type": "transform",
                "time_range_seconds": [0.0, _duration(objective)],
                "keyframes": [
                    {"time_seconds": 0.0, "value": {"translation_m": [0.0, -12.0, 2.0]}, "interpolation": "smooth"},
                    {"time_seconds": _last_time(objective), "value": {"translation_m": [0.0, -7.0, 2.0]}, "interpolation": "smooth"},
                ],
                "path": None,
                "target_id": None,
                "interpolation": "smooth",
                "locked_components": [],
                "source_ref": movement_ref,
            }
        )
        constraints.append(
            {
                "constraint_id": "camera_push_in",
                "type": "camera_motion_direction",
                "strength": "hard",
                "weight": 1.0,
                "subjects": [],
                "time_range_seconds": [0.0, _duration(objective)],
                "parameters": {
                    "camera_id": "camera_main",
                    "target_id": focus_target,
                    "direction": "push_in",
                    "space": "target_relative",
                    "minimum_displacement_m": 3.0,
                },
                "source_status": "explicit",
                "source_ref": movement_ref,
            }
        )
        # 摄影机约束必须在先前约束 Patch 之后单独提交。
        actions.append(("apply_constraint_patch", {"upserts": [constraints[-1]], "remove_ids": []}))

    if any(marker in movement_speed.lower() for marker in ("慢", "slow")):
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
                    "focal_length_mm": 35.0,
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
                "allowed_variables": [],
                "locked_variables": [],
                "profile": "research_default",
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
        value = objective.timeline.get("duration_seconds")
    return float(value) if isinstance(value, (int, float)) and value > 0 else 6.0


def _mock_duration_bounds(messages: list) -> tuple[float | None, float | None]:
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                continue
            start = part.content.find("{")
            if start < 0:
                continue
            try:
                payload = json.loads(part.content[start:])
            except json.JSONDecodeError:
                continue
            minimum = payload.get("minimum_seconds")
            maximum = payload.get("maximum_seconds")
            if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
                return float(minimum), float(maximum)
    return None, None


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
