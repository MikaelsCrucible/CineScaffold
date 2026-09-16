from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticAIContract:
    contract_id: str
    instruction: str


# These rules cross JSON fields, so enum-only schemas cannot communicate them.
# The same registry is rendered into both Semantic model passes and asserted
# against the JSON Schema by tests to prevent prompt/validator drift.
SEMANTIC_AI_CONTRACTS = (
    SemanticAIContract(
        "SEM-SOURCE-EVIDENCE",
        "source_status=explicit/inferred 时 source_text 必须是非空的最小原文证据；"
        "source_status=default/unknown 时 source_text 必须为 null；unknown 的 value 必须为 null。",
    ),
    SemanticAIContract(
        "SEM-UNCERTAINTY-RESOLUTION",
        "uncertainties[].resolution=unresolved 时 selected_value 必须为 null；"
        "resolution=use_default/use_inference 时 selected_value 必须是非空字符串，"
        "并准确记录实际采用的字段值。",
    ),
    SemanticAIContract(
        "SEM-MOTION-COMBINATION",
        "motion_mode、action_kind、motion_type、path_type、local_components、"
        "direction_mode、target_id 与 carrier_id 必须使用动作契约规定的合法组合；"
        "不得只让每个单字段分别合法。",
    ),
    SemanticAIContract(
        "SEM-RELATION-EVENT",
        "空间关系的 timeline_event_id=null 时 temporal_mode 必须为 throughout；"
        "temporal_mode=at_start/at_midpoint/at_end 时必须引用已声明事件。",
    ),
    SemanticAIContract(
        "SEM-CAMERA-MOVEMENT-TARGET",
        "camera.movement.type.value 为 pan/follow/orbit 时 target_id 必须引用场景实体；"
        "static/push_in/pull_out/lateral 时 target_id 必须为 null。",
    ),
    SemanticAIContract(
        "SEM-TEMPORAL-INTEGRITY",
        "事件、动作和摄影机区间必须满足 0 <= start < end <= duration；"
        "动作引用事件时二者区间必须一致，时间关系枚举必须与数值区间一致。",
    ),
    SemanticAIContract(
        "SEM-SUBJECT-STATE-COVERAGE",
        "每个场景实体的 subject_motion 区间并集必须从 0 连续覆盖到 duration，"
        "不得留下会被执行层解释为无依据静止且可见的开头、中间或结尾空窗；"
        "覆盖只要求状态有解释，不要求实体从头可见；中途出现必须以进入前 hidden 状态"
        "及其末端 becomes_visible 明确表达。原文没有等待、延迟出现或提前结束时，"
        "不得自行插入这类状态。",
    ),
    SemanticAIContract(
        "SEM-CARRIED-CARRIER-MOTION",
        "narrative_required 的 carried 表示该实体在世界中随载体移动；其完整区间必须由"
        " carrier_id 对应实体的 self_propelled 动作覆盖。仅建立容纳但载体不移动时，"
        "不得使用 carried 冒充世界位移。",
    ),
    SemanticAIContract(
        "SEM-REFERENCE-INTEGRITY",
        "所有 subject、event、target、carrier、contained_by、focus 和关系引用必须存在、"
        "类型正确且不得非法自引用；所有要求唯一的 ID 和 uncertainty.field 必须唯一。",
    ),
)


def render_semantic_ai_contracts() -> str:
    """Render the code-owned cross-field contract for model-visible prompts."""

    return "\n".join(
        f"- [{item.contract_id}] {item.instruction}"
        for item in SEMANTIC_AI_CONTRACTS
    )


def semantic_ai_contract_ids() -> tuple[str, ...]:
    return tuple(item.contract_id for item in SEMANTIC_AI_CONTRACTS)
