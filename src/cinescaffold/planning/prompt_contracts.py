from __future__ import annotations

from dataclasses import dataclass

from cinescaffold.errors import PromptTemplateError


PLANNING_TOOL_CONTRACTS_PLACEHOLDER = "{{PLANNING_TOOL_CONTRACTS}}"


@dataclass(frozen=True)
class PlanningToolContract:
    tool_name: str
    instruction: str


# The registered Pydantic tool schema remains authoritative for fields and enums.
# This registry owns the state-machine meaning of every model-visible tool so a
# newly registered method cannot silently exist without a prompt explanation.
PLANNING_TOOL_CONTRACTS = (
    PlanningToolContract(
        "submit_scene_skeleton",
        "仅在首次符号设计或 Toolkit 明确允许重提骨架时调用；输入不含米制数值。"
        "Relation/Motion/Camera 的 source_ref 必须复制 Route Context 中对应 binding 的"
        " canonical_source_ref，不得自行拼接字段路径。成功后下一步是 request_design_options。",
    ),
    PlanningToolContract(
        "request_design_options",
        "在骨架被接受后请求 Toolkit 联合求解数值候选；只选择策略和可选尺寸范围，"
        "不得手写 option 数值。成功返回 option 后调用 apply_design_option；若只返回"
        " repair baseline，则调用 begin_design_repair。",
    ),
    PlanningToolContract(
        "begin_design_repair",
        "仅使用 request_design_options 返回的当前 revision 与 baseline_id 物化失败基线；"
        "成功后必须按返回的 required_next_tool 调用 apply_candidate_patch。",
    ),
    PlanningToolContract(
        "apply_design_option",
        "仅使用当前 revision 的未过期 option_id 原子物化候选；不得复制 option 中的坐标。"
        "成功后按返回的 validation/next_actions 继续。",
    ),
    PlanningToolContract(
        "inspect_candidate",
        "只读查看当前或历史 Candidate。view 与过滤参数只缩小返回内容；同一 revision 不得"
        "重复完全相同的读取。它不产生 revision，也不能代替 validate_candidate。",
    ),
    PlanningToolContract(
        "apply_candidate_patch",
        "仅在该工具被动态暴露时做一次原子修复；base_revision 必须是当前 revision。"
        "实体、约束、运动与摄影机可在同一预演事务中联合修改；字段必须遵循动态 Tool Schema"
        " 与 relevant_capabilities，失败不会产生 revision。",
    ),
    PlanningToolContract(
        "solve_candidate",
        "仅在动态暴露时让确定性求解器补齐 layout/camera/all；constraint_ids 与"
        " locked_variables 是范围限制，不是新要求。调用后仍须读取返回验证结果。",
    ),
    PlanningToolContract(
        "validate_candidate",
        "确定性验证指定或当前 revision；空 checks 表示完整验证，非空 checks 只运行所列域。"
        "它不修改 Candidate；hard failure 必须修复，不能用文字宣称通过。",
    ),
    PlanningToolContract(
        "suggest_repairs",
        "仅在 Toolkit 检测到可搜索修复且动态暴露时调用；返回的是绑定当前 revision 的"
        " suggestion，不直接修改 Candidate。随后用 apply_repair 应用所选 suggestion_id。",
    ),
    PlanningToolContract(
        "apply_repair",
        "仅应用 suggest_repairs 对当前 base_revision 返回的未过期 suggestion_id；"
        "成功会产生新 revision 并复验，失败不得猜测 suggestion 内部数值。",
    ),
    PlanningToolContract(
        "restore_candidate",
        "仅在已有历史 revision 明显优于当前失败尝试时，把 source_revision 复制为新的当前"
        " revision；reason 必须说明恢复依据。它不删除历史，也不能恢复未知 revision。",
    ),
)


def planning_tool_contract_names() -> tuple[str, ...]:
    return tuple(item.tool_name for item in PLANNING_TOOL_CONTRACTS)


def render_planning_tool_contracts() -> str:
    return "\n".join(
        f"- `{item.tool_name}`：{item.instruction}"
        for item in PLANNING_TOOL_CONTRACTS
    )


def render_planning_system_prompt(template: str) -> str:
    if PLANNING_TOOL_CONTRACTS_PLACEHOLDER not in template:
        raise PromptTemplateError(
            "Planning 系统提示缺少占位符 "
            + PLANNING_TOOL_CONTRACTS_PLACEHOLDER
        )
    return template.replace(
        PLANNING_TOOL_CONTRACTS_PLACEHOLDER,
        render_planning_tool_contracts(),
    )
