from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cinescaffold.errors import PromptTemplateError


RULES_PLACEHOLDER = "{{CONVERSION_RULES}}"
REVISION_PLACEHOLDERS = {
    "{{SOURCE_LABEL}}",
    "{{SOURCE_TEXT}}",
    "{{DRAFT_JSON}}",
    "{{DIAGNOSTICS_JSON}}",
    "{{REVIEW_PACKET_JSON}}",
}


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str


def build_prompt(
    description: str,
    system_template_path: Path,
    rules_path: Path,
    format_example_path: Path,
    *,
    source_kind: Literal["natural_text", "textual_six"] = "natural_text",
) -> PromptBundle:
    template = system_template_path.read_text(encoding="utf-8")
    if RULES_PLACEHOLDER not in template:
        raise PromptTemplateError(f"系统提示缺少占位符 {RULES_PLACEHOLDER}")

    rules = rules_path.read_text(encoding="utf-8").strip()
    system_prompt = template.replace(RULES_PLACEHOLDER, rules or "（规则暂未提供）")
    example = json.loads(format_example_path.read_text(encoding="utf-8"))
    source_label = "自然语言" if source_kind == "natural_text" else "文本六维"
    source_instruction = (
        "请理解开放描述并按规则整理。"
        if source_kind == "natural_text"
        else (
            "输入已经按六个电影语义维度组织。请忠实保留各部分内容，将其类型化并补充规则要求的来源、"
            "时间和约束；无法结构化的语句必须保留在 uncertainties，不得静默丢弃。"
        )
    )
    user_prompt = (
        f"请将下列{source_label}整理为严格六维 Cinematic Brief，并输出 JSON。{source_instruction}\n\n"
        f"格式示例（仅表示字段形状，不是转换规则）：\n{json.dumps(example, ensure_ascii=False)}\n\n"
        f"{source_label}：\n{description}"
    )
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)


def build_revision_prompt(
    source_text: str,
    draft: dict[str, object],
    diagnostics: list[dict[str, str]],
    system_template_path: Path,
    rules_path: Path,
    revision_template_path: Path,
    review_packet: dict[str, object] | None = None,
    *,
    source_kind: Literal["natural_text", "textual_six"] = "natural_text",
) -> PromptBundle:
    """Build the bounded second-pass semantic review request."""

    template = system_template_path.read_text(encoding="utf-8")
    if RULES_PLACEHOLDER not in template:
        raise PromptTemplateError(f"系统提示缺少占位符 {RULES_PLACEHOLDER}")
    rules = rules_path.read_text(encoding="utf-8").strip()
    system_prompt = template.replace(RULES_PLACEHOLDER, rules or "（规则暂未提供）")

    revision = revision_template_path.read_text(encoding="utf-8")
    missing = sorted(
        placeholder
        for placeholder in REVISION_PLACEHOLDERS
        if placeholder not in revision
    )
    if missing:
        raise PromptTemplateError(
            "语义修正模板缺少占位符：" + ", ".join(missing)
        )
    source_label = "自然语言" if source_kind == "natural_text" else "文本六维"
    values = {
        "{{SOURCE_LABEL}}": source_label,
        "{{SOURCE_TEXT}}": source_text,
        "{{DRAFT_JSON}}": json.dumps(draft, ensure_ascii=False, indent=2),
        "{{DIAGNOSTICS_JSON}}": json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        ),
        "{{REVIEW_PACKET_JSON}}": json.dumps(
            review_packet or {},
            ensure_ascii=False,
            indent=2,
        ),
    }
    for placeholder, value in values.items():
        revision = revision.replace(placeholder, value)
    return PromptBundle(system_prompt=system_prompt, user_prompt=revision)
