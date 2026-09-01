from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cinescaffold.errors import PromptTemplateError


RULES_PLACEHOLDER = "{{CONVERSION_RULES}}"


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
