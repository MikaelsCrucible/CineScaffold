from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
) -> PromptBundle:
    template = system_template_path.read_text(encoding="utf-8")
    if RULES_PLACEHOLDER not in template:
        raise PromptTemplateError(f"系统提示缺少占位符 {RULES_PLACEHOLDER}")

    rules = rules_path.read_text(encoding="utf-8").strip()
    system_prompt = template.replace(RULES_PLACEHOLDER, rules or "（规则暂未提供）")
    example = json.loads(format_example_path.read_text(encoding="utf-8"))
    user_prompt = (
        "请将下列自然语言整理为六维 Cinematic Brief，并输出 JSON。\n\n"
        f"格式示例（仅表示字段形状，不是转换规则）：\n{json.dumps(example, ensure_ascii=False)}\n\n"
        f"自然语言：\n{description}"
    )
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)
