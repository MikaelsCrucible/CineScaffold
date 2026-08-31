from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cinescaffold.prompting import build_prompt
from cinescaffold.providers.base import StructuredOutputProvider
from cinescaffold.schema import load_schema, validate_model_output
from cinescaffold.semantic_rules import (
    apply_translation_rules,
    load_translation_rules,
    translation_rules_sha256,
)


@dataclass(frozen=True)
class SemanticParserConfig:
    system_template_path: Path
    rules_path: Path
    format_example_path: Path
    model_output_schema_path: Path
    translation_rules_path: Path
    translation_parameters_schema_path: Path
    prompt_version: str = "semantic-parser-v0.3"


def parse_cinematic_brief(
    description: str,
    provider: StructuredOutputProvider,
    config: SemanticParserConfig,
) -> dict[str, Any]:
    if not description.strip():
        raise ValueError("自然语言输入不能为空")

    prompt = build_prompt(
        description=description,
        system_template_path=config.system_template_path,
        rules_path=config.rules_path,
        format_example_path=config.format_example_path,
    )
    schema = load_schema(config.model_output_schema_path)
    response = provider.generate(prompt.system_prompt, prompt.user_prompt, schema)
    validate_model_output(response.content, schema)

    translation_rules = load_translation_rules(config.translation_rules_path)
    content, translation_parameters = apply_translation_rules(
        response.content,
        translation_rules,
        description,
    )
    validate_model_output(content, schema)
    translation_schema = load_schema(config.translation_parameters_schema_path)
    validate_model_output(translation_parameters, translation_schema)

    rules_bytes = config.rules_path.read_bytes()
    return {
        "schema_version": "0.2",
        "content": content,
        "translation_parameters": translation_parameters,
        "provenance": {
            "source_prompt": description,
            "provider": provider.name,
            "model": provider.model,
            "parser_prompt_version": config.prompt_version,
            "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
            "translation_rules_sha256": translation_rules_sha256(
                config.translation_rules_path
            ),
            "response_id": response.response_id,
        },
    }
