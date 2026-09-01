from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cinescaffold.prompting import build_prompt
from cinescaffold.providers.base import StructuredOutputProvider
from cinescaffold.schema import load_schema, validate_model_output
from cinescaffold.semantic_rules import (
    apply_translation_rules,
    load_translation_rules,
    translation_rules_sha256,
)
from cinescaffold.textual_six import (
    TextualSixDimensions,
    parse_textual_six,
    render_textual_six,
)


@dataclass(frozen=True)
class SemanticParserConfig:
    system_template_path: Path
    rules_path: Path
    format_example_path: Path
    model_output_schema_path: Path
    translation_rules_path: Path
    translation_parameters_schema_path: Path
    prompt_version: str = "semantic-parser-v0.6"


@dataclass(frozen=True)
class SemanticParseResult:
    brief: dict[str, Any]
    textual_six: TextualSixDimensions


def parse_cinematic_brief(
    description: str,
    provider: StructuredOutputProvider,
    config: SemanticParserConfig,
) -> dict[str, Any]:
    return parse_semantic_input(description, provider, config).brief


def parse_semantic_input(
    source_text: str,
    provider: StructuredOutputProvider,
    config: SemanticParserConfig,
    *,
    source_kind: Literal["natural_text", "textual_six"] = "natural_text",
) -> SemanticParseResult:
    if not source_text.strip():
        raise ValueError("语义输入不能为空")
    source_six = parse_textual_six(source_text) if source_kind == "textual_six" else None

    prompt = build_prompt(
        description=source_text,
        system_template_path=config.system_template_path,
        rules_path=config.rules_path,
        format_example_path=config.format_example_path,
        source_kind=source_kind,
    )
    schema = load_schema(config.model_output_schema_path)
    response = provider.generate(prompt.system_prompt, prompt.user_prompt, schema)
    validate_model_output(response.content, schema)

    translation_rules = load_translation_rules(config.translation_rules_path)
    content, translation_parameters = apply_translation_rules(
        response.content,
        translation_rules,
        source_text,
    )
    validate_model_output(content, schema)
    translation_schema = load_schema(config.translation_parameters_schema_path)
    validate_model_output(translation_parameters, translation_schema)

    rules_bytes = config.rules_path.read_bytes()
    textual_six = source_six or render_textual_six(content)
    brief = {
        "schema_version": "0.5",
        "content": content,
        "translation_parameters": translation_parameters,
        "provenance": {
            "source_prompt": source_text,
            "source_kind": source_kind,
            "textual_six_sha256": textual_six.sha256(),
            "provider": provider.name,
            "model": provider.model,
            "parser_prompt_version": config.prompt_version,
            "rules_sha256": hashlib.sha256(rules_bytes).hexdigest(),
            "translation_rules_sha256": translation_rules_sha256(
                config.translation_rules_path
            ),
            "response_id": response.response_id,
            "provider_usage": response.raw_metadata.get("usage"),
        },
    }
    return SemanticParseResult(brief=brief, textual_six=textual_six)
