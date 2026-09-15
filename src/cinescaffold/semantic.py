from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cinescaffold.errors import SchemaValidationError
from cinescaffold.prompting import build_prompt, build_revision_prompt
from cinescaffold.providers.base import ProviderResponse, StructuredOutputProvider
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
    revision_template_path: Path
    format_example_path: Path
    model_output_schema_path: Path
    translation_rules_path: Path
    translation_parameters_schema_path: Path
    prompt_version: str = "semantic-parser-v0.15"


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
    source_six = (
        parse_textual_six(source_text) if source_kind == "textual_six" else None
    )

    prompt = build_prompt(
        description=source_text,
        system_template_path=config.system_template_path,
        rules_path=config.rules_path,
        format_example_path=config.format_example_path,
        source_kind=source_kind,
    )
    schema = load_schema(config.model_output_schema_path)
    translation_rules = load_translation_rules(config.translation_rules_path)
    draft_response = provider.generate(prompt.system_prompt, prompt.user_prompt, schema)
    diagnostics = _audit_semantic_draft(
        draft_response.content,
        schema,
        translation_rules,
        source_text,
    )
    revision_prompt = build_revision_prompt(
        source_text=source_text,
        draft=draft_response.content,
        diagnostics=diagnostics,
        system_template_path=config.system_template_path,
        rules_path=config.rules_path,
        revision_template_path=config.revision_template_path,
        source_kind=source_kind,
    )
    response = provider.generate(
        revision_prompt.system_prompt,
        revision_prompt.user_prompt,
        schema,
    )
    content, translation_parameters = _validate_and_translate(
        response.content,
        schema,
        translation_rules,
        source_text,
        config.translation_parameters_schema_path,
    )

    rules_bytes = config.rules_path.read_bytes()
    textual_six = source_six or render_textual_six(content)
    brief = {
        "schema_version": "0.7",
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
            "provider_usage": _aggregate_provider_usage(
                [draft_response, response]
            ),
        },
    }
    return SemanticParseResult(brief=brief, textual_six=textual_six)


def _audit_semantic_draft(
    content: dict[str, Any],
    schema: dict[str, Any],
    translation_rules: dict[str, Any],
    source_text: str,
) -> list[dict[str, str]]:
    """Turn first-pass contract failures into bounded revision feedback."""

    try:
        validate_model_output(content, schema)
    except SchemaValidationError as error:
        return [
            {
                "code": "schema_validation_failed",
                "message": str(error),
            }
        ]
    try:
        apply_translation_rules(content, translation_rules, source_text)
    except (SchemaValidationError, ValueError) as error:
        return [
            {
                "code": "semantic_contract_failed",
                "message": str(error),
            }
        ]
    return [
        {
            "code": "independent_semantic_review_required",
            "message": (
                "初稿通过结构与确定性一致性检查；仍须逐句复核原始输入，"
                "检查遗漏、无依据推断和会改变几何结果的歧义。"
            ),
        }
    ]


def _validate_and_translate(
    content: dict[str, Any],
    schema: dict[str, Any],
    translation_rules: dict[str, Any],
    source_text: str,
    translation_schema_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_model_output(content, schema)
    normalized, translation_parameters = apply_translation_rules(
        content,
        translation_rules,
        source_text,
    )
    validate_model_output(normalized, schema)
    translation_schema = load_schema(translation_schema_path)
    validate_model_output(translation_parameters, translation_schema)
    return normalized, translation_parameters


def _aggregate_provider_usage(
    responses: list[ProviderResponse],
) -> dict[str, Any] | None:
    usages = [
        response.raw_metadata.get("usage")
        for response in responses
        if isinstance(response.raw_metadata.get("usage"), dict)
    ]
    if not usages:
        return None

    input_tokens = [
        int(item.get("input_tokens", item.get("prompt_tokens", 0)) or 0)
        for item in usages
    ]
    output_tokens = [
        int(item.get("output_tokens", item.get("completion_tokens", 0)) or 0)
        for item in usages
    ]
    cached_tokens = [
        int(
            item.get("cache_read_tokens")
            or (item.get("input_tokens_details") or {}).get("cached_tokens")
            or (item.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        for item in usages
    ]
    cache_write_tokens = [
        int(item.get("cache_write_tokens", 0) or 0) for item in usages
    ]
    reasoning_tokens = [
        int(
            (item.get("output_tokens_details") or {}).get("reasoning_tokens")
            or (item.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            or 0
        )
        for item in usages
    ]
    return {
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "cache_write_tokens": sum(cache_write_tokens),
        "input_tokens_details": {"cached_tokens": sum(cached_tokens)},
        "output_tokens_details": {"reasoning_tokens": sum(reasoning_tokens)},
        "requests": len(responses),
        "request_input_tokens": input_tokens,
        "request_cache_read_tokens": cached_tokens,
        "request_cache_write_tokens": cache_write_tokens,
    }
