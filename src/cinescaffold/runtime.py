from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from cinescaffold.config import LoadedConfig, resolve_model_settings, resolve_stage_option
from cinescaffold.errors import ConfigurationError
from cinescaffold.execution.runner import ExecutionConfig
from cinescaffold.planning.runner import (
    DEFAULT_MAX_COMMIT_ATTEMPTS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    InterpreterRunConfig,
)
from cinescaffold.planning.trace import CostRates, TraceConfig
from cinescaffold.platforms import default_blender_path, default_mcp_command
from cinescaffold.providers import DeepSeekProvider, MockProvider, OpenAIProvider
from cinescaffold.semantic import SemanticParserConfig
from cinescaffold.workflow import PipelineRunConfig


@dataclass(frozen=True)
class RuntimeResourcePaths:
    semantic_system: Path
    semantic_rules: Path
    semantic_example: Path
    semantic_schema: Path
    translation_rules: Path
    translation_schema: Path
    planning_system: Path

    @classmethod
    def from_root(cls, root: Path) -> RuntimeResourcePaths:
        return cls(
            semantic_system=root / "prompts/semantic_parser/system.md",
            semantic_rules=root / "prompts/semantic_parser/rules.md",
            semantic_example=root / "prompts/semantic_parser/format_example.json",
            semantic_schema=root / "schemas/cinematic_brief_model_output.schema.json",
            translation_rules=root / "prompts/semantic_parser/translation_rules.json",
            translation_schema=root / "schemas/semantic_translation_parameters.schema.json",
            planning_system=root / "prompts/scene_planner/system.md",
        )


def build_pipeline_run_config(
    config: LoadedConfig,
    *,
    output_dir: Path,
    include_semantic: bool,
    overwrite: bool,
    resources: RuntimeResourcePaths | None = None,
    render_profile: str | None = None,
) -> PipelineRunConfig:
    """把已校验的用户配置转换为完整运行配置。"""

    paths = resources or RuntimeResourcePaths.from_root(Path.cwd())
    semantic_provider = None
    semantic_parser = None
    semantic_cost_rates = None
    if include_semantic:
        semantic_provider = _semantic_provider(config, paths)
        semantic_parser = SemanticParserConfig(
            system_template_path=paths.semantic_system,
            rules_path=paths.semantic_rules,
            format_example_path=paths.semantic_example,
            model_output_schema_path=paths.semantic_schema,
            translation_rules_path=paths.translation_rules,
            translation_parameters_schema_path=paths.translation_schema,
        )
        semantic_cost_rates = cost_rates_from_config(config, "semantic")
    return PipelineRunConfig(
        output_dir=output_dir,
        planning=_planning_config(config, output_dir / "planning", paths),
        execution=_execution_config(
            config,
            output_dir / "execution",
            overwrite=overwrite,
            render_profile=render_profile,
        ),
        semantic_provider=semantic_provider,
        semantic_parser=semantic_parser,
        semantic_cost_rates=semantic_cost_rates,
        overwrite=overwrite,
    )


def _semantic_provider(config: LoadedConfig, paths: RuntimeResourcePaths) -> Any:
    settings = resolve_model_settings(
        config,
        "semantic",
        cli_provider=None,
        cli_model=None,
        cli_base_url=None,
    )
    timeout = resolve_stage_option(config, "semantic", "timeout", None) or 60.0
    max_tokens = resolve_stage_option(config, "semantic", "max_tokens", None) or 8192
    thinking = resolve_stage_option(config, "semantic", "thinking_mode", None) or "disabled"
    effort = resolve_stage_option(config, "semantic", "reasoning_effort", None)
    if settings.provider == "mock":
        response = json.loads(paths.semantic_example.read_text(encoding="utf-8"))
        return MockProvider(response)
    if not settings.model:
        raise ConfigurationError(f"{settings.provider} Semantic Provider 需要模型名称")
    if settings.provider == "openai":
        return OpenAIProvider(
            api_key=settings.api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=settings.model,
            base_url=settings.base_url or "https://api.openai.com/v1",
            timeout=float(timeout),
            max_tokens=int(max_tokens),
            reasoning_effort=effort,
        )
    return DeepSeekProvider(
        api_key=settings.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        model=settings.model,
        base_url=settings.base_url or "https://api.deepseek.com",
        timeout=float(timeout),
        max_tokens=int(max_tokens),
        thinking_mode=thinking,
        reasoning_effort=effort,
    )


def _planning_config(
    config: LoadedConfig,
    run_dir: Path,
    paths: RuntimeResourcePaths,
) -> InterpreterRunConfig:
    settings = resolve_model_settings(
        config,
        "planning",
        cli_provider=None,
        cli_model=None,
        cli_base_url=None,
    )
    return InterpreterRunConfig(
        provider=settings.provider,
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        system_prompt_path=paths.planning_system,
        run_dir=run_dir,
        max_requests=_option(config, "max_requests", DEFAULT_MAX_REQUESTS),
        max_tool_calls=_option(config, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS),
        max_input_tokens=_option(config, "max_input_tokens", None),
        max_context_tokens=_option(config, "max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS),
        max_output_tokens=_option(config, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        max_total_tokens=_option(config, "max_total_tokens", None),
        max_seconds=_option(config, "max_seconds", DEFAULT_MAX_SECONDS),
        max_commit_attempts=_option(
            config,
            "max_commit_attempts",
            DEFAULT_MAX_COMMIT_ATTEMPTS,
        ),
        thinking_mode=_option(config, "thinking_mode", None),
        reasoning_effort=_option(config, "reasoning_effort", None),
        model_max_tokens=_option(config, "model_max_tokens", None),
        trace_config=TraceConfig(
            max_event_bytes=_option(config, "trace_max_event_bytes", 32_768),
            max_string_chars=_option(config, "trace_max_string_chars", 4_096),
        ),
        cost_rates=cost_rates_from_config(config, "planning"),
    )


def _execution_config(
    config: LoadedConfig,
    output_dir: Path,
    *,
    overwrite: bool,
    render_profile: str | None,
) -> ExecutionConfig:
    data = config.data
    blender = data.get("execution_blender_path")
    mcp = data.get("execution_mcp_command")
    return ExecutionConfig(
        output_dir=output_dir,
        blender_path=Path(blender) if blender else default_blender_path(),
        mcp_command=Path(mcp) if mcp else default_mcp_command(),
        overwrite=overwrite,
        build_backend=data.get("execution_build_backend", "background"),
        build_timeout_seconds=float(data.get("execution_build_timeout_seconds", "180")),
        render_backend=data.get("execution_render_backend", "background"),
        render_profile=render_profile or data.get("execution_render_profile", "preview"),
        render_timeout_seconds=float(data.get("execution_render_timeout_seconds", "600")),
    )


def _option(config: LoadedConfig, key: str, default: Any) -> Any:
    value = resolve_stage_option(config, "planning", key, None)
    return default if value is None else value


def cost_rates_from_config(
    config: LoadedConfig,
    stage: str,
    *,
    provider: str | None = None,
) -> CostRates | None:
    data = config.data
    input_value = data.get(f"{stage}_input_cost_per_million")
    output_value = data.get(f"{stage}_output_cost_per_million")
    if input_value is None and output_value is None:
        resolved_provider = provider
        if resolved_provider is None:
            resolved_provider = resolve_model_settings(
                config,
                stage,
                cli_provider=None,
                cli_model=None,
                cli_base_url=None,
            ).provider
        if resolved_provider == "mock":
            return CostRates(
                input_per_million=Decimal(0),
                output_per_million=Decimal(0),
                source="mock",
            )
        return None
    if input_value is None or output_value is None:
        raise ConfigurationError("成本估算必须同时填写输入与输出价格")
    return CostRates(
        currency=data.get(f"{stage}_cost_currency", "USD"),
        input_per_million=Decimal(input_value),
        output_per_million=Decimal(output_value),
        cache_read_per_million=_decimal_or_none(
            data.get(f"{stage}_cache_read_cost_per_million")
        ),
        cache_write_per_million=_decimal_or_none(
            data.get(f"{stage}_cache_write_cost_per_million")
        ),
        source=data.get(f"{stage}_price_source", "user_supplied"),
    )


def _decimal_or_none(value: str | None) -> Decimal | None:
    return Decimal(value) if value is not None else None
