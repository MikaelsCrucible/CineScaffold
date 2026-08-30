from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from cinescaffold.errors import CineScaffoldError, ConfigurationError
from cinescaffold.execution.runner import ExecutionConfig, ExecutionRunner
from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunner
from cinescaffold.planning.trace import CostRates, TraceConfig
from cinescaffold.providers import DeepSeekProvider, MockProvider, OpenAIProvider
from cinescaffold.semantic import SemanticParserConfig, parse_cinematic_brief


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "parse":
            return _run_parse(args)
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "execute":
            return _run_execute(args)
        parser.print_help()
        return 2
    except (CineScaffoldError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cinescaffold")
    subparsers = parser.add_subparsers(dest="command")
    parse_parser = subparsers.add_parser("parse", help="将自然语言解析为六维 Cinematic Brief")

    source = parse_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="直接提供自然语言")
    source.add_argument("--input", type=Path, help="从 UTF-8 文本文件读取自然语言")

    parse_parser.add_argument(
        "--provider",
        choices=("mock", "openai", "deepseek"),
        default="mock",
    )
    parse_parser.add_argument("--model", help="真实 Provider 的模型名称")
    parse_parser.add_argument("--base-url", help="覆盖 Provider 官方基础地址")
    parse_parser.add_argument("--timeout", type=float, default=60.0)
    parse_parser.add_argument("--max-tokens", type=int, default=8192)
    parse_parser.add_argument("--rules", type=Path, default=Path("prompts/semantic_parser/rules.md"))
    parse_parser.add_argument(
        "--system-template",
        type=Path,
        default=Path("prompts/semantic_parser/system.md"),
    )
    parse_parser.add_argument(
        "--format-example",
        type=Path,
        default=Path("prompts/semantic_parser/format_example.json"),
    )
    parse_parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/cinematic_brief_model_output.schema.json"),
    )
    parse_parser.add_argument("--mock-response", type=Path)
    parse_parser.add_argument("--output", type=Path)

    plan_parser = subparsers.add_parser(
        "plan",
        help="将 Cinematic Brief 通过 Agent 1 转换为验证后的 Scene IR",
    )
    plan_parser.add_argument("--brief", type=Path, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    plan_parser.add_argument(
        "--provider",
        choices=("mock", "openai", "deepseek"),
        default="mock",
    )
    plan_parser.add_argument("--model", help="真实 Provider 的模型名称")
    plan_parser.add_argument("--base-url", help="覆盖 Provider 官方基础地址")
    plan_parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("prompts/scene_planner/system.md"),
    )
    plan_parser.add_argument("--run-id")
    plan_parser.add_argument("--max-requests", type=int, default=12)
    plan_parser.add_argument("--max-tool-calls", type=int, default=40)
    plan_parser.add_argument("--max-input-tokens", type=int, default=120_000)
    plan_parser.add_argument("--max-output-tokens", type=int, default=30_000)
    plan_parser.add_argument("--max-total-tokens", type=int, default=150_000)
    plan_parser.add_argument("--max-seconds", type=float, default=300.0)
    plan_parser.add_argument("--max-commit-attempts", type=int, default=3)
    plan_parser.add_argument("--trace-max-event-bytes", type=int, default=32_768)
    plan_parser.add_argument("--trace-max-string-chars", type=int, default=4_096)
    plan_parser.add_argument("--input-cost-per-million")
    plan_parser.add_argument("--output-cost-per-million")
    plan_parser.add_argument("--cache-read-cost-per-million")
    plan_parser.add_argument("--cache-write-cost-per-million")
    plan_parser.add_argument("--cost-currency", default="USD")
    plan_parser.add_argument("--price-source", default="user_supplied")

    execute_parser = subparsers.add_parser(
        "execute",
        help="将已提交 Scene IR 通过官方 Blender MCP 渲染为白模视频",
    )
    execute_parser.add_argument("--scene-ir", type=Path, required=True)
    execute_parser.add_argument("--output-dir", type=Path, required=True)
    execute_parser.add_argument(
        "--blender-path",
        type=Path,
        default=Path("/opt/homebrew/bin/blender"),
    )
    execute_parser.add_argument(
        "--mcp-command",
        type=Path,
        default=Path.home() / ".local/bin/blender-mcp",
    )
    execute_parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_parse(args: argparse.Namespace) -> int:
    description = args.text if args.text is not None else args.input.read_text(encoding="utf-8")
    provider = _create_provider(args)
    config = SemanticParserConfig(
        system_template_path=args.system_template,
        rules_path=args.rules,
        format_example_path=args.format_example,
        model_output_schema_path=args.schema,
    )
    brief = parse_cinematic_brief(description, provider, config)
    rendered = json.dumps(brief, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _create_provider(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        path = args.mock_response or args.format_example
        response = json.loads(path.read_text(encoding="utf-8"))
        return MockProvider(response)

    if not args.model:
        raise ConfigurationError(f"{args.provider} Provider 需要 --model")

    if args.provider == "openai":
        return OpenAIProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.model,
            base_url=args.base_url or "https://api.openai.com/v1",
            timeout=args.timeout,
        )

    return DeepSeekProvider(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        model=args.model,
        base_url=args.base_url or "https://api.deepseek.com",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )


def _run_plan(args: argparse.Namespace) -> int:
    brief = json.loads(args.brief.read_text(encoding="utf-8"))
    if not isinstance(brief, dict):
        raise ValueError("Cinematic Brief 根节点必须是对象")
    config = InterpreterRunConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        system_prompt_path=args.system_prompt,
        run_dir=args.output_dir,
        run_id=args.run_id,
        max_requests=args.max_requests,
        max_tool_calls=args.max_tool_calls,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
        max_seconds=args.max_seconds,
        max_commit_attempts=args.max_commit_attempts,
        trace_config=TraceConfig(
            max_event_bytes=args.trace_max_event_bytes,
            max_string_chars=args.trace_max_string_chars,
        ),
        cost_rates=_cost_rates(args),
    )
    result = asyncio.run(InterpreterRunner(config).run(brief))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.status == "success" else 1


def _run_execute(args: argparse.Namespace) -> int:
    payload = json.loads(args.scene_ir.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scene IR 根节点必须是对象")
    config = ExecutionConfig(
        output_dir=args.output_dir,
        blender_path=args.blender_path,
        mcp_command=args.mcp_command,
        overwrite=args.overwrite,
    )
    result = asyncio.run(ExecutionRunner(config).run(payload))
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if result.status == "success" else 1


def _cost_rates(args: argparse.Namespace) -> CostRates | None:
    if args.input_cost_per_million is None and args.output_cost_per_million is None:
        if args.provider == "mock":
            return CostRates(input_per_million=Decimal(0), output_per_million=Decimal(0), source="mock")
        return None
    if args.input_cost_per_million is None or args.output_cost_per_million is None:
        raise ConfigurationError("计算成本必须同时提供输入与输出每百万 token 价格")
    return CostRates(
        currency=args.cost_currency,
        input_per_million=Decimal(args.input_cost_per_million),
        output_per_million=Decimal(args.output_cost_per_million),
        cache_read_per_million=(
            Decimal(args.cache_read_cost_per_million)
            if args.cache_read_cost_per_million is not None
            else None
        ),
        cache_write_per_million=(
            Decimal(args.cache_write_cost_per_million)
            if args.cache_write_cost_per_million is not None
            else None
        ),
        source=args.price_source,
    )
