from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from cinescaffold.errors import CineScaffoldError, ConfigurationError
from cinescaffold.providers import DeepSeekProvider, MockProvider, OpenAIProvider
from cinescaffold.semantic import SemanticParserConfig, parse_cinematic_brief


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "parse":
            return _run_parse(args)
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
