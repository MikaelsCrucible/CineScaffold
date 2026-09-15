from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from cinescaffold import __version__
from cinescaffold.cli_ui import (
    TerminalReporter,
    print_execution_summary,
    print_parse_summary,
    print_pipeline_summary,
    print_planning_summary,
)
from cinescaffold.config import (
    LoadedConfig,
    ModelSettings,
    load_config,
    resolve_model_settings,
    resolve_stage_option,
)
from cinescaffold.errors import CineScaffoldError, ConfigurationError
from cinescaffold.execution.runner import ExecutionConfig, ExecutionRunner
from cinescaffold.planning.runner import (
    DEFAULT_MAX_COMMIT_ATTEMPTS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    InterpreterRunConfig,
    InterpreterRunner,
)
from cinescaffold.planning.trace import CostRates, TraceConfig
from cinescaffold.platforms import default_blender_path, default_mcp_command
from cinescaffold.providers import DeepSeekProvider, MockProvider, OpenAIProvider
from cinescaffold.runtime import RuntimeResourcePaths, cost_rates_from_config
from cinescaffold.semantic import SemanticParseResult, SemanticParserConfig, parse_semantic_input
from cinescaffold.workflow import PipelineRunConfig, PipelineSource, WorkflowRunner


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        loaded_config = load_config(getattr(args, "config", None))
        _apply_config(args, loaded_config)
        if args.command == "parse":
            return _run_parse(args)
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "execute":
            return _run_execute(args)
        if args.command == "run":
            return _run_pipeline(args)
        if args.command == "ui":
            return _run_ui(args, loaded_config)
        parser.print_help()
        return 2
    except (CineScaffoldError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinescaffold",
        description="把自然语言逐步转换为可验证 Scene IR 和 Blender 白模视频。",
        epilog=(
            "常用流程：\n"
            "  cinescaffold parse --provider mock --text \"...\" --output brief.json\n"
            "  cinescaffold plan --provider mock --brief brief.json --output-dir planning\n"
            "  cinescaffold execute --scene-ir planning/final_scene_ir.json "
            "--output-dir execution\n"
            "  cinescaffold run --brief brief.json --output-dir complete-run\n\n"
            "默认显示适合人工阅读的进度和摘要；自动化脚本请使用 --json --quiet。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    parse_parser = subparsers.add_parser("parse", help="将自然语言解析为六维 Cinematic Brief")

    source = parse_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="直接提供自然语言")
    source.add_argument("--input", type=Path, help="从 UTF-8 文本文件读取自然语言")
    source.add_argument("--text-six", help="直接提供包含六个固定部分的文本六维")
    source.add_argument("--text-six-file", type=Path, help="从 UTF-8 文件读取文本六维")

    _add_config_argument(parse_parser)
    _add_model_arguments(parse_parser)
    _add_semantic_arguments(parse_parser)
    parse_parser.add_argument("--output", type=Path)
    parse_parser.add_argument("--text-six-output", type=Path, help="另存人类可读文本六维")
    _add_display_arguments(parse_parser)

    plan_parser = subparsers.add_parser(
        "plan",
        help="将 Cinematic Brief 通过 Agent 1 转换为验证后的 Scene IR",
    )
    plan_parser.add_argument("--brief", type=Path, required=True)
    plan_parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_argument(plan_parser)
    _add_model_arguments(plan_parser)
    _add_planning_arguments(plan_parser)
    _add_display_arguments(plan_parser)

    execute_parser = subparsers.add_parser(
        "execute",
        help="将已提交 Scene IR 通过后台 Blender 构建并渲染为白模视频",
    )
    execute_parser.add_argument("--scene-ir", type=Path, required=True)
    execute_parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_argument(execute_parser)
    _add_execution_arguments(execute_parser)
    _add_display_arguments(execute_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="从自然语言、Cinematic Brief 或 Scene IR 一键运行到白模视频",
    )
    run_source = run_parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument("--text", help="从自然语言开始完整运行")
    run_source.add_argument("--text-six", help="从文本六维开始完整运行")
    run_source.add_argument("--text-six-file", type=Path, help="从文本六维文件开始完整运行")
    run_source.add_argument("--brief", type=Path, help="从 Cinematic Brief 开始运行")
    run_source.add_argument("--scene-ir", "--ir", dest="scene_ir", type=Path, help="从 Scene IR 开始运行")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_argument(run_parser)
    _add_model_arguments(run_parser)
    _add_semantic_arguments(run_parser)
    _add_planning_arguments(run_parser)
    _add_execution_arguments(run_parser)
    _add_display_arguments(run_parser)

    ui_parser = subparsers.add_parser(
        "ui",
        help="启动面向非技术协作者的本地浏览器界面",
    )
    _add_config_argument(ui_parser)
    ui_parser.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅本机可访问")
    ui_parser.add_argument("--port", type=int, default=8080)
    ui_parser.add_argument("--no-open", action="store_true", help="启动后不自动打开浏览器")
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="配置文件路径；缺省时自动读取当前目录 .cinescaffold.conf",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("mock", "openai", "deepseek"))
    parser.add_argument("--model", help="覆盖配置文件中的模型名称")
    parser.add_argument("--base-url", help="覆盖 Provider 基础地址")


def _add_semantic_arguments(parser: argparse.ArgumentParser) -> None:
    resources = RuntimeResourcePaths.from_package()
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--semantic-thinking-mode",
        choices=("enabled", "disabled"),
        help="控制 DeepSeek 语义解析的思考模式；缺省关闭",
    )
    parser.add_argument(
        "--semantic-reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        help="语义解析推理强度；DeepSeek 仅支持 low/high/max",
    )
    parser.add_argument("--rules", type=Path, default=resources.semantic_rules)
    parser.add_argument(
        "--revision-template",
        type=Path,
        default=resources.semantic_revision,
    )
    parser.add_argument(
        "--review-rules",
        type=Path,
        default=resources.semantic_review_rules,
        help="Semantic Revision 的版本化规则召回目录",
    )
    parser.add_argument(
        "--system-template",
        type=Path,
        default=resources.semantic_system,
    )
    parser.add_argument(
        "--format-example",
        type=Path,
        default=resources.semantic_example,
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=resources.semantic_schema,
    )
    parser.add_argument(
        "--translation-rules",
        type=Path,
        default=resources.translation_rules,
        help="四要素到六维的确定性量化表",
    )
    parser.add_argument(
        "--translation-schema",
        type=Path,
        default=resources.translation_schema,
        help="量化快照 Schema",
    )
    parser.add_argument("--mock-response", type=Path)


def _add_planning_arguments(parser: argparse.ArgumentParser) -> None:
    resources = RuntimeResourcePaths.from_package()
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=resources.planning_system,
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="从先前运行的 checkpoint_latest.json 恢复 Candidate",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="累计输入 token 上限；包含每次请求重复发送及缓存命中的上下文",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="单次模型请求的上下文 token 上限",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
    )
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--max-commit-attempts",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled"),
        help="显式设置 DeepSeek 思考模式",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        help="固定模型推理强度",
    )
    parser.add_argument("--model-max-tokens", type=int, help="限制单次模型响应 token 数")
    parser.add_argument(
        "--full-power-diagnostic",
        action="store_true",
        help=(
            "诊断特例：启用 max 思考、流式遥测，并关闭项目侧时间、token、"
            "请求、工具和提交次数上限"
        ),
    )
    parser.add_argument("--trace-max-event-bytes", type=int)
    parser.add_argument("--trace-max-string-chars", type=int)
    parser.add_argument("--input-cost-per-million")
    parser.add_argument("--output-cost-per-million")
    parser.add_argument("--cache-read-cost-per-million")
    parser.add_argument("--cache-write-cost-per-million")
    parser.add_argument("--cost-currency")
    parser.add_argument("--price-source")


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blender-path", type=Path)
    parser.add_argument("--mcp-command", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--build-backend",
        choices=("background", "mcp"),
        default=None,
        help="默认由无窗口 Blender 直接构建；mcp 保留为兼容后端",
    )
    parser.add_argument("--build-timeout-seconds", type=float)
    parser.add_argument(
        "--render-backend",
        choices=("background", "mcp"),
        default=None,
        help="默认由无窗口 Blender 直接渲染；mcp 保留为兼容后端",
    )
    parser.add_argument(
        "--process-mode",
        choices=("fused", "split"),
        default=None,
        help="后台构建和渲染默认合并在一个 Blender 进程；split 用于分阶段诊断",
    )
    parser.add_argument(
        "--render-profile",
        choices=("preview", "control"),
        default=None,
        help="preview 为半分辨率/半采样率诊断视频；control 保持 Scene IR 正式设置",
    )
    parser.add_argument("--render-timeout-seconds", type=float)


def _add_display_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="在 stdout 输出完整机器可读 JSON")
    parser.add_argument("--quiet", action="store_true", help="关闭阶段进度，只保留最终结果")
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")


def _apply_config(args: argparse.Namespace, config: LoadedConfig) -> None:
    args.loaded_config = config
    if args.command == "parse":
        settings = resolve_model_settings(
            config,
            "semantic",
            cli_provider=args.provider,
            cli_model=args.model,
            cli_base_url=args.base_url,
        )
        _assign_model_settings(args, settings)
        _apply_semantic_config(args, config)
    elif args.command == "plan":
        settings = resolve_model_settings(
            config,
            "planning",
            cli_provider=args.provider,
            cli_model=args.model,
            cli_base_url=args.base_url,
        )
        _assign_model_settings(args, settings)
        _apply_planning_config(args, config)
    elif args.command == "execute":
        _apply_execution_config(args, config)
    elif args.command == "run":
        args.semantic_settings = resolve_model_settings(
            config,
            "semantic",
            cli_provider=args.provider,
            cli_model=args.model,
            cli_base_url=args.base_url,
        )
        args.planning_settings = resolve_model_settings(
            config,
            "planning",
            cli_provider=args.provider,
            cli_model=args.model,
            cli_base_url=args.base_url,
        )
        _apply_semantic_config(args, config)
        _apply_planning_config(args, config)
        _apply_execution_config(args, config)
        args.semantic_cost_rates = cost_rates_from_config(
            config,
            "semantic",
            provider=args.semantic_settings.provider,
        )


def _assign_model_settings(args: argparse.Namespace, settings: ModelSettings) -> None:
    args.provider = settings.provider
    args.model = settings.model
    args.base_url = settings.base_url
    args.api_key = settings.api_key


def _apply_planning_config(args: argparse.Namespace, config: LoadedConfig) -> None:
    args.thinking_mode = resolve_stage_option(
        config,
        "planning",
        "thinking_mode",
        args.thinking_mode,
    )
    args.reasoning_effort = resolve_stage_option(
        config,
        "planning",
        "reasoning_effort",
        args.reasoning_effort,
    )
    args.model_max_tokens = resolve_stage_option(
        config,
        "planning",
        "model_max_tokens",
        args.model_max_tokens,
    )
    defaults = {
        "max_requests": DEFAULT_MAX_REQUESTS,
        "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
        "max_input_tokens": None,
        "max_context_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "max_total_tokens": None,
        "max_seconds": DEFAULT_MAX_SECONDS,
        "max_commit_attempts": DEFAULT_MAX_COMMIT_ATTEMPTS,
        "trace_max_event_bytes": 32_768,
        "trace_max_string_chars": 4_096,
        "input_cost_per_million": None,
        "output_cost_per_million": None,
        "cache_read_cost_per_million": None,
        "cache_write_cost_per_million": None,
        "cost_currency": "USD",
        "price_source": "user_supplied",
    }
    for name, default in defaults.items():
        value = resolve_stage_option(config, "planning", name, getattr(args, name))
        setattr(args, name, default if value is None else value)


def _apply_semantic_config(args: argparse.Namespace, config: LoadedConfig) -> None:
    args.semantic_thinking_mode = resolve_stage_option(
        config,
        "semantic",
        "thinking_mode",
        args.semantic_thinking_mode,
    ) or "disabled"
    args.semantic_reasoning_effort = resolve_stage_option(
        config,
        "semantic",
        "reasoning_effort",
        args.semantic_reasoning_effort,
    )
    args.max_tokens = resolve_stage_option(
        config,
        "semantic",
        "max_tokens",
        args.max_tokens,
    ) or 8192
    args.timeout = resolve_stage_option(config, "semantic", "timeout", args.timeout) or 60.0


def _apply_execution_config(args: argparse.Namespace, config: LoadedConfig) -> None:
    blender = args.blender_path or config.data.get("execution_blender_path")
    mcp = args.mcp_command or config.data.get("execution_mcp_command")
    args.blender_path = Path(blender) if blender else default_blender_path()
    args.mcp_command = Path(mcp) if mcp else default_mcp_command()
    args.build_backend = (
        args.build_backend or config.data.get("execution_build_backend") or "background"
    )
    args.render_backend = (
        args.render_backend or config.data.get("execution_render_backend") or "background"
    )
    args.process_mode = (
        args.process_mode or config.data.get("execution_process_mode") or "fused"
    )
    args.render_profile = (
        args.render_profile or config.data.get("execution_render_profile") or "preview"
    )
    build_timeout = (
        args.build_timeout_seconds or config.data.get("execution_build_timeout_seconds")
    )
    args.build_timeout_seconds = float(build_timeout) if build_timeout is not None else 180.0
    timeout = args.render_timeout_seconds or config.data.get("execution_render_timeout_seconds")
    args.render_timeout_seconds = float(timeout) if timeout is not None else 600.0


def _reporter(args: argparse.Namespace) -> TerminalReporter:
    reporter = TerminalReporter(quiet=args.quiet, color=not args.no_color)
    config = getattr(args, "loaded_config", None)
    if config is not None and config.path is not None:
        reporter.success("配置", f"已读取 {config.path}")
    return reporter


def _run_parse(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    source_text, source_kind = _semantic_source(args)
    textual_path = args.text_six_output
    if textual_path is None and args.output is not None:
        textual_path = args.output.with_name("textual_six_dimensions.txt")
    result = _parse_semantic_source(
        args,
        reporter,
        source_text,
        source_kind,
        args.output,
        textual_path,
    )
    brief = result.brief
    rendered = json.dumps(brief, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.json:
            print(rendered, end="")
        else:
            print_parse_summary(brief, args.output, stream=sys.stdout)
    else:
        print(rendered, end="")
        if not args.json:
            print_parse_summary(brief, None, stream=sys.stderr)
    return 0


def _parse_semantic_source(
    args: argparse.Namespace,
    reporter: TerminalReporter,
    source_text: str,
    source_kind: str,
    output_path: Path | None,
    textual_output_path: Path | None,
) -> SemanticParseResult:
    provider = _create_provider(args)
    source_label = "自然语言" if source_kind == "natural_text" else "文本六维"
    reporter.stage(
        "语义解析",
        f"读取 {len(source_text)} 个字符的{source_label}，调用 {provider.name}/{provider.model}",
    )
    config = _semantic_parser_config(args)
    result = parse_semantic_input(
        source_text,
        provider,
        config,
        source_kind=source_kind,
    )
    brief = result.brief
    reporter.success("结构化校验", "Cinematic Brief 已通过关闭 Schema 校验")
    translation = brief.get("translation_parameters", {})
    emotion = translation.get("emotion_class", {})
    reporter.success(
        "规则量化",
        f"{translation.get('rules_version', 'unknown')} · "
        f"{emotion.get('class_id', '?')} {emotion.get('label', '')}".rstrip(),
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(brief, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reporter.success("保存结果", str(output_path.resolve()))
    if textual_output_path:
        textual_output_path.parent.mkdir(parents=True, exist_ok=True)
        textual_output_path.write_text(result.textual_six.render(), encoding="utf-8")
        reporter.success("文本六维", str(textual_output_path.resolve()))
    return result


def _semantic_parser_config(args: argparse.Namespace) -> SemanticParserConfig:
    return SemanticParserConfig(
        system_template_path=args.system_template,
        rules_path=args.rules,
        revision_template_path=args.revision_template,
        review_rules_path=args.review_rules,
        format_example_path=args.format_example,
        model_output_schema_path=args.schema,
        translation_rules_path=args.translation_rules,
        translation_parameters_schema_path=args.translation_schema,
    )


def _create_provider(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        path = args.mock_response or args.format_example
        response = json.loads(path.read_text(encoding="utf-8"))
        return MockProvider(response)

    if not args.model:
        raise ConfigurationError(f"{args.provider} Provider 需要 --model")

    if args.provider == "openai":
        return OpenAIProvider(
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=args.model,
            base_url=args.base_url or "https://api.openai.com/v1",
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            reasoning_effort=args.semantic_reasoning_effort,
        )

    return DeepSeekProvider(
        api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        model=args.model,
        base_url=args.base_url or "https://api.deepseek.com",
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        thinking_mode=args.semantic_thinking_mode,
        reasoning_effort=args.semantic_reasoning_effort,
    )


def _run_plan(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    brief = json.loads(args.brief.read_text(encoding="utf-8"))
    if not isinstance(brief, dict):
        raise ValueError("Cinematic Brief 根节点必须是对象")
    result = _plan_brief(args, reporter, brief, args.output_dir)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print_planning_summary(result, args.output_dir)
    return 0 if result.status == "success" else 1


def _plan_brief(
    args: argparse.Namespace,
    reporter: TerminalReporter,
    brief: dict[str, Any],
    output_dir: Path,
) -> Any:
    config = _planning_run_config(args, output_dir)
    result = asyncio.run(
        InterpreterRunner(config, progress_callback=reporter.event).run(brief)
    )
    return result


def _planning_run_config(
    args: argparse.Namespace,
    output_dir: Path,
) -> InterpreterRunConfig:
    return InterpreterRunConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt_path=args.system_prompt,
        run_dir=output_dir,
        resume_from=args.resume_from,
        run_id=args.run_id,
        max_requests=args.max_requests,
        max_tool_calls=args.max_tool_calls,
        max_input_tokens=args.max_input_tokens,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
        max_seconds=args.max_seconds,
        max_commit_attempts=args.max_commit_attempts,
        thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
        model_max_tokens=args.model_max_tokens,
        full_power_diagnostic=args.full_power_diagnostic,
        trace_config=TraceConfig(
            max_event_bytes=args.trace_max_event_bytes,
            max_string_chars=args.trace_max_string_chars,
        ),
        cost_rates=_cost_rates(args),
    )


def _run_execute(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    payload = json.loads(args.scene_ir.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scene IR 根节点必须是对象")
    result = _execute_scene_ir(args, reporter, payload, args.output_dir)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print_execution_summary(result)
    return 0 if result.status == "success" else 1


def _execute_scene_ir(
    args: argparse.Namespace,
    reporter: TerminalReporter,
    payload: dict[str, Any],
    output_dir: Path,
) -> Any:
    config = _execution_run_config(args, output_dir)
    result = asyncio.run(
        ExecutionRunner(config, progress_callback=reporter.event).run(payload)
    )
    return result


def _execution_run_config(
    args: argparse.Namespace,
    output_dir: Path,
) -> ExecutionConfig:
    return ExecutionConfig(
        output_dir=output_dir,
        blender_path=args.blender_path,
        mcp_command=args.mcp_command,
        overwrite=args.overwrite,
        build_backend=args.build_backend,
        build_timeout_seconds=args.build_timeout_seconds,
        render_backend=args.render_backend,
        process_mode=args.process_mode,
        render_profile=args.render_profile,
        render_timeout_seconds=args.render_timeout_seconds,
    )


def _run_pipeline(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    output_dir = args.output_dir.resolve()
    semantic_args: argparse.Namespace | None = None
    semantic_provider: Any | None = None
    has_textual_six = args.text_six is not None or args.text_six_file is not None
    started_from = (
        "text"
        if args.text is not None
        else "textual_six"
        if has_textual_six
        else "brief"
        if args.brief
        else "scene_ir"
    )

    # 先读取外部输入，避免覆盖时删除位于旧运行目录中的来源文件。
    if args.brief is not None:
        source = PipelineSource(
            kind="brief",
            payload=_read_json_object(args.brief, "Cinematic Brief"),
            artifact_path=args.brief,
        )
    elif args.scene_ir is not None:
        source = PipelineSource(
            kind="scene_ir",
            payload=_read_json_object(args.scene_ir, "Scene IR"),
            artifact_path=args.scene_ir,
        )
    else:
        semantic_args = _stage_args(args, args.semantic_settings)
        source_text, semantic_kind = _semantic_source(args)
        source = PipelineSource(
            kind="text" if semantic_kind == "natural_text" else "textual_six",
            text=source_text,
        )
        semantic_provider = _create_provider(semantic_args)

    planning_args = _stage_args(args, args.planning_settings)
    config = PipelineRunConfig(
        output_dir=output_dir,
        planning=_planning_run_config(planning_args, output_dir / "planning"),
        execution=_execution_run_config(args, output_dir / "execution"),
        semantic_provider=semantic_provider,
        semantic_parser=(
            _semantic_parser_config(semantic_args)
            if semantic_args is not None
            else None
        ),
        semantic_cost_rates=(
            args.semantic_cost_rates if semantic_args is not None else None
        ),
        overwrite=args.overwrite,
    )
    summary = asyncio.run(
        WorkflowRunner(
            config,
            progress_callback=reporter.event,
            planning_runner_type=InterpreterRunner,
            execution_runner_type=ExecutionRunner,
        ).run(source)
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_pipeline_summary(summary)
    return 0 if summary["status"] == "success" else 1


def _run_ui(args: argparse.Namespace, loaded_config: LoadedConfig) -> int:
    from cinescaffold.ui_app import launch_ui

    config_path = args.config or loaded_config.path or Path(".cinescaffold.conf")
    try:
        launch_ui(
            loaded_config,
            config_path=config_path.resolve(),
            project_root=Path.cwd().resolve(),
            host=args.host,
            port=args.port,
            show=not args.no_open,
        )
    except KeyboardInterrupt:
        print("\nCineScaffold Studio 已停止。", file=sys.stderr)
    return 0


def _stage_args(args: argparse.Namespace, settings: ModelSettings) -> argparse.Namespace:
    stage_args = argparse.Namespace(**vars(args))
    _assign_model_settings(stage_args, settings)
    return stage_args


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 根节点必须是对象")
    return value


def _semantic_source(args: argparse.Namespace) -> tuple[str, str]:
    if getattr(args, "text", None) is not None:
        return args.text, "natural_text"
    if getattr(args, "input", None) is not None:
        return args.input.read_text(encoding="utf-8"), "natural_text"
    if getattr(args, "text_six", None) is not None:
        return args.text_six, "textual_six"
    if getattr(args, "text_six_file", None) is not None:
        return args.text_six_file.read_text(encoding="utf-8"), "textual_six"
    raise ValueError("缺少自然语言或文本六维输入")


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
