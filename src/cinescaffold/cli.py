from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
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
from cinescaffold.providers import DeepSeekProvider, MockProvider, OpenAIProvider
from cinescaffold.semantic import SemanticParserConfig, parse_cinematic_brief


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

    _add_config_argument(parse_parser)
    _add_model_arguments(parse_parser)
    _add_semantic_arguments(parse_parser)
    parse_parser.add_argument("--output", type=Path)
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
        help="将已提交 Scene IR 通过官方 Blender MCP 渲染为白模视频",
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
    run_source.add_argument("--brief", type=Path, help="从 Cinematic Brief 开始运行")
    run_source.add_argument("--scene-ir", "--ir", dest="scene_ir", type=Path, help="从 Scene IR 开始运行")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    _add_config_argument(run_parser)
    _add_model_arguments(run_parser)
    _add_semantic_arguments(run_parser)
    _add_planning_arguments(run_parser)
    _add_execution_arguments(run_parser)
    _add_display_arguments(run_parser)
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
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--semantic-thinking-mode",
        choices=("enabled", "disabled"),
        help="控制 DeepSeek 语义解析的思考模式；缺省关闭",
    )
    parser.add_argument(
        "--semantic-reasoning-effort",
        choices=("low", "high", "max"),
        help="语义解析启用思考时的强度",
    )
    parser.add_argument("--rules", type=Path, default=Path("prompts/semantic_parser/rules.md"))
    parser.add_argument(
        "--system-template",
        type=Path,
        default=Path("prompts/semantic_parser/system.md"),
    )
    parser.add_argument(
        "--format-example",
        type=Path,
        default=Path("prompts/semantic_parser/format_example.json"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/cinematic_brief_model_output.schema.json"),
    )
    parser.add_argument(
        "--translation-rules",
        type=Path,
        default=Path("prompts/semantic_parser/translation_rules.json"),
        help="四要素到六维的确定性量化表",
    )
    parser.add_argument(
        "--translation-schema",
        type=Path,
        default=Path("schemas/semantic_translation_parameters.schema.json"),
        help="量化快照 Schema",
    )
    parser.add_argument("--mock-response", type=Path)


def _add_planning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("prompts/scene_planner/system.md"),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="从先前运行的 checkpoint_latest.json 恢复 Candidate",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="累计输入 token 上限；包含每次请求重复发送及缓存命中的上下文",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="单次模型请求的上下文 token 上限",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument(
        "--max-commit-attempts",
        type=int,
        default=DEFAULT_MAX_COMMIT_ATTEMPTS,
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled"),
        help="显式设置 DeepSeek 思考模式",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
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
    parser.add_argument("--trace-max-event-bytes", type=int, default=32_768)
    parser.add_argument("--trace-max-string-chars", type=int, default=4_096)
    parser.add_argument("--input-cost-per-million")
    parser.add_argument("--output-cost-per-million")
    parser.add_argument("--cache-read-cost-per-million")
    parser.add_argument("--cache-write-cost-per-million")
    parser.add_argument("--cost-currency", default="USD")
    parser.add_argument("--price-source", default="user_supplied")


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--blender-path", type=Path, default=Path("/opt/homebrew/bin/blender"))
    parser.add_argument(
        "--mcp-command",
        type=Path,
        default=Path.home() / ".local/bin/blender-mcp",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-backend",
        choices=("background", "mcp"),
        default="background",
        help="默认用无 MCP 调用时限的后台 Blender 渲染；构建仍经官方 MCP",
    )
    parser.add_argument(
        "--render-profile",
        choices=("preview", "control"),
        default="preview",
        help="preview 为半分辨率/半采样率诊断视频；control 保持 Scene IR 正式设置",
    )
    parser.add_argument("--render-timeout-seconds", type=float, default=600.0)


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
    ) or "enabled"
    args.reasoning_effort = resolve_stage_option(
        config,
        "planning",
        "reasoning_effort",
        args.reasoning_effort,
    ) or "low"
    args.model_max_tokens = resolve_stage_option(
        config,
        "planning",
        "model_max_tokens",
        args.model_max_tokens,
    ) or 8192


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


def _reporter(args: argparse.Namespace) -> TerminalReporter:
    reporter = TerminalReporter(quiet=args.quiet, color=not args.no_color)
    config = getattr(args, "loaded_config", None)
    if config is not None and config.path is not None:
        reporter.success("配置", f"已读取 {config.path}")
    return reporter


def _run_parse(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    description = args.text if args.text is not None else args.input.read_text(encoding="utf-8")
    brief = _parse_description(args, reporter, description, args.output)
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


def _parse_description(
    args: argparse.Namespace,
    reporter: TerminalReporter,
    description: str,
    output_path: Path | None,
) -> dict[str, Any]:
    provider = _create_provider(args)
    reporter.stage(
        "自然语言解析",
        f"读取 {len(description)} 个字符，调用 {provider.name}/{provider.model}",
    )
    config = SemanticParserConfig(
        system_template_path=args.system_template,
        rules_path=args.rules,
        format_example_path=args.format_example,
        model_output_schema_path=args.schema,
        translation_rules_path=args.translation_rules,
        translation_parameters_schema_path=args.translation_schema,
    )
    brief = parse_cinematic_brief(description, provider, config)
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
    return brief


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
    config = InterpreterRunConfig(
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
    result = asyncio.run(
        InterpreterRunner(config, progress_callback=reporter.event).run(brief)
    )
    return result


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
    config = ExecutionConfig(
        output_dir=output_dir,
        blender_path=args.blender_path,
        mcp_command=args.mcp_command,
        overwrite=args.overwrite,
        render_backend=args.render_backend,
        render_profile=args.render_profile,
        render_timeout_seconds=args.render_timeout_seconds,
    )
    result = asyncio.run(
        ExecutionRunner(config, progress_callback=reporter.event).run(payload)
    )
    return result


def _run_pipeline(args: argparse.Namespace) -> int:
    reporter = _reporter(args)
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    summary_path = output_dir / "pipeline_summary.json"
    started_from = "text" if args.text is not None else "brief" if args.brief else "scene_ir"

    # 先读取外部输入，避免覆盖时删除位于旧运行目录中的来源文件。
    brief: dict[str, Any] | None = None
    scene_ir_payload: dict[str, Any] | None = None
    if args.brief is not None:
        brief = _read_json_object(args.brief, "Cinematic Brief")
    elif args.scene_ir is not None:
        scene_ir_payload = _read_json_object(args.scene_ir, "Scene IR")

    _prepare_pipeline_output(output_dir, started_from, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter.stage("一键管线", f"从 {started_from} 开始，目标是生成白模视频")
    summary: dict[str, Any] = {
        "schema_version": "0.1",
        "status": "running",
        "started_from": started_from,
        "output_dir": str(output_dir),
        "elapsed_seconds": 0.0,
        "stages": {},
        "artifacts": {},
        "error": None,
    }

    if args.text is not None:
        semantic_args = _stage_args(args, args.semantic_settings)
        brief_path = output_dir / "cinematic_brief.json"
        brief = _parse_description(semantic_args, reporter, args.text, brief_path)
        summary["stages"]["semantic"] = {
            "status": "success",
            "provider": semantic_args.provider,
            "model": semantic_args.model or "mock-cinematic-brief-v0.1",
        }
        summary["artifacts"]["cinematic_brief"] = str(brief_path)
    elif args.brief is not None:
        assert brief is not None
        summary["artifacts"]["cinematic_brief"] = str(args.brief.resolve())
        reporter.success("输入就绪", f"已读取 Cinematic Brief：{args.brief.resolve()}")
    else:
        assert scene_ir_payload is not None
        summary["artifacts"]["scene_ir"] = str(args.scene_ir.resolve())
        reporter.success("输入就绪", f"已读取 Scene IR：{args.scene_ir.resolve()}")

    if brief is not None:
        reporter.stage("场景规划", "将 Cinematic Brief 转换为可提交 Scene IR")
        planning_args = _stage_args(args, args.planning_settings)
        planning_dir = output_dir / "planning"
        planning_result = _plan_brief(planning_args, reporter, brief, planning_dir)
        planning_payload = planning_result.model_dump(mode="json")
        summary["stages"]["planning"] = planning_payload
        if planning_result.status != "success":
            summary["status"] = "planning_failed"
            summary["error"] = _planning_error(planning_result.error)
            return _finish_pipeline(args, reporter, summary, summary_path, started)
        scene_ir_name = planning_result.artifacts.get("scene_ir")
        if not scene_ir_name:
            raise ValueError("规划成功但没有生成 final_scene_ir.json")
        scene_ir_path = planning_dir / scene_ir_name
        scene_ir_payload = _read_json_object(scene_ir_path, "Scene IR")
        summary["artifacts"]["scene_ir"] = str(scene_ir_path)

    if scene_ir_payload is None:
        raise ValueError("一键管线没有获得可执行 Scene IR")
    reporter.stage("Blender 执行", "构建场景、运行验证并渲染白模视频")
    execution_dir = output_dir / "execution"
    execution_result = _execute_scene_ir(args, reporter, scene_ir_payload, execution_dir)
    execution_payload = execution_result.model_dump(mode="json")
    summary["stages"]["execution"] = execution_payload
    summary["status"] = execution_result.status
    summary["error"] = execution_result.error
    if execution_result.render and execution_result.render.get("artifact"):
        summary["artifacts"]["video"] = execution_result.render["artifact"]
    return _finish_pipeline(args, reporter, summary, summary_path, started)


def _prepare_pipeline_output(output_dir: Path, started_from: str, overwrite: bool) -> None:
    owned_paths = [
        output_dir / "pipeline_summary.json",
        output_dir / "execution",
    ]
    if started_from in {"text", "brief"}:
        owned_paths.append(output_dir / "planning")
    if started_from == "text":
        owned_paths.append(output_dir / "cinematic_brief.json")

    conflicts = [path for path in owned_paths if path.exists() or path.is_symlink()]
    if conflicts and not overwrite:
        rendered = ", ".join(str(path) for path in conflicts)
        raise ValueError(f"一键管线产物已存在；请换输出目录或显式使用 --overwrite：{rendered}")
    if not overwrite:
        return

    # 只清理管线拥有的固定路径，不删除输出根目录中的其他资料。
    for path in conflicts:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _stage_args(args: argparse.Namespace, settings: ModelSettings) -> argparse.Namespace:
    stage_args = argparse.Namespace(**vars(args))
    _assign_model_settings(stage_args, settings)
    return stage_args


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 根节点必须是对象")
    return value


def _planning_error(error: dict[str, str] | None) -> str:
    if not error:
        return "场景规划未成功提交 Scene IR"
    return error.get("message", str(error))


def _finish_pipeline(
    args: argparse.Namespace,
    reporter: TerminalReporter,
    summary: dict[str, Any],
    summary_path: Path,
    started: float,
) -> int:
    summary["elapsed_seconds"] = round(time.monotonic() - started, 6)
    summary["artifacts"]["pipeline_summary"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if summary["status"] == "success":
        reporter.success("一键管线", f"全部完成；记录已写入 {summary_path}")
    else:
        reporter.warning("一键管线", f"在状态 {summary['status']} 停止")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_pipeline_summary(summary)
    return 0 if summary["status"] == "success" else 1


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
