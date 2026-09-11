from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal

from cinescaffold.execution.runner import ExecutionConfig, ExecutionRunner
from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunner
from cinescaffold.planning.trace import (
    CostRates,
    ProviderCostLimitExceeded,
    estimate_cost_amount,
    provider_usage_summary,
)
from cinescaffold.providers.base import ProviderResponse, StructuredOutputProvider
from cinescaffold.semantic import SemanticParserConfig, parse_semantic_input


PipelineEventCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class PipelineSource:
    kind: Literal["text", "textual_six", "brief", "scene_ir"]
    text: str | None = None
    payload: dict[str, Any] | None = None
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if self.kind in {"text", "textual_six"}:
            if self.text is None or not self.text.strip() or self.payload is not None:
                raise ValueError("自然语言与文本六维来源必须只提供非空 text")
        elif self.payload is None or self.text is not None:
            raise ValueError("Brief 与 Scene IR 来源必须只提供 payload")


@dataclass(frozen=True)
class PipelineRunConfig:
    output_dir: Path
    planning: InterpreterRunConfig
    execution: ExecutionConfig
    semantic_provider: StructuredOutputProvider | None = None
    semantic_parser: SemanticParserConfig | None = None
    semantic_cost_rates: CostRates | None = None
    max_provider_cost: Decimal | None = None
    provider_cost_currency: str | None = None
    overwrite: bool = False


class WorkflowRunner:
    """CLI 与 UI 共用的一键异步应用服务。"""

    def __init__(
        self,
        config: PipelineRunConfig,
        *,
        progress_callback: PipelineEventCallback | None = None,
        planning_runner_type: type[InterpreterRunner] = InterpreterRunner,
        execution_runner_type: type[ExecutionRunner] = ExecutionRunner,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.planning_runner_type = planning_runner_type
        self.execution_runner_type = execution_runner_type

    async def run(self, source: PipelineSource) -> dict[str, Any]:
        started = time.monotonic()
        output_dir = self.config.output_dir.resolve()
        summary_path = output_dir / "pipeline_summary.json"
        self._prepare_output(output_dir, source.kind)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "schema_version": "0.2",
            "status": "running",
            "delivery_tier": None,
            "started_from": source.kind,
            "output_dir": str(output_dir),
            "elapsed_seconds": 0.0,
            "stages": {},
            "artifacts": {},
            "error": None,
        }
        self._emit("pipeline_started", started_from=source.kind, output_dir=str(output_dir))
        stage = "semantic" if source.kind in {"text", "textual_six"} else "input"
        try:
            self._validate_cost_policy(source)
            brief, scene_ir = await self._resolve_source(source, summary, output_dir)
            if brief is not None:
                stage = "planning"
                initial_cost = _stage_cost(summary.get("stages", {}).get("semantic"))
                if (
                    self.config.max_provider_cost is not None
                    and initial_cost >= self.config.max_provider_cost
                ):
                    raise ProviderCostLimitExceeded(
                        amount=initial_cost,
                        limit=self.config.max_provider_cost,
                        currency=self.config.provider_cost_currency or "unknown",
                    )
                self._emit("pipeline_planning_started")
                planning_config = self.config.planning.model_copy(
                    update={
                        "run_dir": output_dir / "planning",
                        "max_cost": self.config.max_provider_cost,
                        "initial_cost": initial_cost,
                    }
                )
                planning_result = await self.planning_runner_type(
                    planning_config,
                    progress_callback=self.progress_callback,
                ).run(brief)
                summary["stages"]["planning"] = planning_result.model_dump(mode="json")
                self._emit("pipeline_planning_completed", status=planning_result.status)
                if planning_result.status != "success":
                    cost_limited = (
                        isinstance(planning_result.error, dict)
                        and planning_result.error.get("limit_type") == "cost_limit"
                    )
                    summary["status"] = (
                        "cost_limit_exceeded" if cost_limited else "planning_failed"
                    )
                    summary["error"] = _planning_error(planning_result.error)
                    return self._finish(summary, summary_path, started)
                scene_ir_name = planning_result.artifacts.get("scene_ir")
                if not scene_ir_name:
                    raise ValueError("规划成功但没有生成 final_scene_ir.json")
                scene_ir_path = planning_config.run_dir / scene_ir_name
                scene_ir = _read_json_object(scene_ir_path, "Scene IR")
                summary["artifacts"]["scene_ir"] = str(scene_ir_path.resolve())

            if scene_ir is None:
                raise ValueError("一键管线没有获得可执行 Scene IR")
            stage = "execution"
            self._emit("pipeline_execution_started")
            execution_config = replace(
                self.config.execution,
                output_dir=output_dir / "execution",
                overwrite=self.config.overwrite,
            )
            execution_result = await self.execution_runner_type(
                execution_config,
                progress_callback=self.progress_callback,
            ).run(scene_ir)
            summary["stages"]["execution"] = execution_result.model_dump(mode="json")
            summary["status"] = execution_result.status
            summary["error"] = execution_result.error
            if execution_result.status == "success":
                summary["delivery_tier"] = (
                    planning_result.delivery_tier
                    if brief is not None
                    else "standard"
                )
            if execution_result.render and execution_result.render.get("artifact"):
                summary["artifacts"]["video"] = execution_result.render["artifact"]
            self._emit("pipeline_execution_completed", status=execution_result.status)
            return self._finish(summary, summary_path, started)
        except ProviderCostLimitExceeded as error:
            summary["status"] = "cost_limit_exceeded"
            summary["error"] = "Provider cost limit reached"
            self._emit(
                "provider_cost_limit_exceeded",
                stage=stage,
                amount=f"{error.amount:.8f}",
                limit=f"{error.limit:.8f}",
                currency=error.currency,
            )
            return self._finish(summary, summary_path, started)
        except Exception as error:
            summary["status"] = f"{stage}_failed"
            summary["error"] = str(error)
            self._emit("pipeline_failed", stage=stage, error=str(error))
            return self._finish(summary, summary_path, started)

    async def _resolve_source(
        self,
        source: PipelineSource,
        summary: dict[str, Any],
        output_dir: Path,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if source.kind in {"text", "textual_six"}:
            if self.config.semantic_provider is None or self.config.semantic_parser is None:
                raise ValueError("语义入口缺少 Provider 或 Semantic Parser 配置")
            self._emit(
                "pipeline_semantic_started",
                source_kind=source.kind,
                provider=self.config.semantic_provider.name,
                model=self.config.semantic_provider.model,
            )
            semantic_started = time.monotonic()
            metered_provider = _MeteredProvider(
                self.config.semantic_provider,
                rates=self.config.semantic_cost_rates,
                progress_callback=self.progress_callback,
            )
            result = await asyncio.to_thread(
                parse_semantic_input,
                source.text or "",
                metered_provider,
                self.config.semantic_parser,
                source_kind=("natural_text" if source.kind == "text" else "textual_six"),
            )
            brief_path = output_dir / "cinematic_brief.json"
            textual_path = output_dir / "textual_six_dimensions.txt"
            brief_path.write_text(
                json.dumps(result.brief, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            textual_path.write_text(result.textual_six.render(), encoding="utf-8")
            usage = result.brief.get("provenance", {}).get("provider_usage")
            summary["stages"]["semantic"] = {
                "status": "success",
                "provider": self.config.semantic_provider.name,
                "model": self.config.semantic_provider.model,
                "elapsed_seconds": round(time.monotonic() - semantic_started, 6),
                "usage": provider_usage_summary(
                    usage if isinstance(usage, dict) else None,
                    self.config.semantic_cost_rates,
                ),
            }
            summary["artifacts"].update(
                {
                    "cinematic_brief": str(brief_path.resolve()),
                    "textual_six_dimensions": str(textual_path.resolve()),
                }
            )
            self._emit(
                "pipeline_semantic_completed",
                brief_path=str(brief_path.resolve()),
                textual_six_path=str(textual_path.resolve()),
                elapsed_seconds=summary["stages"]["semantic"]["elapsed_seconds"],
            )
            return result.brief, None
        if source.kind == "brief":
            assert source.payload is not None
            if source.artifact_path:
                summary["artifacts"]["cinematic_brief"] = str(source.artifact_path.resolve())
            self._emit("pipeline_input_ready", kind=source.kind, path=_path_text(source.artifact_path))
            return source.payload, None
        assert source.payload is not None
        if source.artifact_path:
            summary["artifacts"]["scene_ir"] = str(source.artifact_path.resolve())
        self._emit("pipeline_input_ready", kind=source.kind, path=_path_text(source.artifact_path))
        return None, source.payload

    def _prepare_output(self, output_dir: Path, source_kind: str) -> None:
        owned = [output_dir / "pipeline_summary.json", output_dir / "execution"]
        if source_kind in {"text", "textual_six", "brief"}:
            owned.append(output_dir / "planning")
        if source_kind in {"text", "textual_six"}:
            owned.extend(
                [output_dir / "cinematic_brief.json", output_dir / "textual_six_dimensions.txt"]
            )
        conflicts = [path for path in owned if path.exists() or path.is_symlink()]
        if conflicts and not self.config.overwrite:
            raise ValueError(
                "一键管线产物已存在；请换输出目录或显式使用覆盖："
                + ", ".join(str(path) for path in conflicts)
            )
        if not self.config.overwrite:
            return
        for path in conflicts:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def _finish(
        self,
        summary: dict[str, Any],
        summary_path: Path,
        started: float,
    ) -> dict[str, Any]:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 6)
        summary["artifacts"]["pipeline_summary"] = str(summary_path.resolve())
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._emit(
            "pipeline_finished",
            status=summary["status"],
            elapsed_seconds=summary["elapsed_seconds"],
            summary_path=str(summary_path.resolve()),
        )
        return summary

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event_type, payload)
        except Exception:
            pass

    def _validate_cost_policy(self, source: PipelineSource) -> None:
        if self.config.max_provider_cost is None:
            return
        if self.config.max_provider_cost <= 0:
            raise ValueError("Provider cost limit must be positive")
        if source.kind == "scene_ir":
            return
        rates = [self.config.planning.cost_rates]
        if source.kind in {"text", "textual_six"}:
            rates.append(self.config.semantic_cost_rates)
        if any(item is None for item in rates):
            raise ValueError("Provider cost limit requires price snapshots for every model stage")
        currencies = {item.currency for item in rates if item is not None}
        if len(currencies) != 1 or self.config.provider_cost_currency not in currencies:
            raise ValueError("Provider cost limit currency does not match stage price snapshots")


class _MeteredProvider:
    """Emit semantic cost immediately after the Provider returns, before parsing."""

    def __init__(
        self,
        wrapped: StructuredOutputProvider,
        *,
        rates: CostRates | None,
        progress_callback: PipelineEventCallback | None,
    ) -> None:
        self._wrapped = wrapped
        self._rates = rates
        self._progress_callback = progress_callback
        self.name = wrapped.name
        self.model = wrapped.model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> ProviderResponse:
        if self._progress_callback is not None:
            self._progress_callback(
                "provider_request_started",
                {"stage": "semantic", "request_index": 1},
            )
        response = self._wrapped.generate(system_prompt, user_prompt, schema)
        usage = response.raw_metadata.get("usage")
        if self._rates is None or not isinstance(usage, dict):
            return response
        summary = provider_usage_summary(usage, self._rates)
        amount = estimate_cost_amount(summary["tokens"], self._rates)
        if self._progress_callback is not None:
            self._progress_callback(
                "provider_cost_incurred",
                {
                    "stage": "semantic",
                    "request_index": 1,
                    "amount": f"{amount:.8f}",
                    "cumulative_amount": f"{amount:.8f}",
                    "currency": self._rates.currency,
                    "pricing_source": self._rates.source,
                },
            )
        return response


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 根节点必须是对象")
    return value


def _planning_error(error: dict[str, str] | None) -> str:
    if not error:
        return "场景规划未成功提交 Scene IR"
    return error.get("message", str(error))


def _stage_cost(stage: Any) -> Decimal:
    if not isinstance(stage, dict):
        return Decimal(0)
    usage = stage.get("usage")
    cost = usage.get("estimated_cost") if isinstance(usage, dict) else None
    amount = cost.get("amount") if isinstance(cost, dict) else None
    return Decimal(amount) if isinstance(amount, str) else Decimal(0)


def _path_text(path: Path | None) -> str | None:
    return str(path.resolve()) if path else None
