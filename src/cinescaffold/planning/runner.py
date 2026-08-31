from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from cinescaffold.planning.agent import PlanningDeps, create_planning_agent
from cinescaffold.planning.checkpoint import (
    load_candidate_checkpoint,
    write_candidate_checkpoint,
)
from cinescaffold.planning.compiler import CommitGateResult, SceneIRCommitGate
from cinescaffold.planning.domain import (
    CommitRequest,
    InfeasibleResult,
    PlanningProfile,
    UnsupportedResult,
)
from cinescaffold.planning.duration import attach_duration_resolution, freeze_brief_duration
from cinescaffold.planning.models import create_planning_model
from cinescaffold.planning.objective import ObjectiveProjection, project_objective_brief
from cinescaffold.planning.toolkit import (
    FULL_VALIDATION_CHECKS,
    TOOLKIT_VERSION,
    ScenePlanningToolkit,
)
from cinescaffold.planning.trace import (
    CostRates,
    StreamTelemetryHandler,
    TraceConfig,
    TraceEventCallback,
    TraceRecorder,
    TracingModel,
    usage_summary,
)


DEFAULT_MAX_REQUESTS = 48
DEFAULT_MAX_TOOL_CALLS = 80
DEFAULT_MAX_CONTEXT_TOKENS = 128_000
DEFAULT_MAX_OUTPUT_TOKENS = 200_000
DEFAULT_MAX_SECONDS = 1_200.0
DEFAULT_MAX_COMMIT_ATTEMPTS = 5


class InterpreterRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str = "mock"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = Field(default=None, exclude=True)
    system_prompt_path: Path = Path("prompts/scene_planner/system.md")
    run_dir: Path
    resume_from: Path | None = None
    run_id: str | None = None
    max_requests: int = Field(default=DEFAULT_MAX_REQUESTS, ge=1)
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_context_tokens: int | None = Field(default=DEFAULT_MAX_CONTEXT_TOKENS, ge=1)
    max_output_tokens: int | None = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_seconds: float = Field(default=DEFAULT_MAX_SECONDS, gt=0)
    max_commit_attempts: int = Field(default=DEFAULT_MAX_COMMIT_ATTEMPTS, ge=1)
    thinking_mode: Literal["enabled", "disabled"] | None = "enabled"
    reasoning_effort: Literal["low", "high", "max"] | None = "low"
    model_max_tokens: int | None = Field(default=8192, ge=1)
    full_power_diagnostic: bool = False
    trace_config: TraceConfig = Field(default_factory=TraceConfig)
    cost_rates: CostRates | None = None


class InterpreterRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: str
    provider: str
    model: str
    terminal_type: str | None
    scene_ir_hash: str | None
    final_revision: int
    usage: dict[str, Any]
    artifacts: dict[str, str]
    error: dict[str, str] | None = None


class InterpreterRunner:
    def __init__(
        self,
        config: InterpreterRunConfig,
        *,
        progress_callback: TraceEventCallback | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback

    async def run(self, cinematic_brief: dict[str, Any]) -> InterpreterRunResult:
        run_id = self.config.run_id or _new_run_id()
        run_dir = self.config.run_dir
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(f"规划输出目录必须为空，避免覆盖实验记录：{run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "planning_agent_tool_trace.jsonl"
        trace = TraceRecorder(
            trace_path,
            run_id,
            self.config.trace_config,
            event_callback=self.progress_callback,
        )
        started = time.monotonic()
        usage = RunUsage()
        projection: ObjectiveProjection | None = None
        toolkit: ScenePlanningToolkit | None = None
        commit_result: CommitGateResult | None = None
        terminal_type: str | None = None
        status = "failed"
        error_payload: dict[str, str] | None = None
        model_label = self.config.model or "mock-scene-planner-v0.1"
        tracing_model: TracingModel | None = None

        try:
            projection = project_objective_brief(cinematic_brief)
            _write_json(run_dir / "cinematic_brief.json", cinematic_brief)
            system_prompt = self.config.system_prompt_path.read_text(encoding="utf-8")
            model_settings = _planning_model_settings(self.config)
            effective_limits = _effective_limits(self.config)
            preserve_thinking_history = _requires_complete_thinking_history(self.config)
            trace.record(
                "run_started",
                provider=self.config.provider,
                model=model_label,
                source_brief_hash=projection.objective_brief.source_brief_sha256,
                system_prompt_sha256=_text_hash(system_prompt),
                toolkit_version=TOOLKIT_VERSION,
                model_settings=model_settings,
                limits=effective_limits,
                full_power_diagnostic=self.config.full_power_diagnostic,
                thinking_history_policy=(
                    "complete_provider_roundtrip"
                    if preserve_thinking_history
                    else "bounded_recent_tool_pairs"
                ),
            )
            if self.config.full_power_diagnostic:
                trace.record(
                    "full_power_diagnostic_enabled",
                    local_time_limit=False,
                    local_usage_limits=False,
                    local_commit_attempt_limit=False,
                    provider_request_timeout=False,
                    stream_telemetry=self.config.provider != "mock",
                    reasoning_content_recorded=True,
                    remaining_external_limits=[
                        "Provider context/output limits",
                        "Provider rate limits and availability",
                        "Operating system and network failures",
                    ],
                )
            trace.record(
                "objective_projection_completed",
                objective_fields=[
                    "subjects",
                    "subject_motion",
                    "scene_design",
                    "composition",
                    "camera",
                    "timeline",
                    "uncertainties",
                ],
                explicit_requirement_count=len(projection.objective_brief.explicit_requirements),
                ignored_subjective_fields=[
                    item.model_dump(mode="json") for item in projection.ignored_subjective_fields
                ],
            )
            profile = PlanningProfile()
            if self.config.resume_from is not None:
                resumed_raw = json.loads(self.config.resume_from.read_text(encoding="utf-8"))
                resolution = resumed_raw["candidate"]["timeline"]["duration_resolution"]
                projection = projection.model_copy(
                    update={
                        "objective_brief": attach_duration_resolution(
                            projection.objective_brief,
                            resolution,
                        )
                    }
                )
                trace.record("duration_resolution_reused", resolution=resolution)
            else:
                resolution = freeze_brief_duration(
                    projection.objective_brief.timeline,
                    fps_numerator=profile.fps_numerator,
                    fps_denominator=profile.fps_denominator,
                )
                projection = projection.model_copy(
                    update={
                        "objective_brief": attach_duration_resolution(
                            projection.objective_brief,
                            resolution,
                        )
                    }
                )
                trace.record(
                    "duration_frozen_from_brief",
                    resolution=resolution.model_dump(mode="json"),
                )
            _write_json(run_dir / "objective_planning_brief.json", projection.model_dump(mode="json"))
            api_key = _provider_api_key(self.config.provider, self.config.api_key)
            raw_model = create_planning_model(
                self.config.provider,
                self.config.model,
                projection.objective_brief,
                api_key=api_key,
                base_url=self.config.base_url,
                disable_request_timeout=self.config.full_power_diagnostic,
            )
            model_label = raw_model.model_name
            tracing_model = TracingModel(
                raw_model,
                trace,
                record_content=self.config.full_power_diagnostic,
            )
            if self.config.full_power_diagnostic:
                # None 会触发 PydanticAI 默认 50 请求上限，必须显式关闭。
                usage_limits = UsageLimits(request_limit=None)
            else:
                usage_limits = UsageLimits(
                    request_limit=self.config.max_requests,
                    tool_calls_limit=self.config.max_tool_calls,
                    input_tokens_limit=self.config.max_input_tokens,
                    per_request_input_tokens_limit=self.config.max_context_tokens,
                    output_tokens_limit=self.config.max_output_tokens,
                    total_tokens_limit=self.config.max_total_tokens,
                )
            resumed_checkpoint = None
            if self.config.resume_from is not None:
                resumed_checkpoint = load_candidate_checkpoint(
                    self.config.resume_from,
                    objective_brief=projection.objective_brief,
                    profile=profile,
                    toolkit_version=TOOLKIT_VERSION,
                )
            toolkit = ScenePlanningToolkit(
                projection.objective_brief,
                profile,
                initial_candidate=(
                    resumed_checkpoint.candidate if resumed_checkpoint is not None else None
                ),
            )
            if resumed_checkpoint is not None:
                trace.record(
                    "candidate_checkpoint_loaded",
                    source_run_id=resumed_checkpoint.run_id,
                    revision=resumed_checkpoint.candidate.revision,
                    candidate_hash=resumed_checkpoint.candidate_hash,
                )
            checkpoint_dir = run_dir / "checkpoints"

            def checkpoint_writer(candidate):
                return write_candidate_checkpoint(
                    checkpoint_dir,
                    run_id=run_id,
                    toolkit_version=TOOLKIT_VERSION,
                    source_brief_sha256=projection.objective_brief.source_brief_sha256,
                    profile_id=profile.profile_id,
                    candidate=candidate,
                )

            checkpoint_writer(toolkit.store.get())
            agent = create_planning_agent(tracing_model, system_prompt)
            deps = PlanningDeps(
                toolkit=toolkit,
                trace=trace,
                deadline_monotonic=(
                    None
                    if self.config.full_power_diagnostic
                    else started + self.config.max_seconds
                ),
                checkpoint_writer=checkpoint_writer,
                preserve_complete_thinking_history=preserve_thinking_history,
            )
            history = None
            prompt = _initial_agent_prompt(
                projection,
                current_revision=toolkit.store.current_revision,
                resumed=resumed_checkpoint is not None,
                resume_summary=(
                    _resume_summary(toolkit) if resumed_checkpoint is not None else None
                ),
            )

            timeout_seconds = (
                None if self.config.full_power_diagnostic else self.config.max_seconds
            )
            attempt = 0
            async with asyncio.timeout(timeout_seconds):
                while (
                    self.config.full_power_diagnostic
                    or attempt < self.config.max_commit_attempts
                ):
                    attempt += 1
                    trace.record(
                        "planning_attempt_started",
                        attempt=attempt,
                        current_revision=toolkit.store.current_revision,
                    )
                    result = await agent.run(
                        prompt,
                        message_history=history,
                        deps=deps,
                        usage=usage,
                        usage_limits=usage_limits,
                        model_settings=model_settings,
                        run_id=f"{run_id}_attempt_{attempt:02d}",
                        conversation_id=run_id,
                        event_stream_handler=(
                            StreamTelemetryHandler(trace, tracing_model)
                            if self.config.full_power_diagnostic
                            and self.config.provider != "mock"
                            else None
                        ),
                    )
                    history = result.all_messages()
                    terminal = result.output
                    terminal_type = terminal.type
                    trace.record(
                        "agent_terminal_received",
                        attempt=attempt,
                        terminal=terminal.model_dump(mode="json"),
                        aggregate_usage=usage_summary(
                            usage,
                            None,
                            tracing_model.request_metrics,
                        ),
                    )
                    if isinstance(terminal, UnsupportedResult):
                        status = "unsupported"
                        break
                    if isinstance(terminal, InfeasibleResult):
                        status = "infeasible"
                        break
                    assert isinstance(terminal, CommitRequest)
                    commit_result = SceneIRCommitGate(toolkit).commit(
                        terminal,
                        agent_run_id=run_id,
                        trace_ref=trace_path.name,
                    )
                    trace.record(
                        "commit_gate_completed",
                        attempt=attempt,
                        requested_revision=terminal.candidate_revision,
                        status=commit_result.status,
                        scene_ir_hash=commit_result.scene_ir_hash,
                        validation=commit_result.validation,
                        violations=commit_result.violations,
                    )
                    if commit_result.status == "success":
                        status = "success"
                        break
                    status = "commit_rejected"
                    prompt = _repair_prompt(commit_result, toolkit.store.current_revision)
        except TimeoutError as error:
            status = "budget_exhausted"
            error_payload = {"type": type(error).__name__, "message": str(error)}
            trace.record("run_failed", status=status, error=error_payload)
        except UsageLimitExceeded as error:
            status = "budget_exhausted"
            error_payload = {
                "type": type(error).__name__,
                "limit_type": _usage_limit_type(str(error)),
                "message": str(error),
            }
            trace.record("run_failed", status=status, error=error_payload)
        except Exception as error:
            status = "failed"
            error_payload = {"type": type(error).__name__, "message": str(error)}
            trace.record("run_failed", status=status, error=error_payload)

        artifacts = _persist_run_artifacts(
            run_dir,
            toolkit,
            commit_result,
        )
        usage_data = usage_summary(
            usage,
            self.config.cost_rates,
            tracing_model.request_metrics if tracing_model is not None else None,
        )
        final_revision = (
            toolkit.store.committed_revision
            if toolkit and toolkit.store.committed_revision is not None
            else toolkit.store.current_revision if toolkit else 0
        )
        summary = InterpreterRunResult(
            run_id=run_id,
            status=status,
            provider=self.config.provider,
            model=model_label,
            terminal_type=terminal_type,
            scene_ir_hash=commit_result.scene_ir_hash if commit_result else None,
            final_revision=final_revision,
            usage=usage_data,
            artifacts=artifacts,
            error=error_payload,
        )
        summary.artifacts["summary"] = "planning_summary.json"
        _write_json(run_dir / "planning_summary.json", summary.model_dump(mode="json"))
        trace.record(
            "run_finished",
            status=status,
            terminal_type=terminal_type,
            final_revision=final_revision,
            scene_ir_hash=summary.scene_ir_hash,
            usage=usage_data,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            artifacts=summary.artifacts,
        )
        return summary


def _initial_agent_prompt(
    projection: ObjectiveProjection,
    *,
    current_revision: int,
    resumed: bool,
    resume_summary: dict[str, Any] | None = None,
) -> str:
    payload = projection.objective_brief.model_dump(mode="json")
    continuation = (
        f"已从可信 checkpoint 恢复 Candidate revision {current_revision}；不要从头重建。"
        if resumed
        else f"当前 Candidate revision 为 {current_revision}。"
    )
    return (
        "请根据以下只含客观内容的 Objective Planning Brief 建立并验证 Candidate。"
        "先检查能力，再通过工具构造；不要处理或猜测已剥离的主观字段。"
        + continuation
        + (
            "恢复摘要如下；不得重复读取 summary，只在修复需要时读取更具体的枚举视图。\n"
            + json.dumps(resume_summary, ensure_ascii=False, separators=(",", ":"))
            if resume_summary is not None
            else ""
        )
        + "\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _resume_summary(toolkit: ScenePlanningToolkit) -> dict[str, Any]:
    state = toolkit.store.get()
    validation = state.validation
    return {
        "revision": state.revision,
        "entity_ids": sorted(state.entities),
        "motion_track_ids": sorted(state.motion_tracks),
        "camera_track_ids": sorted(state.camera.tracks) if state.camera else [],
        "constraint_ids": sorted(state.constraints),
        "validation": (
            {
                "hard_pass": validation.hard_pass,
                "soft_score": validation.soft_score,
                "commit_ready": (
                    validation.hard_pass
                    and validation.soft_score >= toolkit.profile.minimum_soft_score
                ),
                "violation_codes": [item.code for item in validation.violations],
            }
            if validation is not None
            else None
        ),
    }


def _usage_limit_type(message: str) -> str:
    for name in (
        "per_request_input_tokens_limit",
        "input_tokens_limit",
        "output_tokens_limit",
        "total_tokens_limit",
        "request_limit",
        "tool_calls_limit",
        "cost_limit",
    ):
        if name in message:
            return name
    return "unknown_usage_limit"


def _repair_prompt(result: CommitGateResult, current_revision: int) -> str:
    feedback = {
        "current_revision": current_revision,
        "hard_pass": result.validation.get("hard_pass"),
        "soft_score": result.validation.get("soft_score"),
        "violations": result.violations,
    }
    return (
        "Commit Gate 拒绝了提交。只依据以下结构化证据继续检查、修复、求解和验证；"
        "不要放松 explicit hard requirement。\n\n"
        + json.dumps(feedback, ensure_ascii=False, indent=2)
    )


def _persist_run_artifacts(
    run_dir: Path,
    toolkit: ScenePlanningToolkit | None,
    commit_result: CommitGateResult | None,
) -> dict[str, str]:
    artifacts = {"trace": "planning_agent_tool_trace.jsonl"}
    for key, name in (
        ("cinematic_brief", "cinematic_brief.json"),
        ("objective_brief", "objective_planning_brief.json"),
    ):
        if (run_dir / name).exists():
            artifacts[key] = name
    if toolkit is not None:
        constraint_plan_path = run_dir / "constraint_plan.json"
        _write_json(
            constraint_plan_path,
            (
                commit_result.constraint_plan
                if commit_result and commit_result.status == "success"
                else toolkit.store.get().model_dump(mode="json")
            ),
        )
        artifacts["constraint_plan"] = constraint_plan_path.name
        validation = toolkit.validate_candidate(checks=FULL_VALIDATION_CHECKS)
        validation_path = run_dir / "planning_validation.json"
        _write_json(validation_path, validation["data"])
        artifacts["validation"] = validation_path.name
        checkpoint_path = run_dir / "checkpoint_latest.json"
        if checkpoint_path.is_file():
            artifacts["checkpoint"] = checkpoint_path.name
    if commit_result and commit_result.scene_ir is not None:
        scene_ir_path = run_dir / "final_scene_ir.json"
        _write_json(scene_ir_path, commit_result.scene_ir.model_dump(mode="json"))
        artifacts["scene_ir"] = scene_ir_path.name
    return artifacts


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _provider_api_key(provider: str, configured: str | None = None) -> str | None:
    if configured:
        return configured
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY")
    return None


def _planning_model_settings(config: InterpreterRunConfig) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    reasoning_effort = "max" if config.full_power_diagnostic else config.reasoning_effort
    thinking_mode = "enabled" if config.full_power_diagnostic else config.thinking_mode
    model_max_tokens = None if config.full_power_diagnostic else config.model_max_tokens
    if reasoning_effort is not None:
        settings["openai_reasoning_effort"] = reasoning_effort
    if config.provider == "deepseek":
        extra_body: dict[str, Any] = {}
        if thinking_mode is not None:
            extra_body["thinking"] = {"type": thinking_mode}
        if model_max_tokens is not None:
            # DeepSeek Chat Completions 使用 max_tokens。
            extra_body["max_tokens"] = model_max_tokens
        if extra_body:
            settings["extra_body"] = extra_body
    elif model_max_tokens is not None:
        settings["max_tokens"] = model_max_tokens
    return settings


def _effective_limits(config: InterpreterRunConfig) -> dict[str, Any]:
    if config.full_power_diagnostic:
        return {
            "max_requests": None,
            "max_tool_calls": None,
            "max_input_tokens": None,
            "max_context_tokens": None,
            "max_output_tokens": None,
            "max_total_tokens": None,
            "max_seconds": None,
            "max_commit_attempts": None,
        }
    return {
        "max_requests": config.max_requests,
        "max_tool_calls": config.max_tool_calls,
        "max_input_tokens": config.max_input_tokens,
        "max_context_tokens": config.max_context_tokens,
        "max_output_tokens": config.max_output_tokens,
        "max_total_tokens": config.max_total_tokens,
        "max_seconds": config.max_seconds,
        "max_commit_attempts": config.max_commit_attempts,
    }


def _requires_complete_thinking_history(config: InterpreterRunConfig) -> bool:
    """DeepSeek 带工具思考时必须在后续请求中回传完整思考历史。"""

    return config.provider == "deepseek" and (
        config.full_power_diagnostic or config.thinking_mode != "disabled"
    )


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"planning_{timestamp}_{uuid.uuid4().hex[:8]}"


def _text_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
