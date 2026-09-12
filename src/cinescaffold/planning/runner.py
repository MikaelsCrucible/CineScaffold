from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.exceptions import AgentRunError, UsageLimitExceeded
from pydantic_ai.usage import RunUsage, UsageLimits

from cinescaffold.planning.agent import (
    PlanningDeps,
    compact_agent_payload,
    create_planning_agent,
)
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
from cinescaffold.planning.models import (
    build_deterministic_scene_skeleton,
    create_planning_model,
)
from cinescaffold.planning.objective import ObjectiveProjection, project_objective_brief
from cinescaffold.planning.toolkit import (
    EXECUTION_SAFETY_CHECKS,
    FULL_VALIDATION_CHECKS,
    TOOLKIT_VERSION,
    ScenePlanningToolkit,
)
from cinescaffold.planning.trace import (
    CostRates,
    ProviderCostLimitExceeded,
    StreamTelemetryHandler,
    TraceConfig,
    TraceEventCallback,
    TraceRecorder,
    TracingModel,
    usage_summary,
)
from cinescaffold.resources.paths import RuntimeResourcePaths


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
    system_prompt_path: Path = RuntimeResourcePaths.from_package().planning_system
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
    thinking_mode: Literal["enabled", "disabled"] | None = None
    reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None = None
    model_max_tokens: int | None = Field(default=None, ge=1)
    full_power_diagnostic: bool = False
    trace_config: TraceConfig = Field(default_factory=TraceConfig)
    cost_rates: CostRates | None = None
    max_cost: Decimal | None = Field(default=None, gt=0)
    initial_cost: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def require_prices_for_cost_limit(self) -> "InterpreterRunConfig":
        if (
            self.max_cost is not None
            and not self.full_power_diagnostic
            and self.cost_rates is None
        ):
            raise ValueError("Provider cost limit requires a price snapshot")
        return self


class InterpreterRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    status: str
    provider: str
    model: str
    terminal_type: str | None
    scene_ir_hash: str | None
    final_revision: int
    delivery_tier: Literal["standard", "recovered", "simplified"] | None = None
    recovery_context: dict[str, Any] | None = None
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
        delivery_tier: Literal["standard", "recovered", "simplified"] | None = None
        recovery_context: dict[str, Any] | None = None
        profile: PlanningProfile | None = None
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
                cost_rates=self.config.cost_rates,
                max_cost=(None if self.config.full_power_diagnostic else self.config.max_cost),
                initial_cost=self.config.initial_cost,
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
                    try:
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
                    except UsageLimitExceeded:
                        raise
                    except AgentRunError as error:
                        error_payload = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        recovery_context = _recovery_context(
                            toolkit,
                            stage="agent_response",
                            failure_class="invalid_or_truncated_model_response",
                            attempts_remaining=_attempts_remaining(self.config, attempt),
                        )
                        trace.record(
                            "planning_recovery_requested",
                            attempt=attempt,
                            error=error_payload,
                            recovery_context=recovery_context,
                        )
                        if _can_retry(self.config, attempt):
                            prompt = _recovery_prompt(recovery_context)
                            continue
                        status = "failed"
                        break
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
                    elif isinstance(terminal, InfeasibleResult):
                        status = "infeasible"
                    else:
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
                            gate_mode=commit_result.gate_mode,
                            scene_ir_hash=commit_result.scene_ir_hash,
                            validation=commit_result.validation,
                            violations=commit_result.violations,
                        )
                        if commit_result.status == "success":
                            status = "success"
                            delivery_tier = "standard" if attempt == 1 else "recovered"
                            break
                        status = "commit_rejected"

                    recovery_context = _recovery_context(
                        toolkit,
                        stage=("commit_gate" if status == "commit_rejected" else "agent_terminal"),
                        failure_class=status,
                        attempts_remaining=_attempts_remaining(self.config, attempt),
                        violations=(
                            commit_result.violations
                            if status == "commit_rejected" and commit_result
                            else None
                        ),
                    )
                    trace.record(
                        "planning_recovery_requested",
                        attempt=attempt,
                        terminal_type=terminal_type,
                        recovery_context=recovery_context,
                    )
                    if not _can_retry(self.config, attempt):
                        break
                    prompt = _recovery_prompt(recovery_context)

                if status != "success" and not self.config.full_power_diagnostic:
                    toolkit, commit_result, recovery_context = (
                        _commit_simplified_delivery_with_checkpoint(
                            toolkit,
                            projection,
                            profile,
                            run_dir=run_dir,
                            run_id=run_id,
                            trace_path=trace_path,
                            trace=trace,
                            failure_class=status,
                            previous_context=recovery_context,
                        )
                    )
                    status = "success"
                    terminal_type = "simplified_delivery"
                    delivery_tier = "simplified"
                    error_payload = None
        except TimeoutError as error:
            status = "budget_exhausted"
            error_payload = {"type": type(error).__name__, "message": str(error)}
            trace.record("run_failed", status=status, error=error_payload)
        except UsageLimitExceeded as error:
            status = "budget_exhausted"
            limit_type = _usage_limit_type(str(error))
            error_payload = {
                "type": type(error).__name__,
                "limit_type": limit_type,
                "message": str(error),
            }
            trace.record("planning_usage_limit_reached", error=error_payload)
            if (
                not self.config.full_power_diagnostic
                and toolkit is not None
                and projection is not None
                and profile is not None
            ):
                try:
                    toolkit, commit_result, recovery_context = (
                        _commit_simplified_delivery_with_checkpoint(
                            toolkit,
                            projection,
                            profile,
                            run_dir=run_dir,
                            run_id=run_id,
                            trace_path=trace_path,
                            trace=trace,
                            failure_class=f"usage_limit:{limit_type}",
                            previous_context={
                                "stage": "agent_usage_limit",
                                "failure_class": "budget_exhausted",
                                "limit_type": limit_type,
                            },
                        )
                    )
                except Exception as fallback_error:
                    trace.record(
                        "run_failed",
                        status=status,
                        error=error_payload,
                        simplified_delivery_error={
                            "type": type(fallback_error).__name__,
                            "message": str(fallback_error),
                        },
                    )
                else:
                    status = "success"
                    terminal_type = "simplified_delivery"
                    delivery_tier = "simplified"
                    error_payload = None
            else:
                trace.record("run_failed", status=status, error=error_payload)
        except ProviderCostLimitExceeded as error:
            status = "budget_exhausted"
            error_payload = {
                "type": type(error).__name__,
                "limit_type": "cost_limit",
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
            delivery_tier=delivery_tier,
            recovery_context=recovery_context,
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
            delivery_tier=delivery_tier,
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
    workflow = (
        "请继续验证或修复已恢复的 Candidate；不得重新提交 Scene Skeleton。"
        if resumed
        else (
            "先提交不含数值的 Scene Skeleton，再请求并应用 Toolkit Design Option；"
            "不要处理或猜测已剥离的主观字段。"
        )
    )
    return (
        "请根据以下只含客观内容的 Objective Planning Brief 建立并验证 Candidate。"
        + workflow
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
                "violation_codes": sorted(
                    {item.code for item in validation.violations}
                ),
                "violation_count": len(validation.violations),
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


def _can_retry(config: InterpreterRunConfig, attempt: int) -> bool:
    return not config.full_power_diagnostic and attempt < config.max_commit_attempts


def _attempts_remaining(config: InterpreterRunConfig, attempt: int) -> int:
    if config.full_power_diagnostic:
        return 0
    return max(0, config.max_commit_attempts - attempt)


def _recovery_context(
    toolkit: ScenePlanningToolkit,
    *,
    stage: str,
    failure_class: str,
    attempts_remaining: int,
    violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current = toolkit.store.get()
    validation = toolkit.validate_candidate(
        revision=current.revision,
        checks=FULL_VALIDATION_CHECKS,
    )
    best_revision = _best_executable_revision(toolkit)
    effective_violations = violations or validation["violations"]
    context = {
        "stage": stage,
        "failure_class": failure_class,
        "current_revision": current.revision,
        "best_revision": best_revision,
        "hard_pass": validation["data"]["hard_pass"],
        "soft_score": validation["data"]["soft_score"],
        "violations": effective_violations,
        "capability_gaps": validation["capability_gaps"],
        "repair_search_exhausted": toolkit.has_exhausted_repair_search,
        "attempts_remaining": attempts_remaining,
        "allowed_recovery_actions": [
            "inspect_current_or_historical_candidate",
            "apply_validator_scoped_patch_after_repair_search_exhaustion",
            "restore_best_revision",
            "solve_and_validate",
            "request_full_fidelity_commit",
            "allow_deterministic_simplified_delivery",
        ],
    }
    return compact_agent_payload(
        context,
        frame_interval_seconds=(
            current.timeline.fps_denominator / current.timeline.fps_numerator
        ),
    )


def _recovery_prompt(feedback: dict[str, Any]) -> str:
    return (
        "上一轮未被系统接受为完整交付。根据以下 Recovery Context 继续；"
        "不要从头重建，不要放松 explicit hard requirement，也不要重复已穷尽的修复搜索。"
        "若确定性建议已返回 no_change，可仅针对 violation 指向字段使用重新开放的受控 Mutation，"
        "修改后必须重新验证。系统会在完整修复仍不成功时独立生成简化交付。\n\n"
        + json.dumps(feedback, ensure_ascii=False, indent=2)
    )


def _best_executable_revision(toolkit: ScenePlanningToolkit) -> int | None:
    ranked: list[tuple[int, float, int]] = []
    for revision in toolkit.store.revisions:
        state = toolkit.store.get(revision)
        if not state.entities or state.camera is None:
            continue
        safety = toolkit.validate_candidate(
            revision=revision,
            checks=EXECUTION_SAFETY_CHECKS,
        )["data"]
        if not safety["hard_pass"]:
            continue
        fidelity = toolkit.validate_candidate(
            revision=revision,
            checks=FULL_VALIDATION_CHECKS,
        )["data"]
        hard_count = sum(
            item["severity"] == "hard" for item in fidelity["violations"]
        )
        ranked.append((hard_count, -float(fidelity["soft_score"]), -revision))
    if not ranked:
        return None
    return -min(ranked)[2]


def _commit_simplified_delivery(
    toolkit: ScenePlanningToolkit,
    projection: ObjectiveProjection,
    profile: PlanningProfile,
    *,
    run_id: str,
    trace_path: Path,
    trace: TraceRecorder,
    failure_class: str,
    previous_context: dict[str, Any] | None,
) -> tuple[ScenePlanningToolkit, CommitGateResult, dict[str, Any]]:
    revision = _best_executable_revision(toolkit)
    source = "best_agent_candidate"
    if revision is None:
        source = "deterministic_brief_fallback"
        toolkit = ScenePlanningToolkit(projection.objective_brief, profile)
        skeleton = build_deterministic_scene_skeleton(projection.objective_brief)
        submitted = toolkit.submit_scene_skeleton(skeleton)
        if submitted["status"] != "ok":
            raise RuntimeError(
                f"deterministic fallback skeleton was rejected: {submitted['warnings']}"
            )
        options = toolkit.request_design_options(preference="balanced", max_options=3)
        candidates = options["data"].get("options", [])
        if not candidates:
            raise RuntimeError(
                f"deterministic fallback produced no design option: {options['warnings']}"
            )
        selected = min(
            candidates,
            key=lambda item: (
                item["predicted"]["hard_violation_count"],
                -float(item["predicted"]["soft_score"]),
                item["option_id"],
            ),
        )
        applied = toolkit.apply_design_option(
            selected["base_revision"],
            selected["option_id"],
        )
        if applied["status"] != "ok":
            raise RuntimeError(
                f"deterministic fallback option was rejected: {applied['warnings']}"
            )
        revision = toolkit.store.current_revision

    request = CommitRequest(
        type="commit_request",
        candidate_revision=revision,
        summary="Deterministic simplified delivery after planning recovery was exhausted.",
    )
    result = SceneIRCommitGate(toolkit).commit_simplified(
        request,
        agent_run_id=run_id,
        trace_ref=trace_path.name,
    )
    if result.status != "success":
        raise RuntimeError("no Candidate passed the execution safety gate")
    context = {
        "stage": "simplified_delivery",
        "failure_class": failure_class,
        "current_revision": toolkit.store.current_revision,
        "best_revision": revision,
        "source": source,
        "gate_mode": result.gate_mode,
        "fidelity_violations": result.violations,
        "previous": previous_context,
    }
    trace.record(
        "simplified_delivery_committed",
        requested_revision=revision,
        source=source,
        gate_mode=result.gate_mode,
        scene_ir_hash=result.scene_ir_hash,
        fidelity_validation=result.validation,
        fidelity_violations=result.violations,
    )
    return toolkit, result, context


def _commit_simplified_delivery_with_checkpoint(
    toolkit: ScenePlanningToolkit,
    projection: ObjectiveProjection,
    profile: PlanningProfile,
    *,
    run_dir: Path,
    run_id: str,
    trace_path: Path,
    trace: TraceRecorder,
    failure_class: str,
    previous_context: dict[str, Any] | None,
) -> tuple[ScenePlanningToolkit, CommitGateResult, dict[str, Any]]:
    toolkit, result, context = _commit_simplified_delivery(
        toolkit,
        projection,
        profile,
        run_id=run_id,
        trace_path=trace_path,
        trace=trace,
        failure_class=failure_class,
        previous_context=previous_context,
    )
    checkpoint_directory = (
        run_dir / "fallback_checkpoints"
        if context["source"] == "deterministic_brief_fallback"
        else run_dir / "checkpoints"
    )
    write_candidate_checkpoint(
        checkpoint_directory,
        run_id=run_id,
        toolkit_version=TOOLKIT_VERSION,
        source_brief_sha256=projection.objective_brief.source_brief_sha256,
        profile_id=profile.profile_id,
        candidate=toolkit.store.get(),
    )
    return toolkit, result, context


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
        artifact_revision = (
            toolkit.store.committed_revision
            if toolkit.store.committed_revision is not None
            else toolkit.store.current_revision
        )
        constraint_plan_path = run_dir / "constraint_plan.json"
        _write_json(
            constraint_plan_path,
            (
                commit_result.constraint_plan
                if commit_result and commit_result.status == "success"
                else toolkit.store.get(artifact_revision).model_dump(mode="json")
            ),
        )
        artifacts["constraint_plan"] = constraint_plan_path.name
        validation = toolkit.validate_candidate(
            revision=artifact_revision,
            checks=FULL_VALIDATION_CHECKS,
        )
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
            "max_cost": None,
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
        "max_cost": str(config.max_cost) if config.max_cost is not None else None,
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
