from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage

from cinescaffold.planning.models import create_planning_model
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.planning.runner import (
    InterpreterRunConfig,
    InterpreterRunner,
    _effective_limits,
    _planning_model_settings,
    _route_context_index,
    _requires_complete_thinking_history,
    _usage_limit_type,
)
from cinescaffold.planning.toolkit import NARRATIVE_FIDELITY_CHECKS
from cinescaffold.planning.trace import (
    CostRates,
    ProviderCostLimitExceeded,
    StreamTelemetryHandler,
    TraceConfig,
    TraceRecorder,
    TracingModel,
    provider_usage_summary,
)
from tests.helpers import ROOT, valid_planning_brief


class InterpreterRunnerTest(unittest.TestCase):
    def test_route_context_indexes_motion_and_shared_events_without_coordinates(self) -> None:
        brief = json.loads(
            (ROOT / "examples/cinematic_briefs/roadside_pickup_12s.json").read_text(
                encoding="utf-8"
            )
        )
        objective = project_objective_brief(brief).objective_brief

        context = _route_context_index(objective)

        self.assertIn("car", context["motion_timelines"])
        self.assertTrue(context["shared_events"])
        self.assertTrue(context["spatial_relations"])
        self.assertNotIn("translation_m", json.dumps(context))

    def test_context_limit_falls_back_without_another_model_request(self) -> None:
        calls = 0

        def callback(messages, info) -> ModelResponse:
            del messages
            nonlocal calls
            calls += 1
            output_tool = next(
                item for item in info.output_tools if item.name.endswith("InfeasibleResult")
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        output_tool.name,
                        {
                            "type": "infeasible",
                            "conflicting_constraint_ids": ["unprocessed"],
                            "evidence": ["the response crossed the context limit"],
                            "attempted_revisions": [0],
                        },
                        tool_call_id="over_context_terminal",
                    )
                ],
                usage=RequestUsage(input_tokens=200, output_tokens=10),
                finish_reason="stop",
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = FunctionModel(callback, model_name="context-limit-test")
            config = InterpreterRunConfig(
                provider="mock",
                run_dir=run_dir,
                run_id="context_limit_fallback_test",
                max_context_tokens=100,
            )
            with patch(
                "cinescaffold.planning.runner.create_planning_model",
                return_value=model,
            ):
                result = asyncio.run(InterpreterRunner(config).run(valid_planning_brief()))
            trace = (run_dir / "planning_agent_tool_trace.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(calls, 1)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.delivery_tier, "simplified")
        self.assertEqual(result.terminal_type, "simplified_delivery")
        self.assertEqual(
            result.recovery_context["failure_class"],
            "usage_limit:per_request_input_tokens_limit",
        )
        self.assertIn("planning_usage_limit_reached", trace)
        self.assertIn("simplified_delivery_committed", trace)

    def test_infeasible_terminal_falls_back_to_executable_scene_ir(self) -> None:
        request_messages: list[str] = []

        def callback(messages, info) -> ModelResponse:
            request_messages.append(repr(messages))
            output_tool = next(
                item for item in info.output_tools if item.name.endswith("InfeasibleResult")
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        output_tool.name,
                        {
                            "type": "infeasible",
                            "conflicting_constraint_ids": ["claimed_conflict"],
                            "evidence": ["model gave up before building a Candidate"],
                            "attempted_revisions": [0],
                        },
                        tool_call_id="infeasible_terminal",
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=10),
                finish_reason="stop",
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = FunctionModel(callback, model_name="infeasible-test")
            config = InterpreterRunConfig(
                provider="mock",
                run_dir=run_dir,
                run_id="simplified_delivery_test",
                max_commit_attempts=2,
            )
            with patch(
                "cinescaffold.planning.runner.create_planning_model",
                return_value=model,
            ):
                result = asyncio.run(InterpreterRunner(config).run(valid_planning_brief()))
            scene_ir = json.loads(
                (run_dir / result.artifacts["scene_ir"]).read_text(encoding="utf-8")
            )
            trace = (run_dir / "planning_agent_tool_trace.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.delivery_tier, "simplified")
        self.assertEqual(result.terminal_type, "simplified_delivery")
        self.assertEqual(result.recovery_context["stage"], "simplified_delivery")
        self.assertEqual(
            scene_ir["acceptance"]["required_validators"],
            NARRATIVE_FIDELITY_CHECKS,
        )
        self.assertEqual(len(request_messages), 2)
        second_request = request_messages[1]
        self.assertIn("repair_packet_version", second_request)
        self.assertIn("narrative_motions", second_request)
        self.assertNotIn("infeasible_terminal", second_request)
        self.assertIn("planning_recovery_requested", trace)
        self.assertIn("simplified_delivery_committed", trace)

    def test_openai_provider_usage_uses_cache_aware_cost_snapshot(self) -> None:
        result = provider_usage_summary(
            {
                "input_tokens": 10_000,
                "input_tokens_details": {"cached_tokens": 4_000},
                "output_tokens": 2_000,
                "output_tokens_details": {"reasoning_tokens": 500},
            },
            CostRates(
                input_per_million=Decimal("2"),
                output_per_million=Decimal("12"),
                cache_read_per_million=Decimal("0.2"),
                cache_write_per_million=Decimal("2.5"),
                source="openai_terra_test",
            ),
        )

        self.assertEqual(result["tokens"]["cache_read_tokens"], 4_000)
        self.assertEqual(result["tokens"]["details"]["reasoning_tokens"], 500)
        self.assertEqual(result["estimated_cost"]["amount"], "0.03680000")
        self.assertEqual(result["pricing_snapshot"]["source"], "openai_terra_test")

    def test_tracing_model_stops_before_request_after_postpaid_cost_limit(self) -> None:
        calls = 0

        def callback(messages, info) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                parts=[TextPart("done")],
                usage=RequestUsage(input_tokens=10, output_tokens=3),
            )

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace = TraceRecorder(trace_path, "cost_limit_test")
            model = TracingModel(
                FunctionModel(callback, model_name="probe"),
                trace,
                cost_rates=CostRates(
                    currency="CNY",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                    source="test",
                ),
                max_cost=Decimal("0.00001"),
            )
            agent = Agent(model, system_prompt="system")
            asyncio.run(agent.run("first"))
            with self.assertRaises(ProviderCostLimitExceeded):
                asyncio.run(agent.run("second"))
            rendered = trace_path.read_text(encoding="utf-8")

        self.assertEqual(calls, 1)
        self.assertIn("provider_cost_incurred", rendered)
        self.assertIn("provider_cost_limit_exceeded", rendered)

    def test_cost_limit_requires_a_price_snapshot(self) -> None:
        with self.assertRaisesRegex(ValueError, "price snapshot"):
            InterpreterRunConfig(
                run_dir=Path("unused"),
                max_cost=Decimal("2"),
            )

    def test_real_provider_models_build_without_network_call(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief

        openai_model = create_planning_model(
            "openai", "gpt-test", objective, api_key="test-key"
        )
        deepseek_model = create_planning_model(
            "deepseek", "deepseek-chat", objective, api_key="test-key"
        )
        deepseek_flash_model = create_planning_model(
            "deepseek", "deepseek-flash", objective, api_key="test-key"
        )

        self.assertEqual(openai_model.provider.name, "openai")
        self.assertEqual(deepseek_model.provider.name, "deepseek")
        self.assertEqual(deepseek_flash_model.provider.name, "deepseek")
        self.assertTrue(deepseek_flash_model.profile["supports_thinking"])
        self.assertFalse(
            deepseek_flash_model.profile["openai_supports_tool_choice_required"]
        )

    def test_deepseek_settings_use_chat_completion_fields(self) -> None:
        config = InterpreterRunConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            run_dir=Path("unused"),
            thinking_mode="enabled",
            reasoning_effort="high",
            model_max_tokens=8192,
        )
        self.assertEqual(
            _planning_model_settings(config),
            {
                "openai_reasoning_effort": "high",
                "extra_body": {
                    "thinking": {"type": "enabled"},
                    "max_tokens": 8192,
                },
            },
        )

    def test_full_power_diagnostic_overrides_local_limits_and_model_caps(self) -> None:
        config = InterpreterRunConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            run_dir=Path("unused"),
            thinking_mode="disabled",
            reasoning_effort="low",
            model_max_tokens=1024,
            full_power_diagnostic=True,
        )

        self.assertEqual(
            _planning_model_settings(config),
            {
                "openai_reasoning_effort": "max",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        )
        self.assertTrue(all(value is None for value in _effective_limits(config).values()))
        self.assertTrue(_requires_complete_thinking_history(config))

        disabled = config.model_copy(
            update={"full_power_diagnostic": False, "thinking_mode": "disabled"}
        )
        self.assertFalse(_requires_complete_thinking_history(disabled))

    def test_stream_telemetry_records_reasoning_and_text_content(self) -> None:
        async def events():
            yield PartStartEvent(index=0, part=ThinkingPart("private-one"))
            yield PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="private-two"),
            )
            yield PartStartEvent(index=1, part=TextPart("answer-text"))
            yield PartDeltaEvent(
                index=1,
                delta=TextPartDelta(content_delta="-more"),
            )

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace = TraceRecorder(trace_path, "stream_metadata_test")
            started = time.monotonic()
            model = SimpleNamespace(
                active_stream_request_index=1,
                active_stream_started=started,
                record_content=True,
            )
            asyncio.run(StreamTelemetryHandler(trace, model)(None, events()))
            rendered = trace_path.read_text(encoding="utf-8")
            payload = json.loads(rendered)["payload"]

        self.assertIn("private-one", rendered)
        self.assertIn("private-two", rendered)
        self.assertIn("answer-text", rendered)
        self.assertIn("-more", rendered)
        self.assertEqual(payload["reasoning_content"], "private-oneprivate-two")
        self.assertEqual(payload["text_content"], "answer-text-more")
        self.assertEqual(payload["reasoning_chunks"], 2)
        self.assertEqual(payload["reasoning_chars"], 22)
        self.assertTrue(payload["reasoning_content_recorded"])
        self.assertTrue(payload["text_content_recorded"])

    def test_stream_telemetry_stall_heartbeat_carries_partial_content(self) -> None:
        async def events():
            yield PartStartEvent(index=0, part=ThinkingPart("thinking-so-far"))
            await asyncio.sleep(0.2)
            yield PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta="done"),
            )

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace = TraceRecorder(trace_path, "stream_stall_test")
            started = time.monotonic()
            model = SimpleNamespace(
                active_stream_request_index=1,
                active_stream_started=started,
                record_content=True,
            )
            asyncio.run(
                StreamTelemetryHandler(trace, model, heartbeat_seconds=0.05)(None, events())
            )
            rendered = trace_path.read_text(encoding="utf-8")

        stalled = json.loads(
            next(
                line
                for line in rendered.splitlines()
                if '"event_type":"model_stream_stalled"' in line
            )
        )["payload"]
        self.assertEqual(stalled["reasoning_content"], "thinking-so-far")
        completed = json.loads(
            next(
                line
                for line in rendered.splitlines()
                if '"event_type":"model_stream_telemetry_completed"' in line
            )
        )["payload"]
        self.assertEqual(completed["reasoning_content"], "thinking-so-fardone")

    def test_stream_telemetry_integration_via_agent_run(self) -> None:
        """真实 Agent 集成路径：event_stream_handler 必须写出遥测事件。"""

        async def stream_function(messages, info):
            yield "text-chunk-1"
            yield "text-chunk-2"

        def callback(messages, info) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart("done")],
                usage=RequestUsage(input_tokens=10, output_tokens=3),
            )

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace = TraceRecorder(trace_path, "stream_integration_test")
            model = TracingModel(
                FunctionModel(callback, stream_function=stream_function, model_name="probe"),
                trace,
                record_content=True,
            )
            agent = Agent(model, system_prompt="system")
            handler = StreamTelemetryHandler(trace, model)
            asyncio.run(agent.run("user", event_stream_handler=handler))
            rendered = trace_path.read_text(encoding="utf-8")

        self.assertIn("model_stream_telemetry_completed", rendered)
        completed = json.loads(
            next(
                line
                for line in rendered.splitlines()
                if '"event_type":"model_stream_telemetry_completed"' in line
            )
        )["payload"]
        self.assertEqual(completed["text_content"], "text-chunk-1text-chunk-2")
        self.assertTrue(completed["text_content_recorded"])

    def test_stream_telemetry_consumes_tool_node_streams(self) -> None:
        """handler 必须能排空工具调用节点的流（回归：曾因 continue 漏建任务而死锁）。"""

        async def stream_function(messages, info):
            returns = [
                part
                for message in messages
                if hasattr(message, "parts")
                for part in message.parts
                if isinstance(part, ToolReturnPart)
            ]
            if not returns:
                yield {0: DeltaToolCall(name="probe_tool", json_args="{}", tool_call_id="call_1")}
            else:
                yield "done"

        def callback(messages, info) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart("done")],
                usage=RequestUsage(input_tokens=5, output_tokens=3),
            )

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.jsonl"
            trace = TraceRecorder(trace_path, "stream_tool_node_test")
            model = TracingModel(
                FunctionModel(callback, stream_function=stream_function, model_name="probe"),
                trace,
                record_content=True,
            )
            agent = Agent(model, system_prompt="system")

            def probe_tool(ctx) -> str:
                return "tool-result"

            agent.tool(probe_tool)
            handler = StreamTelemetryHandler(trace, model)
            asyncio.run(agent.run("user", event_stream_handler=handler))
            rendered = trace_path.read_text(encoding="utf-8")

        # 两次模型请求各记一个遥测事件；工具节点流被排空但不产生遥测。
        self.assertEqual(rendered.count("model_stream_telemetry_completed"), 2)

    def test_mock_agent_completes_loop_and_records_bounded_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = InterpreterRunConfig(
                provider="mock",
                run_dir=run_dir,
                run_id="test_planning_run",
                system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
                trace_config=TraceConfig(max_event_bytes=8_192, max_string_chars=1_024),
                cost_rates=CostRates(
                    input_per_million=Decimal("1.25"),
                    output_per_million=Decimal("5.0"),
                    source="test_snapshot",
                ),
            )

            result = asyncio.run(InterpreterRunner(config).run(valid_planning_brief()))
            trace_path = run_dir / "planning_agent_tool_trace.jsonl"
            trace_lines = trace_path.read_text(encoding="utf-8").splitlines()
            trace = [json.loads(line) for line in trace_lines]
            scene_ir = json.loads((run_dir / "final_scene_ir.json").read_text(encoding="utf-8"))
            checkpoint = json.loads(
                (run_dir / "checkpoint_latest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.status, "success", result.error)
        self.assertEqual(result.terminal_type, "commit_request")
        self.assertGreater(result.usage["tokens"]["input_tokens"], 0)
        self.assertGreater(result.usage["context"]["max_request_input_tokens"], 0)
        self.assertGreaterEqual(result.usage["context"]["cache_hit_ratio"], 0.0)
        self.assertEqual(result.usage["pricing_snapshot"]["source"], "test_snapshot")
        self.assertGreater(Decimal(result.usage["estimated_cost"]["amount"]), 0)
        self.assertTrue(all(len(line.encode("utf-8")) <= 8_192 for line in trace_lines))
        self.assertTrue(any(item["event_type"] == "model_request_completed" for item in trace))
        self.assertTrue(any(item["event_type"] == "tool_call_completed" for item in trace))
        self.assertTrue(any(item["event_type"] == "commit_gate_completed" for item in trace))
        run_started = next(item for item in trace if item["event_type"] == "run_started")
        self.assertEqual(run_started["payload"]["toolkit_version"], "0.29")
        self.assertNotIn("孤独", "\n".join(trace_lines))
        # 普通运行保持精简日志：不记录对话内容，response 只记类型与规模。
        request_started = next(
            item for item in trace if item["event_type"] == "model_request_started"
        )
        self.assertNotIn("messages", request_started["payload"])
        completed = next(
            item for item in trace if item["event_type"] == "model_request_completed"
        )
        summaries = completed["payload"]["response_parts"]
        self.assertTrue(all("content_recorded" not in item for item in summaries))
        self.assertTrue(
            any(item["kind"] == "tool_call" for item in summaries)
        )
        self.assertEqual(scene_ir["schema_version"], "0.1")
        self.assertEqual(len(scene_ir["camera"]["state_track"]["samples"]), 144)
        self.assertEqual(checkpoint["candidate"]["revision"], result.final_revision)

    def test_mock_design_plans_all_reference_scenarios_in_three_tools(self) -> None:
        filenames = [
            "desert_ship_10s.json",
            "solar_system_10s.json",
            "roadside_pickup_12s.json",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in filenames:
                brief = json.loads(
                    (
                        ROOT / "examples" / "cinematic_briefs" / filename
                    ).read_text(encoding="utf-8")
                )
                run_dir = root / filename.removesuffix(".json")
                result = asyncio.run(
                    InterpreterRunner(
                        InterpreterRunConfig(
                            provider="mock",
                            run_dir=run_dir,
                            run_id=f"mock_design_{filename}",
                            system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
                        )
                    ).run(brief)
                )
                trace = [
                    json.loads(line)
                    for line in (run_dir / "planning_agent_tool_trace.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                tool_names = [
                    item["payload"]["tool_name"]
                    for item in trace
                    if item["event_type"] == "tool_call_completed"
                ]

                self.assertEqual(result.status, "success", (filename, result.error))
                self.assertEqual(
                    tool_names,
                    [
                        "submit_scene_skeleton",
                        "request_design_options",
                        "apply_design_option",
                    ],
                )

    def test_resume_uses_checkpoint_revision_with_fresh_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(
                        provider="mock",
                        run_dir=root / "first",
                        run_id="checkpoint_source",
                        system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
                    )
                ).run(valid_planning_brief())
            )
            second_dir = root / "second"
            second = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(
                        provider="mock",
                        run_dir=second_dir,
                        run_id="checkpoint_resume",
                        resume_from=root / "first" / "checkpoint_latest.json",
                        system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
                    )
                ).run(valid_planning_brief())
            )
            trace = (second_dir / "planning_agent_tool_trace.jsonl").read_text(
                encoding="utf-8"
            )

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success", second.error)
        self.assertEqual(second.final_revision, first.final_revision)
        self.assertIn("candidate_checkpoint_loaded", trace)

    def test_design_option_handles_explicit_environment_without_repair_retry(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["environment"] = {
            "value": "荒漠",
            "source_status": "explicit",
            "source_text": "荒漠里",
        }
        with tempfile.TemporaryDirectory() as directory:
            config = InterpreterRunConfig(
                provider="mock",
                run_dir=Path(directory),
                run_id="test_repair_run",
                system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
            )

            result = asyncio.run(InterpreterRunner(config).run(brief))

        self.assertEqual(result.status, "success")
        self.assertIsNone(result.error)

    def test_default_limits_distinguish_context_from_cumulative_usage(self) -> None:
        config = InterpreterRunConfig(run_dir=Path("unused"))

        self.assertIsNone(config.max_input_tokens)
        self.assertEqual(config.max_requests, 48)
        self.assertEqual(config.max_tool_calls, 80)
        self.assertEqual(config.max_context_tokens, 128_000)
        self.assertEqual(config.max_output_tokens, 200_000)
        self.assertEqual(config.max_seconds, 1_200.0)
        self.assertEqual(config.max_commit_attempts, 5)
        self.assertIsNone(config.max_total_tokens)
        self.assertEqual(
            _usage_limit_type("Exceeded the per_request_input_tokens_limit of 128000"),
            "per_request_input_tokens_limit",
        )

    def test_full_power_diagnostic_mock_keeps_mock_non_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = InterpreterRunConfig(
                provider="mock",
                run_dir=run_dir,
                run_id="mock_full_power",
                system_prompt_path=ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md",
                full_power_diagnostic=True,
            )

            result = asyncio.run(InterpreterRunner(config).run(valid_planning_brief()))
            trace_path = run_dir / "planning_agent_tool_trace.jsonl"
            trace_text = trace_path.read_text(encoding="utf-8")
            trace = [
                json.loads(line)
                for line in trace_text.splitlines()
            ]

        self.assertEqual(result.status, "success", result.error)
        self.assertIn('"event_type":"full_power_diagnostic_enabled"', trace_text)
        # 诊断模式记录完整对话内容与响应参数。
        request_started = next(
            item for item in trace if item["event_type"] == "model_request_started"
        )
        self.assertIn("messages", request_started["payload"])
        self.assertTrue(
            any(
                part["kind"] == "user-prompt" and part.get("content")
                for message in request_started["payload"]["messages"]
                for part in message.get("parts", [])
            )
        )
        completed = next(
            item for item in trace if item["event_type"] == "model_request_completed"
        )
        first_call = next(
            item
            for item in completed["payload"]["response_parts"]
            if item["kind"] == "tool-call"
        )
        self.assertIn("skeleton", json.dumps(first_call.get("args"), ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
