from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
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
    _requires_complete_thinking_history,
    _usage_limit_type,
)
from cinescaffold.planning.trace import (
    CostRates,
    StreamTelemetryHandler,
    TraceConfig,
    TraceRecorder,
    TracingModel,
)
from tests.helpers import ROOT, valid_planning_brief


class InterpreterRunnerTest(unittest.TestCase):
    def test_real_provider_models_build_without_network_call(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief

        openai_model = create_planning_model(
            "openai", "gpt-test", objective, api_key="test-key"
        )
        deepseek_model = create_planning_model(
            "deepseek", "deepseek-chat", objective, api_key="test-key"
        )

        self.assertEqual(openai_model.provider.name, "openai")
        self.assertEqual(deepseek_model.provider.name, "deepseek")

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
                system_prompt_path=ROOT / "prompts/scene_planner/system.md",
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
        self.assertEqual(run_started["payload"]["toolkit_version"], "0.14")
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

    def test_resume_uses_checkpoint_revision_with_fresh_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(
                        provider="mock",
                        run_dir=root / "first",
                        run_id="checkpoint_source",
                        system_prompt_path=ROOT / "prompts/scene_planner/system.md",
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
                        system_prompt_path=ROOT / "prompts/scene_planner/system.md",
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

    def test_commit_repair_attempts_use_distinct_agent_run_ids(self) -> None:
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
                system_prompt_path=ROOT / "prompts/scene_planner/system.md",
            )

            result = asyncio.run(InterpreterRunner(config).run(brief))

        self.assertEqual(result.status, "commit_rejected")
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
                system_prompt_path=ROOT / "prompts/scene_planner/system.md",
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
        self.assertIn("sections", json.dumps(first_call.get("args"), ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
