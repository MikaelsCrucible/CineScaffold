from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from cinescaffold.planning.runner import (
    InterpreterRunConfig,
    InterpreterRunner,
    _planning_model_settings,
    _usage_limit_type,
)
from cinescaffold.planning.models import create_planning_model
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.planning.trace import CostRates, TraceConfig
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
        self.assertEqual(run_started["payload"]["toolkit_version"], "0.9")
        self.assertNotIn("孤独", "\n".join(trace_lines))
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
        self.assertEqual(config.max_requests, 24)
        self.assertEqual(config.max_context_tokens, 32_000)
        self.assertIsNone(config.max_total_tokens)
        self.assertEqual(
            _usage_limit_type("Exceeded the per_request_input_tokens_limit of 32000"),
            "per_request_input_tokens_limit",
        )


if __name__ == "__main__":
    unittest.main()
