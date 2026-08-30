from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunner
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

        self.assertEqual(result.status, "success", result.error)
        self.assertEqual(result.terminal_type, "commit_request")
        self.assertGreater(result.usage["tokens"]["input_tokens"], 0)
        self.assertEqual(result.usage["pricing_snapshot"]["source"], "test_snapshot")
        self.assertGreater(Decimal(result.usage["estimated_cost"]["amount"]), 0)
        self.assertTrue(all(len(line.encode("utf-8")) <= 8_192 for line in trace_lines))
        self.assertTrue(any(item["event_type"] == "model_request_completed" for item in trace))
        self.assertTrue(any(item["event_type"] == "tool_call_completed" for item in trace))
        self.assertTrue(any(item["event_type"] == "commit_gate_completed" for item in trace))
        self.assertNotIn("孤独", "\n".join(trace_lines))
        self.assertEqual(scene_ir["schema_version"], "0.1")
        self.assertEqual(len(scene_ir["camera"]["state_track"]["samples"]), 144)

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


if __name__ == "__main__":
    unittest.main()
