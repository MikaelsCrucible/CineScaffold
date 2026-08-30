from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from cinescaffold.planning.duration import duration_request, freeze_duration
from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunner
from tests.helpers import valid_planning_brief


class PlanningDurationTest(unittest.TestCase):
    def test_exact_duration_is_frozen_without_duration_model_request(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["timeline"].update(
            {
                "duration_seconds": 5.0,
                "duration_range_seconds": None,
                "duration_source_status": "explicit",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(provider="mock", run_dir=run_dir)
                ).run(brief)
            )
            scene_ir = json.loads((run_dir / "final_scene_ir.json").read_text(encoding="utf-8"))
            trace = [
                json.loads(line)
                for line in (run_dir / "planning_agent_tool_trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(result.status, "success")
        self.assertEqual(scene_ir["timeline"]["duration_seconds"], 5.0)
        self.assertEqual(scene_ir["timeline"]["duration_resolution"]["resolution_method"], "user_exact")
        first_request = next(item for item in trace if item["event_type"] == "model_request_started")
        resolved = next(item for item in trace if item["event_type"] == "duration_resolved")
        self.assertLess(resolved["sequence"], first_request["sequence"])

    def test_range_duration_uses_agent_choice_and_stays_in_range(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["timeline"].update(
            {
                "duration_seconds": None,
                "duration_range_seconds": {
                    "minimum_seconds": 7.0,
                    "maximum_seconds": 9.0,
                },
                "duration_source_status": "explicit",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(provider="mock", run_dir=run_dir)
                ).run(brief)
            )
            scene_ir = json.loads((run_dir / "final_scene_ir.json").read_text(encoding="utf-8"))

        self.assertEqual(result.status, "success")
        self.assertEqual(scene_ir["timeline"]["duration_seconds"], 8.0)
        self.assertEqual(
            scene_ir["timeline"]["duration_resolution"]["request"],
            {"mode": "range", "minimum_seconds": 7.0, "maximum_seconds": 9.0},
        )

    def test_conflicting_exact_and_range_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能同时"):
            duration_request(
                {
                    "duration_seconds": 6.0,
                    "duration_range_seconds": {
                        "minimum_seconds": 5.0,
                        "maximum_seconds": 8.0,
                    },
                }
            )

    def test_narrow_range_must_contain_a_frame_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "窄于一个"):
            freeze_duration(
                request_mode="range",
                proposed_seconds=1.01,
                reason="测试",
                fps_numerator=24,
                fps_denominator=1,
                minimum_seconds=1.01,
                maximum_seconds=1.02,
            )


if __name__ == "__main__":
    unittest.main()
