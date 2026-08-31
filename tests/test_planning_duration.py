from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from cinescaffold.planning.duration import freeze_brief_duration
from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunner
from tests.helpers import valid_planning_brief


class PlanningDurationTest(unittest.TestCase):
    def test_brief_duration_is_frozen_before_first_model_request(self) -> None:
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
        self.assertEqual(
            scene_ir["timeline"]["duration_resolution"]["resolution_method"],
            "brief_explicit",
        )
        first_request = next(item for item in trace if item["event_type"] == "model_request_started")
        frozen = next(
            item for item in trace if item["event_type"] == "duration_frozen_from_brief"
        )
        self.assertLess(frozen["sequence"], first_request["sequence"])

    def test_missing_brief_duration_fails_without_model_request(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["timeline"].update(
            {"duration_seconds": None, "duration_source_status": "unknown"}
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = asyncio.run(
                InterpreterRunner(
                    InterpreterRunConfig(provider="mock", run_dir=run_dir)
                ).run(brief)
            )
            trace = (run_dir / "planning_agent_tool_trace.jsonl").read_text(encoding="utf-8")

        self.assertEqual(result.status, "failed")
        self.assertIn("自然语言解析阶段", result.error["message"])
        self.assertNotIn('"event_type":"model_request_started"', trace)

    def test_resolved_duration_must_stay_inside_source_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须位于"):
            freeze_brief_duration(
                {
                    "duration_seconds": 10.0,
                    "duration_source_status": "inferred",
                    "duration_range_seconds": {
                        "minimum_seconds": 5.0,
                        "maximum_seconds": 8.0,
                    },
                },
                fps_numerator=24,
                fps_denominator=1,
            )

    def test_upper_bound_only_range_may_start_at_zero(self) -> None:
        resolution = freeze_brief_duration(
            {
                "duration_seconds": 10.0,
                "duration_range_seconds": {
                    "minimum_seconds": 0.0,
                    "maximum_seconds": 10.0,
                },
                "duration_source_status": "inferred",
            },
            fps_numerator=24,
            fps_denominator=1,
        )

        self.assertEqual(resolution.resolved_duration_seconds, 10.0)


if __name__ == "__main__":
    unittest.main()
