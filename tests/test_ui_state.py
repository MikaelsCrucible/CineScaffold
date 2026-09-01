from __future__ import annotations

import json
import unittest

from cinescaffold.ui_state import (
    UiRunInputs,
    UiRunMetrics,
    event_message,
    event_progress,
    source_from_inputs,
)


class UiStateTest(unittest.TestCase):
    def test_builds_each_source_without_file_system_coupling(self) -> None:
        text = source_from_inputs(UiRunInputs(source_kind="text", natural_text="测试场景"))
        textual = source_from_inputs(
            UiRunInputs(source_kind="textual_six", textual_six="主体：测试")
        )
        brief = source_from_inputs(
            UiRunInputs(source_kind="brief", brief_json=json.dumps({"schema_version": "0.5"}))
        )
        scene_ir = source_from_inputs(
            UiRunInputs(source_kind="scene_ir", scene_ir_json=json.dumps({"schema_version": "0.1"}))
        )

        self.assertEqual(text.kind, "text")
        self.assertEqual(textual.kind, "textual_six")
        self.assertEqual(brief.payload["schema_version"], "0.5")
        self.assertEqual(scene_ir.payload["schema_version"], "0.1")

    def test_rejects_invalid_json_before_starting_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON 无效"):
            source_from_inputs(UiRunInputs(source_kind="brief", brief_json="{"))

    def test_progress_is_monotonic_and_bounded(self) -> None:
        progress = 0.0
        for event in (
            "pipeline_started",
            "pipeline_semantic_started",
            "pipeline_semantic_completed",
            "pipeline_planning_started",
            "model_request_completed",
            "tool_call_completed",
            "pipeline_execution_started",
            "render_started",
            "pipeline_finished",
        ):
            next_progress = event_progress(event, progress)
            self.assertGreaterEqual(next_progress, progress)
            self.assertLessEqual(next_progress, 1.0)
            progress = next_progress
        self.assertEqual(progress, 1.0)

    def test_metrics_keep_semantic_and_planning_costs_separate(self) -> None:
        summary = {
            "elapsed_seconds": 12.5,
            "stages": {
                "semantic": {
                    "usage": {
                        "tokens": {"input_tokens": 100, "output_tokens": 50},
                        "estimated_cost": {"amount": "0.01", "currency": "USD"},
                    }
                },
                "planning": {
                    "usage": {
                        "tokens": {
                            "input_tokens": 1000,
                            "output_tokens": 500,
                            "requests": 4,
                            "tool_calls": 6,
                        },
                        "estimated_cost": {"amount": "0.25", "currency": "USD"},
                    }
                },
            },
        }

        metrics = UiRunMetrics.from_summary(summary)

        self.assertEqual(metrics.semantic_tokens, 150)
        self.assertEqual(metrics.planning_tokens, 1500)
        self.assertEqual(metrics.requests, 4)
        self.assertEqual(metrics.tool_calls, 6)
        self.assertEqual(metrics.semantic_cost, "0.01 USD")
        self.assertEqual(metrics.planning_cost, "0.25 USD")

    def test_event_log_formats_model_usage(self) -> None:
        message = event_message(
            "model_request_completed",
            {
                "request_index": 2,
                "usage": {"input_tokens": 123, "output_tokens": 45},
            },
        )
        self.assertEqual(message, "模型响应 #2：输入 123 / 输出 45 tokens")


if __name__ == "__main__":
    unittest.main()
