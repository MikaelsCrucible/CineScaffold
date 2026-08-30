from __future__ import annotations

import io
import unittest

from cinescaffold.cli_ui import TerminalReporter


class TerminalReporterTest(unittest.TestCase):
    def test_model_event_renders_compact_token_statistics(self) -> None:
        stream = io.StringIO()
        reporter = TerminalReporter(color=False, stream=stream)

        reporter.event(
            "model_request_completed",
            {
                "request_index": 3,
                "usage": {"input_tokens": 1234, "output_tokens": 56},
                "duration_ms": 1250,
            },
        )

        rendered = stream.getvalue()
        self.assertIn("模型响应", rendered)
        self.assertIn("输入 1,234", rendered)
        self.assertIn("输出 56 tokens", rendered)
        self.assertIn("1.25s", rendered)

    def test_quiet_reporter_emits_nothing(self) -> None:
        stream = io.StringIO()
        reporter = TerminalReporter(quiet=True, color=False, stream=stream)

        reporter.stage("测试", "不应显示")
        reporter.event("run_started", {"provider": "mock", "model": "mock"})

        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
