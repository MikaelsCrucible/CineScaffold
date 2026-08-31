from __future__ import annotations

import io
import unittest

from cinescaffold.cli_ui import TerminalReporter


class TerminalReporterTest(unittest.TestCase):
    def test_model_request_uses_neutral_label(self) -> None:
        stream = io.StringIO()
        reporter = TerminalReporter(color=False, stream=stream)

        reporter.event("model_request_started", {"request_index": 2, "message_count": 3})

        rendered = stream.getvalue()
        self.assertIn("模型请求", rendered)
        self.assertNotIn("模型思考", rendered)

    def test_non_tool_text_compaction_is_visible(self) -> None:
        stream = io.StringIO()
        reporter = TerminalReporter(color=False, stream=stream)

        reporter.event(
            "model_non_tool_text_compacted",
            {"omitted_chars": 29026, "authoritative_revision": 6},
        )

        rendered = stream.getvalue()
        self.assertIn("异常正文", rendered)
        self.assertIn("29026 字符", rendered)
        self.assertIn("revision 6", rendered)

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
