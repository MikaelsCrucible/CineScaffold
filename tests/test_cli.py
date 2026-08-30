from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from cinescaffold.cli import main
from tests.helpers import ROOT, valid_planning_brief


class CliTest(unittest.TestCase):
    def test_mock_parse_writes_brief(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "parse",
                        "--provider",
                        "mock",
                        "--text",
                        "测试描述",
                        "--system-template",
                        str(ROOT / "prompts/semantic_parser/system.md"),
                        "--rules",
                        str(ROOT / "prompts/semantic_parser/rules.md"),
                        "--format-example",
                        str(ROOT / "prompts/semantic_parser/format_example.json"),
                        "--schema",
                        str(ROOT / "schemas/cinematic_brief_model_output.schema.json"),
                        "--output",
                        str(output),
                        "--quiet",
                    ]
                )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(result["provenance"]["provider"], "mock")

    def test_mock_plan_writes_scene_ir_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            output_dir = root / "run"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(output_dir),
                        "--system-prompt",
                        str(ROOT / "prompts/scene_planner/system.md"),
                        "--quiet",
                    ]
                )
            summary = json.loads(
                (output_dir / "planning_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "success")
        self.assertGreater(summary["usage"]["tokens"]["input_tokens"], 0)

    def test_mock_plan_default_output_is_human_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(root / "run"),
                        "--system-prompt",
                        str(ROOT / "prompts/scene_planner/system.md"),
                        "--no-color",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertIn("CineScaffold 规划结果", stdout.getvalue())
        self.assertIn("Tokens", stdout.getvalue())
        self.assertIn("下一步", stdout.getvalue())
        self.assertIn("Agent 工具", stderr.getvalue())
        self.assertIn("Commit Gate", stderr.getvalue())

    def test_mock_plan_json_quiet_is_clean_machine_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(root / "run"),
                        "--system-prompt",
                        str(ROOT / "prompts/scene_planner/system.md"),
                        "--json",
                        "--quiet",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "success")
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
