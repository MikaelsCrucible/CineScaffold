from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cinescaffold.cli import main
from tests.helpers import ROOT


class CliTest(unittest.TestCase):
    def test_mock_parse_writes_brief(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief.json"
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
                ]
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(result["provenance"]["provider"], "mock")


if __name__ == "__main__":
    unittest.main()
