from __future__ import annotations

import unittest

from cinescaffold.errors import SchemaValidationError
from cinescaffold.providers.mock import MockProvider
from cinescaffold.semantic import SemanticParserConfig, parse_cinematic_brief
from tests.helpers import ROOT, valid_model_output


class SemanticParserTest(unittest.TestCase):
    def test_mock_provider_builds_provenance_envelope(self) -> None:
        provider = MockProvider(valid_model_output())
        config = SemanticParserConfig(
            system_template_path=ROOT / "prompts/semantic_parser/system.md",
            rules_path=ROOT / "prompts/semantic_parser/rules.md",
            format_example_path=ROOT / "prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT / "schemas/cinematic_brief_model_output.schema.json",
        )
        result = parse_cinematic_brief("测试自然语言", provider, config)

        self.assertEqual(result["schema_version"], "0.1")
        self.assertEqual(result["provenance"]["provider"], "mock")
        self.assertEqual(result["provenance"]["source_prompt"], "测试自然语言")
        self.assertEqual(len(provider.calls), 1)

    def test_invalid_mock_response_is_rejected_locally(self) -> None:
        provider = MockProvider({"summary": "不完整"})
        config = SemanticParserConfig(
            system_template_path=ROOT / "prompts/semantic_parser/system.md",
            rules_path=ROOT / "prompts/semantic_parser/rules.md",
            format_example_path=ROOT / "prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT / "schemas/cinematic_brief_model_output.schema.json",
        )
        with self.assertRaises(SchemaValidationError):
            parse_cinematic_brief("测试自然语言", provider, config)


if __name__ == "__main__":
    unittest.main()
