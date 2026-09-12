from __future__ import annotations

import unittest

from cinescaffold.errors import SchemaValidationError
from cinescaffold.providers.mock import MockProvider
from cinescaffold.semantic import SemanticParserConfig, parse_cinematic_brief, parse_semantic_input
from cinescaffold.textual_six import SECTION_LABELS
from tests.test_textual_six import TEXTUAL_SIX
from tests.helpers import ROOT, valid_model_output


class SemanticParserTest(unittest.TestCase):
    def test_mock_provider_builds_provenance_envelope(self) -> None:
        provider = MockProvider(valid_model_output())
        config = SemanticParserConfig(
            system_template_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
            rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
            format_example_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json",
            translation_rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json",
            translation_parameters_schema_path=ROOT / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json",
        )
        result = parse_cinematic_brief("测试自然语言", provider, config)

        self.assertEqual(result["schema_version"], "0.6")
        self.assertEqual(
            result["content"]["camera"]["view_relation_to_motion"],
            {
                "value": "unspecified",
                "source_status": "default",
                "source_text": None,
            },
        )
        self.assertEqual(result["provenance"]["provider"], "mock")
        self.assertEqual(result["provenance"]["source_prompt"], "测试自然语言")
        self.assertEqual(result["provenance"]["source_kind"], "natural_text")
        self.assertRegex(result["provenance"]["textual_six_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertIsNone(result["provenance"]["provider_usage"])
        self.assertEqual(result["translation_parameters"]["emotion_class"]["class_id"], "E6")
        self.assertFalse(
            result["translation_parameters"]["lighting"]["applied_to_blender_preview"]
        )
        self.assertEqual(len(provider.calls), 1)

    def test_textual_six_input_is_preserved_and_labeled(self) -> None:
        provider = MockProvider(valid_model_output())
        config = SemanticParserConfig(
            system_template_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
            rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
            format_example_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json",
            translation_rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json",
            translation_parameters_schema_path=ROOT / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json",
        )

        result = parse_semantic_input(TEXTUAL_SIX, provider, config, source_kind="textual_six")

        self.assertEqual(result.brief["provenance"]["source_kind"], "textual_six")
        self.assertEqual(result.textual_six.render(), TEXTUAL_SIX)
        self.assertTrue(all(label in provider.calls[0]["user_prompt"] for label in SECTION_LABELS))

    def test_invalid_mock_response_is_rejected_locally(self) -> None:
        provider = MockProvider({"summary": "不完整"})
        config = SemanticParserConfig(
            system_template_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
            rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
            format_example_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json",
            translation_rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json",
            translation_parameters_schema_path=ROOT / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json",
        )
        with self.assertRaises(SchemaValidationError):
            parse_cinematic_brief("测试自然语言", provider, config)


if __name__ == "__main__":
    unittest.main()
