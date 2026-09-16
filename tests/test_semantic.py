from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any

from cinescaffold.errors import SemanticContractError
from cinescaffold.providers.base import ProviderResponse
from cinescaffold.providers.mock import MockProvider
from cinescaffold.semantic import (
    SemanticParserConfig,
    parse_cinematic_brief,
    parse_semantic_input,
)
from cinescaffold.semantic_contracts import semantic_ai_contract_ids
from cinescaffold.schema import load_schema
from cinescaffold.textual_six import SECTION_LABELS
from tests.test_textual_six import TEXTUAL_SIX
from tests.helpers import ROOT, valid_model_output


class _SequenceProvider:
    name = "sequence"
    model = "sequence-model"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(self, system_prompt, user_prompt, schema):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "schema": schema,
            }
        )
        index = len(self.calls) - 1
        return ProviderResponse(
            content=deepcopy(self.responses[index]),
            response_id=f"sequence-{index + 1}",
            raw_metadata={
                "usage": {
                    "input_tokens": 100 + index,
                    "output_tokens": 20 + index,
                }
            },
        )


class SemanticParserTest(unittest.TestCase):
    def _config(self) -> SemanticParserConfig:
        return SemanticParserConfig(
            system_template_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
            rules_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
            revision_template_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/revision.md",
            review_rules_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/review_rules.json",
            format_example_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
            model_output_schema_path=ROOT
            / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json",
            translation_rules_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json",
            translation_parameters_schema_path=ROOT
            / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json",
        )

    def test_mock_provider_builds_provenance_envelope(self) -> None:
        provider = MockProvider(valid_model_output())
        config = self._config()
        result = parse_cinematic_brief("测试自然语言", provider, config)

        self.assertEqual(result["schema_version"], "0.8")
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
        review = result["provenance"]["semantic_review"]
        self.assertEqual(review["catalog_version"], "semantic-review-rules-v0.2")
        self.assertEqual(
            review["selection_policy"],
            "attention_only_no_semantic_inference",
        )
        self.assertIn("SOURCE.COVERAGE", review["selected_rule_ids"])
        self.assertRegex(review["rules_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(result["translation_parameters"]["emotion_class"]["class_id"], "E6")
        self.assertFalse(
            result["translation_parameters"]["lighting"]["applied_to_blender_preview"]
        )
        self.assertEqual(len(provider.calls), 2)
        self.assertIn(
            "independent_semantic_review_required",
            provider.calls[1]["user_prompt"],
        )

    def test_textual_six_input_is_preserved_and_labeled(self) -> None:
        provider = MockProvider(valid_model_output())
        config = self._config()

        result = parse_semantic_input(
            TEXTUAL_SIX,
            provider,
            config,
            source_kind="textual_six",
        )

        self.assertEqual(result.brief["provenance"]["source_kind"], "textual_six")
        self.assertEqual(result.textual_six.render(), TEXTUAL_SIX)
        self.assertTrue(all(label in provider.calls[0]["user_prompt"] for label in SECTION_LABELS))

    def test_invalid_mock_response_is_rejected_locally(self) -> None:
        provider = MockProvider({"summary": "不完整"})
        with self.assertRaises(SemanticContractError):
            parse_cinematic_brief("测试自然语言", provider, self._config())
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("schema_validation_failed", provider.calls[1]["user_prompt"])

    def test_invalid_draft_can_be_replaced_by_one_revision(self) -> None:
        provider = _SequenceProvider([{"summary": "不完整"}, valid_model_output()])

        result = parse_cinematic_brief("测试自然语言", provider, self._config())

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result["provenance"]["response_id"], "sequence-2")
        usage = result["provenance"]["provider_usage"]
        self.assertEqual(usage["requests"], 2)
        self.assertEqual(usage["input_tokens"], 201)
        self.assertEqual(usage["output_tokens"], 41)

    def test_semantic_contract_failure_is_sent_to_revision(self) -> None:
        inconsistent = valid_model_output()
        inconsistent["scene_dynamics"]["mode"] = "dynamic"
        provider = _SequenceProvider([inconsistent, valid_model_output()])

        parse_cinematic_brief("静止场景", provider, self._config())

        self.assertEqual(len(provider.calls), 2)
        self.assertIn("semantic_contract_failed", provider.calls[1]["user_prompt"])
        self.assertIn(
            "semantic_scene_dynamics_contract_failed",
            provider.calls[1]["user_prompt"],
        )
        self.assertIn("scene_dynamics.mode", provider.calls[1]["user_prompt"])
        self.assertIn('"kind": "confirmed_error"', provider.calls[1]["user_prompt"])

    def test_revision_receives_text_recalled_review_without_story_mapping(self) -> None:
        provider = _SequenceProvider([valid_model_output(), valid_model_output()])

        result = parse_cinematic_brief(
            "男生和女生擦肩而过。",
            provider,
            self._config(),
        )

        revision_prompt = provider.calls[1]["user_prompt"]
        self.assertIn('"rule_id": "MULTI_ENTITY.SHARED_FACTS"', revision_prompt)
        self.assertIn('"rule_id": "TIMELINE.LOCAL_EVENT_SCOPE"', revision_prompt)
        self.assertIn('"kind": "review_risk"', revision_prompt)
        self.assertIn(
            "MULTI_ENTITY.SHARED_FACTS",
            result["provenance"]["semantic_review"]["selected_rule_ids"],
        )

    def test_revision_does_not_add_story_specific_motion_semantics(self) -> None:
        prompt_text = "\n".join(
            [
                self._config().system_template_path.read_text(encoding="utf-8"),
                self._config().rules_path.read_text(encoding="utf-8"),
                self._config().revision_template_path.read_text(encoding="utf-8"),
            ]
        )

        for forbidden in ("crossing", "pass_by", "擦肩而过"):
            self.assertNotIn(forbidden, prompt_text)

    def test_cross_field_contracts_are_visible_to_both_passes_and_schema(self) -> None:
        provider = _SequenceProvider([valid_model_output(), valid_model_output()])

        parse_cinematic_brief("测试自然语言", provider, self._config())

        schema_text = str(load_schema(self._config().model_output_schema_path))
        for contract_id in semantic_ai_contract_ids():
            with self.subTest(contract_id=contract_id):
                self.assertIn(contract_id, provider.calls[0]["system_prompt"])
                self.assertIn(contract_id, provider.calls[1]["system_prompt"])
                self.assertIn(contract_id, schema_text)


if __name__ == "__main__":
    unittest.main()
