from __future__ import annotations

import unittest

from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.semantic_rules import apply_translation_rules, load_translation_rules
from tests.helpers import valid_model_output
from tests.helpers import ROOT


class ObjectiveProjectionTest(unittest.TestCase):
    def test_subjective_content_never_enters_agent_brief(self) -> None:
        brief = _valid_brief()
        brief["content"]["summary"] = "男人孤独地面对飞船"
        brief["content"]["mood"]["emotional_tones"] = [
            {
                "value": "孤独",
                "source_status": "explicit",
                "source_text": "感觉很孤独",
            }
        ]

        result = project_objective_brief(brief)
        payload = result.objective_brief.model_dump()

        self.assertNotIn("summary", payload)
        self.assertNotIn("mood", payload)
        self.assertNotIn("source_prompt", payload["source_metadata"])
        self.assertEqual(
            [item.path for item in result.ignored_subjective_fields],
            ["content.summary", "content.mood"],
        )

    def test_explicit_objective_requirements_keep_source_paths(self) -> None:
        brief = _valid_brief()
        brief["content"]["camera"]["movement"]["type"] = {
            "value": "缓慢推近",
            "source_status": "explicit",
            "source_text": "镜头慢慢推近",
        }

        result = project_objective_brief(brief)

        requirement = result.objective_brief.explicit_requirements[0]
        self.assertEqual(requirement.path, "content.camera.movement.type")
        self.assertEqual(requirement.value, "缓慢推近")

    def test_rejects_incomplete_brief(self) -> None:
        brief = _valid_brief()
        del brief["content"]["camera"]

        with self.assertRaisesRegex(ValueError, "缺少客观字段"):
            project_objective_brief(brief)

    def test_explicit_duration_is_indexed_as_objective_requirement(self) -> None:
        brief = _valid_brief()
        brief["content"]["timeline"]["duration_seconds"] = 6.0
        brief["content"]["timeline"]["duration_source_status"] = "explicit"

        result = project_objective_brief(brief)

        self.assertIn(
            "content.timeline.duration_seconds",
            [item.path for item in result.objective_brief.explicit_requirements],
        )

    def test_v02_passes_quantitative_objectives_but_strips_feeling_and_lighting(self) -> None:
        brief = _valid_brief()
        brief["content"]["mood"]["emotional_tones"] = [
            {
                "value": "孤独",
                "source_status": "explicit",
                "source_text": "感觉很孤独",
            }
        ]
        normalized, parameters = apply_translation_rules(
            brief["content"],
            load_translation_rules(ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"),
        )
        brief["schema_version"] = "0.2"
        brief["content"] = normalized
        brief["translation_parameters"] = parameters
        brief["provenance"]["translation_rules_sha256"] = "1" * 64

        result = project_objective_brief(brief)
        objective = result.objective_brief

        self.assertEqual(objective.schema_version, "0.2")
        self.assertEqual(objective.translation_parameters["camera"]["height_m"], 1.2)
        self.assertNotIn("lighting", objective.translation_parameters)
        self.assertNotIn("emotion_class", objective.translation_parameters)
        self.assertNotIn("feeling", objective.translation_parameters["input_slots"])

    def test_projection_repairs_legacy_camera_parameters_that_conflict_with_explicit_motion(self) -> None:
        brief = _valid_brief()
        brief["content"]["camera"]["movement"]["type"] = {
            "value": "缓慢推近",
            "source_status": "explicit",
            "source_text": "镜头慢慢推近",
        }
        normalized, parameters = apply_translation_rules(
            brief["content"],
            load_translation_rules(
                ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
            ),
        )
        parameters["camera"].update(
            movement="static",
            speed_mps=0.0,
            start_distance_m=15.0,
            end_distance_m=15.0,
            source_status="default",
        )
        brief["schema_version"] = "0.6"
        brief["content"] = normalized
        brief["translation_parameters"] = parameters
        brief["provenance"]["translation_rules_sha256"] = "1" * 64

        camera = project_objective_brief(brief).objective_brief.translation_parameters["camera"]

        self.assertEqual(camera["movement"], "push_in")
        self.assertLess(camera["end_distance_m"], camera["start_distance_m"])
        self.assertGreater(camera["speed_mps"], 0.0)
        self.assertEqual(camera["source_status"], "explicit")


def _valid_brief() -> dict:
    return {
        "schema_version": "0.1",
        "content": valid_model_output(),
        "provenance": {
            "source_prompt": "不应进入 Agent 1",
            "provider": "mock",
            "model": "mock-cinematic-brief-v0.1",
            "parser_prompt_version": "semantic-parser-v0.1",
            "rules_sha256": "0" * 64,
            "response_id": "mock-response-001",
        },
    }


if __name__ == "__main__":
    unittest.main()
