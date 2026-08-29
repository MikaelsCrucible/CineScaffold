from __future__ import annotations

import unittest

from cinescaffold.planning.objective import project_objective_brief
from tests.helpers import valid_model_output


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
