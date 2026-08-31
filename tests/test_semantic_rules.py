from __future__ import annotations

import unittest

from cinescaffold.schema import load_schema, validate_model_output
from cinescaffold.semantic_rules import apply_translation_rules, load_translation_rules
from tests.helpers import ROOT, valid_model_output


class SemanticRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_translation_rules(
            ROOT / "prompts/semantic_parser/translation_rules.json"
        )
        self.schema = load_schema(
            ROOT / "schemas/semantic_translation_parameters.schema.json"
        )

    def test_loneliness_maps_to_fixed_profile_without_preview_lighting(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man_01",
                "category": self._annotated("男人", "一个男人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
        ]
        content["subject_motion"] = [self._motion("man_01", "站立")]
        content["scene_design"]["environment"] = self._annotated("荒漠", "荒漠里")
        content["mood"]["emotional_tones"] = [
            self._statement("孤独", "感觉很孤独")
        ]
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )

        normalized, parameters = apply_translation_rules(content, self.rules)

        validate_model_output(parameters, self.schema)
        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(normalized["camera"]["camera_height"]["value"], "1.2 m")
        self.assertEqual(normalized["camera"]["camera_height"]["source_status"], "inferred")
        self.assertIn(
            "仅供最终视频生成",
            normalized["mood"]["lighting_intent"][-1]["value"],
        )
        self.assertEqual(parameters["emotion_class"]["class_id"], "E1")
        self.assertEqual(parameters["camera"]["height_m"], 1.2)
        self.assertEqual(parameters["composition"]["subject_frame_ratio"], [0.01, 0.05])
        self.assertEqual(parameters["scene"]["asset_key"], "desert")
        self.assertEqual(parameters["subjects"][0]["reference_height_m"], 1.75)
        self.assertEqual(parameters["subjects"][0]["facing_direction_world"], [0.0, -1.0, 0.0])
        self.assertEqual(parameters["lighting"]["application_scope"], "final_video_generation_only")
        self.assertFalse(parameters["lighting"]["applied_to_blender_preview"])

    def test_missing_slots_use_declared_defaults(self) -> None:
        content = valid_model_output()

        normalized, parameters = apply_translation_rules(content, self.rules)

        self.assertEqual(normalized["subjects"][0]["category"]["value"], "人")
        self.assertEqual(normalized["scene_design"]["environment"]["value"], "空白空间")
        self.assertEqual(normalized["timeline"]["duration_seconds"], 15.0)
        self.assertEqual(parameters["emotion_class"]["class_id"], "E6")
        self.assertEqual(parameters["motions"][0]["motion_type"], "static")
        self.assertEqual(parameters["motions"][0]["speed_range_mps"], [0.0, 0.0])
        default_fields = {item["field"] for item in normalized["uncertainties"]}
        self.assertIn("subjects", default_fields)
        self.assertIn("timeline.duration_seconds", default_fields)

    def test_explicit_camera_is_recorded_as_override(self) -> None:
        content = valid_model_output()
        content["camera"]["view_angle"] = self._annotated("俯拍", "使用俯拍")
        content["mood"]["emotional_tones"] = [
            self._statement("孤独", "感觉孤独")
        ]

        _, parameters = apply_translation_rules(content, self.rules)

        self.assertIn("camera.view_angle", parameters["explicit_override_paths"])
        normalized, _ = apply_translation_rules(content, self.rules)
        self.assertEqual(normalized["camera"]["view_angle"]["value"], "俯拍")

    def test_duration_upper_bound_resolves_to_its_maximum(self) -> None:
        content = valid_model_output()
        content["timeline"]["duration_range_seconds"] = {
            "minimum_seconds": 0.0,
            "maximum_seconds": 10.0,
        }
        content["timeline"]["duration_source_status"] = "explicit"

        normalized, _ = apply_translation_rules(content, self.rules)

        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(normalized["timeline"]["duration_source_status"], "inferred")
        uncertainty = next(
            item
            for item in normalized["uncertainties"]
            if item["field"] == "timeline.duration_seconds"
        )
        self.assertEqual(uncertainty["resolution"], "use_inference")
        self.assertEqual(uncertainty["selected_value"], "10.0")

    def test_target_motion_uses_relative_direction(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man",
                "category": self._annotated("男人", "一个男人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "ship",
                "category": self._annotated("飞船", "一艘飞船"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [self._motion("man", "走向飞船")]

        _, parameters = apply_translation_rules(content, self.rules)

        motion = parameters["motions"][0]
        self.assertEqual(motion["motion_type"], "walking")
        self.assertEqual(motion["target_id"], "ship")
        self.assertEqual(motion["direction_mode"], "toward_or_relative_to_target")
        self.assertIsNone(motion["direction_vector_world"])
        self.assertEqual(parameters["subjects"][1]["minimum_footprint_m"], [10.0, 10.0])
        self.assertEqual(parameters["subjects"][1]["default_scene_depth_ratio"], 0.5)

    @staticmethod
    def _annotated(value: str, source_text: str) -> dict:
        return {"value": value, "source_status": "explicit", "source_text": source_text}

    @staticmethod
    def _statement(value: str, source_text: str) -> dict:
        return {"value": value, "source_status": "explicit", "source_text": source_text}

    @staticmethod
    def _unknown() -> dict:
        return {"value": None, "source_status": "unknown", "source_text": None}

    @classmethod
    def _motion(cls, subject_id: str, action: str) -> dict:
        return {
            "subject_id": subject_id,
            "action": cls._annotated(action, action),
            "direction": cls._unknown(),
            "speed": cls._unknown(),
            "trajectory": cls._unknown(),
            "start_time_seconds": None,
            "end_time_seconds": None,
            "secondary_motion": [],
        }


if __name__ == "__main__":
    unittest.main()
