from __future__ import annotations

import unittest

from cinescaffold.errors import SchemaValidationError
from cinescaffold.schema import load_schema, validate_model_output
from tests.helpers import ROOT, valid_model_output


class SchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_schema(ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json")

    def test_valid_output_passes(self) -> None:
        validate_model_output(valid_model_output(), self.schema)

    def test_missing_dimension_fails(self) -> None:
        value = valid_model_output()
        del value["camera"]
        with self.assertRaisesRegex(SchemaValidationError, "camera 缺失"):
            validate_model_output(value, self.schema)

    def test_extra_field_fails(self) -> None:
        value = valid_model_output()
        value["unexpected"] = True
        with self.assertRaisesRegex(SchemaValidationError, "额外字段"):
            validate_model_output(value, self.schema)

    def test_fine_grained_narrative_motion_kind_fails(self) -> None:
        value = valid_model_output()
        value["subject_motion"] = [
            {
                "motion_id": "subject_depart",
                "subject_id": "subject",
                "action": _annotated("离开"),
                "motion_semantics": {
                    "action_kind": "depart",
                    "motion_type": "moving",
                    "motion_mode": "self_propelled",
                    "direction_mode": "none",
                    "target_id": None,
                    "carrier_id": None,
                    "path_type": "linear",
                    "local_components": [],
                    "timeline_event_id": None,
                    "narrative_required": True,
                    "postconditions": {
                        "contained_by_id": None,
                        "external_visibility": "unchanged",
                    },
                    "source_status": "explicit",
                    "source_text": "离开",
                },
                "direction": _unknown(),
                "speed": _unknown(),
                "trajectory": _unknown(),
                "start_time_seconds": 0.0,
                "end_time_seconds": 1.0,
                "secondary_motion": [],
            }
        ]

        with self.assertRaises(SchemaValidationError):
            validate_model_output(value, self.schema)

    def test_null_timeline_event_range_fails(self) -> None:
        value = valid_model_output()
        value["timeline"]["events"] = [
            {
                "id": "event_01",
                "description": "事件",
                "start_time_seconds": None,
                "end_time_seconds": None,
                "reference_ids": [],
                "source_status": "inferred",
                "source_text": "事件",
            }
        ]

        with self.assertRaises(SchemaValidationError):
            validate_model_output(value, self.schema)

    def test_redundant_relationship_types_are_not_part_of_semantic_contract(
        self,
    ) -> None:
        for relation_type in ("ground_support", "orbit_around", "carried_by"):
            with self.subTest(relation_type=relation_type):
                value = valid_model_output()
                value["scene_design"]["relationships"] = [
                    {
                        "type": relation_type,
                        "subject_id": "subject",
                        "reference_id": "reference",
                        "source_status": "inferred",
                        "source_text": None,
                        "timeline_event_id": None,
                        "temporal_mode": "throughout",
                    }
                ]

                with self.assertRaises(SchemaValidationError):
                    validate_model_output(value, self.schema)

    def test_semantic_contract_rejects_planning_only_source_status(self) -> None:
        value = valid_model_output()
        value["summary"] = "测试"
        value["camera"]["movement"]["type"]["source_status"] = "agent_selected"

        with self.assertRaises(SchemaValidationError):
            validate_model_output(value, self.schema)

    def test_semantic_relationship_rejects_free_text_strength(self) -> None:
        value = valid_model_output()
        value["scene_design"]["relationships"] = [
            {
                "type": "proximity",
                "subject_id": "a",
                "reference_id": "b",
                "strength": "非常近",
                "source_status": "explicit",
                "source_text": "二者非常近",
                "timeline_event_id": None,
                "temporal_mode": "throughout",
            }
        ]

        with self.assertRaises(SchemaValidationError):
            validate_model_output(value, self.schema)

    def test_non_executable_camera_fields_are_closed_in_schema(self) -> None:
        for field in ("direction", "trajectory", "easing"):
            with self.subTest(field=field):
                value = valid_model_output()
                value["camera"]["movement"][field] = _annotated("自由文本")

                with self.assertRaises(SchemaValidationError):
                    validate_model_output(value, self.schema)

    def test_executable_composition_fields_reject_synonyms(self) -> None:
        value = valid_model_output()
        value["composition"]["screen_placements"] = [
            {
                "subject_id": "subject",
                "horizontal": _annotated("画面左边"),
                "vertical": _annotated("center"),
            }
        ]

        with self.assertRaises(SchemaValidationError):
            validate_model_output(value, self.schema)


def _annotated(value: str) -> dict[str, str]:
    return {"value": value, "source_status": "explicit", "source_text": value}


def _unknown() -> dict[str, str | None]:
    return {"value": None, "source_status": "unknown", "source_text": None}


if __name__ == "__main__":
    unittest.main()
