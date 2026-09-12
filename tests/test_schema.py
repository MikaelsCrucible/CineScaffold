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


def _annotated(value: str) -> dict[str, str]:
    return {"value": value, "source_status": "explicit", "source_text": value}


def _unknown() -> dict[str, str | None]:
    return {"value": None, "source_status": "unknown", "source_text": None}


if __name__ == "__main__":
    unittest.main()
