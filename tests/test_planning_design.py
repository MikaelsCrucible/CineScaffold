from __future__ import annotations

import unittest

from pydantic import ValidationError

from cinescaffold.planning.design import (
    SceneSkeleton,
    skeleton_hash,
    task_capability_slice,
    validate_scene_skeleton,
)
from cinescaffold.planning.domain import PlanningProfile
from cinescaffold.planning.objective import project_objective_brief
from tests.helpers import valid_planning_brief


class PlanningDesignTest(unittest.TestCase):
    def test_scene_skeleton_keeps_only_symbolic_scene_choices(self) -> None:
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())

        self.assertEqual(skeleton.entities[1].proxy_family, "human_capsule")
        self.assertEqual(skeleton.relations[1].kind, "camera_depth_order")
        self.assertNotIn("translation_m", skeleton.model_dump(mode="json"))
        self.assertTrue(skeleton_hash(skeleton).startswith("sha256:"))

    def test_scene_skeleton_rejects_unknown_references(self) -> None:
        value = _desert_skeleton()
        value["relations"][1]["subject_id"] = "missing_ship"

        with self.assertRaises(ValidationError):
            SceneSkeleton.model_validate(value)

    def test_scene_skeleton_must_include_every_brief_subject(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        skeleton = skeleton.model_copy(
            update={
                "entities": [
                    item for item in skeleton.entities if item.entity_id != "ship_01"
                ],
                "relations": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "ship_01"):
            validate_scene_skeleton(objective, skeleton)

    def test_task_capabilities_are_filtered_by_skeleton(self) -> None:
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        result = task_capability_slice(skeleton, PlanningProfile())

        self.assertEqual(result["required_path_families"], [])
        self.assertIn("camera_depth_order", result["required_relation_kinds"])
        self.assertNotIn("constraint_parameter_schemas", result)
        self.assertEqual(result["next_tool"], "apply_design_option")


def _desert_skeleton() -> dict:
    return {
        "entities": [
            {
                "entity_id": "ground",
                "semantic_type": "desert_ground",
                "role": "environment",
                "proxy_family": "ground_plane",
                "scale_intent": "large",
                "source_refs": [],
            },
            {
                "entity_id": "man_01",
                "semantic_type": "human",
                "role": "protagonist",
                "proxy_family": "human_capsule",
                "scale_intent": "human",
                "source_refs": [],
            },
            {
                "entity_id": "ship_01",
                "semantic_type": "spaceship",
                "role": "background_element",
                "proxy_family": "vehicle_box",
                "scale_intent": "huge",
                "source_refs": [],
            },
        ],
        "relations": [
            {
                "relation_id": "man_ground",
                "kind": "ground_support",
                "subject_id": "man_01",
                "reference_id": "ground",
                "source_status": "inferred",
                "source_ref": "translation_parameters.subjects[0].ground_contact_position_m",
            },
            {
                "relation_id": "ship_behind_man",
                "kind": "camera_depth_order",
                "subject_id": "ship_01",
                "reference_id": "man_01",
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[0]",
            },
        ],
        "motion_phases": [
            {
                "phase_id": "man_hold",
                "subject_id": "man_01",
                "kind": "hold",
                "timeline_event_id": None,
                "target_id": None,
                "carrier_id": None,
                "direction_mode": "none",
                "path_family": "stationary",
                "source_status": "inferred",
                "source_ref": "content.subject_motion[0].action",
            }
        ],
        "camera_intent": {
            "movement": "push_in",
            "focus_target_id": "man_01",
            "view_relation_to_motion": "unspecified",
            "source_status": "explicit",
            "source_ref": "content.camera.movement.type",
        },
    }


if __name__ == "__main__":
    unittest.main()
