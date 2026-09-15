from __future__ import annotations

import unittest

from cinescaffold.planning.models import build_deterministic_scene_skeleton
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.relationships import (
    classify_relationship,
    normalize_scene_relationships,
)
from tests.helpers import valid_planning_brief


class RelationshipBoundaryTest(unittest.TestCase):
    def test_duplicate_canonical_far_relations_collapse(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "far_from",
                        "subject_id": "ship_01",
                        "reference_id": "man_01",
                        "source_status": "inferred",
                        "source_text": None,
                        "timeline_event_id": None,
                        "temporal_mode": "throughout",
                    },
                    {
                        "type": "far_from",
                        "subject_id": "ship_01",
                        "reference_id": "man_01",
                        "source_status": "explicit",
                        "source_text": "远处有飞船",
                        "timeline_event_id": None,
                        "temporal_mode": "throughout",
                    },
                ],
            }
        }

        normalize_scene_relationships(content)

        self.assertEqual(
            content["scene_design"]["relationships"],
            [
                {
                    "type": "far_from",
                    "subject_id": "ship_01",
                    "reference_id": "man_01",
                    "source_status": "explicit",
                    "source_text": "远处有飞船",
                    "timeline_event_id": None,
                    "temporal_mode": "throughout",
                }
            ],
        )

    def test_noncanonical_distance_is_not_reinterpreted_from_strength(self) -> None:
        self.assertIsNone(
            classify_relationship({"type": "distance", "strength": "near"})
        )

    def test_canonical_scale_dominance_preserves_declared_orientation(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "scale_dominance",
                        "subject_id": "moon",
                        "reference_id": "earth",
                        "source_status": "explicit",
                        "source_text": "月亮在画面中占主导",
                        "timeline_event_id": None,
                        "temporal_mode": "throughout",
                    }
                ]
            }
        }

        normalize_scene_relationships(content)

        relation = content["scene_design"]["relationships"][0]
        self.assertEqual(relation["type"], "scale_dominance")
        self.assertEqual(relation["subject_id"], "moon")
        self.assertEqual(relation["reference_id"], "earth")

    def test_malformed_relationship_item_is_not_silently_deleted(self) -> None:
        content = {"scene_design": {"relationships": ["not-an-object"]}}

        normalize_scene_relationships(content)

        self.assertEqual(content["scene_design"]["relationships"], ["not-an-object"])

    def test_unknown_relationship_is_rejected_at_current_brief_boundary(
        self,
    ) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["relationships"] = [
            {
                "type": "visually_echoes",
                "subject_id": "ship_01",
                "reference_id": "man_01",
                "source_status": "inferred",
                "source_text": None,
                "timeline_event_id": None,
                "temporal_mode": "throughout",
            }
        ]
        with self.assertRaisesRegex(ValueError, "不支持的类型"):
            project_objective_brief(brief)

    def test_direct_far_and_near_conflict_is_rejected_before_planning(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "far_from",
                        "subject_id": "a",
                        "reference_id": "b",
                    },
                    {
                        "type": "proximity",
                        "subject_id": "b",
                        "reference_id": "a",
                    },
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "同时被标记为远离和靠近"):
            normalize_scene_relationships(content)

    def test_reversed_far_relations_are_not_silently_deduplicated(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "far_from",
                        "subject_id": "a",
                        "reference_id": "b",
                    },
                    {
                        "type": "far_from",
                        "subject_id": "b",
                        "reference_id": "a",
                    },
                ]
            }
        }

        with self.assertRaisesRegex(ValueError, "相反的远景方向"):
            normalize_scene_relationships(content)

    def test_absolute_visual_scale_does_not_create_an_arbitrary_pairwise_relation(
        self,
    ) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["relationships"] = []
        brief["content"]["composition"]["visual_scales"] = [
            {
                "subject_id": "ship_01",
                "scale": {
                    "value": "10%-20%",
                    "source_status": "explicit",
                    "source_text": "飞船占画幅10%-20%",
                },
            }
        ]
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertFalse(
            any(
                relation["kind"] == "scale_dominance"
                for relation in skeleton["relations"]
            )
        )

    def test_space_environment_does_not_invent_a_ground_plane(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["environment"] = {
            "value": "太空",
            "source_status": "explicit",
            "source_text": "在太空中",
        }
        objective = project_objective_brief(brief).objective_brief.model_copy(
            update={"translation_parameters": {"scene": {"asset_key": "space"}}}
        )

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertFalse(
            any(
                entity["proxy_family"] == "ground_plane"
                for entity in skeleton["entities"]
            )
        )

    def test_ground_contact_is_derived_without_a_semantic_relationship(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["subjects"][1]["id"] = "road_01"
        brief["content"]["subjects"][1]["category"] = {
            "value": "道路",
            "source_status": "explicit",
            "source_text": "路面上",
        }
        brief["content"]["scene_design"]["relationships"] = []
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertFalse(skeleton["relations"])
        road = next(
            item for item in skeleton["entities"] if item["entity_id"] == "road_01"
        )
        self.assertEqual(road["proxy_family"], "ground_plane")

    def test_deterministic_fallback_maps_passenger_and_vehicle_categories(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["subjects"][0]["category"] = {
            "value": "乘客",
            "source_status": "explicit",
            "source_text": "一名乘客",
        }
        brief["content"]["subjects"][1]["category"] = {
            "value": "载具",
            "source_status": "explicit",
            "source_text": "一辆载具",
        }
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)
        families = {
            item["entity_id"]: item["proxy_family"] for item in skeleton["entities"]
        }

        self.assertEqual(families["man_01"], "human_capsule")
        self.assertEqual(families["ship_01"], "vehicle_box")

    def test_unknown_asset_key_falls_back_to_environment_ground_semantics(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["environment"] = {
            "value": "路边街道",
            "source_status": "explicit",
            "source_text": "在路边",
        }
        objective = project_objective_brief(brief).objective_brief.model_copy(
            update={
                "translation_parameters": {
                    "scene": {"asset_key": "custom_roadside_pack"}
                }
            }
        )

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertTrue(
            any(
                entity["proxy_family"] == "ground_plane"
                and entity["role"] == "environment"
                for entity in skeleton["entities"]
            )
        )

    def test_deterministic_fallback_downgrades_unknown_source_status(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["relationships"][0]["source_status"] = (
            "unknown"
        )
        brief["content"]["subject_motion"] = [
            {
                "motion_id": "motion_man_wait",
                "subject_id": "man_01",
                "action": {
                    "value": "等待",
                    "source_status": "unknown",
                    "source_text": None,
                },
                "motion_semantics": {
                    "action_kind": "hold",
                    "motion_type": "static",
                    "motion_mode": "stationary",
                    "direction_mode": "none",
                    "target_id": None,
                    "carrier_id": None,
                    "path_type": "stationary",
                    "timeline_event_id": None,
                    "narrative_required": False,
                    "postconditions": {
                        "contained_by_id": None,
                        "external_visibility": "unchanged",
                    },
                    "source_status": "unknown",
                    "source_text": None,
                },
                "direction": {
                    "value": None,
                    "source_status": "unknown",
                    "source_text": None,
                },
                "speed": {
                    "value": None,
                    "source_status": "unknown",
                    "source_text": None,
                },
                "trajectory": {
                    "value": None,
                    "source_status": "unknown",
                    "source_text": None,
                },
                "start_time_seconds": 0.0,
                "end_time_seconds": 6.0,
                "secondary_motion": [],
            }
        ]
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertEqual(skeleton["relations"][0]["source_status"], "inferred")
        self.assertEqual(skeleton["motion_phases"][0]["source_status"], "inferred")


if __name__ == "__main__":
    unittest.main()
