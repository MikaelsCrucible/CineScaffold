from __future__ import annotations

import unittest

from cinescaffold.planning.models import build_deterministic_scene_skeleton
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.relationships import normalize_scene_relationships
from tests.helpers import valid_planning_brief


class RelationshipBoundaryTest(unittest.TestCase):
    def test_distance_strength_and_reverse_synonym_collapse_to_one_far_relation(
        self,
    ) -> None:
        content = {
            "scene_design": {
                "spatial_layers": [
                    {"layer": "远景", "content_ids": ["ship_01"]},
                    {"layer": "前景", "content_ids": ["man_01"]},
                ],
                "relationships": [
                    {
                        "type": "distance",
                        "strength": "far",
                        "subject_id": "man_01",
                        "reference_id": "ship_01",
                        "source_status": "explicit",
                    },
                    {
                        "type": "far_from",
                        "strength": "scene_relative",
                        "subject_id": "ship_01",
                        "reference_id": "man_01",
                        "source_status": "explicit",
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
                    "strength": "scene_relative",
                    "subject_id": "ship_01",
                    "reference_id": "man_01",
                    "source_status": "explicit",
                    "timeline_event_id": None,
                    "temporal_mode": "throughout",
                }
            ],
        )

    def test_generic_distance_uses_near_strength(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "distance",
                        "strength": "near",
                        "subject_id": "a",
                        "reference_id": "b",
                    }
                ]
            }
        }

        normalize_scene_relationships(content)

        self.assertEqual(
            content["scene_design"]["relationships"][0]["type"], "proximity"
        )

    def test_smaller_than_is_oriented_toward_the_dominant_entity(self) -> None:
        content = {
            "scene_design": {
                "relationships": [
                    {
                        "type": "smaller_than",
                        "subject_id": "moon",
                        "reference_id": "earth",
                    }
                ]
            }
        }

        normalize_scene_relationships(content)

        relation = content["scene_design"]["relationships"][0]
        self.assertEqual(relation["type"], "scale_dominance")
        self.assertEqual(relation["subject_id"], "earth")
        self.assertEqual(relation["reference_id"], "moon")

    def test_malformed_relationship_item_is_not_silently_deleted(self) -> None:
        content = {"scene_design": {"relationships": ["not-an-object"]}}

        normalize_scene_relationships(content)

        self.assertEqual(content["scene_design"]["relationships"], ["not-an-object"])

    def test_unknown_relationship_is_preserved_but_not_guessed_as_proximity(
        self,
    ) -> None:
        brief = valid_planning_brief()
        brief["content"]["scene_design"]["relationships"] = [
            {
                "type": "visually_echoes",
                "strength": "subtle",
                "subject_id": "ship_01",
                "reference_id": "man_01",
                "source_status": "inferred",
                "source_text": None,
                "timeline_event_id": None,
                "temporal_mode": "throughout",
            }
        ]
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertEqual(
            objective.scene_design["relationships"][0]["type"], "visually_echoes"
        )
        self.assertFalse(
            any(relation["kind"] == "proximity" for relation in skeleton["relations"])
        )

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

    def test_explicit_ground_support_is_not_duplicated_by_environment_fallback(
        self,
    ) -> None:
        brief = valid_planning_brief()
        brief["content"]["subjects"][1]["id"] = "road_01"
        brief["content"]["subjects"][1]["category"] = {
            "value": "道路",
            "source_status": "explicit",
            "source_text": "路面上",
        }
        brief["content"]["scene_design"]["relationships"] = [
            {
                "type": "ground_support",
                "subject_id": "man_01",
                "reference_id": "road_01",
                "strength": None,
                "source_status": "explicit",
                "source_text": "男人站在路面上",
                "timeline_event_id": None,
                "temporal_mode": "throughout",
            }
        ]
        objective = project_objective_brief(brief).objective_brief

        skeleton = build_deterministic_scene_skeleton(objective)

        self.assertEqual(
            [
                item
                for item in skeleton["relations"]
                if item["kind"] == "ground_support"
                and item["subject_id"] == "man_01"
            ],
            [
                item
                for item in skeleton["relations"]
                if item["source_status"] == "explicit"
            ],
        )
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
            item["entity_id"]: item["proxy_family"]
            for item in skeleton["entities"]
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


if __name__ == "__main__":
    unittest.main()
