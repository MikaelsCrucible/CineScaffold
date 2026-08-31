from __future__ import annotations

import unittest

from cinescaffold.planning.compiler import SceneIRCommitGate, compile_scene_ir
from cinescaffold.planning.domain import CommitRequest
from cinescaffold.planning.duration import attach_duration_resolution, freeze_brief_duration
from cinescaffold.planning.objective import ObjectiveRequirement, project_objective_brief
from cinescaffold.planning.toolkit import (
    FULL_VALIDATION_CHECKS,
    ScenePlanningToolkit,
    _direction_matches,
    _entity_transform_at,
)
from tests.helpers import valid_model_output


class ScenePlanningToolkitTest(unittest.TestCase):
    def test_capabilities_expose_frozen_half_open_timeline(self) -> None:
        result = _toolkit().get_capabilities(["limits"])

        self.assertEqual(
            result["data"]["timeline"],
            {
                "fps_numerator": 24,
                "fps_denominator": 1,
                "frame_count": 144,
                "duration_seconds": 6.0,
                "time_domain": "half_open",
                "time_range_seconds": [0.0, 6.0],
                "last_frame_time_seconds": 143 / 24,
            },
        )
        self.assertIn("camera", result["data"]["inspect_views"])
        self.assertEqual(result["data"]["acceptance"]["minimum_soft_score"], 0.75)
        self.assertFalse(result["data"]["acceptance"]["commit_ready"])
        self.assertEqual(result["data"]["coordinate_system"]["up_axis"], "+Z")
        self.assertEqual(
            result["data"]["coordinate_system"]["default_path_plane_normal"],
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(
            result["data"]["semantic_distinctions"]["relative_position_front_behind"],
            "规范世界 -Y/+Y；不表示摄影机深度",
        )
        self.assertNotIn("constraint_parameter_schemas", result["data"])

        constraints = _toolkit().get_capabilities(["constraints"])
        relative_schema = constraints["data"]["constraint_parameter_schemas"]["relative_position"]
        self.assertEqual(relative_schema["allowed_values"]["space"], ["world"])
        self.assertIn("front", relative_schema["allowed_values"]["relation"])
        self.assertNotIn("supported_geometry", constraints["data"])

    def test_capability_sections_reject_non_planning_domains(self) -> None:
        result = _toolkit().get_capabilities(["blender"])

        self.assertEqual(result["status"], "rejected")
        self.assertIn("不属于 Scene Planning", result["warnings"][0])

    def test_world_motion_direction_uses_negative_y_as_forward(self) -> None:
        self.assertTrue(
            _direction_matches(
                (0.0, -3.0, 0.0),
                "forward",
                2.0,
                1e-8,
                space="world",
            )
        )
        self.assertFalse(
            _direction_matches(
                (0.0, 3.0, 0.0),
                "forward",
                2.0,
                1e-8,
                space="world",
            )
        )

    def test_hold_checks_motion_between_matching_endpoints(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "man_detour",
                    "target_entity_id": "man_01",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": {"translation_m": [0.0, 0.0, 1.0]}},
                        {"time_seconds": 3.0, "value": {"translation_m": [4.0, 0.0, 1.0]}},
                        {"time_seconds": 5.999, "value": {"translation_m": [0.0, 0.0, 1.0]}},
                    ],
                }
            ],
            [],
        )
        _apply_hold_constraint(toolkit, ["translation"])

        validation = toolkit.validate_candidate(checks=["motion"])

        violation = next(
            item for item in validation["violations"] if item["code"] == "HOLD_VIOLATED"
        )
        self.assertGreater(violation["actual"]["translation_delta_m"], 0.0)

    def test_hold_checks_rotation_and_scale_components(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "man_transform_change",
                    "target_entity_id": "man_01",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {
                                "translation_m": [0.0, 0.0, 1.0],
                                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "scale": [1.0, 1.0, 1.0],
                            },
                        },
                        {
                            "time_seconds": 3.0,
                            "value": {
                                "translation_m": [0.0, 0.0, 1.0],
                                "rotation_quaternion_wxyz": [0.70710678, 0.0, 0.0, 0.70710678],
                                "scale": [2.0, 1.0, 1.0],
                            },
                        },
                    ],
                }
            ],
            [],
        )
        _apply_hold_constraint(toolkit, ["rotation", "scale"])

        validation = toolkit.validate_candidate(checks=["motion"])

        violation = next(
            item for item in validation["violations"] if item["code"] == "HOLD_VIOLATED"
        )
        self.assertGreater(violation["actual"]["rotation_delta_degrees"], 0.0)
        self.assertGreater(violation["actual"]["scale_delta"], 0.0)

    def test_hold_checks_visibility_component(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "man_visibility",
                    "target_entity_id": "man_01",
                    "type": "visibility",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": True, "interpolation": "step"},
                        {"time_seconds": 2.0, "value": False, "interpolation": "step"},
                        {"time_seconds": 4.0, "value": True, "interpolation": "step"},
                    ],
                }
            ],
            [],
        )
        _apply_hold_constraint(toolkit, ["visibility"])

        validation = toolkit.validate_candidate(checks=["motion"])

        violation = next(
            item for item in validation["violations"] if item["code"] == "HOLD_VIOLATED"
        )
        self.assertEqual(violation["actual"]["visibility"]["expected"], True)
        self.assertEqual(violation["actual"]["visibility"]["actual"], False)

    def test_hold_rejects_empty_component_list(self) -> None:
        toolkit = _toolkit()

        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "empty_hold",
                    "type": "hold",
                    "strength": "soft",
                    "subjects": ["man_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {"target_id": "man_01", "components": []},
                    "source_status": "agent_selected",
                    "source_ref": "agent.invalid_hold",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("components", result["warnings"][0])

    def test_unimplemented_relative_position_space_is_rejected(self) -> None:
        toolkit = _toolkit()

        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "camera_relative_position",
                    "type": "relative_position",
                    "strength": "soft",
                    "subjects": ["man_01", "ship_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "subject_id": "man_01",
                        "reference_id": "ship_01",
                        "relation": "front",
                        "space": "camera",
                    },
                    "source_status": "agent_selected",
                    "source_ref": "agent.layout",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("parameters.space", result["warnings"][0])

    def test_relative_position_rejects_reversed_gap_range(self) -> None:
        toolkit = _toolkit()

        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "invalid_gap_range",
                    "type": "relative_position",
                    "strength": "soft",
                    "subjects": ["man_01", "ship_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "subject_id": "man_01",
                        "reference_id": "ship_01",
                        "relation": "front",
                        "minimum_gap": 5.0,
                        "maximum_gap": 2.0,
                    },
                    "source_status": "agent_selected",
                    "source_ref": "agent.layout",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("间距上下界颠倒", result["warnings"][0])

    def test_relative_position_enforces_maximum_gap(self) -> None:
        toolkit = _toolkit()
        man = _man_entity() | {
            "solved_transform": {
                "translation_m": [0.0, -5.0, 1.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        ship = _ship_entity() | {
            "solved_transform": {
                "translation_m": [0.0, 0.0, 1.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        toolkit.apply_entity_patch([man, ship], [])
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0, "focus_target_id": "ship_01"},
            tracks=[
                {
                    "track_id": "camera_static",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {"translation_m": [0.0, -20.0, 10.0]},
                        }
                    ],
                }
            ],
            remove_track_ids=[],
        )
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "front_gap",
                    "type": "relative_position",
                    "strength": "soft",
                    "subjects": ["man_01", "ship_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "subject_id": "man_01",
                        "reference_id": "ship_01",
                        "relation": "front",
                        "minimum_gap": 2.0,
                        "maximum_gap": 4.0,
                    },
                    "source_status": "agent_selected",
                    "source_ref": "agent.layout",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=[])

        violation = next(
            item
            for item in validation["violations"]
            if item["code"] == "RELATIVE_POSITION_VIOLATED"
        )
        self.assertEqual(violation["actual"]["signed_gap_m"], 5.0)

    def test_constraint_validation_returns_only_actionable_union_branch(self) -> None:
        result = _toolkit().apply_constraint_patch(
            [
                {
                    "constraint_id": "bad_relative",
                    "type": "relative_position",
                    "strength": "hard",
                    "subjects": ["man_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "subject_id": "man_01",
                        "reference_id": "ship_01",
                        "relation": "left",
                        "range_m": [0.0, 4.0],
                    },
                    "source_status": "explicit",
                    "source_ref": "content.scene_design.relationships[0]",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("parameters.range_m", result["warnings"][0])
        self.assertLess(len(result["warnings"][0]), 500)

    def test_camera_patch_rejects_overlapping_singleton_tracks_atomically(self) -> None:
        toolkit = _toolkit()
        result = toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[
                {
                    "track_id": "camera_fast",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [],
                },
                {
                    "track_id": "camera_slow",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [],
                },
            ],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["revision_after"], 0)
        self.assertIn("remove_track_ids", result["warnings"][0])
        self.assertIsNone(toolkit.store.get().camera)

    def test_validator_rejects_ambiguous_tracks_from_legacy_checkpoint(self) -> None:
        toolkit = _toolkit()
        accepted = toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[
                {
                    "track_id": "camera_fast",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [],
                }
            ],
        )
        self.assertEqual(accepted["status"], "ok")

        def inject_legacy_track(state):
            original = state.camera.tracks["camera_fast"]
            state.camera.tracks["camera_slow"] = original.model_copy(
                update={"track_id": "camera_slow"}
            )
            return ([{"operation": "inject", "path": "camera.tracks.camera_slow"}], [])

        toolkit.store.apply(inject_legacy_track)
        validation = toolkit.validate_candidate(checks=["camera"])

        self.assertFalse(validation["data"]["hard_pass"])
        self.assertTrue(
            any(
                item["code"] == "AMBIGUOUS_CAMERA_TRACKS"
                for item in validation["violations"]
            )
        )

    def test_motion_patch_rejects_overlapping_entity_tracks_atomically(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        result = toolkit.apply_motion_patch(
            [
                {
                    "track_id": "man_move_a",
                    "target_entity_id": "man_01",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [],
                },
                {
                    "track_id": "man_move_b",
                    "target_entity_id": "man_01",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [],
                },
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("remove_ids", result["warnings"][0])
        self.assertEqual(toolkit.store.get().motion_tracks, {})

    def test_same_id_remove_and_upsert_atomically_replaces_entity_and_track(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        replaced_entity = _man_entity() | {"label": "替换后的人物"}

        entity_result = toolkit.apply_entity_patch([replaced_entity], ["man_01"])
        first_track = {
            "track_id": "man_move",
            "target_entity_id": "man_01",
            "type": "transform",
            "time_range_seconds": [0.0, 6.0],
            "keyframes": [],
        }
        toolkit.apply_motion_patch([first_track], [])
        replaced_track = first_track | {
            "keyframes": [
                {
                    "time_seconds": 0.0,
                    "value": {"translation_m": [0.0, 0.0, 0.9]},
                }
            ]
        }
        track_result = toolkit.apply_motion_patch([replaced_track], ["man_move"])

        self.assertEqual(entity_result["status"], "ok")
        self.assertEqual(toolkit.store.get().entities["man_01"].label, "替换后的人物")
        self.assertEqual(track_result["status"], "ok")
        self.assertEqual(len(toolkit.store.get().motion_tracks["man_move"].keyframes), 1)

    def test_entity_replacement_cannot_change_proxy_topology(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_ship_entity()], [])
        changed = _ship_entity() | {
            "proxy": {
                "type": "capsule",
                "radius_m": 30.0,
                "segment_length_m": 80.0,
                "axis": "+Z",
            }
        }

        result = toolkit.apply_entity_patch([changed], ["ship_01"])

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(toolkit.store.get().entities["ship_01"].proxy.type, "box")
        self.assertIn("不得更换代理拓扑", result["warnings"][0])

    def test_non_explicit_hard_constraint_is_rejected(self) -> None:
        toolkit = _toolkit()
        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "invented_ground_height",
                    "type": "position_at_time",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": ["man_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "target_id": "man_01",
                        "position_m": [0.0, 0.0, 0.4],
                    },
                    "source_status": "agent_selected",
                    "source_ref": "agent.layout",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["revision_after"], 0)
        self.assertIn("explicit requirement", result["warnings"][0])

    def test_hard_constraint_rejects_semantically_incompatible_explicit_source(self) -> None:
        toolkit = _toolkit()

        def add_environment_requirement(state):
            state.required_source_refs.append("content.scene_design.environment")
            return ([{"operation": "test", "path": "required_source_refs"}], [])

        toolkit.store.apply(add_environment_requirement)
        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "invented_environment_depth",
                    "type": "depth_order",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": ["man_01", "ship_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "near_entity_id": "man_01",
                        "far_entity_id": "ship_01",
                        "minimum_depth_gap_meters": 10.0,
                    },
                    "source_status": "explicit",
                    "source_ref": "content.scene_design.environment",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("来源不兼容", result["warnings"][0])

    def test_far_relationship_requires_camera_depth_order(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "distance_only",
                    "type": "distance_range",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": ["man_01", "ship_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "entity_ids": ["man_01", "ship_01"],
                        "minimum_meters": 30.0,
                        "maximum_meters": 200.0,
                    },
                    "source_status": "explicit",
                    "source_ref": "content.scene_design.relationships[0]",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["hard_semantics"])

        violation = next(
            item
            for item in validation["violations"]
            if item["code"] == "EXPLICIT_REQUIREMENT_CONSTRAINT_INCOMPLETE"
        )
        self.assertEqual(violation["expected"]["constraint_types"], ["depth_order"])

    def test_entity_fully_below_horizontal_ground_is_rejected(self) -> None:
        toolkit = _toolkit()
        ground = {
            "entity_id": "desert_ground",
            "label": "荒漠地面",
            "role": "environment",
            "proxy": {"type": "plane", "size_xy_m": [400.0, 400.0]},
            "tags": ["environment", "ground"],
            "source_refs": [],
            "solved_transform": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        ship = _ship_entity() | {
            "solved_transform": {
                "translation_m": [0.0, 100.0, -100.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        toolkit.apply_entity_patch([ground, ship], [])

        validation = toolkit.validate_candidate(checks=["transforms"])

        self.assertIn(
            "ENTITY_FULLY_BELOW_GROUND",
            {item["code"] for item in validation["violations"]},
        )

    def test_entity_partially_intersecting_ground_is_rejected(self) -> None:
        toolkit = _toolkit()
        ground = {
            "entity_id": "desert_ground",
            "label": "荒漠地面",
            "role": "environment",
            "proxy": {"type": "plane", "size_xy_m": [400.0, 400.0]},
            "tags": ["environment", "ground"],
            "source_refs": [],
            "solved_transform": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        ship = _ship_entity() | {
            "proxy": {"type": "box", "size_xyz_m": [180.0, 180.0, 200.0]},
            "solved_transform": {
                "translation_m": [180.0, 0.0, 10.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        toolkit.apply_entity_patch([ground, ship], [])

        validation = toolkit.validate_candidate(checks=["transforms"])

        self.assertIn(
            "ENTITY_INTERSECTS_GROUND",
            {item["code"] for item in validation["violations"]},
        )

    def test_explicit_embedded_entity_may_cross_ground_without_being_cut(self) -> None:
        toolkit = _toolkit()
        ground = _ground_entity()
        ship = _ship_entity() | {
            "ground_interaction": {
                "mode": "embedded",
                "ground_entity_id": "desert_ground",
                "minimum_penetration_m": 4.0,
                "maximum_penetration_m": 6.0,
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[0]",
            },
            "solved_transform": {
                "translation_m": [0.0, 20.0, 2.5],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        toolkit.apply_entity_patch([ground, ship], [])

        validation = toolkit.validate_candidate(checks=["transforms"])

        self.assertNotIn(
            "ENTITY_INTERSECTS_GROUND",
            {item["code"] for item in validation["violations"]},
        )
        self.assertNotIn(
            "ENTITY_EMBEDDING_RANGE_VIOLATED",
            {item["code"] for item in validation["violations"]},
        )
        stored = toolkit.store.get().entities["ship_01"]
        self.assertEqual(stored.proxy.size_xyz_m, (80.0, 30.0, 15.0))

    def test_agent_selected_ground_exception_is_rejected(self) -> None:
        toolkit = _toolkit()
        ship = _ship_entity() | {
            "ground_interaction": {
                "mode": "may_intersect",
                "maximum_penetration_m": 5.0,
                "source_status": "agent_selected",
            }
        }

        result = toolkit.apply_entity_patch([_ground_entity(), ship], [])

        self.assertEqual(result["status"], "rejected")
        self.assertIn("只能来自 Brief 的明确地面交互要求", result["warnings"][0])

    def test_must_touch_rejects_floating_entity(self) -> None:
        toolkit = _toolkit()
        man = _man_entity() | {
            "ground_interaction": {
                "mode": "must_touch",
                "source_status": "agent_selected",
            },
            "solved_transform": {
                "translation_m": [0.0, 0.0, 2.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            },
        }
        toolkit.apply_entity_patch([_ground_entity(), man], [])

        validation = toolkit.validate_candidate(checks=["transforms"])

        self.assertIn(
            "ENTITY_GROUND_CONTACT_VIOLATED",
            {item["code"] for item in validation["violations"]},
        )

    def test_camera_subject_is_not_reported_as_missing_entity(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[],
        )
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "camera_speed",
                    "type": "speed_range",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": ["camera_main"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "target_id": "camera_main",
                        "minimum_mps": 0.01,
                        "maximum_mps": 2.0,
                        "space": "world",
                    },
                    "source_status": "explicit",
                    "source_ref": "content.camera.movement.speed",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["references"])

        self.assertNotIn(
            "CONSTRAINT_REFERENCE_MISSING",
            {item["code"] for item in validation["violations"]},
        )

    def test_position_at_time_supports_camera_target(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[
                {
                    "track_id": "camera_position",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {"translation_m": [0.0, -30.0, 2.5]},
                        }
                    ],
                }
            ],
        )
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "camera_position_hint",
                    "type": "position_at_time",
                    "strength": "soft",
                    "weight": 1.0,
                    "subjects": ["camera_main"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "target_id": "camera_main",
                        "position_m": [0.0, -30.0, 2.5],
                        "tolerance_m": 0.01,
                    },
                    "source_status": "inferred",
                    "source_ref": "content.camera.angle",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["projection"])

        self.assertNotIn(
            "CONSTRAINT_PARAMETER_INVALID",
            {item["code"] for item in validation["violations"]},
        )

    def test_transform_keyframe_rejects_unknown_position_alias(self) -> None:
        toolkit = _toolkit()
        result = toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[
                {
                    "track_id": "invalid_camera_track",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {"position": [0.0, -12.0, 2.0]},
                            "interpolation": "linear",
                        }
                    ],
                }
            ],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["revision_after"], 0)
        self.assertIn("position", result["warnings"][0])

    def test_mutations_create_revisions_and_restore_keeps_history(self) -> None:
        toolkit = _toolkit()
        first = toolkit.apply_entity_patch([_man_entity()], [])
        second = toolkit.apply_entity_patch([_ship_entity()], [])

        restored = toolkit.restore_candidate(first["revision_after"], "撤销飞船尝试")

        self.assertEqual(first["revision_after"], 1)
        self.assertEqual(second["revision_after"], 2)
        self.assertEqual(restored["revision_after"], 3)
        self.assertEqual(toolkit.store.revisions, [0, 1, 2, 3])
        self.assertEqual(sorted(toolkit.store.get(2).entities), ["man_01", "ship_01"])
        self.assertEqual(sorted(toolkit.store.get(3).entities), ["man_01"])

    def test_unsupported_constraint_does_not_mutate_candidate(self) -> None:
        toolkit = _toolkit()
        result = toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "cloth_01",
                    "type": "collision_clearance",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": [],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {},
                    "source_status": "explicit",
                    "source_ref": "content.subject_motion[0]",
                }
            ],
            [],
        )

        self.assertEqual(result["status"], "unsupported")
        self.assertEqual(result["revision_after"], 0)
        self.assertIn("constraint:collision_clearance", result["capability_gaps"])

    def test_commit_gate_bakes_complete_scene_ir(self) -> None:
        toolkit = _solved_toolkit()
        validation = toolkit.validate_candidate(checks=FULL_VALIDATION_CHECKS)
        revision = toolkit.store.current_revision

        committed = SceneIRCommitGate(toolkit).commit(
            CommitRequest(
                type="commit_request",
                candidate_revision=revision,
                summary="硬约束已通过",
            ),
            agent_run_id="test_run",
            trace_ref="planning_agent_tool_trace.jsonl",
        )

        self.assertTrue(validation["data"]["hard_pass"], validation["violations"])
        self.assertEqual(committed.status, "success")
        self.assertIsNotNone(committed.scene_ir_hash)
        scene_ir = committed.scene_ir
        self.assertEqual(len(scene_ir.entities), 2)
        self.assertEqual(
            len(scene_ir.camera.state_track.samples),
            scene_ir.timeline.frame_count,
        )
        self.assertEqual(scene_ir.camera.state_track.samples[0].frame, 1)
        self.assertEqual(
            scene_ir.camera.state_track.samples[-1].frame,
            scene_ir.timeline.frame_end,
        )
        self.assertNotIn("mood", scene_ir.model_dump_json())
        self.assertEqual(scene_ir.lighting.purpose, "technical_preview")
        self.assertEqual(scene_ir.lighting.mode, "neutral_camera_rig")
        self.assertEqual(scene_ir.lighting.rig_id, "neutral_camera_rig_v0.1")
        self.assertFalse(scene_ir.lighting.cast_shadows)
        self.assertEqual(scene_ir.lighting.lights, [])
        self.assertEqual(scene_ir.entities[0].ground_interaction.mode, "must_be_above")

    def test_unmapped_explicit_requirement_blocks_commit(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity() | {"source_refs": []}], [])
        toolkit.solve_candidate()

        validation = toolkit.validate_candidate(checks=["hard_semantics"])

        self.assertFalse(validation["data"]["hard_pass"])
        self.assertTrue(
            any(item["code"] == "UNMAPPED_EXPLICIT_REQUIREMENT" for item in validation["violations"])
        )

    def test_push_in_uses_target_distance_instead_of_world_axis(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity()], [])
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "push_in_x_axis",
                    "type": "camera_motion_direction",
                    "strength": "hard",
                    "weight": 1.0,
                    "subjects": [],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "camera_id": "camera_main",
                        "target_id": "man_01",
                        "direction": "push_in",
                        "space": "camera",
                        "minimum_displacement_m": 5.0,
                    },
                    "source_status": "explicit",
                    "source_ref": "content.camera.movement.type",
                }
            ],
            [],
        )
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={
                "focal_length_mm": 50.0,
                "sensor_width_mm": 36.0,
                "focus_target_id": "man_01",
            },
            tracks=[
                {
                    "track_id": "camera_move_x",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": {"translation_m": [-120.0, 2.0, 6.0]}},
                        {"time_seconds": 143 / 24, "value": {"translation_m": [-60.0, 2.0, 6.0]}},
                    ],
                }
            ],
        )
        toolkit.solve_candidate()

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertTrue(validation["data"]["hard_pass"], validation["violations"])

    def test_projection_tolerance_accepts_numerically_full_visibility(self) -> None:
        toolkit = _solved_toolkit()
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "man_fully_visible",
                    "type": "keep_in_frame",
                    "strength": "soft",
                    "weight": 1.0,
                    "subjects": ["man_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "entity_id": "man_01",
                        "minimum_inside_fraction": 1.0,
                    },
                    "source_status": "agent_selected",
                    "source_ref": "content.composition",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["visibility"])

        codes = {item["code"] for item in validation["violations"]}
        self.assertNotIn("ENTITY_OUT_OF_FRAME", codes)

    def test_solver_preserves_unspecified_box_orientation(self) -> None:
        toolkit = _toolkit()
        man = _man_entity() | {
            "solved_transform": {
                "translation_m": [0.0, 0.0, 0.9],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        ship = _ship_entity() | {
            "solved_transform": {
                "translation_m": [120.0, 0.0, 20.0],
                "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        toolkit.apply_entity_patch([man, ship], [])
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={
                "focal_length_mm": 50.0,
                "sensor_width_mm": 36.0,
                "focus_target_id": "man_01",
            },
            tracks=[
                {
                    "track_id": "camera_move_x",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": {"translation_m": [-120.0, 2.0, 6.0]}},
                        {"time_seconds": 143 / 24, "value": {"translation_m": [-60.0, 2.0, 6.0]}},
                    ],
                }
            ],
        )

        toolkit.solve_candidate()
        validation = toolkit.validate_candidate(checks=["transforms", "projection"])
        ship_rotation = toolkit.store.get().entities["ship_01"].solved_transform.rotation_quaternion_wxyz

        self.assertEqual(ship_rotation, (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(validation["data"]["hard_pass"], validation["violations"])

    def test_speed_requirement_rejects_unrelated_camera_distance_mapping(self) -> None:
        toolkit = _toolkit()

        def add_speed_requirement(state):
            state.required_source_refs.append("content.camera.movement.speed")
            return ([{"operation": "test", "path": "required_source_refs"}], [])

        toolkit.store.apply(add_speed_requirement)
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "wrong_slow_mapping",
                    "type": "camera_distance",
                    "strength": "soft",
                    "weight": 1.0,
                    "subjects": ["man_01"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "camera_id": "camera_main",
                        "target_id": "man_01",
                        "minimum_meters": 90.0,
                        "maximum_meters": 200.0,
                    },
                    "source_status": "explicit",
                    "source_ref": "content.camera.movement.speed",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["hard_semantics"])

        self.assertTrue(
            any(
                item["code"] == "EXPLICIT_REQUIREMENT_MAPPING_INCOMPATIBLE"
                for item in validation["violations"]
            )
        )

    def test_nested_target_relative_orbits_compile_to_parent_local_ir(self) -> None:
        toolkit = _relative_motion_toolkit()

        earth_quarter = _entity_transform_at(toolkit.store.get(), "earth", 1.5)
        moon_quarter = _entity_transform_at(toolkit.store.get(), "moon", 1.5)
        scene_ir = compile_scene_ir(
            toolkit,
            toolkit.store.get(),
            agent_run_id="relative_motion_test",
            trace_ref="test_trace.jsonl",
        )
        entities = {item.entity_id: item for item in scene_ir.entities}

        self.assertAlmostEqual(earth_quarter.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(earth_quarter.translation_m[1], 10.0, places=8)
        self.assertAlmostEqual(moon_quarter.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(moon_quarter.translation_m[1], 8.0, places=8)
        earth_frame = entities["earth"].local_state_track.samples[36].value
        moon_frame = entities["moon"].local_state_track.samples[36].value
        self.assertAlmostEqual(earth_frame.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(earth_frame.translation_m[1], 10.0, places=8)
        self.assertEqual(earth_frame.translation_m[2], 0.0)
        self.assertAlmostEqual(moon_frame.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(moon_frame.translation_m[1], -2.0, places=8)
        self.assertEqual(moon_frame.translation_m[2], 0.0)

    def test_nested_orbit_phase_lock_is_rejected_as_unreadable(self) -> None:
        toolkit = _relative_motion_toolkit(moon_cycle_count=1.0)

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertFalse(validation["data"]["hard_pass"])
        violation = next(
            item
            for item in validation["violations"]
            if item["code"] == "NESTED_ORBIT_PHASE_LOCKED"
        )
        self.assertEqual(violation["actual"]["child_cycle_count"], 1.0)
        self.assertIn("cycle_count", violation["adjustable_variables"][0])

    def test_different_nested_orbit_cycle_counts_are_readable(self) -> None:
        toolkit = _relative_motion_toolkit(moon_cycle_count=3.0)

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertTrue(validation["data"]["hard_pass"], validation["violations"])

    def test_inferred_camera_rejects_nearly_edge_on_orbit_projection(self) -> None:
        toolkit = _relative_motion_toolkit(
            plane_normal=(0.0, 1.0, 0.0),
            camera_position=(0.0, 9.0, 36.0),
        )

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertFalse(validation["data"]["hard_pass"])
        violation = next(
            item
            for item in validation["violations"]
            if item["code"] == "ORBIT_PLANE_NEAR_EDGE_ON"
        )
        self.assertEqual(violation["severity"], "hard")
        self.assertLess(
            violation["actual"]["median_absolute_view_normal_dot"],
            violation["expected"]["minimum_median_absolute_view_normal_dot"],
        )

    def test_explicit_camera_view_keeps_edge_on_orbit_as_warning(self) -> None:
        toolkit = _relative_motion_toolkit(
            plane_normal=(0.0, 1.0, 0.0),
            camera_position=(0.0, 9.0, 36.0),
        )
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "explicit_requirements": [
                    *toolkit.objective_brief.explicit_requirements,
                    ObjectiveRequirement(
                        path="content.camera.view_angle",
                        value="轨道侧视",
                        source_text="从轨道侧面看",
                    ),
                ]
            }
        )

        validation = toolkit.validate_candidate(checks=["motion"])

        violation = next(
            item
            for item in validation["violations"]
            if item["code"] == "ORBIT_PLANE_NEAR_EDGE_ON"
        )
        self.assertTrue(validation["data"]["hard_pass"])
        self.assertEqual(violation["severity"], "warning")

    def test_polygon_cannot_masquerade_as_default_orbit(self) -> None:
        toolkit = _relative_motion_toolkit(earth_radius_m=20.0)

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertIn(
            "ORBIT_TRAJECTORY_NOT_ANALYTIC",
            {item["code"] for item in validation["violations"]},
        )

    def test_explicit_figure_eight_orbit_may_use_lemniscate(self) -> None:
        toolkit = _relative_motion_toolkit()
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "subject_motion": [
                    {
                        "subject_id": "earth",
                        "trajectory": {
                            "value": "∞形",
                            "source_status": "explicit",
                            "source_text": "沿∞形运动",
                        },
                    }
                ]
            }
        )
        replacement = {
            "track_id": "earth_orbit",
            "target_entity_id": "earth",
            "type": "path_follow",
            "time_range_seconds": [0.0, 6.0],
            "path": {
                "representation": "lemniscate",
                "space": "target_relative",
                "target_id": "sun",
                "width_m": 20.0,
                "height_m": 10.0,
            },
            "source_ref": "content.subject_motion[0].trajectory",
        }
        toolkit.apply_motion_patch([replacement], ["earth_orbit"])

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertNotIn(
            "ORBIT_TRAJECTORY_NOT_ANALYTIC",
            {item["code"] for item in validation["violations"]},
        )

    def test_relative_motion_cycle_is_rejected_atomically(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_sphere_entity("earth"), _sphere_entity("moon")], [])
        revision = toolkit.store.current_revision

        result = toolkit.apply_motion_patch(
            [
                _orbit_track("earth_orbit", "earth", "moon", 10.0),
                _orbit_track("moon_orbit", "moon", "earth", 2.0),
            ],
            [],
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(toolkit.store.current_revision, revision)
        self.assertEqual(toolkit.store.get().motion_tracks, {})
        self.assertIn("参考系依赖存在循环", result["warnings"][0])

    def test_local_motion_composes_parent_rotation(self) -> None:
        toolkit = _toolkit()
        half_root = 2**-0.5
        parent = _sphere_entity("parent") | {
            "solved_transform": {
                "translation_m": [10.0, 0.0, 0.0],
                "rotation_quaternion_wxyz": [half_root, 0.0, 0.0, half_root],
                "scale": [1.0, 1.0, 1.0],
            }
        }
        child = _sphere_entity("child") | {"parent_id": "parent"}
        toolkit.apply_entity_patch([parent, child], [])
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "child_local",
                    "target_entity_id": "child",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {
                                "translation_m": [1.0, 0.0, 0.0],
                                "space": "local",
                            },
                        }
                    ],
                }
            ],
            [],
        )

        child_world = _entity_transform_at(toolkit.store.get(), "child", 0.0)

        self.assertAlmostEqual(child_world.translation_m[0], 10.0, places=8)
        self.assertAlmostEqual(child_world.translation_m[1], 1.0, places=8)

    def test_camera_relative_entity_uses_current_camera_transform(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_sphere_entity("marker")], [])
        toolkit.apply_camera_patch(
            camera_id="camera_main",
            projection="perspective",
            active=True,
            static={"focal_length_mm": 35.0},
            tracks=[
                {
                    "track_id": "camera_world",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {
                            "time_seconds": 0.0,
                            "value": {"translation_m": [10.0, 0.0, 0.0]},
                        }
                    ],
                }
            ],
            remove_track_ids=[],
        )
        result = toolkit.apply_motion_patch(
            [
                {
                    "track_id": "marker_camera_relative",
                    "target_entity_id": "marker",
                    "type": "path_follow",
                    "time_range_seconds": [0.0, 6.0],
                    "path": {
                        "space": "camera",
                        "control_points": [[0.0, 0.0, -5.0], [0.0, 0.0, -5.0]],
                    },
                }
            ],
            [],
        )

        marker = _entity_transform_at(toolkit.store.get(), "marker", 0.0)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(marker.translation_m, (10.0, 0.0, -5.0))

    def test_distance_range_checks_every_frozen_frame(self) -> None:
        toolkit = _relative_motion_toolkit(earth_radius_m=20.0)
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "earth_orbit_radius",
                    "type": "distance_range",
                    "strength": "soft",
                    "weight": 1.0,
                    "subjects": ["sun", "earth"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "entity_ids": ["sun", "earth"],
                        "minimum_meters": 9.0,
                        "maximum_meters": 11.0,
                    },
                    "source_status": "agent_selected",
                    "source_ref": "agent.orbit_radius",
                }
            ],
            [],
        )

        validation = toolkit.validate_candidate(checks=["motion"])

        self.assertIn(
            "DISTANCE_RANGE_VIOLATED",
            {item["code"] for item in validation["violations"]},
        )


def _solved_toolkit() -> ScenePlanningToolkit:
    toolkit = _toolkit()
    toolkit.apply_entity_patch([_man_entity(), _ship_entity()], [])
    toolkit.apply_constraint_patch(
        [
            {
                "constraint_id": "ship_behind_man",
                "type": "depth_order",
                "strength": "soft",
                "weight": 1.0,
                "subjects": ["man_01", "ship_01"],
                "time_range_seconds": [0.0, 6.0],
                "parameters": {
                    "near_entity_id": "man_01",
                    "far_entity_id": "ship_01",
                    "minimum_depth_gap_meters": 20.0,
                },
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[0]",
            },
            {
                "constraint_id": "camera_push_in",
                "type": "camera_motion_direction",
                "strength": "hard",
                "weight": 1.0,
                "subjects": [],
                "time_range_seconds": [0.0, 6.0],
                "parameters": {"direction": "push_in", "minimum_displacement_m": 3.0},
                "source_status": "explicit",
                "source_ref": "content.camera.movement.type",
            },
        ],
        [],
    )
    toolkit.apply_camera_patch(
        camera_id="camera_main",
        projection="perspective",
        active=True,
        static={
            "focal_length_mm": 35.0,
            "sensor_width_mm": 36.0,
            "focus_target_id": "man_01",
            "source_refs": [],
        },
        tracks=[
            {
                "track_id": "camera_move_01",
                "type": "transform",
                "time_range_seconds": [0.0, 6.0],
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "value": {"translation_m": [0.0, -12.0, 2.0]},
                        "interpolation": "smooth",
                    },
                    {
                        "time_seconds": 5.999,
                        "value": {"translation_m": [0.0, -7.0, 2.0]},
                        "interpolation": "smooth",
                    },
                ],
                "path": None,
                "target_id": None,
                "interpolation": "smooth",
                "locked_components": [],
                "source_ref": "content.camera.movement.type",
            }
        ],
        remove_track_ids=[],
    )
    toolkit.solve_candidate()
    return toolkit


def _apply_hold_constraint(
    toolkit: ScenePlanningToolkit,
    components: list[str],
) -> None:
    result = toolkit.apply_constraint_patch(
        [
            {
                "constraint_id": "hold_man",
                "type": "hold",
                "strength": "soft",
                "subjects": ["man_01"],
                "time_range_seconds": [0.0, 6.0],
                "parameters": {
                    "target_id": "man_01",
                    "components": components,
                },
                "source_status": "agent_selected",
                "source_ref": "agent.hold_test",
            }
        ],
        [],
    )
    if result["status"] != "ok":
        raise AssertionError(result)


def _toolkit() -> ScenePlanningToolkit:
    content = valid_model_output()
    content["subjects"] = [
        {
            "id": "man_01",
            "category": _annotated("男人", "一个男人"),
            "description": _unknown(),
            "narrative_role": _unknown(),
            "attributes": [],
        },
        {
            "id": "ship_01",
            "category": _annotated("飞船", "巨大的飞船"),
            "description": _unknown(),
            "narrative_role": _unknown(),
            "attributes": [],
        },
    ]
    content["scene_design"]["relationships"] = [
        {
            "type": "远处",
            "subject_id": "ship_01",
            "reference_id": "man_01",
            "strength": "明显",
            "source_status": "explicit",
            "source_text": "远处有飞船",
        }
    ]
    content["camera"]["movement"]["type"] = _annotated("缓慢推近", "镜头慢慢推近")
    content["timeline"].update(
        {"duration_seconds": 6.0, "duration_source_status": "inferred"}
    )
    brief = {
        "schema_version": "0.1",
        "content": content,
        "provenance": {
            "source_prompt": "一个男人站在荒漠里，远处有飞船，镜头慢慢推近。",
            "provider": "mock",
            "model": "mock-cinematic-brief-v0.1",
            "parser_prompt_version": "semantic-parser-v0.1",
            "rules_sha256": "0" * 64,
            "response_id": "mock-response-001",
        },
    }
    objective = project_objective_brief(brief).objective_brief
    resolution = freeze_brief_duration(
        objective.timeline,
        fps_numerator=24,
        fps_denominator=1,
    )
    return ScenePlanningToolkit(attach_duration_resolution(objective, resolution))


def _man_entity() -> dict:
    return {
        "entity_id": "man_01",
        "label": "男人代理",
        "role": "primary_subject",
        "proxy": {"type": "capsule", "radius_m": 0.3, "segment_length_m": 1.2, "axis": "+Z"},
        "parent_id": None,
        "tags": ["person"],
        "locked_fields": [],
        "source_refs": ["content.subjects[0].category"],
        "solved_transform": {},
    }


def _ship_entity() -> dict:
    return {
        "entity_id": "ship_01",
        "label": "飞船代理",
        "role": "background_subject",
        "proxy": {"type": "box", "size_xyz_m": [80.0, 30.0, 15.0]},
        "parent_id": None,
        "tags": ["spaceship"],
        "locked_fields": [],
        "source_refs": ["content.subjects[1].category"],
        "solved_transform": {},
    }


def _ground_entity() -> dict:
    return {
        "entity_id": "desert_ground",
        "label": "荒漠地面",
        "role": "environment",
        "proxy": {"type": "plane", "size_xy_m": [400.0, 400.0]},
        "parent_id": None,
        "tags": ["environment", "ground"],
        "locked_fields": [],
        "source_refs": [],
        "solved_transform": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }


def _sphere_entity(entity_id: str, *, parent_id: str | None = None) -> dict:
    return {
        "entity_id": entity_id,
        "label": entity_id,
        "role": "subject",
        "proxy": {"type": "sphere", "radius_m": 1.0},
        "parent_id": parent_id,
        "tags": [],
        "locked_fields": [],
        "source_refs": [],
        "solved_transform": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }


def _orbit_track(
    track_id: str,
    entity_id: str,
    target_id: str,
    radius_m: float,
    *,
    cycle_count: float = 1.0,
    plane_normal: tuple[float, float, float] | None = None,
) -> dict:
    track = {
        "track_id": track_id,
        "target_entity_id": entity_id,
        "type": "path_follow",
        "time_range_seconds": [0.0, 6.0],
        "path": {
            "representation": "circle",
            "space": "target_relative",
            "target_id": target_id,
            "radius_m": radius_m,
            "cycle_count": cycle_count,
            "parameterization": "arc_length",
            "orientation_mode": "keep",
        },
        "source_ref": "content.scene_design.relationships[0]",
    }
    if plane_normal is not None:
        track["path"]["plane_normal"] = list(plane_normal)
    return track


def _relative_motion_toolkit(
    *,
    earth_radius_m: float = 10.0,
    moon_cycle_count: float = 3.0,
    plane_normal: tuple[float, float, float] | None = None,
    camera_position: tuple[float, float, float] = (0.0, -45.0, 35.0),
) -> ScenePlanningToolkit:
    toolkit = _toolkit()
    toolkit.objective_brief = toolkit.objective_brief.model_copy(
        update={
            "scene_design": {
                "relationships": [
                    {
                        "type": "orbit_around",
                        "subject_id": "earth",
                        "reference_id": "sun",
                    },
                    {
                        "type": "orbit_around",
                        "subject_id": "moon",
                        "reference_id": "earth",
                    },
                ]
            }
        }
    )
    toolkit.apply_entity_patch(
        [
            _sphere_entity("sun"),
            _sphere_entity("earth", parent_id="sun"),
            _sphere_entity("moon", parent_id="earth"),
        ],
        [],
    )
    earth_track = _orbit_track(
        "earth_orbit",
        "earth",
        "sun",
        10.0,
        plane_normal=plane_normal,
    )
    if earth_radius_m != 10.0:
        earth_track["path"] = {
            "representation": "polyline",
            "space": "target_relative",
            "target_id": "sun",
            "control_points": [
                [10.0, 0.0, 0.0],
                [earth_radius_m, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [0.0, -10.0, 0.0],
            ],
            "closed": True,
            "parameterization": "normalized_time",
        }
    motion_result = toolkit.apply_motion_patch(
        [
            earth_track,
            _orbit_track(
                "moon_orbit",
                "moon",
                "earth",
                2.0,
                cycle_count=moon_cycle_count,
                plane_normal=plane_normal,
            ),
        ],
        [],
    )
    if motion_result["status"] != "ok":
        raise AssertionError(motion_result)
    camera_result = toolkit.apply_camera_patch(
        camera_id="camera_main",
        projection="perspective",
        active=True,
        static={"focal_length_mm": 35.0, "focus_target_id": "sun"},
        tracks=[
            {
                "track_id": "camera_static",
                "type": "transform",
                "time_range_seconds": [0.0, 6.0],
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "value": {"translation_m": list(camera_position)},
                    }
                ],
            }
        ],
        remove_track_ids=[],
    )
    if camera_result["status"] != "ok":
        raise AssertionError(camera_result)
    return toolkit


def _annotated(value: str, source_text: str) -> dict:
    return {"value": value, "source_status": "explicit", "source_text": source_text}


def _unknown() -> dict:
    return {"value": None, "source_status": "unknown", "source_text": None}


if __name__ == "__main__":
    unittest.main()
