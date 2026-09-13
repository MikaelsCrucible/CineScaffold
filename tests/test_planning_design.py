from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from cinescaffold.planning.compiler import compile_scene_ir
from cinescaffold.planning.design import (
    SceneSkeleton,
    skeleton_hash,
    task_capability_slice,
    validate_scene_skeleton,
)
from cinescaffold.planning.domain import PlanningProfile, ValidationReport, Violation
from cinescaffold.planning.duration import attach_duration_resolution, freeze_brief_duration
from cinescaffold.planning.geometry import surface_clearance_ratio
from cinescaffold.planning.models import _mock_scene_skeleton
from cinescaffold.planning.objective import ObjectiveRequirement, project_objective_brief
from cinescaffold.planning.toolkit import ScenePlanningToolkit, _route_anchor_failures
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

    def test_scene_skeleton_allows_unspecified_camera_focus(self) -> None:
        value = _solar_skeleton()
        value["camera_intent"]["focus_target_id"] = None

        skeleton = SceneSkeleton.model_validate(value)

        self.assertIsNone(skeleton.camera_intent.focus_target_id)

    def test_scene_skeleton_rejects_duplicate_ids_and_non_ground_support(self) -> None:
        duplicate = _desert_skeleton()
        duplicate["relations"].append(dict(duplicate["relations"][0]))
        wrong_ground = _desert_skeleton()
        wrong_ground["relations"][0]["reference_id"] = "ship_01"

        with self.assertRaisesRegex(ValidationError, "Relation ID"):
            SceneSkeleton.model_validate(duplicate)
        with self.assertRaisesRegex(ValidationError, "ground_plane"):
            SceneSkeleton.model_validate(wrong_ground)

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

    def test_v06_skeleton_must_keep_every_required_narrative_motion(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        objective = objective.model_copy(
            update={
                "schema_version": "0.6",
                "subject_motion": [
                    {
                        "motion_id": "required_hold",
                        "subject_id": "man_01",
                        "motion_semantics": {"narrative_required": True},
                    }
                ],
            }
        )
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())

        with self.assertRaisesRegex(ValueError, "required_hold"):
            validate_scene_skeleton(objective, skeleton)

    def test_scene_skeleton_rejects_unknown_timeline_event(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        skeleton.relations[0].timeline_event_id = "missing_event"

        with self.assertRaisesRegex(ValueError, "missing_event"):
            validate_scene_skeleton(objective, skeleton)

    def test_static_subject_scene_rejects_changing_skeleton_phase(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        value = _desert_skeleton()
        value["motion_phases"][0].update(
            kind="linear_move",
            direction_mode="world_forward",
            path_family="linear",
            speed_intent="slow",
        )
        skeleton = SceneSkeleton.model_validate(value)

        with self.assertRaisesRegex(ValueError, "scene_dynamics=static"):
            validate_scene_skeleton(objective, skeleton)

    def test_dynamic_scene_rejects_hold_only_skeleton(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        objective = objective.model_copy(
            update={
                "scene_dynamics": {
                    "mode": "dynamic",
                    "source_status": "inferred",
                    "reason": "主体发生局部状态变化",
                }
            }
        )
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())

        with self.assertRaisesRegex(ValueError, "scene_dynamics=dynamic"):
            validate_scene_skeleton(objective, skeleton)

    def test_required_local_interaction_cannot_be_replaced_by_hold(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        objective = objective.model_copy(
            update={
                "schema_version": "0.6",
                "scene_dynamics": {
                    "mode": "dynamic",
                    "source_status": "inferred",
                    "reason": "主体发生局部变化",
                },
                "subject_motion": [
                    {
                        "motion_id": "gesture_01",
                        "subject_id": "man_01",
                        "motion_semantics": {
                            "motion_mode": "local_interaction",
                            "path_type": "stationary",
                            "narrative_required": True,
                        },
                    }
                ],
            }
        )
        value = _desert_skeleton()
        value["motion_phases"][0].update(
            motion_id="gesture_01",
            kind="visibility",
            visibility_state="hidden",
            transition_at="at_end",
        )
        skeleton = SceneSkeleton.model_validate(value)

        with self.assertRaisesRegex(ValueError, "错误表达关键叙事动作"):
            validate_scene_skeleton(objective, skeleton)

        value["motion_phases"][0].update(
            kind="local_transform",
            local_components=["rotation"],
            visibility_state=None,
            transition_at=None,
        )
        skeleton = SceneSkeleton.model_validate(value)
        validate_scene_skeleton(objective, skeleton)

    def test_parabolic_path_family_is_representable(self) -> None:
        value = _desert_skeleton()
        value["motion_phases"][0].update(
            kind="linear_move",
            direction_mode="world_forward",
            path_family="parabolic",
            speed_intent="medium",
        )

        skeleton = SceneSkeleton.model_validate(value)

        self.assertEqual(skeleton.motion_phases[0].path_family, "parabolic")

    def test_deterministic_local_interaction_builds_observable_transform(self) -> None:
        source = _desert_toolkit()
        objective = source.objective_brief.model_copy(
            update={
                "schema_version": "0.6",
                "scene_dynamics": {
                    "mode": "dynamic",
                    "source_status": "inferred",
                    "reason": "主体发生局部变化",
                },
                "subject_motion": [
                    {
                        "motion_id": "gesture_01",
                        "subject_id": "man_01",
                        "action": {
                            "value": "挥手",
                            "source_status": "inferred",
                            "source_text": None,
                        },
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 6.0,
                        "motion_semantics": {
                            "motion_type": "interactive",
                            "motion_mode": "local_interaction",
                            "direction_mode": "none",
                            "target_id": None,
                            "carrier_id": None,
                            "path_type": "stationary",
                            "timeline_event_id": None,
                            "source_status": "inferred",
                            "narrative_required": True,
                        },
                    }
                ],
            }
        )
        toolkit = ScenePlanningToolkit(objective)
        skeleton = _mock_scene_skeleton(objective)

        self.assertEqual(skeleton["motion_phases"][0]["kind"], "local_transform")
        accepted = toolkit.submit_scene_skeleton(skeleton)
        self.assertEqual(accepted["status"], "ok", accepted)
        options = toolkit.request_design_options(max_options=1)
        candidates = [item.candidate for item in toolkit._design_options.values()]
        candidates.extend(
            item.candidate for item in toolkit._design_repair_baselines.values()
        )
        self.assertTrue(candidates, options)
        track = candidates[0].motion_tracks["design_motion_man_01"]
        rotations = [item.value.rotation_quaternion_wxyz for item in track.keyframes]
        self.assertGreater(len(set(rotations)), 1)

    def test_failed_hard_design_exposes_only_a_repair_baseline(self) -> None:
        source = _desert_toolkit()
        first_ref = "content.composition.visual_scales[0].scale"
        second_ref = "content.composition.visual_scales[1].scale"
        objective = source.objective_brief.model_copy(
            update={
                "composition": source.objective_brief.composition
                | {
                    "visual_scales": [
                        {
                            "subject_id": "man_01",
                            "scale": {
                                "value": "1%-2%",
                                "source_status": "explicit",
                                "source_text": "人物占画幅1%-2%",
                            },
                        },
                        {
                            "subject_id": "man_01",
                            "scale": {
                                "value": "80%-90%",
                                "source_status": "explicit",
                                "source_text": "人物占画幅80%-90%",
                            },
                        },
                    ]
                },
                "explicit_requirements": [
                    *source.objective_brief.explicit_requirements,
                    ObjectiveRequirement(path=first_ref, value="1%-2%"),
                    ObjectiveRequirement(path=second_ref, value="80%-90%"),
                ],
            }
        )
        toolkit = ScenePlanningToolkit(objective)
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(max_options=1)

        self.assertEqual(result["data"]["options"], [])
        self.assertEqual(result["data"]["next_tool"], "begin_design_repair")
        baseline = result["data"]["repair_baselines"][0]
        begun = toolkit.begin_design_repair(
            baseline["base_revision"], baseline["baseline_id"]
        )
        self.assertEqual(begun["status"], "ok", begun)
        self.assertEqual(begun["data"]["next_tool"], "apply_candidate_patch")
        self.assertTrue(toolkit.store.get().entities)

    def test_explicit_overhead_view_changes_actual_camera_pitch(self) -> None:
        brief = valid_planning_brief()
        brief["content"]["camera"]["view_angle"] = {
            "value": "垂直俯拍",
            "source_status": "explicit",
            "source_text": "从正上方俯拍",
        }
        objective = project_objective_brief(brief).objective_brief
        resolution = freeze_brief_duration(
            objective.timeline,
            fps_numerator=24,
            fps_denominator=1,
        )
        toolkit = ScenePlanningToolkit(attach_duration_resolution(objective, resolution))
        toolkit.submit_scene_skeleton(_desert_skeleton())

        options = toolkit.request_design_options(max_options=1)

        candidates = [item.candidate for item in toolkit._design_options.values()]
        candidates.extend(
            item.candidate for item in toolkit._design_repair_baselines.values()
        )
        self.assertTrue(candidates, options)
        candidate = candidates[0]
        camera = candidate.camera
        self.assertIsNotNone(camera)
        assert camera is not None
        focus = candidate.entities["man_01"].solved_transform.translation_m
        position = camera.solved_transform.translation_m
        horizontal = math.hypot(position[0] - focus[0], position[1] - focus[1])
        pitch = math.degrees(math.atan2(position[2] - focus[2], horizontal))
        self.assertGreaterEqual(pitch, 65.0)

    def test_orbit_and_lateral_camera_intents_create_real_tracks(self) -> None:
        for movement, text, expected_track in (
            ("orbit", "环绕", "path_follow"),
            ("lateral", "横移", "transform"),
        ):
            with self.subTest(movement=movement):
                source = _desert_toolkit()
                objective = source.objective_brief.model_copy(
                    update={
                        "scene_design": source.objective_brief.scene_design
                        | {"relationships": []},
                        "camera": source.objective_brief.camera
                        | {
                            "movement": source.objective_brief.camera["movement"]
                            | {
                                "type": {
                                    "value": text,
                                    "source_status": "explicit",
                                    "source_text": text,
                                }
                            }
                        },
                        "explicit_requirements": [
                            ObjectiveRequirement(
                                path="content.camera.movement.type", value=text
                            )
                        ],
                    }
                )
                toolkit = ScenePlanningToolkit(objective)
                skeleton = _desert_skeleton()
                skeleton["relations"] = skeleton["relations"][:1]
                skeleton["camera_intent"]["movement"] = movement
                toolkit.submit_scene_skeleton(skeleton)

                options = toolkit.request_design_options(max_options=1)

                self.assertTrue(options["data"]["options"], options)
                candidate = toolkit._design_options[
                    options["data"]["options"][0]["option_id"]
                ].candidate
                self.assertIn(
                    expected_track,
                    {track.type for track in candidate.camera.tracks.values()},
                )

    def test_follow_camera_uses_target_relative_track(self) -> None:
        source = _desert_toolkit()
        objective = source.objective_brief.model_copy(
            update={
                "schema_version": "0.6",
                "scene_dynamics": {
                    "mode": "dynamic",
                    "source_status": "inferred",
                    "reason": "主体位移",
                },
                "scene_design": source.objective_brief.scene_design
                | {"relationships": []},
                "subject_motion": [
                    {
                        "motion_id": "walk_01",
                        "subject_id": "man_01",
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 6.0,
                        "motion_semantics": {
                            "motion_type": "walking",
                            "motion_mode": "self_propelled",
                            "direction_mode": "world_forward",
                            "target_id": None,
                            "carrier_id": None,
                            "path_type": "linear",
                            "timeline_event_id": None,
                            "narrative_required": False,
                        },
                    }
                ],
                "camera": source.objective_brief.camera
                | {
                    "movement": source.objective_brief.camera["movement"]
                    | {
                        "type": {
                            "value": "跟拍",
                            "source_status": "explicit",
                            "source_text": "跟拍人物",
                        }
                    }
                },
                "explicit_requirements": [
                    ObjectiveRequirement(
                        path="content.camera.movement.type", value="跟拍"
                    )
                ],
            }
        )
        toolkit = ScenePlanningToolkit(objective)
        skeleton = _desert_skeleton()
        skeleton["relations"] = skeleton["relations"][:1]
        skeleton["motion_phases"][0].update(
            motion_id="walk_01",
            kind="linear_move",
            direction_mode="world_forward",
            path_family="linear",
            speed_intent="slow",
        )
        skeleton["camera_intent"].update(
            movement="follow",
            speed_intent="match_subject",
        )
        toolkit.submit_scene_skeleton(skeleton)

        options = toolkit.request_design_options(max_options=1)

        self.assertTrue(options["data"]["options"], options)
        candidate = toolkit._design_options[
            options["data"]["options"][0]["option_id"]
        ].candidate
        path = candidate.camera.tracks["design_camera_path"].path
        self.assertEqual(path.space, "target_relative")
        self.assertEqual(path.target_id, "man_01")

    def test_explicit_screen_placement_becomes_axis_specific_hard_constraints(self) -> None:
        source = _desert_toolkit()
        horizontal_ref = "content.composition.screen_placements[0].horizontal"
        vertical_ref = "content.composition.screen_placements[0].vertical"
        objective = source.objective_brief.model_copy(
            update={
                "composition": source.objective_brief.composition
                | {
                    "screen_placements": [
                        {
                            "subject_id": "man_01",
                            "horizontal": {
                                "value": "左侧",
                                "source_status": "explicit",
                                "source_text": "人物在左侧",
                            },
                            "vertical": {
                                "value": "下方",
                                "source_status": "explicit",
                                "source_text": "人物在下方",
                            },
                        }
                    ]
                },
                "explicit_requirements": [
                    *source.objective_brief.explicit_requirements,
                    ObjectiveRequirement(path=horizontal_ref, value="左侧"),
                    ObjectiveRequirement(path=vertical_ref, value="下方"),
                ],
            }
        )
        toolkit = ScenePlanningToolkit(objective)
        toolkit.submit_scene_skeleton(_desert_skeleton())
        result = toolkit.request_design_options(max_options=1)
        candidates = [item.candidate for item in toolkit._design_options.values()]
        candidates.extend(
            item.candidate for item in toolkit._design_repair_baselines.values()
        )
        self.assertTrue(candidates, result)

        constraints = {
            item.source_ref: item
            for item in candidates[0].constraints.values()
            if item.type == "screen_region"
        }
        self.assertEqual(constraints[horizontal_ref].strength, "hard")
        self.assertEqual(constraints[vertical_ref].strength, "hard")

    def test_motion_phase_direction_contract_rejects_accidental_target(self) -> None:
        value = _pickup_skeleton()
        phase = value["motion_phases"][1]
        phase.update(direction_mode="world_forward", target_id=None)

        skeleton = SceneSkeleton.model_validate(value)
        self.assertEqual(skeleton.motion_phases[1].direction_mode, "world_forward")

        phase["direction_mode"] = "none"
        phase["target_id"] = "man"
        with self.assertRaisesRegex(ValidationError, "不接受 target_id"):
            SceneSkeleton.model_validate(value)

    def test_task_capabilities_are_filtered_by_skeleton(self) -> None:
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        result = task_capability_slice(skeleton, PlanningProfile())

        self.assertEqual(result["required_path_families"], [])
        self.assertIn("camera_depth_order", result["required_relation_kinds"])
        self.assertNotIn("constraint_parameter_schemas", result)
        self.assertIn("custom_size_requests", result["size_design"])
        self.assertEqual(
            result["route_planning"]["decision_owner"],
            "planning_agent",
        )
        self.assertFalse(
            result["motion_readability"]["subject_translation_present"]
        )
        self.assertFalse(
            result["motion_readability"][
                "maximize_motion_readability_applicable"
            ]
        )
        self.assertIsNone(
            result["acceptance"][
                "minimum_view_subject_motion_obliqueness_degrees"
            ]
        )
        self.assertEqual(result["next_tool"], "apply_design_option")

    def test_scene_skeleton_rejects_route_anchor_on_unrelated_phase(self) -> None:
        value = _pickup_skeleton()
        value["route_intents"][0]["anchors"][0]["phase_id"] = "man_wait"

        with self.assertRaisesRegex(ValidationError, "linear_move"):
            SceneSkeleton.model_validate(value)

    def test_skeleton_keeps_qualitative_proportion_without_numeric_size(self) -> None:
        value = _desert_skeleton()
        ship = next(item for item in value["entities"] if item["entity_id"] == "ship_01")
        ship["proxy_family"] = "generic_box"
        ship["proportion_intent"] = "flat"
        skeleton = SceneSkeleton.model_validate(value)

        dumped = skeleton.model_dump(mode="json")

        self.assertEqual(skeleton.entities[2].proportion_intent, "flat")
        self.assertNotIn("size_xyz_m", json.dumps(dumped))

    def test_design_options_are_read_only_and_validator_predicted(self) -> None:
        toolkit = _desert_toolkit()
        accepted = toolkit.submit_scene_skeleton(_desert_skeleton())
        result = toolkit.request_design_options(
            preference="preserve_composition",
            max_options=2,
        )

        self.assertEqual(accepted["revision_after"], 0)
        self.assertEqual(result["revision_after"], 0)
        self.assertEqual(toolkit.store.current_revision, 0)
        self.assertEqual(len(result["data"]["options"]), 2)
        self.assertEqual(
            result["data"]["options"][0]["strategy"],
            "preserve_composition",
        )
        self.assertIn(
            "camera_distance_m",
            result["data"]["options"][0]["numeric_envelopes"],
        )
        self.assertIn("hard_pass", result["data"]["options"][0]["predicted"])

    def test_identical_design_candidates_are_exposed_only_once(self) -> None:
        source = _desert_toolkit()
        source.submit_scene_skeleton(_desert_skeleton())
        first = source.request_design_options(max_options=1)
        source_option = source._design_options[first["data"]["options"][0]["option_id"]]

        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        fixed_result = (
            source_option.candidate,
            source_option.numeric_envelopes,
            source_option.assumptions,
        )
        with patch(
            "cinescaffold.planning.toolkit.build_design_candidate",
            return_value=fixed_result,
        ):
            result = toolkit.request_design_options(max_options=3)

        self.assertEqual(len(result["data"]["options"]), 1)
        self.assertEqual(result["data"]["collapsed_duplicate_count"], 1)
        self.assertEqual(len(result["data"]["duplicate_strategies"]), 1)

    def test_apply_design_option_materializes_atomically(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        suggested = toolkit.request_design_options(max_options=1)
        option = suggested["data"]["options"][0]

        applied = toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()

        self.assertEqual(applied["status"], "ok")
        self.assertEqual(applied["revision_after"], 1)
        self.assertEqual(set(state.entities), {"ground", "man_01", "ship_01"})
        self.assertIsNotNone(state.camera)
        self.assertTrue(applied["data"]["prediction_matched"])
        self.assertEqual(
            applied["changes"][0]["operation"],
            "materialize_design",
        )

    def test_design_ground_has_canonical_identity_and_survives_semantic_patch(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        suggested = toolkit.request_design_options(max_options=1)
        option = suggested["data"]["options"][0]
        toolkit.apply_design_option(0, option["option_id"])
        before = toolkit.store.get()
        man = before.entities["man_01"].model_dump(
            mode="json",
            exclude={"solved_transform"},
        )
        man["label"] = "更新后的人物"

        result = toolkit.apply_entity_patch([man], [])
        after = toolkit.store.get()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            set(after.entities["ground"].tags) & {"environment", "ground"},
            {"environment", "ground"},
        )
        self.assertEqual(
            after.entities["man_01"].solved_transform,
            before.entities["man_01"].solved_transform,
        )
        self.assertIn("man_01", result["data"]["preserved_solved_transform_ids"])
        safety = toolkit.validate_candidate(checks=[
            "schema",
            "references",
            "timeline",
            "hierarchy",
            "transforms",
            "camera",
            "rebuildability",
        ])
        self.assertTrue(safety["data"]["hard_pass"], safety["violations"])

    def test_custom_size_request_is_selected_inside_validated_option(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        suggested = toolkit.request_design_options(
            max_options=1,
            custom_size_requests=[
                {
                    "entity_id": "ship_01",
                    "minimum_xyz_m": [30.0, 10.0, 5.0],
                    "maximum_xyz_m": [40.0, 12.0, 6.0],
                    "preferred_xyz_m": [35.0, 11.0, 5.5],
                    "rationale": "大型背景飞船需要宽扁代理体",
                }
            ],
        )
        option = suggested["data"]["options"][0]
        size_range = option["numeric_envelopes"]["entity_size_ranges_m"]["ship_01"]

        applied = toolkit.apply_design_option(0, option["option_id"])
        geometry = toolkit.store.get().entities["ship_01"].proxy

        self.assertEqual(applied["status"], "ok")
        self.assertEqual(geometry.size_xyz_m, (35.0, 11.0, 5.5))
        self.assertEqual(size_range["minimum_xyz"], [30.0, 10.0, 5.0])
        self.assertEqual(size_range["maximum_xyz"], [40.0, 12.0, 6.0])
        self.assertEqual(size_range["selected_xyz"], [35.0, 11.0, 5.5])
        self.assertEqual(suggested["data"]["custom_size_request_count"], 1)

    def test_far_gap_uses_scene_reference_not_large_proxy_extent(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(
            max_options=1,
            custom_size_requests=[
                {
                    "entity_id": "ship_01",
                    "minimum_xyz_m": [60.0, 18.0, 8.0],
                    "maximum_xyz_m": [80.0, 30.0, 12.0],
                    "preferred_xyz_m": [70.0, 24.0, 10.0],
                    "rationale": "验证远景净空来自场景参考系而非巨型代理自身尺寸",
                }
            ],
        )

        self.assertEqual(result["status"], "ok")
        option = result["data"]["options"][0]
        toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()
        ratio, clearance_m, characteristic_extent_m = surface_clearance_ratio(
            state.entities["man_01"].proxy,
            state.entities["man_01"].solved_transform,
            state.entities["ship_01"].proxy,
            state.entities["ship_01"].solved_transform,
            ground_plane=True,
        )
        constraint = next(
            item
            for item in state.constraints.values()
            if item.type == "collision_clearance"
        )
        self.assertAlmostEqual(clearance_m, constraint.parameters.minimum_meters)
        self.assertLessEqual(ratio, 0.5)
        self.assertGreater(characteristic_extent_m, 20.0)
        self.assertLess(
            math.dist(
                state.camera.solved_transform.translation_m,
                state.entities["man_01"].solved_transform.translation_m,
            ),
            100.0,
        )
        self.assertGreaterEqual(state.entities["ground"].proxy.size_xy_m[0], 1000.0)

    def test_far_layout_uses_scene_depth_not_oblique_camera_azimuth(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(
            preference="preserve_composition",
            max_options=1,
        )
        option = result["data"]["options"][0]
        state = toolkit._design_options[option["option_id"]].candidate
        man_position = state.entities["man_01"].solved_transform.translation_m
        ship_position = state.entities["ship_01"].solved_transform.translation_m

        self.assertAlmostEqual(ship_position[0], man_position[0])
        self.assertGreater(ship_position[1], man_position[1])
        self.assertEqual(
            state.entities["ship_01"].solved_transform.rotation_quaternion_wxyz,
            (1.0, 0.0, 0.0, 0.0),
        )

    def test_static_scene_rejects_motion_readability_strategy_and_aligns_scene_depth(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "scene_dynamics": {
                    "mode": "static",
                    "source_status": "inferred",
                    "reason": "只有摄影机推近，主体状态不变",
                }
            }
        )
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(
            preference="maximize_motion_readability",
            max_options=3,
        )

        self.assertEqual(result["data"]["preference_resolution"]["applied"], "balanced")
        self.assertNotIn(
            "maximize_motion_readability",
            [item["strategy"] for item in result["data"]["options"]],
        )
        option = result["data"]["options"][0]
        state = toolkit._design_options[option["option_id"]].candidate
        camera_position = state.camera.solved_transform.translation_m
        focus_position = state.entities["man_01"].solved_transform.translation_m
        yaw = math.degrees(
            math.atan2(
                camera_position[0] - focus_position[0],
                -(camera_position[1] - focus_position[1]),
            )
        )
        self.assertAlmostEqual(yaw, 0.0)
        ship_position = state.entities["ship_01"].solved_transform.translation_m
        self.assertAlmostEqual(camera_position[0], focus_position[0])
        self.assertAlmostEqual(ship_position[0], focus_position[0])
        self.assertLess(camera_position[1], focus_position[1])
        self.assertGreater(ship_position[1], focus_position[1])
        self.assertTrue(
            any("沿规范场景纵深轴" in item for item in option["assumptions"])
        )

    def test_non_explicit_projected_size_does_not_force_camera_retreat(self) -> None:
        for source_status in ("default", "inferred", "explicit"):
            with self.subTest(source_status=source_status):
                toolkit = _desert_toolkit()
                toolkit.objective_brief = toolkit.objective_brief.model_copy(
                    update={
                        "scene_dynamics": {
                            "mode": "static",
                            "source_status": "inferred",
                            "reason": "静态构图回归",
                        },
                        "translation_parameters": _static_desert_translation_parameters(
                            composition_source_status=source_status,
                            major_object_frame_ratio=[0.01, 0.02],
                        ),
                    }
                )
                toolkit.submit_scene_skeleton(_desert_skeleton())

                result = toolkit.request_design_options(max_options=1)
                option = result["data"]["options"][0]
                state = toolkit._design_options[option["option_id"]].candidate
                camera_position = state.camera.solved_transform.translation_m
                focus_position = state.entities["man_01"].solved_transform.translation_m

                self.assertLess(math.dist(camera_position, focus_position), 100.0)
                self.assertEqual(
                    state.constraints[
                        "design_major_object_projected_size"
                    ].source_status,
                    "inferred" if source_status == "explicit" else source_status,
                )
                self.assertIn(
                    "PROJECTED_SIZE_VIOLATED",
                    option["predicted"]["violation_codes"],
                )

    def test_static_camera_fit_keeps_default_height_above_open_ground(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "scene_dynamics": {
                    "mode": "static",
                    "source_status": "inferred",
                    "reason": "主体静止，只有摄影机推近",
                },
                "translation_parameters": _static_desert_translation_parameters(
                    composition_source_status="default",
                    major_object_frame_ratio=[0.15, 0.3],
                ),
            }
        )
        skeleton = _desert_skeleton()
        skeleton["camera_intent"]["focus_target_id"] = None
        toolkit.submit_scene_skeleton(skeleton)

        result = toolkit.request_design_options(max_options=1)
        candidates = [item.candidate for item in toolkit._design_options.values()]
        candidates.extend(
            item.candidate for item in toolkit._design_repair_baselines.values()
        )

        self.assertTrue(candidates, result)
        camera = candidates[0].camera
        self.assertIsNotNone(camera)
        assert camera is not None
        transforms = [camera.solved_transform]
        transforms.extend(
            keyframe.value
            for track in camera.tracks.values()
            if track.type == "transform"
            for keyframe in track.keyframes
        )
        self.assertTrue(
            all(transform.translation_m[2] >= 1.5 for transform in transforms)
        )

    def test_camera_below_environment_ground_is_one_hard_violation(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        result = toolkit.request_design_options(max_options=1)
        option = result["data"]["options"][0]
        state = toolkit._design_options[option["option_id"]].candidate.model_copy(
            deep=True
        )
        assert state.camera is not None
        state.camera.solved_transform.translation_m = (0.0, -10.0, -1.0)
        for track in state.camera.tracks.values():
            if track.type != "transform":
                continue
            for keyframe in track.keyframes:
                keyframe.value.translation_m = (
                    keyframe.value.translation_m[0],
                    keyframe.value.translation_m[1],
                    -1.0,
                )

        report = toolkit._validate(state, ["camera"])
        violations = [
            item for item in report.violations
            if item.code == "CAMERA_GROUND_CLEARANCE_VIOLATED"
        ]

        self.assertFalse(report.hard_pass)
        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0].time_range_seconds,
            (
                0.0,
                state.timeline.duration_seconds
                - state.timeline.fps_denominator / state.timeline.fps_numerator,
            ),
        )

    def test_open_ground_grows_to_cover_a_distant_camera_track(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "scene_dynamics": {
                    "mode": "static",
                    "source_status": "inferred",
                    "reason": "静态构图回归",
                },
                "composition": {
                    "patterns": [],
                    "visual_scales": [
                        {
                            "subject_id": "ship_01",
                            "scale": {
                                "value": "1%-2%",
                                "source_status": "explicit",
                                "source_text": "飞船只占画面 1%-2%",
                            },
                        }
                    ],
                    "visibility_requirements": [],
                },
                "translation_parameters": _static_desert_translation_parameters(
                    composition_source_status="explicit",
                    major_object_frame_ratio=[0.01, 0.02],
                ),
            }
        )
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(max_options=1)
        option = result["data"]["options"][0]
        state = toolkit._design_options[option["option_id"]].candidate
        ground = state.entities["ground"]
        camera_track = next(iter(state.camera.tracks.values()))
        camera_positions = [
            keyframe.value.translation_m for keyframe in camera_track.keyframes
        ]

        self.assertGreaterEqual(ground.proxy.size_xy_m[0], 1000.0)
        self.assertGreaterEqual(ground.proxy.size_xy_m[1], 1000.0)
        self.assertTrue(
            all(
                abs(position[0]) < ground.proxy.size_xy_m[0] / 2.0
                and abs(position[1]) < ground.proxy.size_xy_m[1] / 2.0
                for position in camera_positions
            )
        )

    def test_static_composition_keeps_small_subject_and_major_object_visible(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "scene_dynamics": {
                    "mode": "static",
                    "source_status": "inferred",
                    "reason": "静态构图回归",
                },
                "composition": {
                    "patterns": [],
                    "visual_scales": [
                        {
                            "subject_id": "man_01",
                            "scale": {
                                "value": "画面中小比例",
                                "source_status": "inferred",
                                "source_text": "人物小比例",
                            },
                        },
                        {
                            "subject_id": "ship_01",
                            "scale": {
                                "value": "相对男人巨大",
                                "source_status": "explicit",
                                "source_text": "巨大的飞船",
                            },
                        },
                    ],
                    "visibility_requirements": [],
                },
                "translation_parameters": {
                    "composition": {
                        "subject_frame_ratio": [0.01, 0.05],
                        "major_object_frame_ratio": [0.1, 0.3],
                        "negative_space_ratio": [0.7, 1.0],
                        "source_status": "inferred",
                    }
                },
            }
        )
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(max_options=1)
        option = result["data"]["options"][0]
        state = toolkit._design_options[option["option_id"]].candidate

        self.assertTrue(option["predicted"]["hard_pass"])
        self.assertTrue(option["predicted"]["commit_ready"])
        self.assertLess(option["predicted"]["soft_score"], 1.0)
        self.assertEqual(
            state.constraints["design_major_object_projected_size"].subjects,
            ["ship_01"],
        )
        self.assertIn("design_negative_space", state.constraints)
        self.assertIn("design_composition_presence_ship_01", state.constraints)

    def test_custom_size_request_rejects_unknown_entity(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())

        result = toolkit.request_design_options(
            custom_size_requests=[
                {
                    "entity_id": "missing",
                    "minimum_xyz_m": [1.0, 1.0, 1.0],
                    "maximum_xyz_m": [2.0, 2.0, 2.0],
                    "rationale": "测试未知引用",
                }
            ]
        )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("missing", result["warnings"][0])

    def test_revised_size_request_invalidates_previous_options(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        initial = toolkit.request_design_options(max_options=1)
        initial_option = initial["data"]["options"][0]
        revised = toolkit.request_design_options(
            max_options=1,
            custom_size_requests=[
                {
                    "entity_id": "ship_01",
                    "minimum_xyz_m": [30.0, 10.0, 5.0],
                    "maximum_xyz_m": [40.0, 12.0, 6.0],
                    "rationale": "首批候选比例不足",
                }
            ],
        )
        revised_option = revised["data"]["options"][0]

        stale = toolkit.apply_design_option(0, initial_option["option_id"])
        applied = toolkit.apply_design_option(0, revised_option["option_id"])

        self.assertEqual(stale["status"], "rejected")
        self.assertEqual(applied["status"], "ok")

    def test_duplicate_design_request_is_rejected(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        toolkit.request_design_options(max_options=1)

        duplicate = toolkit.request_design_options(max_options=1)

        self.assertEqual(duplicate["status"], "rejected")
        self.assertIn("重复", duplicate["warnings"][0])

    def test_flat_generic_proxy_is_not_materialized_as_cube(self) -> None:
        value = _desert_skeleton()
        ship = next(item for item in value["entities"] if item["entity_id"] == "ship_01")
        ship["proxy_family"] = "generic_box"
        ship["scale_intent"] = "large"
        ship["proportion_intent"] = "flat"
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(value)
        suggested = toolkit.request_design_options(max_options=1)
        option = suggested["data"]["options"][0]

        toolkit.apply_design_option(0, option["option_id"])
        dimensions = toolkit.store.get().entities["ship_01"].proxy.size_xyz_m

        self.assertGreater(dimensions[0], dimensions[1])
        self.assertGreater(dimensions[1], dimensions[2] * 10.0)

    def test_design_option_is_stale_after_candidate_changes(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        suggested = toolkit.request_design_options(max_options=1)
        option = suggested["data"]["options"][0]
        toolkit.apply_entity_patch(
            [
                {
                    "entity_id": "temporary",
                    "role": "subject",
                    "proxy": {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
                    "source_refs": [],
                    "solved_transform": {},
                }
            ],
            [],
        )

        result = toolkit.apply_design_option(0, option["option_id"])

        self.assertEqual(result["status"], "rejected")
        self.assertIn("revision 0", result["warnings"][0])

    def test_nested_orbit_option_is_commit_ready_and_target_relative(self) -> None:
        toolkit = _example_toolkit("solar_system_10s.json")
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "translation_parameters": {
                    "camera": {
                        "height_m": 1.5,
                        "focal_length_mm": 50.0,
                        "movement": "static",
                        "start_distance_m": 15.0,
                        "end_distance_m": 15.0,
                        "source_status": "default",
                    }
                }
            }
        )
        toolkit.submit_scene_skeleton(_solar_skeleton())

        result = toolkit.request_design_options(
            preference="maximize_motion_readability",
            max_options=1,
        )
        option = result["data"]["options"][0]
        applied = toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()

        self.assertTrue(applied["data"]["commit_ready"], applied["violations"])
        self.assertEqual(
            state.motion_tracks["design_orbit_earth"].path.target_id,
            "sun",
        )
        self.assertEqual(
            state.motion_tracks["design_orbit_moon"].path.target_id,
            "earth",
        )
        self.assertGreater(
            state.motion_tracks["design_orbit_moon"].path.cycle_count,
            state.motion_tracks["design_orbit_earth"].path.cycle_count,
        )
        self.assertGreater(
            state.entities["sun"].proxy.radius_m,
            state.entities["earth"].proxy.radius_m,
        )
        self.assertGreater(
            state.entities["earth"].proxy.radius_m,
            state.entities["moon"].proxy.radius_m,
        )
        earth_orbit = state.motion_tracks["design_orbit_earth"].path.radius_m
        moon_orbit = state.motion_tracks["design_orbit_moon"].path.radius_m
        minimum_moon_sun_clearance = (
            earth_orbit
            - moon_orbit
            - state.entities["sun"].proxy.radius_m
            - state.entities["moon"].proxy.radius_m
        )
        self.assertGreaterEqual(
            minimum_moon_sun_clearance,
            toolkit.profile.orbit_surface_clearance_m,
        )

    def test_unspecified_camera_focus_uses_fixed_scene_anchor(self) -> None:
        toolkit = _example_toolkit("solar_system_10s.json")
        skeleton = _solar_skeleton()
        skeleton["camera_intent"]["focus_target_id"] = None
        toolkit.submit_scene_skeleton(skeleton)
        suggested = toolkit.request_design_options(max_options=1)
        toolkit.apply_design_option(0, suggested["data"]["options"][0]["option_id"])
        state = toolkit.store.get()

        scene_ir = compile_scene_ir(
            toolkit,
            state,
            agent_run_id="fixed_scene_anchor_test",
            trace_ref="test_trace.jsonl",
        )
        camera_samples = scene_ir.camera.state_track.samples

        self.assertIsNone(state.camera.static.focus_target_id)
        self.assertEqual(
            camera_samples[0].value.rotation_quaternion_wxyz,
            camera_samples[-1].value.rotation_quaternion_wxyz,
        )

    def test_static_entity_focus_does_not_become_implicit_tracking(self) -> None:
        toolkit = _example_toolkit("solar_system_10s.json")
        skeleton = _solar_skeleton()
        skeleton["camera_intent"]["focus_target_id"] = "earth"
        toolkit.submit_scene_skeleton(skeleton)
        suggested = toolkit.request_design_options(max_options=1)
        toolkit.apply_design_option(0, suggested["data"]["options"][0]["option_id"])
        scene_ir = compile_scene_ir(
            toolkit,
            toolkit.store.get(),
            agent_run_id="fixed_entity_focus_test",
            trace_ref="test_trace.jsonl",
        )
        camera_samples = scene_ir.camera.state_track.samples

        self.assertEqual(
            camera_samples[0].value.rotation_quaternion_wxyz,
            camera_samples[-1].value.rotation_quaternion_wxyz,
        )

    def test_projected_size_keeps_visual_scale_subject_identity(self) -> None:
        toolkit = _example_toolkit("solar_system_10s.json")
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "composition": {
                    "visual_scales": [
                        {
                            "subject_id": "moon",
                            "scale": {
                                "value": "5%-10%",
                                "source_status": "default",
                                "source_text": None,
                            },
                        }
                    ],
                    "visibility_requirements": [],
                },
                "translation_parameters": {
                    "composition": {
                        "subject_frame_ratio": [0.05, 0.1],
                        "source_status": "default",
                    }
                },
            }
        )
        toolkit.submit_scene_skeleton(_solar_skeleton())
        suggested = toolkit.request_design_options(max_options=1)
        toolkit.apply_design_option(0, suggested["data"]["options"][0]["option_id"])

        constraint = toolkit.store.get().constraints["design_subject_projected_size"]

        self.assertEqual(constraint.subjects, ["moon"])
        self.assertEqual(constraint.parameters.entity_id, "moon")
        self.assertEqual(constraint.parameters.minimum, 0.05)
        self.assertEqual(constraint.parameters.maximum, 0.1)
        self.assertEqual(
            constraint.source_ref,
            "content.composition.visual_scales[0].scale",
        )

    def test_pickup_option_composes_boarding_from_generic_motion(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        toolkit.submit_scene_skeleton(_pickup_skeleton())

        result = toolkit.request_design_options(max_options=1)
        option = result["data"]["options"][0]
        applied = toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()
        car_track = state.motion_tracks["design_motion_car"]
        man_track = state.motion_tracks["design_motion_man"]
        visibility = state.motion_tracks["design_visibility_man"]

        self.assertTrue(applied["data"]["commit_ready"], applied["violations"])
        self.assertEqual(set(state.entities), {"road", "man", "car"})
        self.assertEqual(
            [item.time_seconds for item in car_track.keyframes],
            [0.0, 4.0, 7.0, 11.958333333333334],
        )
        self.assertEqual(
            [item.time_seconds for item in man_track.keyframes],
            [0.0, 4.0, 7.0],
        )
        self.assertEqual(
            man_track.keyframes[0].value.translation_m,
            man_track.keyframes[1].value.translation_m,
        )
        self.assertNotEqual(
            man_track.keyframes[0].value.translation_m,
            man_track.keyframes[-1].value.translation_m,
        )
        self.assertFalse(visibility.keyframes[-1].value)
        self.assertAlmostEqual(visibility.keyframes[-1].time_seconds, 7.0)
        self.assertNotIn("design_carried_man_man_carried", state.motion_tracks)

    def test_motion_phase_uses_narrower_motion_interval_than_shared_event(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        toolkit.objective_brief.subject_motion[1].update(
            start_time_seconds=3.0,
            end_time_seconds=6.0,
        )
        toolkit.objective_brief.subject_motion[2].update(
            start_time_seconds=6.0,
            end_time_seconds=8.0,
        )
        toolkit.objective_brief.subject_motion[3].update(
            start_time_seconds=6.0,
            end_time_seconds=8.0,
        )
        shared_event = {
            "id": "arrival_and_boarding",
            "description": "车辆到达后人物上车",
            "start_time_seconds": 3.0,
            "end_time_seconds": 8.0,
            "reference_ids": ["car", "man"],
            "source_status": "inferred",
        }
        toolkit.objective_brief.timeline["events"].append(shared_event)
        for index in (1, 2):
            toolkit.objective_brief.subject_motion[index]["motion_semantics"] = {
                "action_kind": "approach" if index == 1 else "board",
                "motion_type": "moving" if index == 1 else "interactive",
                "motion_mode": "self_propelled" if index == 1 else "local_interaction",
                "direction_mode": "toward_target",
                "target_id": "man" if index == 1 else "car",
                "carrier_id": None,
                "path_type": "linear" if index == 1 else "stationary",
                "timeline_event_id": "arrival_and_boarding",
                "postconditions": {
                    "contained_by_id": "car" if index == 2 else None,
                    "external_visibility": "hidden" if index == 2 else "unchanged",
                },
                "source_status": "inferred",
            }
        skeleton = _pickup_skeleton()
        skeleton["motion_phases"][1]["timeline_event_id"] = "arrival_and_boarding"
        skeleton["motion_phases"][2]["timeline_event_id"] = "arrival_and_boarding"
        skeleton["motion_phases"][3]["timeline_event_id"] = "arrival_and_boarding"
        skeleton["relations"][1]["timeline_event_id"] = "arrival_and_boarding"
        skeleton["relations"][2]["timeline_event_id"] = "arrival_and_boarding"
        toolkit.submit_scene_skeleton(skeleton)

        options = toolkit.request_design_options(max_options=1)
        toolkit.apply_design_option(0, options["data"]["options"][0]["option_id"])
        state = toolkit.store.get()

        self.assertEqual(
            [item.time_seconds for item in state.motion_tracks["design_motion_car"].keyframes[:2]],
            [3.0, 6.0],
        )
        self.assertEqual(
            [item.time_seconds for item in state.motion_tracks["design_motion_man"].keyframes],
            [0.0, 4.0, 6.0, 8.0],
        )
        codes = {item["code"] for item in toolkit.validate_candidate()["violations"]}
        self.assertNotIn("MOTION_MODE_STATIONARY_VIOLATED", codes)

    def test_explicit_visibility_becomes_hard_keep_in_frame_constraint(self) -> None:
        base = _desert_toolkit()
        visibility_ref = "content.composition.visibility_requirements[0].requirement"
        objective = base.objective_brief.model_copy(
            update={
                "composition": base.objective_brief.composition
                | {
                    "visibility_requirements": [
                        {
                            "subject_id": "ship_01",
                            "requirement": {
                                "value": "远处可见",
                                "source_status": "explicit",
                                "source_text": "远处有巨大的飞船",
                            },
                        }
                    ]
                },
                "explicit_requirements": [
                    *base.objective_brief.explicit_requirements,
                    ObjectiveRequirement(path=visibility_ref, value="远处可见"),
                ],
            }
        )
        toolkit = ScenePlanningToolkit(objective)
        toolkit.submit_scene_skeleton(_desert_skeleton())

        options = toolkit.request_design_options(max_options=1)
        applied = toolkit.apply_design_option(0, options["data"]["options"][0]["option_id"])
        constraint = toolkit.store.get().constraints["design_visibility_ship_01_0"]

        self.assertEqual(applied["status"], "ok")
        self.assertEqual(constraint.type, "keep_in_frame")
        self.assertEqual(constraint.strength, "hard")
        self.assertEqual(constraint.source_ref, visibility_ref)
        self.assertEqual(constraint.parameters.minimum_inside_fraction, 0.01)

    def test_design_handles_late_approach_world_forward_and_carried_binding(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        events = toolkit.objective_brief.timeline["events"]
        events[0].update(start_time_seconds=2.5, end_time_seconds=5.0)
        events[1].update(start_time_seconds=5.0, end_time_seconds=7.5)
        events[2].update(start_time_seconds=7.5, end_time_seconds=12.0)
        semantics = {
            1: {
                "action_kind": "approach",
                "motion_type": "moving",
                "motion_mode": "self_propelled",
                "direction_mode": "toward_target",
                "target_id": "man",
                "carrier_id": None,
                "path_type": "linear",
                "timeline_event_id": "wait_and_arrive",
                "postconditions": {
                    "contained_by_id": None,
                    "external_visibility": "unchanged",
                },
                "source_status": "inferred",
            },
            2: {
                "action_kind": "board",
                "motion_type": "moving",
                "motion_mode": "self_propelled",
                "direction_mode": "toward_target",
                "target_id": "car",
                "carrier_id": None,
                "path_type": "linear",
                "timeline_event_id": "boarding",
                "postconditions": {
                    "contained_by_id": "car",
                    "external_visibility": "hidden",
                },
                "source_status": "inferred",
            },
            4: {
                "action_kind": "depart",
                "motion_type": "moving",
                "motion_mode": "self_propelled",
                "direction_mode": "away_from_target",
                "target_id": "man",
                "carrier_id": None,
                "path_type": "linear",
                "timeline_event_id": "departure",
                "postconditions": {
                    "contained_by_id": None,
                    "external_visibility": "unchanged",
                },
                "source_status": "inferred",
            },
            5: {
                "action_kind": "transport",
                "motion_type": "carried",
                "motion_mode": "carried",
                "direction_mode": "world_forward",
                "target_id": None,
                "carrier_id": "car",
                "path_type": "linear",
                "timeline_event_id": "departure",
                "postconditions": {
                    "contained_by_id": "car",
                    "external_visibility": "hidden",
                },
                "source_status": "inferred",
            },
        }
        for index, value in semantics.items():
            toolkit.objective_brief.subject_motion[index]["motion_semantics"] = value
        skeleton = _pickup_skeleton()
        skeleton["motion_phases"][5]["target_id"] = None
        skeleton["motion_phases"][5]["direction_mode"] = "world_right"
        toolkit.submit_scene_skeleton(skeleton)

        options = toolkit.request_design_options(max_options=1)
        toolkit.apply_design_option(0, options["data"]["options"][0]["option_id"])
        state = toolkit.store.get()
        car_track = state.motion_tracks["design_motion_car"]
        man_position = state.entities["man"].solved_transform.translation_m
        approach_start = car_track.keyframes[0].value.translation_m
        approach_end = car_track.keyframes[1].value.translation_m
        departure_start = car_track.keyframes[2].value.translation_m
        departure_end = car_track.keyframes[3].value.translation_m

        self.assertGreater(
            abs(approach_start[0] - man_position[0]),
            abs(approach_end[0] - man_position[0]),
        )
        self.assertGreater(
            abs(departure_end[0] - departure_start[0])
            + abs(departure_end[1] - departure_start[1]),
            0.5,
        )
        self.assertNotIn("design_carried_man_man_carried", state.motion_tracks)
        codes = {item["code"] for item in toolkit.validate_candidate()["violations"]}
        self.assertNotIn("MOTION_DIRECTION_SEMANTICS_UNMET", codes)
        self.assertNotIn("MOTION_POSTCONDITION_CONTAINMENT_UNMET", codes)
        self.assertNotIn("CARRIED_SUBJECT_UNBOUND", codes)

    def test_direction_unspecified_pickup_uses_one_route_across_stop(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        events = toolkit.objective_brief.timeline["events"]
        events[0].update(start_time_seconds=0.0, end_time_seconds=4.0)
        events[1].update(start_time_seconds=4.0, end_time_seconds=7.0)
        events[2].update(start_time_seconds=7.0, end_time_seconds=12.0)
        for index in (1, 4):
            toolkit.objective_brief.subject_motion[index]["motion_semantics"] = {
                "action_kind": "locomotion",
                "motion_type": "moving",
                "motion_mode": "self_propelled",
                "direction_mode": "none",
                "target_id": None,
                "carrier_id": None,
                "path_type": "linear",
                "timeline_event_id": "wait_and_arrive" if index == 1 else "departure",
                "postconditions": {
                    "contained_by_id": None,
                    "external_visibility": "unchanged",
                },
                "source_status": "inferred",
            }

        skeleton = _pickup_skeleton()
        skeleton["motion_phases"][1].update(direction_mode="none", target_id=None)
        skeleton["motion_phases"][5].update(direction_mode="none", target_id=None)
        # 推断的进入用显隐和后续载运表达，不虚构朝载体移动。
        del skeleton["motion_phases"][2]
        toolkit.submit_scene_skeleton(skeleton)

        options = toolkit.request_design_options(max_options=1)
        self.assertEqual(len(options["data"]["options"]), 1, options)
        toolkit.apply_design_option(0, options["data"]["options"][0]["option_id"])
        state = toolkit.store.get()
        car_track = state.motion_tracks["design_motion_car"]
        points = [item.value.translation_m for item in car_track.keyframes]
        approach = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        departure = (points[3][0] - points[2][0], points[3][1] - points[2][1])
        self.assertGreater(approach[0] * departure[0] + approach[1] * departure[1], 0.0)
        man = state.entities["man"].solved_transform.translation_m
        self.assertLessEqual(
            ((points[1][0] - man[0]) ** 2 + (points[1][1] - man[1]) ** 2) ** 0.5,
            3.0,
        )

    def test_route_anchor_can_bind_motion_to_a_later_interaction_event(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        skeleton = _pickup_skeleton()
        skeleton["relations"][1].update(
            timeline_event_id="boarding",
        )
        skeleton["motion_phases"][1].update(
            direction_mode="none",
            target_id=None,
        )
        skeleton["motion_phases"][5].update(
            direction_mode="none",
            target_id=None,
        )
        skeleton["route_intents"][0]["continuity"] = "preserve_direction"
        del skeleton["motion_phases"][2]
        accepted = toolkit.submit_scene_skeleton(skeleton)

        self.assertEqual(accepted["data"]["route_intent_count"], 1)
        options = toolkit.request_design_options(max_options=1)
        self.assertEqual(len(options["data"]["options"]), 1, options)
        option = options["data"]["options"][0]
        anchored_violation = "skeleton_car_stops_beside_man"
        toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()
        self.assertNotIn(
            anchored_violation,
            {item.constraint_id for item in state.validation.violations},
        )
        car_track = state.motion_tracks["design_motion_car"]
        points = [item.value.translation_m for item in car_track.keyframes]
        man = state.entities["man"].solved_transform.translation_m

        self.assertLessEqual(
            ((points[1][0] - man[0]) ** 2 + (points[1][1] - man[1]) ** 2) ** 0.5,
            3.0,
        )
        approach = (points[1][0] - points[0][0], points[1][1] - points[0][1])
        departure = (points[3][0] - points[2][0], points[3][1] - points[2][1])
        self.assertGreater(approach[0] * departure[0] + approach[1] * departure[1], 0.0)

    def test_route_can_take_its_axis_from_an_existing_reference_entity(self) -> None:
        source = _example_toolkit("roadside_pickup_12s.json")
        objective = source.objective_brief.model_copy(
            deep=True,
            update={
                "translation_parameters": {
                    "scene": {"dimensions_m": [20.0, 100.0]}
                }
            },
        )
        toolkit = ScenePlanningToolkit(objective)
        skeleton = _pickup_skeleton()
        skeleton["relations"][1].update(timeline_event_id="boarding")
        skeleton["motion_phases"][1].update(direction_mode="none", target_id=None)
        skeleton["motion_phases"][5].update(direction_mode="none", target_id=None)
        skeleton["route_intents"][0].update(
            axis_reference_id="road",
            continuity="preserve_direction",
        )
        del skeleton["motion_phases"][2]
        toolkit.submit_scene_skeleton(skeleton)

        options = toolkit.request_design_options(max_options=1)
        self.assertEqual(len(options["data"]["options"]), 1, options)
        toolkit.apply_design_option(0, options["data"]["options"][0]["option_id"])
        points = [
            item.value.translation_m
            for item in toolkit.store.get()
            .motion_tracks["design_motion_car"]
            .keyframes
        ]
        approach = (points[1][0] - points[0][0], points[1][1] - points[0][1])

        self.assertGreater(abs(approach[1]), abs(approach[0]))

    def test_declared_route_anchor_rejects_even_a_soft_relation_violation(self) -> None:
        skeleton = SceneSkeleton.model_validate(_pickup_skeleton())
        report = ValidationReport(
            revision=1,
            hard_pass=True,
            soft_score=0.99,
            checks=["motion"],
            violations=[
                Violation(
                    id="violation_1",
                    code="DISTANCE_RANGE_VIOLATED",
                    severity="soft",
                    constraint_id="skeleton_car_stops_beside_man",
                    entity_ids=["car", "man"],
                    message="declared route waypoint was missed",
                )
            ],
        )

        failures = _route_anchor_failures(skeleton, report)

        self.assertEqual([item.id for item in failures], ["violation_1"])

    def test_scene_skeleton_rejects_special_board_motion_kind(self) -> None:
        value = _pickup_skeleton()
        value["motion_phases"][2]["kind"] = "board"

        with self.assertRaises(ValidationError):
            SceneSkeleton.model_validate(value)

    def test_mock_decomposes_board_state_without_collision_path(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        objective = toolkit.objective_brief.model_copy(deep=True)
        objective.subject_motion[2]["motion_semantics"] = {
            "action_kind": "board",
            "motion_type": "walking",
            "motion_mode": "self_propelled",
            "target_id": "car",
            "carrier_id": None,
            "path_type": "linear",
            "direction_mode": "toward_target",
            "timeline_event_id": "boarding",
            "postconditions": {
                "contained_by_id": "car",
                "external_visibility": "hidden",
            },
            "source_status": "explicit",
        }
        skeleton = _mock_scene_skeleton(objective)
        phases = skeleton["motion_phases"]

        self.assertEqual(
            {item["entity_id"] for item in skeleton["entities"]},
            {"environment_ground", "man", "car"},
        )
        self.assertNotIn("board", {item["kind"] for item in phases})
        self.assertTrue(
            any(
                item["subject_id"] == "man"
                and item["kind"] == "hold"
                and item["timeline_event_id"] == "boarding"
                and item["target_id"] is None
                for item in phases
            )
        )
        self.assertTrue(
            any(
                item["subject_id"] == "man"
                and item["kind"] == "visibility"
                and item["visibility_state"] == "hidden"
                for item in phases
            )
        )


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


def _desert_toolkit() -> ScenePlanningToolkit:
    objective = project_objective_brief(valid_planning_brief()).objective_brief
    resolution = freeze_brief_duration(
        objective.timeline,
        fps_numerator=24,
        fps_denominator=1,
    )
    return ScenePlanningToolkit(attach_duration_resolution(objective, resolution))


def _static_desert_translation_parameters(
    *,
    composition_source_status: str,
    major_object_frame_ratio: list[float],
) -> dict:
    return {
        "scene": {
            "asset_key": "desert",
            "dimensions_m": [200.0, 200.0],
            "source_status": "inferred",
        },
        "camera": {
            "height_m": 1.5,
            "focal_length_mm": 50.0,
            "movement": "push_in",
            "start_distance_m": 15.0,
            "end_distance_m": 5.0,
            "source_status": "default",
        },
        "composition": {
            "subject_frame_ratio": [0.05, 0.1],
            "major_object_frame_ratio": major_object_frame_ratio,
            "negative_space_ratio": [0.6, 0.7],
            "source_status": composition_source_status,
        },
    }


def _example_toolkit(filename: str) -> ScenePlanningToolkit:
    root = Path(__file__).resolve().parents[1]
    brief = json.loads(
        (root / "examples" / "cinematic_briefs" / filename).read_text(encoding="utf-8")
    )
    objective = project_objective_brief(brief).objective_brief
    resolution = freeze_brief_duration(
        objective.timeline,
        fps_numerator=24,
        fps_denominator=1,
    )
    return ScenePlanningToolkit(attach_duration_resolution(objective, resolution))


def _solar_skeleton() -> dict:
    return {
        "entities": [
            _symbolic_entity("sun", "sun", "center", "celestial_sphere", "large", 0),
            _symbolic_entity("earth", "earth", "planet", "celestial_sphere", "small", 1),
            _symbolic_entity("moon", "moon", "satellite", "celestial_sphere", "tiny", 2),
        ],
        "relations": [
            _symbolic_relation("earth_sun", "orbit_around", "earth", "sun", 0),
            _symbolic_relation("moon_earth", "orbit_around", "moon", "earth", 1),
        ],
        "motion_phases": [
            _symbolic_phase("sun_hold", "sun", "hold", 0, status="inferred"),
            _symbolic_phase(
                "earth_orbit", "earth", "orbit", 1, target_id="sun", path="circle"
            ),
            _symbolic_phase(
                "moon_orbit", "moon", "orbit", 2, target_id="earth", path="circle"
            ),
        ],
        "camera_intent": _symbolic_camera("sun"),
    }


def _pickup_skeleton() -> dict:
    return {
        "entities": [
            {
                "entity_id": "road",
                "semantic_type": "road",
                "role": "environment",
                "proxy_family": "ground_plane",
                "scale_intent": "large",
                "source_refs": [],
            },
            _symbolic_entity("man", "human", "passenger", "human_capsule", "human", 0),
            _symbolic_entity("car", "car", "vehicle", "vehicle_box", "large", 1),
        ],
        "relations": [
            {
                "relation_id": "man_road",
                "kind": "ground_support",
                "subject_id": "man",
                "reference_id": "road",
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[0]",
            },
            {
                "relation_id": "car_stops_beside_man",
                "kind": "proximity",
                "subject_id": "car",
                "reference_id": "man",
                "timeline_event_id": "wait_and_arrive",
                "temporal_mode": "at_end",
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[1]",
            },
            {
                "relation_id": "man_boards_car",
                "kind": "proximity",
                "subject_id": "man",
                "reference_id": "car",
                "timeline_event_id": "boarding",
                "temporal_mode": "at_end",
                "source_status": "explicit",
                "source_ref": "content.scene_design.relationships[2]",
            },
        ],
        "motion_phases": [
            _symbolic_phase("man_wait", "man", "hold", 0, event="wait_and_arrive"),
            _symbolic_phase(
                "car_arrive",
                "car",
                "linear_move",
                1,
                event="wait_and_arrive",
                target_id="man",
                direction="toward_target",
                path="linear",
            ),
            _symbolic_phase(
                "man_approach_car",
                "man",
                "linear_move",
                2,
                event="boarding",
                target_id="car",
                direction="toward_target",
                path="linear",
            ),
            {
                **_symbolic_phase(
                    "man_hidden_after_entry",
                    "man",
                    "visibility",
                    2,
                    event="boarding",
                ),
                "visibility_state": "hidden",
                "transition_at": "at_end",
            },
            _symbolic_phase(
                "car_wait", "car", "hold", 3, event="boarding", status="inferred"
            ),
            _symbolic_phase(
                "car_depart",
                "car",
                "linear_move",
                4,
                event="departure",
                target_id="man",
                direction="away_from_target",
                path="linear",
            ),
            _symbolic_phase(
                "man_carried",
                "man",
                "carried",
                5,
                event="departure",
                carrier_id="car",
                status="inferred",
            ),
        ],
        "route_intents": [
            {
                "route_id": "car_route",
                "subject_id": "car",
                "anchors": [
                    {
                        "anchor_id": "car_arrival_waypoint",
                        "phase_id": "car_arrive",
                        "boundary": "at_end",
                        "relation_id": "car_stops_beside_man",
                    }
                ],
                "continuity": "allow_turns",
            }
        ],
        "camera_intent": _symbolic_camera("car"),
    }


def _symbolic_entity(
    entity_id: str,
    semantic_type: str,
    role: str,
    proxy_family: str,
    scale_intent: str,
    subject_index: int,
) -> dict:
    return {
        "entity_id": entity_id,
        "semantic_type": semantic_type,
        "role": role,
        "proxy_family": proxy_family,
        "scale_intent": scale_intent,
        "source_refs": [f"content.subjects[{subject_index}].category"],
    }


def _symbolic_relation(
    relation_id: str,
    kind: str,
    subject_id: str,
    reference_id: str,
    relation_index: int,
) -> dict:
    return {
        "relation_id": relation_id,
        "kind": kind,
        "subject_id": subject_id,
        "reference_id": reference_id,
        "source_status": "explicit",
        "source_ref": f"content.scene_design.relationships[{relation_index}]",
    }


def _symbolic_phase(
    phase_id: str,
    subject_id: str,
    kind: str,
    motion_index: int,
    *,
    event: str | None = None,
    target_id: str | None = None,
    carrier_id: str | None = None,
    direction: str = "none",
    path: str = "stationary",
    status: str = "explicit",
) -> dict:
    return {
        "phase_id": phase_id,
        "subject_id": subject_id,
        "kind": kind,
        "timeline_event_id": event,
        "target_id": target_id,
        "carrier_id": carrier_id,
        "direction_mode": "orbit_around" if kind == "orbit" else direction,
        "path_family": path,
        "source_status": status,
        "source_ref": f"content.subject_motion[{motion_index}].action",
    }


def _symbolic_camera(focus_target_id: str) -> dict:
    return {
        "movement": "static",
        "focus_target_id": focus_target_id,
        "view_relation_to_motion": "unspecified",
        "source_status": "inferred",
        "source_ref": "content.camera.movement.type",
    }


if __name__ == "__main__":
    unittest.main()
