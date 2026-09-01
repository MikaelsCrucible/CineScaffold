from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from cinescaffold.planning.design import (
    SceneSkeleton,
    skeleton_hash,
    task_capability_slice,
    validate_scene_skeleton,
)
from cinescaffold.planning.domain import PlanningProfile
from cinescaffold.planning.duration import attach_duration_resolution, freeze_brief_duration
from cinescaffold.planning.models import _mock_scene_skeleton
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.planning.toolkit import ScenePlanningToolkit
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

    def test_scene_skeleton_rejects_unknown_timeline_event(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        skeleton.relations[0].timeline_event_id = "missing_event"

        with self.assertRaisesRegex(ValueError, "missing_event"):
            validate_scene_skeleton(objective, skeleton)

    def test_task_capabilities_are_filtered_by_skeleton(self) -> None:
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        result = task_capability_slice(skeleton, PlanningProfile())

        self.assertEqual(result["required_path_families"], [])
        self.assertIn("camera_depth_order", result["required_relation_kinds"])
        self.assertNotIn("constraint_parameter_schemas", result)
        self.assertIn("custom_size_requests", result["size_design"])
        self.assertEqual(result["next_tool"], "apply_design_option")

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

    def test_custom_size_request_is_selected_inside_validated_option(self) -> None:
        toolkit = _desert_toolkit()
        toolkit.submit_scene_skeleton(_desert_skeleton())
        suggested = toolkit.request_design_options(
            max_options=1,
            custom_size_requests=[
                {
                    "entity_id": "ship_01",
                    "minimum_xyz_m": [60.0, 18.0, 8.0],
                    "maximum_xyz_m": [80.0, 30.0, 12.0],
                    "preferred_xyz_m": [70.0, 24.0, 10.0],
                    "rationale": "大型背景飞船需要宽扁代理体",
                }
            ],
        )
        option = suggested["data"]["options"][0]
        size_range = option["numeric_envelopes"]["entity_size_ranges_m"]["ship_01"]

        applied = toolkit.apply_design_option(0, option["option_id"])
        geometry = toolkit.store.get().entities["ship_01"].proxy

        self.assertEqual(applied["status"], "ok")
        self.assertEqual(geometry.size_xyz_m, (70.0, 24.0, 10.0))
        self.assertEqual(size_range["minimum_xyz"], [60.0, 18.0, 8.0])
        self.assertEqual(size_range["maximum_xyz"], [80.0, 30.0, 12.0])
        self.assertEqual(size_range["selected_xyz"], [70.0, 24.0, 10.0])
        self.assertEqual(suggested["data"]["custom_size_request_count"], 1)

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
                    "minimum_xyz_m": [60.0, 18.0, 8.0],
                    "maximum_xyz_m": [80.0, 30.0, 12.0],
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
        carried = state.motion_tracks["design_carried_man_man_carried"]

        self.assertTrue(applied["data"]["commit_ready"], applied["violations"])
        self.assertEqual(set(state.entities), {"road", "man", "car"})
        self.assertEqual(
            [item.time_seconds for item in car_track.keyframes],
            [0.0, 4.0, 7.0, 11.958333333333334],
        )
        self.assertEqual(
            [item.time_seconds for item in man_track.keyframes],
            [4.0, 7.0],
        )
        self.assertNotEqual(
            man_track.keyframes[0].value.translation_m,
            man_track.keyframes[-1].value.translation_m,
        )
        self.assertFalse(visibility.keyframes[-1].value)
        self.assertAlmostEqual(visibility.keyframes[-1].time_seconds, 7.0)
        self.assertEqual(carried.path.space, "target_relative")
        self.assertEqual(carried.path.target_id, "car")

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
                "direction_mode": "world_forward",
                "target_id": None,
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
        skeleton["motion_phases"][4]["target_id"] = None
        skeleton["motion_phases"][4]["direction_mode"] = "screen_left_to_right"
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
        carried = state.motion_tracks["design_carried_man_man_carried"]

        self.assertGreater(
            abs(approach_start[0] - man_position[0]),
            abs(approach_end[0] - man_position[0]),
        )
        self.assertLess(departure_end[1], departure_start[1])
        self.assertEqual(carried.path.target_id, "car")
        codes = {item["code"] for item in toolkit.validate_candidate()["violations"]}
        self.assertNotIn("MOTION_DIRECTION_SEMANTICS_UNMET", codes)
        self.assertNotIn("MOTION_POSTCONDITION_CONTAINMENT_UNMET", codes)
        self.assertNotIn("CARRIED_SUBJECT_UNBOUND", codes)

    def test_scene_skeleton_rejects_special_board_motion_kind(self) -> None:
        value = _pickup_skeleton()
        value["motion_phases"][2]["kind"] = "board"

        with self.assertRaises(ValidationError):
            SceneSkeleton.model_validate(value)

    def test_mock_decomposes_board_semantics_without_extra_entity(self) -> None:
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
                and item["kind"] == "linear_move"
                and item["timeline_event_id"] == "boarding"
                and item["target_id"] == "car"
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
