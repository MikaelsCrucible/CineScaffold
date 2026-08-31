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

    def test_task_capabilities_are_filtered_by_skeleton(self) -> None:
        skeleton = SceneSkeleton.model_validate(_desert_skeleton())
        result = task_capability_slice(skeleton, PlanningProfile())

        self.assertEqual(result["required_path_families"], [])
        self.assertIn("camera_depth_order", result["required_relation_kinds"])
        self.assertNotIn("constraint_parameter_schemas", result)
        self.assertEqual(result["next_tool"], "apply_design_option")

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

    def test_pickup_option_preserves_arrive_wait_board_depart_sequence(self) -> None:
        toolkit = _example_toolkit("roadside_pickup_12s.json")
        toolkit.submit_scene_skeleton(_pickup_skeleton())

        result = toolkit.request_design_options(max_options=1)
        option = result["data"]["options"][0]
        applied = toolkit.apply_design_option(0, option["option_id"])
        state = toolkit.store.get()
        car_track = state.motion_tracks["design_motion_car"]
        visibility = state.motion_tracks["design_visibility_man"]

        self.assertTrue(applied["data"]["commit_ready"], applied["violations"])
        self.assertEqual(
            [item.time_seconds for item in car_track.keyframes],
            [0.0, 4.0, 7.0, 11.958333333333334],
        )
        self.assertFalse(visibility.keyframes[-1].value)
        self.assertAlmostEqual(visibility.keyframes[-1].time_seconds, 7.0)


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
                "man_board",
                "man",
                "board",
                2,
                event="boarding",
                target_id="car",
            ),
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
