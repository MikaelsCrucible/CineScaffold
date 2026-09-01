from __future__ import annotations

import ast
import unittest
from pathlib import Path

from cinescaffold.planning.compiler import (
    _validate_compiled_equivalence,
    compile_scene_ir,
)
from cinescaffold.planning.duration import attach_duration_resolution, freeze_brief_duration
from cinescaffold.planning.objective import (
    BriefSourceMetadata,
    ObjectivePlanningBrief,
    ObjectiveRequirement,
)
from cinescaffold.planning.toolkit import ScenePlanningToolkit
from tests.test_planning_toolkit import _solved_toolkit


class ValidationGeneralityTest(unittest.TestCase):
    def test_validation_layers_do_not_reference_experiment_identifiers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        production_files = [
            root / "src/cinescaffold/planning/toolkit.py",
            root / "src/cinescaffold/planning/compiler.py",
            root / "src/cinescaffold/execution/validation.py",
        ]
        forbidden = {
            "man_01",
            "ship_01",
            "sun",
            "earth",
            "moon",
            "man",
            "ship",
            "car",
            "road",
            "wait_and_arrive",
            "boarding",
            "roadside_pickup",
        }

        for path in production_files:
            content = path.read_text(encoding="utf-8")
            literals = {
                node.value
                for node in ast.walk(ast.parse(content))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            for identifier in forbidden:
                self.assertNotIn(identifier, literals, f"{path.name}: {identifier}")

    def test_typed_direction_validation_is_identifier_invariant(self) -> None:
        first = _typed_motion_toolkit("scout", "beacon", end_x=-5.0)
        second = _typed_motion_toolkit("courier", "monument", end_x=-5.0)

        first_codes = _hard_codes(first.validate_candidate(checks=["motion"]))
        second_codes = _hard_codes(second.validate_candidate(checks=["motion"]))

        self.assertEqual(first_codes, second_codes)
        self.assertIn("MOTION_DIRECTION_SEMANTICS_UNMET", first_codes)

    def test_stationary_semantics_reject_mid_interval_detour(self) -> None:
        toolkit = _typed_motion_toolkit(
            "performer",
            "marker",
            end_x=0.0,
            motion_mode="stationary",
            motion_type="static",
            direction_mode="none",
            path_type="stationary",
            intermediate_x=3.0,
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["motion"]))

        self.assertIn("MOTION_MODE_STATIONARY_VIOLATED", codes)

    def test_unseen_s_curve_requires_matching_path_family(self) -> None:
        toolkit = _typed_motion_toolkit(
            "drone",
            "tower",
            end_x=5.0,
            direction_mode="none",
            path_type="s_curve",
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["motion"]))

        self.assertIn("MOTION_PATH_FAMILY_UNMET", codes)

    def test_unseen_s_curve_accepts_generic_catmull_rom_path(self) -> None:
        toolkit = _typed_motion_toolkit(
            "probe",
            "station",
            end_x=5.0,
            direction_mode="none",
            path_type="s_curve",
        )
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "probe_s_curve",
                    "target_entity_id": "probe",
                    "type": "path_follow",
                    "time_range_seconds": [0.0, 6.0],
                    "path": {
                        "representation": "catmull_rom",
                        "space": "world",
                        "control_points": [
                            [0.0, 0.0, 0.5],
                            [2.0, 2.0, 0.5],
                            [4.0, -2.0, 0.5],
                            [6.0, 0.0, 0.5],
                        ],
                    },
                }
            ],
            ["probe_motion"],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["motion"]))

        self.assertNotIn("MOTION_PATH_FAMILY_UNMET", codes)
        self.assertNotIn("SELF_PROPELLED_MOTION_MISSING", codes)

    def test_straight_catmull_rom_cannot_impersonate_s_curve(self) -> None:
        toolkit = _typed_motion_toolkit(
            "sensor",
            "dock",
            end_x=5.0,
            direction_mode="none",
            path_type="s_curve",
        )
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "sensor_fake_s_curve",
                    "target_entity_id": "sensor",
                    "type": "path_follow",
                    "time_range_seconds": [0.0, 6.0],
                    "path": {
                        "representation": "catmull_rom",
                        "space": "world",
                        "control_points": [
                            [0.0, 0.0, 0.5],
                            [2.0, 0.0, 0.5],
                            [4.0, 0.0, 0.5],
                            [6.0, 0.0, 0.5],
                        ],
                    },
                }
            ],
            ["sensor_motion"],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["motion"]))

        self.assertIn("MOTION_PATH_FAMILY_UNMET", codes)

    def test_subject_source_ref_cannot_be_attached_to_another_entity(self) -> None:
        toolkit = _typed_motion_toolkit("actor", "statue", end_x=2.0)
        source_ref = "content.subjects[0].category"
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "explicit_requirements": [
                    ObjectiveRequirement(path=source_ref, value="actor")
                ]
            }
        )

        def require_source(state):
            state.required_source_refs = [source_ref]
            return ([{"operation": "replace", "path": "required_source_refs"}], [])

        toolkit.store.apply(require_source)
        toolkit.apply_entity_patch(
            [
                _entity("actor", x=0.0),
                _entity("statue", x=10.0) | {"source_refs": [source_ref]},
            ],
            [],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["hard_semantics"]))

        self.assertIn("EXPLICIT_REQUIREMENT_ENTITY_BINDING_MISMATCH", codes)

    def test_relationship_source_ref_cannot_validate_the_wrong_pair(self) -> None:
        toolkit = _typed_motion_toolkit("traveler", "gate", end_x=2.0)
        source_ref = "content.scene_design.relationships[0]"
        toolkit.objective_brief = toolkit.objective_brief.model_copy(
            update={
                "subjects": [
                    {"id": "traveler"},
                    {"id": "gate"},
                    {"id": "tree"},
                ],
                "scene_design": {
                    "relationships": [
                        {
                            "type": "near",
                            "subject_id": "traveler",
                            "reference_id": "gate",
                            "source_status": "explicit",
                        }
                    ]
                },
                "explicit_requirements": [
                    ObjectiveRequirement(path=source_ref, value={"type": "near"})
                ],
            }
        )

        def require_source(state):
            state.required_source_refs = [source_ref]
            return ([{"operation": "replace", "path": "required_source_refs"}], [])

        toolkit.store.apply(require_source)
        toolkit.apply_entity_patch([_entity("tree", x=12.0)], [])
        toolkit.apply_constraint_patch(
            [
                {
                    "constraint_id": "wrong_near_pair",
                    "type": "distance_range",
                    "strength": "hard",
                    "subjects": ["gate", "tree"],
                    "time_range_seconds": [0.0, 6.0],
                    "parameters": {
                        "entity_ids": ["gate", "tree"],
                        "minimum_meters": 0.0,
                        "maximum_meters": 5.0,
                    },
                    "source_status": "explicit",
                    "source_ref": source_ref,
                }
            ],
            [],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["hard_semantics"]))

        self.assertIn("EXPLICIT_REQUIREMENT_ENTITY_BINDING_MISMATCH", codes)

    def test_generic_containment_cannot_be_faked_by_hiding_subject(self) -> None:
        toolkit = _typed_motion_toolkit(
            "parcel",
            "locker",
            end_x=8.5,
            direction_mode="none",
            contained_by_id="locker",
            external_visibility="hidden",
        )
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "parcel_visibility",
                    "target_entity_id": "parcel",
                    "type": "visibility",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": True, "interpolation": "step"},
                        {
                            "time_seconds": 143 / 24,
                            "value": False,
                            "interpolation": "step",
                        },
                    ],
                }
            ],
            [],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["motion"]))

        self.assertNotIn("MOTION_POSTCONDITION_VISIBILITY_UNMET", codes)
        self.assertIn("MOTION_POSTCONDITION_CONTAINMENT_UNMET", codes)

    def test_ground_validation_checks_every_frozen_frame(self) -> None:
        toolkit = _typed_motion_toolkit("parcel", "marker", end_x=0.0)
        toolkit.apply_entity_patch(
            [
                {
                    "entity_id": "platform",
                    "role": "environment",
                    "proxy": {"type": "plane", "size_xy_m": [20.0, 20.0]},
                    "tags": ["environment"],
                    "solved_transform": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0],
                    },
                },
                _entity(
                    "parcel",
                    x=0.0,
                    ground_entity_id="platform",
                ),
            ],
            [],
        )
        toolkit.apply_motion_patch(
            [
                {
                    "track_id": "parcel_motion",
                    "target_entity_id": "parcel",
                    "type": "transform",
                    "time_range_seconds": [0.0, 6.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": {"translation_m": [0.0, 0.0, 0.5]}},
                        {"time_seconds": 1.5, "value": {"translation_m": [0.0, 0.0, -1.0]}},
                        {"time_seconds": 3.0, "value": {"translation_m": [0.0, 0.0, 0.5]}},
                        {"time_seconds": 143 / 24, "value": {"translation_m": [0.0, 0.0, 0.5]}},
                    ],
                }
            ],
            [],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["transforms"]))

        self.assertIn("ENTITY_INTERSECTS_GROUND", codes)

    def test_unsupported_sloped_ground_is_rejected_instead_of_skipped(self) -> None:
        toolkit = _typed_motion_toolkit("crate", "marker", end_x=1.0)
        toolkit.apply_entity_patch(
            [
                {
                    "entity_id": "slope",
                    "role": "environment",
                    "proxy": {"type": "plane", "size_xy_m": [20.0, 20.0]},
                    "tags": ["environment"],
                    "solved_transform": {
                        "translation_m": [0.0, 0.0, 0.0],
                        "rotation_quaternion_wxyz": [
                            0.9238795325,
                            0.3826834324,
                            0.0,
                            0.0,
                        ],
                        "scale": [1.0, 1.0, 1.0],
                    },
                },
                _entity("crate", x=0.0, ground_entity_id="slope"),
            ],
            [],
        )

        codes = _hard_codes(toolkit.validate_candidate(checks=["transforms"]))

        self.assertIn("GROUND_ORIENTATION_UNSUPPORTED", codes)

    def test_compiled_scene_ir_must_match_candidate_per_frame(self) -> None:
        toolkit = _solved_toolkit()
        state = toolkit.store.get()
        scene_ir = compile_scene_ir(
            toolkit,
            state,
            agent_run_id="generality_audit",
            trace_ref="trace.jsonl",
        )
        _validate_compiled_equivalence(scene_ir, state, toolkit.profile)
        tampered = scene_ir.model_copy(deep=True)
        first_entity = tampered.entities[0]
        first_sample = first_entity.local_state_track.samples[0]
        changed = first_sample.value.model_copy(
            update={
                "translation_m": (
                    first_sample.value.translation_m[0] + 1.0,
                    first_sample.value.translation_m[1],
                    first_sample.value.translation_m[2],
                )
            }
        )
        first_entity.local_state_track.samples[0] = first_sample.model_copy(
            update={"value": changed}
        )

        with self.assertRaisesRegex(ValueError, "不等价"):
            _validate_compiled_equivalence(tampered, state, toolkit.profile)


def _typed_motion_toolkit(
    subject_id: str,
    target_id: str,
    *,
    end_x: float,
    motion_mode: str = "self_propelled",
    motion_type: str = "moving",
    direction_mode: str = "toward_target",
    path_type: str = "linear",
    intermediate_x: float | None = None,
    contained_by_id: str | None = None,
    external_visibility: str = "unchanged",
) -> ScenePlanningToolkit:
    semantics = {
        "action_kind": "locomotion",
        "motion_type": motion_type,
        "motion_mode": motion_mode,
        "direction_mode": direction_mode,
        "target_id": target_id if direction_mode != "none" else None,
        "carrier_id": None,
        "path_type": path_type,
        "timeline_event_id": None,
        "postconditions": {
            "contained_by_id": contained_by_id,
            "external_visibility": external_visibility,
        },
        "source_status": "inferred",
        "source_text": "generic audit motion",
    }
    motion = {
        "subject_id": subject_id,
        "action": {"value": "generic motion", "source_status": "explicit", "source_text": "generic motion"},
        "motion_semantics": semantics,
        "start_time_seconds": 0.0,
        "end_time_seconds": 6.0,
    }
    objective = ObjectivePlanningBrief(
        schema_version="0.4",
        source_brief_sha256="sha256:generality-audit",
        subjects=[{"id": subject_id}, {"id": target_id}],
        subject_motion=[motion],
        scene_design={"relationships": []},
        composition={},
        camera={},
        timeline={
            "duration_seconds": 6.0,
            "duration_source_status": "explicit",
            "events": [],
        },
        uncertainties=[],
        translation_parameters={
            "motions": [
                {
                    "motion_index": 0,
                    "subject_id": subject_id,
                    "motion_type": motion_type,
                    "motion_mode": motion_mode,
                    "direction_mode": direction_mode,
                    "target_id": target_id if direction_mode != "none" else None,
                    "path_type": path_type,
                    "start_time_seconds": 0.0,
                    "end_time_seconds": 6.0,
                }
            ]
        },
        explicit_requirements=[],
        source_metadata=BriefSourceMetadata(
            provider="mock",
            model="generality-audit",
            parser_prompt_version="test",
            rules_sha256="sha256:test",
            translation_rules_sha256="sha256:test",
            response_id=None,
        ),
    )
    resolution = freeze_brief_duration(
        objective.timeline,
        fps_numerator=24,
        fps_denominator=1,
    )
    toolkit = ScenePlanningToolkit(attach_duration_resolution(objective, resolution))
    toolkit.apply_entity_patch(
        [_entity(subject_id, x=0.0), _entity(target_id, x=10.0)],
        [],
    )
    keyframes = [
        {"time_seconds": 0.0, "value": {"translation_m": [0.0, 0.0, 0.5]}},
    ]
    if intermediate_x is not None:
        keyframes.append(
            {"time_seconds": 1.5, "value": {"translation_m": [intermediate_x, 0.0, 0.5]}}
        )
        keyframes.append(
            {"time_seconds": 3.0, "value": {"translation_m": [0.0, 0.0, 0.5]}}
        )
    keyframes.append(
        {"time_seconds": 143 / 24, "value": {"translation_m": [end_x, 0.0, 0.5]}}
    )
    toolkit.apply_motion_patch(
        [
            {
                "track_id": f"{subject_id}_motion",
                "target_entity_id": subject_id,
                "type": "transform",
                "time_range_seconds": [0.0, 6.0],
                "keyframes": keyframes,
            }
        ],
        [],
    )
    return toolkit


def _entity(
    entity_id: str,
    *,
    x: float,
    ground_entity_id: str | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "role": "subject",
        "proxy": {"type": "box", "size_xyz_m": [1.0, 1.0, 1.0]},
        "tags": [],
        "ground_interaction": {
            "mode": "must_be_above",
            "ground_entity_id": ground_entity_id,
        },
        "solved_transform": {
            "translation_m": [x, 0.0, 0.5],
            "rotation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
    }


def _hard_codes(result: dict) -> set[str]:
    return {
        item["code"]
        for item in result["violations"]
        if item["severity"] == "hard"
    }
