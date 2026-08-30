from __future__ import annotations

import unittest

from cinescaffold.planning.compiler import SceneIRCommitGate
from cinescaffold.planning.domain import CommitRequest
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.planning.toolkit import FULL_VALIDATION_CHECKS, ScenePlanningToolkit
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

    def test_unmapped_explicit_requirement_blocks_commit(self) -> None:
        toolkit = _toolkit()
        toolkit.apply_entity_patch([_man_entity() | {"source_refs": []}], [])
        toolkit.solve_candidate()

        validation = toolkit.validate_candidate(checks=["hard_semantics"])

        self.assertFalse(validation["data"]["hard_pass"])
        self.assertTrue(
            any(item["code"] == "UNMAPPED_EXPLICIT_REQUIREMENT" for item in validation["violations"])
        )


def _solved_toolkit() -> ScenePlanningToolkit:
    toolkit = _toolkit()
    toolkit.apply_entity_patch([_man_entity(), _ship_entity()], [])
    toolkit.apply_constraint_patch(
        [
            {
                "constraint_id": "ship_behind_man",
                "type": "depth_order",
                "strength": "hard",
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
    return ScenePlanningToolkit(project_objective_brief(brief).objective_brief)


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


def _annotated(value: str, source_text: str) -> dict:
    return {"value": value, "source_status": "explicit", "source_text": source_text}


def _unknown() -> dict:
    return {"value": None, "source_status": "unknown", "source_text": None}


if __name__ == "__main__":
    unittest.main()
