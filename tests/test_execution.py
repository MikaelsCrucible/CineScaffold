from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from cinescaffold.blender.runtime import (
    _blender_render_engine,
    _configure_workbench_preview,
    _render_profile_plan,
    _validate_mesh_geometry,
)
from cinescaffold.errors import ExecutionError
from cinescaffold.execution.mcp import OfficialBlenderMCPAdapter
from cinescaffold.execution.runner import ExecutionConfig, ExecutionRunner
from cinescaffold.execution.validation import validate_scene_ir_for_execution
from cinescaffold.planning.compiler import SceneIRCommitGate
from cinescaffold.planning.domain import CommitRequest
from cinescaffold.planning.ir import SceneIR
from tests.test_planning_toolkit import _solved_toolkit


class _FakeAdapter:
    async def apply_scene_ir(self, *, scene_ir, scene_ir_hash, template_blend, output_dir):
        (output_dir / "scene.blend").write_bytes(b"fake-blend")
        (output_dir / "runtime_snapshot.json").write_text("{}\n", encoding="utf-8")
        (output_dir / "runtime_validation.json").write_text(
            '{"passed": true}\n',
            encoding="utf-8",
        )
        return {
            "status": "ok",
            "scene_ir_hash": scene_ir_hash,
            "validation_passed": True,
        }

    async def render_video(self, scene_blend, render_profile):
        name = "diagnostic_preview.mp4" if render_profile == "preview" else "clay_preview.mp4"
        preview = scene_blend.parent / name
        preview.write_bytes(b"fake-video")
        return {"status": "ok", "artifact": str(preview), "render_profile": render_profile}


class _FailingAdapter:
    async def apply_scene_ir(self, **kwargs):
        raise RuntimeError("MCP unavailable")


class _FakeExecutionRunner(ExecutionRunner):
    def _create_factory_template(self, target: Path) -> None:
        target.write_bytes(b"fake-template")
        (self.output_dir / "template_blender.log").write_text("fake\n", encoding="utf-8")


class _FakeShading:
    light = ""
    color_type = ""
    show_shadows = True
    show_cavity = True
    cavity_type = ""
    show_specular_highlight = True
    background_type = ""


class _FakeScene:
    class Display:
        shading = _FakeShading()

    display = Display()


class ExecutionTest(unittest.TestCase):
    def test_runner_emits_user_facing_stage_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events: list[str] = []
            result = asyncio.run(
                _FakeExecutionRunner(
                    ExecutionConfig(
                        output_dir=Path(directory),
                        render_backend="mcp",
                    ),
                    adapter=_FakeAdapter(),
                    progress_callback=lambda event, payload: events.append(event),
                ).run(self.scene_ir.model_dump(mode="json"))
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(events[0], "execution_validation_started")
        self.assertIn("mcp_build_started", events)
        self.assertIn("render_started", events)
        self.assertEqual(events[-1], "execution_finished")

    @classmethod
    def setUpClass(cls) -> None:
        toolkit = _solved_toolkit()
        committed = SceneIRCommitGate(toolkit).commit(
            CommitRequest(
                type="commit_request",
                candidate_revision=toolkit.store.current_revision,
                summary="执行测试",
            ),
            agent_run_id="execution_test",
            trace_ref="planning_agent_tool_trace.jsonl",
        )
        assert committed.scene_ir is not None
        cls.scene_ir = committed.scene_ir

    def test_execution_validation_accepts_committed_ir(self) -> None:
        validate_scene_ir_for_execution(self.scene_ir)

    def test_runner_upgrades_legacy_timeline_and_ground_defaults(self) -> None:
        payload = self.scene_ir.model_dump(mode="json")
        del payload["timeline"]["duration_resolution"]
        for entity in payload["entities"]:
            del entity["ground_interaction"]
        with tempfile.TemporaryDirectory() as directory:
            runner = _FakeExecutionRunner(
                ExecutionConfig(output_dir=Path(directory), render_backend="mcp"),
                adapter=_FakeAdapter(),
            )
            result = asyncio.run(runner.run(payload))

        self.assertEqual(result.status, "success")

    def test_scene_ir_engine_maps_to_blender_5_2_enum(self) -> None:
        self.assertEqual(_blender_render_engine("BLENDER_EEVEE_NEXT"), "BLENDER_EEVEE")

    def test_preview_profile_halves_resolution_and_samples_without_changing_duration(self) -> None:
        plan = _render_profile_plan(
            "preview",
            frame_start=1,
            frame_end=144,
            fps=24,
            fps_base=1.0,
            render_engine="BLENDER_EEVEE",
            resolution_x=1280,
            resolution_y=720,
        )

        self.assertEqual(plan["frame_step"], 2)
        self.assertEqual(plan["rendered_frame_count"], 72)
        self.assertEqual(plan["fps_base"], 2.0)
        self.assertEqual(plan["render_engine"], "BLENDER_WORKBENCH")
        self.assertEqual((plan["resolution_x"], plan["resolution_y"]), (640, 360))

    def test_runtime_geometry_rejects_flat_mesh_for_box(self) -> None:
        violations = []
        _validate_mesh_geometry(
            {"type": "box", "size_xyz_m": [60.0, 20.0, 20.0]},
            {
                "vertex_count": 4,
                "polygon_count": 1,
                "local_bounds_size_m": [60.0, 20.0, 0.0],
            },
            "entities.ship_01.mesh",
            violations,
        )

        self.assertEqual(len(violations), 1)
        self.assertEqual(
            violations[0]["path"],
            "entities.ship_01.mesh.local_bounds_size_m",
        )

    def test_execution_validation_rejects_incomplete_track(self) -> None:
        payload = self.scene_ir.model_dump(mode="json")
        payload["camera"]["state_track"]["samples"].pop()
        scene_ir = SceneIR.model_validate(payload)

        with self.assertRaisesRegex(ValueError, "没有完整覆盖"):
            validate_scene_ir_for_execution(scene_ir)

    def test_neutral_preview_lighting_rejects_directional_shadows(self) -> None:
        payload = self.scene_ir.model_dump(mode="json")
        payload["lighting"]["cast_shadows"] = True

        with self.assertRaisesRegex(ValueError, "不得生成方向性投影"):
            SceneIR.model_validate(payload)

    def test_workbench_preview_disables_shadow_and_cavity_shading(self) -> None:
        scene = _FakeScene()

        _configure_workbench_preview(scene)

        self.assertEqual(scene.display.shading.light, "STUDIO")
        self.assertFalse(scene.display.shading.show_shadows)
        self.assertFalse(scene.display.shading.show_cavity)
        self.assertFalse(scene.display.shading.show_specular_highlight)

    def test_mcp_bootstrap_is_fixed_and_syntax_valid(self) -> None:
        adapter = OfficialBlenderMCPAdapter(
            mcp_command=Path("/tmp/blender-mcp"),
            blender_path=Path("/tmp/blender"),
            source_root=Path("/tmp/cinescaffold-src"),
        )
        code = adapter._apply_code(self.scene_ir, "sha256:test", Path("/tmp/run"))

        compile(code, "<mcp-bootstrap>", "exec")
        self.assertIn("from cinescaffold.blender.runtime import apply_scene_ir", code)
        self.assertNotIn("bpy.ops", code)

    def test_runner_completes_without_agent_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution"
            runner = _FakeExecutionRunner(
                ExecutionConfig(output_dir=output_dir, render_backend="mcp"),
                adapter=_FakeAdapter(),
            )
            result = asyncio.run(runner.run(self.scene_ir.model_dump(mode="json")))
            manifest = json.loads(
                (output_dir / "execution_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(manifest["status"], "success")
        self.assertIn("diagnostic_preview", manifest["artifact_sha256"])
        self.assertEqual(manifest["render_backend"], "mcp")
        self.assertEqual(manifest["render_profile"], "preview")

    def test_control_profile_keeps_formal_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "execution"
            runner = _FakeExecutionRunner(
                ExecutionConfig(
                    output_dir=output_dir,
                    render_backend="mcp",
                    render_profile="control",
                ),
                adapter=_FakeAdapter(),
            )
            result = asyncio.run(runner.run(self.scene_ir.model_dump(mode="json")))

        self.assertEqual(result.status, "success")
        self.assertIn("clay_preview", result.artifacts)
        self.assertNotIn("diagnostic_preview", result.artifacts)

    def test_background_render_bootstrap_is_fixed_and_syntax_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = ExecutionRunner(
                ExecutionConfig(output_dir=Path(directory)),
                adapter=_FakeAdapter(),
            )
            code = runner._background_render_code(Path(directory) / "result.json")

        compile(code, "<background-render>", "exec")
        self.assertIn("render_clay_video('preview')", code)
        self.assertNotIn("bpy.ops", code)

    def test_runner_requires_explicit_overwrite_for_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "scene.blend").write_bytes(b"existing")
            runner = _FakeExecutionRunner(
                ExecutionConfig(output_dir=output_dir),
                adapter=_FakeAdapter(),
            )

            with self.assertRaisesRegex(ExecutionError, "--overwrite"):
                asyncio.run(runner.run(self.scene_ir.model_dump(mode="json")))

    def test_runner_records_mcp_failure_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner = _FakeExecutionRunner(
                ExecutionConfig(output_dir=output_dir),
                adapter=_FailingAdapter(),
            )
            result = asyncio.run(runner.run(self.scene_ir.model_dump(mode="json")))
            manifest = json.loads(
                (output_dir / "execution_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.status, "execution_failed")
        self.assertEqual(manifest["build"]["stage"], "mcp_build")


if __name__ == "__main__":
    unittest.main()
