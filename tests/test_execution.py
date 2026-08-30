from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from cinescaffold.blender.runtime import _blender_render_engine
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

    async def render_preview(self, scene_blend):
        preview = scene_blend.parent / "clay_preview.mp4"
        preview.write_bytes(b"fake-video")
        return {"status": "ok", "artifact": str(preview)}


class _FailingAdapter:
    async def apply_scene_ir(self, **kwargs):
        raise RuntimeError("MCP unavailable")


class _FakeExecutionRunner(ExecutionRunner):
    def _create_factory_template(self, target: Path) -> None:
        target.write_bytes(b"fake-template")
        (self.output_dir / "template_blender.log").write_text("fake\n", encoding="utf-8")


class ExecutionTest(unittest.TestCase):
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

    def test_scene_ir_engine_maps_to_blender_5_2_enum(self) -> None:
        self.assertEqual(_blender_render_engine("BLENDER_EEVEE_NEXT"), "BLENDER_EEVEE")

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
                ExecutionConfig(output_dir=output_dir),
                adapter=_FakeAdapter(),
            )
            result = asyncio.run(runner.run(self.scene_ir.model_dump(mode="json")))
            manifest = json.loads(
                (output_dir / "execution_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(manifest["status"], "success")
        self.assertIn("clay_preview", manifest["artifact_sha256"])

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
