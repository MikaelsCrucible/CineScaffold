from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cinescaffold.cli import main
from cinescaffold.config import LoadedConfig, load_config
from cinescaffold.execution.runner import ExecutionResult
from tests.helpers import ROOT, valid_planning_brief
from tests.test_textual_six import TEXTUAL_SIX


class _PipelineExecutionRunner:
    def __init__(self, config, *, progress_callback=None) -> None:
        self.config = config
        self.progress_callback = progress_callback

    async def run(self, payload):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        video = self.config.output_dir / "diagnostic_preview.mp4"
        video.write_bytes(b"fake-video")
        if self.progress_callback:
            self.progress_callback(
                "execution_finished",
                {"status": "success", "elapsed_seconds": 0.01},
            )
        return ExecutionResult(
            status="success",
            scene_ir_hash="sha256:test",
            build={"status": "ok", "blender_version": "test", "violation_count": 0},
            render={
                "status": "ok",
                "artifact": str(video),
                "rendered_frame_count": 72,
                "resolution_x": 640,
                "resolution_y": 360,
                "fps": 12,
            },
            artifacts={"diagnostic_preview": video.name},
            elapsed_seconds=0.01,
        )


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        # CLI 离线测试不得继承开发者机器上的默认配置。
        self.config_patch = patch(
            "cinescaffold.cli.load_config",
            side_effect=lambda path: load_config(path) if path else LoadedConfig(None, {}),
        )
        self.config_patch.start()

    def tearDown(self) -> None:
        self.config_patch.stop()

    def test_ui_shutdown_is_clean(self) -> None:
        with patch("cinescaffold.ui_app.launch_ui", side_effect=KeyboardInterrupt):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                status = main(["ui", "--no-open"])

        self.assertEqual(status, 0)
        self.assertIn("Studio 已停止", stderr.getvalue())

    def test_execute_passes_independent_build_and_render_backends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_ir = root / "scene_ir.json"
            scene_ir.write_text("{}\n", encoding="utf-8")
            configs = []

            def runner_factory(config, *, progress_callback=None):
                configs.append(config)
                return _PipelineExecutionRunner(
                    config,
                    progress_callback=progress_callback,
                )

            with patch("cinescaffold.cli.ExecutionRunner", side_effect=runner_factory):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "execute",
                            "--scene-ir",
                            str(scene_ir),
                            "--output-dir",
                            str(root / "execution"),
                            "--build-backend",
                            "mcp",
                            "--build-timeout-seconds",
                            "45",
                            "--render-backend",
                            "background",
                            "--process-mode",
                            "split",
                            "--quiet",
                        ]
                    )

        self.assertEqual(status, 0)
        self.assertEqual(configs[0].build_backend, "mcp")
        self.assertEqual(configs[0].build_timeout_seconds, 45.0)
        self.assertEqual(configs[0].render_backend, "background")
        self.assertEqual(configs[0].process_mode, "split")

    def test_parse_uses_simple_config_without_exposing_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / ".cinescaffold.conf"
            output = root / "brief.json"
            config_path.write_text(
                "provider = mock\nmodel = local-model\napi_key = must-not-appear\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "parse",
                        "--config",
                        str(config_path),
                        "--text",
                        "测试描述",
                        "--output",
                        str(output),
                        "--no-color",
                    ]
                )
            result = json.loads(output.read_text(encoding="utf-8"))

        rendered = stdout.getvalue() + stderr.getvalue() + str(result)
        self.assertEqual(status, 0)
        self.assertEqual(result["provenance"]["provider"], "mock")
        self.assertIn("已读取", stderr.getvalue())
        self.assertIn("规则量化", stderr.getvalue())
        self.assertNotIn("must-not-appear", rendered)

    def test_mock_parse_writes_brief(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "brief.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "parse",
                        "--provider",
                        "mock",
                        "--text",
                        "测试描述",
                        "--system-template",
                        str(ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md"),
                        "--rules",
                        str(ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md"),
                        "--format-example",
                        str(ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json"),
                        "--schema",
                        str(ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json"),
                        "--output",
                        str(output),
                        "--quiet",
                    ]
                )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(result["provenance"]["provider"], "mock")
        self.assertIn("10.0 秒", stdout.getvalue())

    def test_mock_parse_uses_packaged_resources_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "brief.json"
            with chdir(root):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "parse",
                            "--provider",
                            "mock",
                            "--text",
                            "测试描述",
                            "--output",
                            str(output),
                            "--quiet",
                        ]
                    )

            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(result["provenance"]["provider"], "mock")

    def test_mock_plan_writes_scene_ir_and_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            output_dir = root / "run"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(output_dir),
                        "--system-prompt",
                        str(ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md"),
                        "--quiet",
                    ]
                )
            summary = json.loads(
                (output_dir / "planning_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(status, 0)
        self.assertEqual(summary["status"], "success")
        self.assertGreater(summary["usage"]["tokens"]["input_tokens"], 0)

    def test_mock_plan_default_output_is_human_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(root / "run"),
                        "--system-prompt",
                        str(ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md"),
                        "--no-color",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertIn("CineScaffold 规划结果", stdout.getvalue())
        self.assertIn("Tokens", stdout.getvalue())
        self.assertIn("下一步", stdout.getvalue())
        self.assertIn("Agent 工具", stderr.getvalue())
        self.assertIn("Commit Gate", stderr.getvalue())

    def test_mock_plan_json_quiet_is_clean_machine_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "plan",
                        "--provider",
                        "mock",
                        "--brief",
                        str(brief_path),
                        "--output-dir",
                        str(root / "run"),
                        "--system-prompt",
                        str(ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md"),
                        "--json",
                        "--quiet",
                    ]
                )
            result = json.loads(stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "success")
        self.assertEqual(stderr.getvalue(), "")

    def test_run_from_text_completes_all_three_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock_response = root / "mock_response.json"
            mock_response.write_text(
                json.dumps(valid_planning_brief()["content"], ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    status = main(
                        [
                            "run",
                            "--text",
                            "测试描述",
                            "--provider",
                            "mock",
                            "--mock-response",
                            str(mock_response),
                            "--output-dir",
                            str(root / "run"),
                            "--json",
                            "--quiet",
                        ]
                    )
            result = json.loads(stdout.getvalue())
            textual_six_exists = Path(result["artifacts"]["textual_six_dimensions"]).is_file()

        self.assertEqual(status, 0)
        self.assertEqual(result["status"], "success")
        self.assertIn("semantic", result["stages"])
        self.assertEqual(
            result["stages"]["semantic"]["usage"]["estimated_cost"]["amount"],
            "0.00000000",
        )
        self.assertIn("elapsed_seconds", result["stages"]["semantic"])
        self.assertIn("planning", result["stages"])
        self.assertIn("execution", result["stages"])
        self.assertTrue(textual_six_exists)
        self.assertEqual(stderr.getvalue(), "")

    def test_run_from_textual_six_records_source_and_completes_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock_response = root / "mock_response.json"
            text_six = root / "scene.txt"
            mock_response.write_text(
                json.dumps(valid_planning_brief()["content"], ensure_ascii=False),
                encoding="utf-8",
            )
            text_six.write_text(TEXTUAL_SIX, encoding="utf-8")
            stdout = io.StringIO()
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "run",
                            "--text-six-file",
                            str(text_six),
                            "--provider",
                            "mock",
                            "--mock-response",
                            str(mock_response),
                            "--output-dir",
                            str(root / "run"),
                            "--json",
                            "--quiet",
                        ]
                    )
            result = json.loads(stdout.getvalue())
            brief = json.loads(Path(result["artifacts"]["cinematic_brief"]).read_text(encoding="utf-8"))
            saved_textual_six = Path(result["artifacts"]["textual_six_dimensions"]).read_text(encoding="utf-8")

        self.assertEqual(status, 0)
        self.assertEqual(result["started_from"], "textual_six")
        self.assertEqual(brief["provenance"]["source_kind"], "textual_six")
        self.assertEqual(saved_textual_six, TEXTUAL_SIX)

    def test_run_from_brief_skips_semantic_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "run",
                            "--brief",
                            str(brief_path),
                            "--provider",
                            "mock",
                            "--output-dir",
                            str(root / "run"),
                            "--json",
                            "--quiet",
                        ]
                    )
            result = json.loads(stdout.getvalue())

        self.assertEqual(status, 0)
        self.assertNotIn("semantic", result["stages"])
        self.assertIn("planning", result["stages"])
        self.assertIn("execution", result["stages"])

    def test_run_from_ir_skips_llm_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            planning_dir = root / "planning-source"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                plan_status = main(
                    [
                        "plan",
                        "--brief",
                        str(brief_path),
                        "--provider",
                        "mock",
                        "--output-dir",
                        str(planning_dir),
                        "--quiet",
                    ]
                )
            stdout = io.StringIO()
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "run",
                            "--ir",
                            str(planning_dir / "final_scene_ir.json"),
                            "--output-dir",
                            str(root / "run"),
                            "--json",
                            "--quiet",
                        ]
                    )
            result = json.loads(stdout.getvalue())

        self.assertEqual(plan_status, 0)
        self.assertEqual(status, 0)
        self.assertNotIn("semantic", result["stages"])
        self.assertNotIn("planning", result["stages"])
        self.assertIn("execution", result["stages"])

    def test_run_overwrite_replaces_owned_stages_and_preserves_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            output_dir = root / "run"
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            arguments = [
                "run",
                "--brief",
                str(brief_path),
                "--provider",
                "mock",
                "--output-dir",
                str(output_dir),
                "--json",
                "--quiet",
            ]
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    first_status = main(arguments)
            marker = output_dir / "planning" / "stale-marker.txt"
            marker.write_text("旧实验", encoding="utf-8")
            notes = output_dir / "notes.txt"
            notes.write_text("保留", encoding="utf-8")

            stdout = io.StringIO()
            with patch("cinescaffold.cli.ExecutionRunner", _PipelineExecutionRunner):
                with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                    second_status = main([*arguments, "--overwrite"])
            result = json.loads(stdout.getvalue())

            self.assertEqual(first_status, 0)
            self.assertEqual(second_status, 0)
            self.assertEqual(result["status"], "success")
            self.assertFalse(marker.exists())
            self.assertEqual(notes.read_text(encoding="utf-8"), "保留")

    def test_run_rejects_existing_stage_before_pipeline_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            brief_path = root / "brief.json"
            output_dir = root / "run"
            planning_dir = output_dir / "planning"
            planning_dir.mkdir(parents=True)
            (planning_dir / "partial.json").write_text("{}", encoding="utf-8")
            brief_path.write_text(
                json.dumps(valid_planning_brief(), ensure_ascii=False),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                status = main(
                    [
                        "run",
                        "--brief",
                        str(brief_path),
                        "--provider",
                        "mock",
                        "--output-dir",
                        str(output_dir),
                        "--quiet",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("一键管线产物已存在", stderr.getvalue())
            self.assertFalse((output_dir / "pipeline_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
