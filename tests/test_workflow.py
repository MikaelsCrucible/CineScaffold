from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from cinescaffold.errors import ProviderHTTPError
from cinescaffold.execution.runner import ExecutionConfig, ExecutionResult
from cinescaffold.planning.runner import InterpreterRunConfig, InterpreterRunResult
from cinescaffold.planning.trace import CostRates
from cinescaffold.providers.base import ProviderResponse
from cinescaffold.providers import MockProvider
from cinescaffold.semantic import SemanticParserConfig
from cinescaffold.workflow import PipelineRunConfig, PipelineSource, WorkflowRunner
from tests.helpers import ROOT


class _PlanningRunner:
    def __init__(self, config, *, progress_callback=None) -> None:
        self.config = config
        self.progress_callback = progress_callback

    async def run(self, _brief):
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        (self.config.run_dir / "final_scene_ir.json").write_text("{}\n", encoding="utf-8")
        return InterpreterRunResult(
            run_id="workflow-test",
            status="success",
            provider="mock",
            model="mock",
            terminal_type="commit_request",
            scene_ir_hash="sha256:test",
            final_revision=1,
            delivery_tier="standard",
            usage={"tokens": {"requests": 1, "tool_calls": 1}},
            artifacts={"scene_ir": "final_scene_ir.json"},
        )


class _ExecutionRunner:
    def __init__(self, config, *, progress_callback=None) -> None:
        self.config = config
        self.progress_callback = progress_callback

    async def run(self, _payload):
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        video = self.config.output_dir / "diagnostic_preview.mp4"
        blend = self.config.output_dir / "scene.blend"
        glb = self.config.output_dir / "scene.glb"
        viewer_manifest = self.config.output_dir / "viewer_manifest.json"
        video.write_bytes(b"video")
        blend.write_bytes(b"blend")
        glb.write_bytes(b"glb")
        viewer_manifest.write_text("{}\n", encoding="utf-8")
        return ExecutionResult(
            status="success",
            scene_ir_hash="sha256:test",
            build={"status": "ok"},
            render={"status": "ok", "artifact": str(video)},
            artifacts={
                "blend": str(blend),
                "diagnostic_preview": video.name,
                "glb_preview": str(glb),
                "viewer_manifest": str(viewer_manifest),
            },
            elapsed_seconds=0.01,
        )


class _FailedPlanningRunner:
    def __init__(self, config, *, progress_callback=None) -> None:
        self.config = config

    async def run(self, _brief):
        return InterpreterRunResult(
            run_id="workflow-failed-test",
            status="failed",
            provider="deepseek",
            model="deepseek-flash",
            terminal_type=None,
            scene_ir_hash=None,
            final_revision=0,
            usage={"tokens": {"requests": 1, "tool_calls": 0}},
            artifacts={},
            error={
                "type": "ModelHTTPError",
                "message": "provider response stays internal",
                "failure_code": "provider_overloaded",
            },
        )


class _UsageProvider:
    name = "usage-test"
    model = "usage-test-model"

    def __init__(self, content):
        self.content = content

    def generate(self, _system_prompt, _user_prompt, _schema):
        return ProviderResponse(
            content=self.content,
            response_id="usage-test-1",
            raw_metadata={
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 100,
                }
            },
        )


class _RejectedProvider:
    name = "rejection-test"
    model = "rejection-test-model"

    def __init__(self, status_code: int = 402) -> None:
        self.status_code = status_code

    def generate(self, _system_prompt, _user_prompt, _schema):
        raise ProviderHTTPError(self.status_code, "provider rejection")


class WorkflowRunnerTest(unittest.IsolatedAsyncioTestCase):
    def _config(self, root: Path, *, semantic: bool = False) -> PipelineRunConfig:
        provider = None
        parser = None
        if semantic:
            response = json.loads(
                (ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json").read_text(encoding="utf-8")
            )
            provider = MockProvider(response)
            parser = SemanticParserConfig(
                system_template_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
                rules_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
                format_example_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
                model_output_schema_path=ROOT
                / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json",
                translation_rules_path=ROOT
                / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json",
                translation_parameters_schema_path=ROOT
                / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json",
            )
        return PipelineRunConfig(
            output_dir=root / "run",
            planning=InterpreterRunConfig(provider="mock", run_dir=root / "unused-planning"),
            execution=ExecutionConfig(output_dir=root / "unused-execution"),
            semantic_provider=provider,
            semantic_parser=parser,
        )

    async def test_natural_text_runs_all_stages_and_saves_both_semantic_forms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events: list[str] = []
            runner = WorkflowRunner(
                self._config(root, semantic=True),
                progress_callback=lambda name, _payload: events.append(name),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            )
            summary = await runner.run(PipelineSource(kind="text", text="一个人在荒漠里等待"))

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["delivery_tier"], "standard")
            self.assertIsNone(summary["failure_code"])
            self.assertTrue((root / "run/cinematic_brief.json").is_file())
            self.assertTrue((root / "run/textual_six_dimensions.txt").is_file())
            self.assertTrue(Path(summary["artifacts"]["video"]).is_file())
            self.assertTrue(Path(summary["artifacts"]["scene_blend"]).is_file())
            self.assertTrue(Path(summary["artifacts"]["glb_preview"]).is_file())
            self.assertTrue(Path(summary["artifacts"]["viewer_manifest"]).is_file())
            self.assertIn("pipeline_semantic_completed", events)
            self.assertIn("pipeline_planning_started", events)
            self.assertIn("pipeline_execution_started", events)

    async def test_scene_ir_skips_semantic_and_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = WorkflowRunner(
                self._config(root),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            )
            summary = await runner.run(PipelineSource(kind="scene_ir", payload={}))

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["delivery_tier"], "standard")
            self.assertNotIn("semantic", summary["stages"])
            self.assertNotIn("planning", summary["stages"])
            self.assertIn("execution", summary["stages"])

    async def test_scene_ir_does_not_require_unused_provider_prices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            config = PipelineRunConfig(
                **{
                    **config.__dict__,
                    "max_provider_cost": Decimal("2"),
                    "provider_cost_currency": "CNY",
                }
            )
            summary = await WorkflowRunner(
                config,
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="scene_ir", payload={}))

        self.assertEqual(summary["status"], "success")

    async def test_semantic_failure_is_recorded_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = WorkflowRunner(
                self._config(root),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            )
            summary = await runner.run(PipelineSource(kind="text", text="测试"))

            self.assertEqual(summary["status"], "semantic_failed")
            self.assertIn("缺少 Provider", summary["error"])
            self.assertEqual(summary["failure_code"], "pipeline_failed")
            self.assertTrue((root / "run/pipeline_summary.json").is_file())

    async def test_semantic_cost_is_emitted_and_stops_before_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, semantic=True)
            assert config.semantic_provider is not None
            rates = CostRates(
                currency="CNY",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
                source="test",
            )
            config = PipelineRunConfig(
                **{
                    **config.__dict__,
                    "semantic_provider": _UsageProvider(config.semantic_provider.response),
                    "semantic_cost_rates": rates,
                    "planning": config.planning.model_copy(update={"cost_rates": rates}),
                    "max_provider_cost": Decimal("0.0001"),
                    "provider_cost_currency": "CNY",
                }
            )
            events: list[str] = []
            summary = await WorkflowRunner(
                config,
                progress_callback=lambda name, _payload: events.append(name),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="text", text="测试"))

        self.assertEqual(summary["status"], "cost_limit_exceeded")
        self.assertIn("provider_cost_incurred", events)
        self.assertIn("provider_cost_limit_exceeded", events)
        self.assertNotIn("pipeline_planning_started", events)

    async def test_planning_provider_failure_code_reaches_pipeline_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = await WorkflowRunner(
                self._config(root, semantic=True),
                planning_runner_type=_FailedPlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="text", text="测试"))

        self.assertEqual(summary["status"], "planning_failed")
        self.assertEqual(summary["failure_code"], "provider_overloaded")

    async def test_semantic_cost_is_emitted_before_invalid_response_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, semantic=True)
            assert config.semantic_provider is not None
            rates = CostRates(
                currency="CNY",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
                source="test",
            )
            config = PipelineRunConfig(
                **{
                    **config.__dict__,
                    "semantic_provider": _UsageProvider({"invalid": True}),
                    "semantic_cost_rates": rates,
                    "planning": config.planning.model_copy(update={"cost_rates": rates}),
                    "max_provider_cost": Decimal("2"),
                    "provider_cost_currency": "CNY",
                }
            )
            events: list[str] = []
            summary = await WorkflowRunner(
                config,
                progress_callback=lambda name, _payload: events.append(name),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="text", text="测试"))

        self.assertEqual(summary["status"], "semantic_failed")
        self.assertIn("provider_cost_incurred", events)
        self.assertNotIn("pipeline_planning_started", events)

    async def test_semantic_http_rejection_closes_cost_as_confirmed_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, semantic=True)
            rates = CostRates(
                currency="CNY",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
                source="test",
            )
            config = PipelineRunConfig(
                **{
                    **config.__dict__,
                    "semantic_provider": _RejectedProvider(),
                    "semantic_cost_rates": rates,
                    "planning": config.planning.model_copy(update={"cost_rates": rates}),
                    "max_provider_cost": Decimal("2"),
                    "provider_cost_currency": "CNY",
                }
            )
            events: list[tuple[str, dict[str, object]]] = []
            summary = await WorkflowRunner(
                config,
                progress_callback=lambda name, payload: events.append((name, payload)),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="text", text="测试"))

        self.assertEqual(summary["status"], "semantic_failed")
        self.assertEqual(summary["failure_code"], "provider_balance_exhausted")
        settlements = [payload for name, payload in events if name == "provider_cost_incurred"]
        self.assertEqual(len(settlements), 1)
        self.assertEqual(settlements[0]["amount"], "0.00000000")
        self.assertEqual(settlements[0]["billing_resolution"], "confirmed_not_billed")
        self.assertEqual(settlements[0]["http_status"], 402)

    async def test_semantic_overload_exposes_safe_failure_code_without_zero_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, semantic=True)
            rates = CostRates(
                currency="CNY",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
                source="test",
            )
            config = PipelineRunConfig(
                **{
                    **config.__dict__,
                    "semantic_provider": _RejectedProvider(503),
                    "semantic_cost_rates": rates,
                    "planning": config.planning.model_copy(update={"cost_rates": rates}),
                    "max_provider_cost": Decimal("2"),
                    "provider_cost_currency": "CNY",
                }
            )
            events: list[tuple[str, dict[str, object]]] = []
            summary = await WorkflowRunner(
                config,
                progress_callback=lambda name, payload: events.append((name, payload)),
                planning_runner_type=_PlanningRunner,
                execution_runner_type=_ExecutionRunner,
            ).run(PipelineSource(kind="text", text="测试"))

        self.assertEqual(summary["status"], "semantic_failed")
        self.assertEqual(summary["failure_code"], "provider_overloaded")
        self.assertFalse(any(name == "provider_cost_incurred" for name, _ in events))
        failure = next(payload for name, payload in events if name == "pipeline_failed")
        self.assertEqual(failure["failure_code"], "provider_overloaded")


if __name__ == "__main__":
    unittest.main()
