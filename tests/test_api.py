from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from cinescaffold.api import execute_scene_ir, run_pipeline
from cinescaffold.workflow import PipelineSource


class PublicApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_pipeline_delegates_to_workflow_service(self) -> None:
        source = PipelineSource(kind="scene_ir", payload={})
        config = object()
        expected = {"status": "success"}

        with patch("cinescaffold.api.WorkflowRunner") as runner_type:
            runner_type.return_value.run = AsyncMock(return_value=expected)
            result = await run_pipeline(source, config)  # type: ignore[arg-type]

        self.assertIs(result, expected)
        runner_type.assert_called_once_with(config, progress_callback=None)
        runner_type.return_value.run.assert_awaited_once_with(source)

    async def test_execute_scene_ir_delegates_to_execution_service(self) -> None:
        scene_ir = {"schema_version": "0.2"}
        config = object()
        expected = object()

        with patch("cinescaffold.api.ExecutionRunner") as runner_type:
            runner_type.return_value.run = AsyncMock(return_value=expected)
            result = await execute_scene_ir(scene_ir, config)  # type: ignore[arg-type]

        self.assertIs(result, expected)
        runner_type.assert_called_once_with(config, progress_callback=None)
        runner_type.return_value.run.assert_awaited_once_with(scene_ir)


if __name__ == "__main__":
    unittest.main()
