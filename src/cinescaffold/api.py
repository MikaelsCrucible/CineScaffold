"""Stable application-facing API for embedding CineScaffold."""

from __future__ import annotations

from typing import Any

from cinescaffold.config import LoadedConfig, load_config
from cinescaffold.execution.runner import ExecutionConfig, ExecutionResult, ExecutionRunner
from cinescaffold.runtime import RuntimeResourcePaths, build_pipeline_run_config
from cinescaffold.workflow import (
    PipelineEventCallback,
    PipelineRunConfig,
    PipelineSource,
    WorkflowRunner,
)


async def run_pipeline(
    source: PipelineSource,
    config: PipelineRunConfig,
    *,
    progress_callback: PipelineEventCallback | None = None,
) -> dict[str, Any]:
    """Run the public pipeline without depending on CLI or UI internals."""

    return await WorkflowRunner(
        config,
        progress_callback=progress_callback,
    ).run(source)


async def execute_scene_ir(
    scene_ir: dict[str, Any],
    config: ExecutionConfig,
    *,
    progress_callback: PipelineEventCallback | None = None,
) -> ExecutionResult:
    """Build a validated Scene IR and optionally render its video."""

    return await ExecutionRunner(
        config,
        progress_callback=progress_callback,
    ).run(scene_ir)


__all__ = [
    "ExecutionConfig",
    "ExecutionResult",
    "LoadedConfig",
    "PipelineEventCallback",
    "PipelineRunConfig",
    "PipelineSource",
    "RuntimeResourcePaths",
    "build_pipeline_run_config",
    "execute_scene_ir",
    "load_config",
    "run_pipeline",
]
