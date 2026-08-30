from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from cinescaffold.errors import ExecutionError
from cinescaffold.execution.mcp import MCP_BUILD_TOOL, OfficialBlenderMCPAdapter
from cinescaffold.execution.validation import validate_scene_ir_for_execution
from cinescaffold.planning.ir import SceneIR
from cinescaffold.planning.store import canonical_hash


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    scene_ir_hash: str
    build: dict[str, Any]
    render: dict[str, Any] | None
    artifacts: dict[str, str]
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class ExecutionConfig:
    output_dir: Path
    blender_path: Path = Path("/opt/homebrew/bin/blender")
    mcp_command: Path = Path.home() / ".local/bin/blender-mcp"
    overwrite: bool = False
    template_timeout_seconds: float = 60.0


class ExecutionRunner:
    """不使用 Agent 的固定 Scene IR 执行状态机。"""

    def __init__(
        self,
        config: ExecutionConfig,
        *,
        adapter: OfficialBlenderMCPAdapter | None = None,
    ) -> None:
        self.config = config
        self.output_dir = config.output_dir.resolve()
        self.adapter = adapter or OfficialBlenderMCPAdapter(
            mcp_command=config.mcp_command,
            blender_path=config.blender_path,
            log_path=self.output_dir / "blender_mcp.log",
        )

    async def run(self, payload: dict[str, Any]) -> ExecutionResult:
        started = time.monotonic()
        scene_ir = SceneIR.model_validate(payload)
        validate_scene_ir_for_execution(scene_ir)
        scene_ir_hash = canonical_hash(scene_ir)
        self._prepare_output_dir()

        normalized_ir_path = self.output_dir / "scene_ir.json"
        normalized_ir_path.write_text(
            json.dumps(scene_ir.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        template_path = self.output_dir / "factory_template.blend"
        self._create_factory_template(template_path)

        try:
            build = await self.adapter.apply_scene_ir(
                scene_ir=scene_ir,
                scene_ir_hash=scene_ir_hash,
                template_blend=template_path,
                output_dir=self.output_dir,
            )
        except Exception as error:
            result = ExecutionResult(
                status="execution_failed",
                scene_ir_hash=scene_ir_hash,
                build={"status": "failed", "stage": "mcp_build", "message": str(error)},
                render=None,
                artifacts=self._existing_artifacts(),
                elapsed_seconds=time.monotonic() - started,
                error=f"Blender MCP 构建失败：{error}",
            )
            self._write_manifest(scene_ir, result)
            return result
        if build.get("status") != "ok" or build.get("validation_passed") is not True:
            result = ExecutionResult(
                status="runtime_mismatch",
                scene_ir_hash=scene_ir_hash,
                build=build,
                render=None,
                artifacts=self._existing_artifacts(),
                elapsed_seconds=time.monotonic() - started,
                error="Blender Runtime 与 Scene IR 不一致",
            )
            self._write_manifest(scene_ir, result)
            return result

        scene_blend = self.output_dir / "scene.blend"
        try:
            render = await self.adapter.render_preview(scene_blend)
        except Exception as error:
            result = ExecutionResult(
                status="render_failed",
                scene_ir_hash=scene_ir_hash,
                build=build,
                render={"status": "failed", "stage": "mcp_render", "message": str(error)},
                artifacts=self._existing_artifacts(),
                elapsed_seconds=time.monotonic() - started,
                error=f"Blender MCP 渲染失败：{error}",
            )
            self._write_manifest(scene_ir, result)
            return result
        status = "success" if render.get("status") == "ok" else "render_failed"
        result = ExecutionResult(
            status=status,
            scene_ir_hash=scene_ir_hash,
            build=build,
            render=render,
            artifacts=self._existing_artifacts(),
            elapsed_seconds=time.monotonic() - started,
            error=None if status == "success" else "白模视频渲染失败",
        )
        self._write_manifest(scene_ir, result)
        return result

    def _prepare_output_dir(self) -> None:
        reserved = (
            "scene_ir.json",
            "factory_template.blend",
            "scene.blend",
            "runtime_snapshot.json",
            "runtime_validation.json",
            "clay_preview.mp4",
            "execution_manifest.json",
            "blender_mcp.log",
            "template_blender.log",
        )
        conflicts = [name for name in reserved if (self.output_dir / name).exists()]
        if conflicts and not self.config.overwrite:
            raise ExecutionError(
                "输出目录已有执行产物；如需覆盖请传入 --overwrite：" + ", ".join(conflicts)
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.overwrite:
            for name in conflicts:
                path = self.output_dir / name
                if path.is_file():
                    path.unlink()

    def _create_factory_template(self, target: Path) -> None:
        blender_path = self.config.blender_path.resolve()
        if not blender_path.is_file():
            raise ExecutionError(f"Blender 命令不存在：{blender_path}")
        code = (
            "import bpy; "
            f"bpy.ops.wm.save_as_mainfile(filepath={str(target)!r}, check_existing=False)"
        )
        completed = subprocess.run(
            [
                str(blender_path),
                "--background",
                "--factory-startup",
                "--python-expr",
                code,
            ],
            capture_output=True,
            text=True,
            timeout=self.config.template_timeout_seconds,
            check=False,
        )
        (self.output_dir / "template_blender.log").write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not target.is_file():
            raise ExecutionError("Blender 无法创建 factory template；请查看 template_blender.log")

    def _existing_artifacts(self) -> dict[str, str]:
        names = {
            "scene_ir": "scene_ir.json",
            "blend": "scene.blend",
            "runtime_snapshot": "runtime_snapshot.json",
            "runtime_validation": "runtime_validation.json",
            "clay_preview": "clay_preview.mp4",
            "mcp_log": "blender_mcp.log",
            "template_log": "template_blender.log",
        }
        return {
            artifact_id: str((self.output_dir / name).resolve())
            for artifact_id, name in names.items()
            if (self.output_dir / name).is_file()
        }

    def _write_manifest(self, scene_ir: SceneIR, result: ExecutionResult) -> None:
        artifact_hashes = {
            artifact_id: _sha256(Path(path))
            for artifact_id, path in result.artifacts.items()
        }
        manifest = {
            "manifest_version": "0.1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": result.status,
            "scene_ir_hash": result.scene_ir_hash,
            "scene_ir_schema_version": scene_ir.schema_version,
            "executor_api_version": "0.1",
            "mcp_tool": MCP_BUILD_TOOL,
            "blender_expected": scene_ir.provenance.expected_blender,
            "elapsed_seconds": result.elapsed_seconds,
            "artifacts": result.artifacts,
            "artifact_sha256": artifact_hashes,
            "build": result.build,
            "render": result.render,
            "error": result.error,
        }
        (self.output_dir / "execution_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
