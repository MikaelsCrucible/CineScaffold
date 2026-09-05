from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from cinescaffold.errors import ExecutionError
from cinescaffold.planning.ir import SceneIR
from cinescaffold.planning.store import canonical_hash
from cinescaffold.platforms import default_blender_path, default_mcp_command


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    scene_ir_hash: str
    build: dict[str, Any]
    render: dict[str, Any] | None
    artifacts: dict[str, str]
    elapsed_seconds: float
    error: str | None = None


class BlenderMCPAdapter(Protocol):
    async def apply_scene_ir(
        self,
        *,
        scene_ir: SceneIR,
        scene_ir_hash: str,
        template_blend: Path,
        output_dir: Path,
    ) -> dict[str, Any]: ...

    async def render_video(
        self,
        scene_blend: Path,
        render_profile: str,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExecutionConfig:
    output_dir: Path
    blender_path: Path = default_blender_path()
    mcp_command: Path = default_mcp_command()
    overwrite: bool = False
    template_timeout_seconds: float = 60.0
    build_timeout_seconds: float = 180.0
    render_timeout_seconds: float = 600.0
    build_backend: Literal["background", "mcp"] = "background"
    render_backend: Literal["background", "mcp"] = "background"
    process_mode: Literal["fused", "split"] = "fused"
    render_profile: Literal["preview", "control"] = "preview"


class ExecutionRunner:
    """不使用 Agent 的固定 Scene IR 执行状态机。"""

    def __init__(
        self,
        config: ExecutionConfig,
        *,
        adapter: BlenderMCPAdapter | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.output_dir = config.output_dir.resolve()
        self.adapter = adapter

    async def run(self, payload: dict[str, Any]) -> ExecutionResult:
        # Avoid coupling package import order between the planner commit gate and executor.
        from cinescaffold.execution.validation import validate_scene_ir_for_execution

        started = time.monotonic()
        self._emit("execution_validation_started")
        try:
            payload = _upgrade_legacy_scene_ir(payload)
            scene_ir = SceneIR.model_validate(payload)
            validate_scene_ir_for_execution(scene_ir)
        except Exception as error:
            self._emit("execution_validation_failed", error=str(error))
            raise
        scene_ir_hash = canonical_hash(scene_ir)
        self._emit(
            "execution_validation_completed",
            scene_ir_hash=scene_ir_hash,
            entity_count=len(scene_ir.entities),
            frame_count=scene_ir.timeline.frame_count,
            duration_seconds=scene_ir.timeline.duration_seconds,
        )
        self._emit("execution_workspace_started", output_dir=str(self.output_dir))
        self._prepare_output_dir()

        normalized_ir_path = self.output_dir / "scene_ir.json"
        normalized_ir_path.write_text(
            json.dumps(scene_ir.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._emit("execution_workspace_completed", output_dir=str(self.output_dir))
        process_mode = self._effective_process_mode()
        fused_render: dict[str, Any] | None = None
        fused_render_error: str | None = None
        try:
            self._emit(
                "scene_build_started",
                backend=self.config.build_backend,
                process_mode=process_mode,
                timeout_seconds=self.config.build_timeout_seconds,
            )
            if process_mode == "fused":
                build, fused_render, fused_render_error = (
                    await self._build_and_render_background(
                        normalized_ir_path,
                        scene_ir_hash,
                    )
                )
            elif self.config.build_backend == "mcp":
                build = await self._build_scene_mcp(scene_ir, scene_ir_hash)
            else:
                build = await self._build_scene_background(normalized_ir_path, scene_ir_hash)
        except Exception as error:
            self._emit(
                "scene_build_failed",
                backend=self.config.build_backend,
                error=str(error),
            )
            result = ExecutionResult(
                status="execution_failed",
                scene_ir_hash=scene_ir_hash,
                build={
                    "status": "failed",
                    "stage": f"{self.config.build_backend}_build",
                    "message": str(error),
                },
                render=None,
                artifacts=self._existing_artifacts(),
                elapsed_seconds=time.monotonic() - started,
                error=f"Blender 场景构建失败：{error}",
            )
            self._write_manifest(scene_ir, result)
            self._emit("execution_finished", status=result.status, elapsed_seconds=result.elapsed_seconds)
            return result
        self._emit(
            "scene_build_completed",
            backend=self.config.build_backend,
            status=build.get("status"),
            blender_version=build.get("blender_version"),
            validation_passed=build.get("validation_passed"),
            violation_count=build.get("violation_count"),
        )
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
            self._emit("execution_finished", status=result.status, elapsed_seconds=result.elapsed_seconds)
            return result

        scene_blend = self.output_dir / "scene.blend"
        try:
            self._emit(
                "render_started",
                backend=self.config.render_backend,
                process_mode=process_mode,
                profile=self.config.render_profile,
                frame_count=scene_ir.timeline.frame_count,
                timeout_seconds=self.config.render_timeout_seconds,
            )
            if process_mode == "fused":
                if fused_render_error is not None:
                    raise ExecutionError(fused_render_error)
                if fused_render is None:
                    raise ExecutionError("合并后台 Blender 未返回渲染结果")
                render = fused_render
            elif self.config.render_backend == "mcp":
                render = await self._mcp_adapter().render_video(
                    scene_blend,
                    self.config.render_profile,
                )
            else:
                render = await self._render_video_background(scene_blend)
        except Exception as error:
            self._emit("render_failed", error=str(error))
            result = ExecutionResult(
                status="render_failed",
                scene_ir_hash=scene_ir_hash,
                build=build,
                render={
                    "status": "failed",
                    "stage": f"{self.config.render_backend}_render",
                    "message": str(error),
                },
                artifacts=self._existing_artifacts(),
                elapsed_seconds=time.monotonic() - started,
                error=f"Blender 渲染失败：{error}",
            )
            self._write_manifest(scene_ir, result)
            self._emit("execution_finished", status=result.status, elapsed_seconds=result.elapsed_seconds)
            return result
        self._emit(
            "render_completed",
            status=render.get("status"),
            artifact=render.get("artifact"),
            rendered_frame_count=render.get("rendered_frame_count"),
            fps=render.get("fps"),
            resolution_x=render.get("resolution_x"),
            resolution_y=render.get("resolution_y"),
        )
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
        self._emit("execution_finished", status=result.status, elapsed_seconds=result.elapsed_seconds)
        return result

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        # 终端展示错误不应改变构建与渲染结果。
        try:
            self.progress_callback(event_type, payload)
        except Exception:
            pass

    async def _build_scene_background(
        self,
        scene_ir_path: Path,
        scene_ir_hash: str,
    ) -> dict[str, Any]:
        blender_path = self._validated_blender_path()
        result_path = self.output_dir / "background_build_result.json"
        code = self._background_build_code(scene_ir_path, scene_ir_hash, result_path)
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [
                    str(blender_path),
                    "--background",
                    "--factory-startup",
                    "--python-expr",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.build_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._write_blender_log(
                self.output_dir / "build_blender.log",
                error.stdout or "",
                error.stderr or "",
            )
            raise ExecutionError(
                f"后台 Blender 构建超过 {self.config.build_timeout_seconds:g} 秒"
            ) from error
        self._write_blender_log(
            self.output_dir / "build_blender.log",
            completed.stdout,
            completed.stderr,
        )
        result = self._read_background_result(
            result_path,
            completed.returncode,
            "后台 Blender 构建失败；请查看 build_blender.log",
        )
        if not (self.output_dir / "scene.blend").is_file():
            raise ExecutionError("后台 Blender 构建未生成 scene.blend")
        return result | {"transport": "background_blender"}

    async def _build_and_render_background(
        self,
        scene_ir_path: Path,
        scene_ir_hash: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
        """在一个 Blender 进程中构建、校验、保存并渲染。"""
        blender_path = self._validated_blender_path()
        build_result_path = self.output_dir / "background_build_result.json"
        render_result_path = self.output_dir / "background_render_result.json"
        log_path = self.output_dir / "fused_blender.log"
        code = self._background_fused_code(
            scene_ir_path,
            scene_ir_hash,
            build_result_path,
            render_result_path,
        )
        timeout_seconds = (
            self.config.build_timeout_seconds + self.config.render_timeout_seconds
        )
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [
                    str(blender_path),
                    "--background",
                    "--factory-startup",
                    "--python-expr",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._write_blender_log(log_path, error.stdout or "", error.stderr or "")
            if build_result_path.is_file():
                build = self._read_background_result(
                    build_result_path,
                    0,
                    "合并后台 Blender 构建结果无效；请查看 fused_blender.log",
                )
                return (
                    build | {"transport": "background_blender"},
                    None,
                    f"合并后台 Blender 渲染超过 {timeout_seconds:g} 秒总预算",
                )
            raise ExecutionError(
                f"合并后台 Blender 构建超过 {timeout_seconds:g} 秒总预算"
            ) from error

        self._write_blender_log(log_path, completed.stdout, completed.stderr)
        build = self._read_background_result(
            build_result_path,
            0,
            "合并后台 Blender 构建失败；请查看 fused_blender.log",
        )
        build = build | {"transport": "background_blender"}
        if not (self.output_dir / "scene.blend").is_file():
            raise ExecutionError("合并后台 Blender 构建未生成 scene.blend")
        if build.get("status") != "ok" or build.get("validation_passed") is not True:
            return build, None, None
        if completed.returncode != 0 or not render_result_path.is_file():
            return build, None, "合并后台 Blender 渲染失败；请查看 fused_blender.log"
        try:
            render = self._read_background_result(
                render_result_path,
                0,
                "合并后台 Blender 渲染失败；请查看 fused_blender.log",
            )
        except ExecutionError as error:
            return build, None, str(error)
        expected = self._render_output_path()
        if render.get("status") != "ok" or not expected.is_file():
            return build, None, "合并后台 Blender 未生成白模视频"
        return build, render | {"transport": "background_blender"}, None

    def _background_fused_code(
        self,
        scene_ir_path: Path,
        scene_ir_hash: str,
        build_result_path: Path,
        render_result_path: Path,
    ) -> str:
        source_root = Path(__file__).resolve().parents[2]
        return "\n".join(
            (
                "import json, sys, time",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(source_root)!r})",
                (
                    "from cinescaffold.blender.runtime import "
                    "apply_scene_ir, render_clay_video"
                ),
                (
                    f"scene_ir = json.loads(Path({str(scene_ir_path.resolve())!r})"
                    ".read_text(encoding='utf-8'))"
                ),
                "build_started = time.monotonic()",
                (
                    "build = apply_scene_ir("
                    f"scene_ir, {str(self.output_dir)!r}, {scene_ir_hash!r})"
                ),
                "build['elapsed_seconds'] = time.monotonic() - build_started",
                "build['transport'] = 'background_blender'",
                (
                    f"Path({str(build_result_path.resolve())!r}).write_text("
                    "json.dumps(build, ensure_ascii=False), encoding='utf-8')"
                ),
                "if build.get('status') == 'ok' and build.get('validation_passed') is True:",
                "    render_started = time.monotonic()",
                f"    render = render_clay_video({self.config.render_profile!r})",
                "    render['elapsed_seconds'] = time.monotonic() - render_started",
                "    render['transport'] = 'background_blender'",
                (
                    f"    Path({str(render_result_path.resolve())!r}).write_text("
                    "json.dumps(render, ensure_ascii=False), encoding='utf-8')"
                ),
            )
        )

    def _background_build_code(
        self,
        scene_ir_path: Path,
        scene_ir_hash: str,
        result_path: Path,
    ) -> str:
        source_root = Path(__file__).resolve().parents[2]
        return "\n".join(
            (
                "import json, sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(source_root)!r})",
                "from cinescaffold.blender.runtime import apply_scene_ir",
                (
                    f"scene_ir = json.loads(Path({str(scene_ir_path.resolve())!r})"
                    ".read_text(encoding='utf-8'))"
                ),
                (
                    "result = apply_scene_ir("
                    f"scene_ir, {str(self.output_dir)!r}, {scene_ir_hash!r})"
                ),
                (
                    f"Path({str(result_path.resolve())!r}).write_text("
                    "json.dumps(result, ensure_ascii=False), encoding='utf-8')"
                ),
            )
        )

    async def _build_scene_mcp(
        self,
        scene_ir: SceneIR,
        scene_ir_hash: str,
    ) -> dict[str, Any]:
        template_path = self.output_dir / "factory_template.blend"
        self._emit("factory_template_started", blender_path=str(self.config.blender_path))
        try:
            await asyncio.to_thread(self._create_factory_template, template_path)
        except Exception as error:
            self._emit("factory_template_failed", error=str(error))
            raise
        self._emit("factory_template_completed", path=str(template_path))
        return await self._mcp_adapter().apply_scene_ir(
            scene_ir=scene_ir,
            scene_ir_hash=scene_ir_hash,
            template_blend=template_path,
            output_dir=self.output_dir,
        )

    def _mcp_adapter(self) -> BlenderMCPAdapter:
        if self.adapter is None:
            # Keep the MCP client import off the default background-only path.
            from cinescaffold.execution.mcp import OfficialBlenderMCPAdapter

            self.adapter = OfficialBlenderMCPAdapter(
                mcp_command=self.config.mcp_command,
                blender_path=self.config.blender_path,
                log_path=self.output_dir / "blender_mcp.log",
                build_timeout_seconds=self.config.build_timeout_seconds,
                render_timeout_seconds=self.config.render_timeout_seconds,
            )
        return self.adapter

    async def _render_video_background(self, scene_blend: Path) -> dict[str, Any]:
        blender_path = self._validated_blender_path()
        result_path = self.output_dir / "background_render_result.json"
        code = self._background_render_code(result_path)
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                [
                    str(blender_path),
                    "--background",
                    str(scene_blend.resolve()),
                    "--python-expr",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=self.config.render_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._write_blender_log(
                self.output_dir / "render_blender.log",
                error.stdout or "",
                error.stderr or "",
            )
            raise ExecutionError(
                f"后台 Blender 渲染超过 {self.config.render_timeout_seconds:g} 秒"
            ) from error
        self._write_blender_log(
            self.output_dir / "render_blender.log",
            completed.stdout,
            completed.stderr,
        )
        result = self._read_background_result(
            result_path,
            completed.returncode,
            "后台 Blender 渲染失败；请查看 render_blender.log",
        )
        expected = self._render_output_path()
        if result.get("status") != "ok" or not expected.is_file():
            raise ExecutionError("后台 Blender 未生成白模视频")
        return result | {"transport": "background_blender"}

    def _background_render_code(self, result_path: Path) -> str:
        source_root = Path(__file__).resolve().parents[2]
        return "\n".join(
            (
                "import json, sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(source_root)!r})",
                "from cinescaffold.blender.runtime import render_clay_video",
                f"result = render_clay_video({self.config.render_profile!r})",
                (
                    f"Path({str(result_path.resolve())!r}).write_text("
                    "json.dumps(result, ensure_ascii=False), encoding='utf-8')"
                ),
            )
        )

    def _validated_blender_path(self) -> Path:
        blender_path = self.config.blender_path.resolve()
        if not blender_path.is_file():
            raise ExecutionError(f"Blender 命令不存在：{blender_path}")
        return blender_path

    @staticmethod
    def _read_background_result(
        result_path: Path,
        returncode: int,
        failure_message: str,
    ) -> dict[str, Any]:
        if returncode != 0 or not result_path.is_file():
            raise ExecutionError(failure_message)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExecutionError(f"{failure_message}；结果文件无效") from error
        if not isinstance(result, dict):
            raise ExecutionError(f"{failure_message}；结果不是 JSON 对象")
        return result

    @staticmethod
    def _write_blender_log(path: Path, stdout: str | bytes, stderr: str | bytes) -> None:
        def normalize(value: str | bytes) -> str:
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

        path.write_text(
            normalize(stdout) + normalize(stderr),
            encoding="utf-8",
        )

    def _prepare_output_dir(self) -> None:
        reserved = (
            "scene_ir.json",
            "factory_template.blend",
            "scene.blend",
            "runtime_snapshot.json",
            "runtime_validation.json",
            "clay_preview.mp4",
            "diagnostic_preview.mp4",
            "execution_manifest.json",
            "blender_mcp.log",
            "template_blender.log",
            "render_blender.log",
            "background_render_result.json",
            "background_build_result.json",
            "build_blender.log",
            "fused_blender.log",
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
            "diagnostic_preview": "diagnostic_preview.mp4",
            "mcp_log": "blender_mcp.log",
            "template_log": "template_blender.log",
            "render_log": "render_blender.log",
            "background_render_result": "background_render_result.json",
            "background_build_result": "background_build_result.json",
            "build_log": "build_blender.log",
            "fused_log": "fused_blender.log",
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
            "manifest_version": "0.3",
            "created_at": datetime.now(UTC).isoformat(),
            "status": result.status,
            "scene_ir_hash": result.scene_ir_hash,
            "scene_ir_schema_version": scene_ir.schema_version,
            "executor_api_version": "0.5",
            "build_backend": self.config.build_backend,
            "render_backend": self.config.render_backend,
            "requested_process_mode": self.config.process_mode,
            "process_mode": self._effective_process_mode(),
            "render_profile": self.config.render_profile,
            "blender_expected": scene_ir.provenance.expected_blender,
            "elapsed_seconds": result.elapsed_seconds,
            "artifacts": result.artifacts,
            "artifact_sha256": artifact_hashes,
            "build": result.build,
            "render": result.render,
            "error": result.error,
        }
        if self.config.build_backend == "mcp" or self.config.render_backend == "mcp":
            from cinescaffold.execution.mcp import MCP_BUILD_TOOL

            manifest["mcp_tool"] = MCP_BUILD_TOOL
        (self.output_dir / "execution_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _render_output_path(self) -> Path:
        name = "diagnostic_preview.mp4" if self.config.render_profile == "preview" else "clay_preview.mp4"
        return self.output_dir / name

    def _effective_process_mode(self) -> Literal["fused", "split"]:
        if (
            self.config.process_mode == "fused"
            and self.config.build_backend == "background"
            and self.config.render_backend == "background"
        ):
            return "fused"
        return "split"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _upgrade_legacy_scene_ir(payload: dict[str, Any]) -> dict[str, Any]:
    timeline = payload.get("timeline")
    if not isinstance(timeline, dict) or "duration_resolution" in timeline:
        return payload
    duration = timeline.get("duration_seconds")
    frame_count = timeline.get("frame_count")
    if not isinstance(duration, (int, float)) or not isinstance(frame_count, int):
        return payload
    upgraded = copy.deepcopy(payload)
    upgraded["timeline"]["duration_resolution"] = {
        "request": {"mode": "legacy_frozen"},
        "resolution_method": "legacy_frozen",
        "proposed_duration_seconds": duration,
        "resolved_duration_seconds": duration,
        "frame_count": frame_count,
        "reason": "从旧版已冻结 Scene IR 迁移；原始时长请求类型未知。",
    }
    return upgraded
