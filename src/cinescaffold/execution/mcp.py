from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp.client.transports import StdioTransport
from pydantic_ai.mcp import MCPToolset

from cinescaffold.planning.ir import SceneIR


MCP_BUILD_TOOL = "execute_blender_code_for_cli"


class OfficialBlenderMCPAdapter:
    """只允许调用项目维护的固定 Blender 入口。"""

    def __init__(
        self,
        *,
        mcp_command: Path,
        blender_path: Path,
        source_root: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.mcp_command = mcp_command.resolve()
        self.blender_path = blender_path.resolve()
        self.source_root = (source_root or Path(__file__).resolve().parents[2]).resolve()
        self.log_path = log_path.resolve() if log_path else None

    async def apply_scene_ir(
        self,
        *,
        scene_ir: SceneIR,
        scene_ir_hash: str,
        template_blend: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        code = self._apply_code(scene_ir, scene_ir_hash, output_dir)
        return await self._call_cli_tool(template_blend, code)

    async def render_video(self, scene_blend: Path, render_profile: str) -> dict[str, Any]:
        return await self._call_cli_tool(scene_blend, self._render_code(render_profile))

    async def render_preview(self, scene_blend: Path) -> dict[str, Any]:
        """兼容旧调用，保持完整控制视频。"""
        return await self.render_video(scene_blend, "control")

    async def _call_cli_tool(self, blend_file: Path, code: str) -> dict[str, Any]:
        self._validate_runtime_paths(blend_file)
        transport = StdioTransport(
            command=str(self.mcp_command),
            args=[],
            env={"BLENDER_PATH": str(self.blender_path)},
            keep_alive=False,
            log_file=self.log_path,
        )
        toolset = MCPToolset(
            transport,
            tool_error_behavior="error",
            read_timeout=180.0,
        )
        tools = await toolset.list_tools()
        if MCP_BUILD_TOOL not in {tool.name for tool in tools}:
            raise RuntimeError(f"官方 Blender MCP 缺少工具：{MCP_BUILD_TOOL}")
        result = await toolset.direct_call_tool(
            MCP_BUILD_TOOL,
            {"blend_file": str(blend_file.resolve()), "code": code},
        )
        if not isinstance(result, dict):
            raise RuntimeError("Blender MCP 返回值不是对象")
        return result

    def _apply_code(self, scene_ir: SceneIR, scene_ir_hash: str, output_dir: Path) -> str:
        payload = json.dumps(
            scene_ir.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "\n".join(
            (
                "import json, sys",
                f"sys.path.insert(0, {str(self.source_root)!r})",
                "from cinescaffold.blender.runtime import apply_scene_ir",
                (
                    "result = apply_scene_ir("
                    f"json.loads({payload!r}), {str(output_dir.resolve())!r}, {scene_ir_hash!r})"
                ),
            )
        )

    def _render_code(self, render_profile: str) -> str:
        if render_profile not in {"preview", "control"}:
            raise ValueError(f"未知渲染档位：{render_profile}")
        return "\n".join(
            (
                "import sys",
                f"sys.path.insert(0, {str(self.source_root)!r})",
                "from cinescaffold.blender.runtime import render_clay_video",
                f"result = render_clay_video({render_profile!r})",
            )
        )

    def _validate_runtime_paths(self, blend_file: Path) -> None:
        if not self.mcp_command.is_file():
            raise FileNotFoundError(f"Blender MCP 命令不存在：{self.mcp_command}")
        if not self.blender_path.is_file():
            raise FileNotFoundError(f"Blender 命令不存在：{self.blender_path}")
        if not blend_file.is_file():
            raise FileNotFoundError(f"输入 .blend 不存在：{blend_file}")
