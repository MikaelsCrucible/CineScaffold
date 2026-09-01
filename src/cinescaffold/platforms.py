from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def default_blender_path() -> Path:
    """按当前平台寻找 Blender；未安装时返回可编辑的常见位置。"""

    discovered = shutil.which("blender")
    if discovered:
        return Path(discovered)
    if sys.platform == "darwin":
        homebrew = Path("/opt/homebrew/bin/blender")
        return homebrew if homebrew.exists() else Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if sys.platform == "win32":
        program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        candidates = sorted(
            (program_files / "Blender Foundation").glob("Blender */blender.exe"),
            reverse=True,
        )
        return candidates[0] if candidates else program_files / "Blender Foundation/Blender/blender.exe"
    return Path("/usr/bin/blender")


def default_mcp_command() -> Path:
    """查找 uv 安装的 Blender MCP 命令。"""

    discovered = shutil.which("blender-mcp")
    if discovered:
        return Path(discovered)
    suffix = ".exe" if sys.platform == "win32" else ""
    return Path.home() / ".local/bin" / f"blender-mcp{suffix}"


def open_local_path(path: Path) -> None:
    """使用平台文件管理器打开文件或目录。"""

    target = path.resolve()
    if not target.exists():
        raise FileNotFoundError(f"路径不存在：{target}")
    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    command = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen(
        [command, str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
