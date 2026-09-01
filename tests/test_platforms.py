from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from cinescaffold.platforms import default_blender_path, default_mcp_command


class PlatformDefaultsTest(unittest.TestCase):
    def test_windows_defaults_use_executable_suffixes(self) -> None:
        with patch("cinescaffold.platforms.shutil.which", return_value=None), patch(
            "cinescaffold.platforms.sys.platform", "win32"
        ), patch("cinescaffold.platforms.Path.home", return_value=Path("/fake-home")), patch.dict(
            os.environ, {"PROGRAMFILES": r"C:\Apps"}
        ):
            blender = default_blender_path()
            mcp = default_mcp_command()

        self.assertEqual(blender.name, "blender.exe")
        self.assertEqual(mcp.name, "blender-mcp.exe")

    def test_path_lookup_wins_over_platform_fallback(self) -> None:
        with patch("cinescaffold.platforms.shutil.which", return_value="/custom/blender"):
            self.assertEqual(default_blender_path(), Path("/custom/blender"))
