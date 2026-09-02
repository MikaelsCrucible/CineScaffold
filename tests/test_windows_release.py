from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from tests.helpers import ROOT


class WindowsReleaseTest(unittest.TestCase):
    def test_windows_lock_preserves_versions_and_replaces_uvloop(self) -> None:
        ui_versions = _locked_versions(ROOT / "requirements-ui.lock")
        windows_versions = _locked_versions(ROOT / "requirements-windows.lock")

        shared = ui_versions.keys() & windows_versions.keys()
        self.assertEqual(
            {name: windows_versions[name] for name in shared},
            {name: ui_versions[name] for name in shared},
        )
        self.assertNotIn("uvloop", windows_versions)
        self.assertEqual(windows_versions["pywin32"], "312")
        self.assertEqual(windows_versions["pywin32-ctypes"], "0.2.3")

    def test_builder_creates_self_contained_install_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            wheel = temporary / "cinescaffold-0.7.0-py3-none-any.whl"
            uv_exe = temporary / "uv.exe"
            wheel.write_bytes(b"wheel-placeholder")
            uv_exe.write_bytes(b"uv-placeholder")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/build_windows_release.py"),
                    "--wheel",
                    str(wheel),
                    "--uv-exe",
                    str(uv_exe),
                    "--output-dir",
                    str(temporary / "dist"),
                    "--version",
                    "0.7.0",
                    "--commit",
                    "abc123",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            archive = temporary / "dist/CineScaffold-Windows-x64-v0.7.0.zip"
            checksum = archive.with_suffix(".zip.sha256")

            self.assertTrue(archive.is_file())
            self.assertTrue(checksum.is_file())
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                prefix = "CineScaffold-Windows-x64-v0.7.0/"
                for required in (
                    "Install-CineScaffold.cmd",
                    "Start-CineScaffold.cmd",
                    "README-Windows.md",
                    "tools/uv.exe",
                    "app/requirements-windows.lock",
                    "app/cinescaffold-0.7.0-py3-none-any.whl",
                    "prompts/scene_planner/system.md",
                    "schemas/scene_ir.schema.json",
                    "BUILD_INFO.json",
                ):
                    self.assertIn(prefix + required, names)
                build_info = json.loads(bundle.read(prefix + "BUILD_INFO.json"))
            self.assertEqual(build_info["platform"], "windows-x64")
            self.assertEqual(build_info["commit"], "abc123")


def _locked_versions(path: Path) -> dict[str, str]:
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^ \\\n]+)")
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            versions[match.group(1).lower()] = match.group(2)
    return versions


if __name__ == "__main__":
    unittest.main()
