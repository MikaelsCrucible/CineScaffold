from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="组装 CineScaffold Windows x64 发行 ZIP。")
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--uv-exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    package_name = f"CineScaffold-Windows-x64-v{args.version}"
    output_dir = args.output_dir.resolve()
    package_dir = output_dir / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    _copy_tree(root / "packaging/windows", package_dir)
    _copy_tree(root / "src/cinescaffold/resources/prompts", package_dir / "prompts")
    _copy_tree(root / "src/cinescaffold/resources/schemas", package_dir / "schemas")
    _copy_tree(root / "examples", package_dir / "examples")
    shutil.copy2(root / ".cinescaffold.example.conf", package_dir / ".cinescaffold.example.conf")

    app_dir = package_dir / "app"
    tools_dir = package_dir / "tools"
    app_dir.mkdir()
    tools_dir.mkdir()
    shutil.copy2(root / "requirements-windows.lock", app_dir / "requirements-windows.lock")
    shutil.copy2(args.wheel.resolve(), app_dir / args.wheel.name)
    shutil.copy2(args.uv_exe.resolve(), tools_dir / "uv.exe")

    build_info = {
        "name": "CineScaffold",
        "version": args.version,
        "platform": "windows-x64",
        "commit": args.commit,
        "python": "3.12",
    }
    (package_dir / "BUILD_INFO.json").write_text(
        json.dumps(build_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive = Path(shutil.make_archive(str(output_dir / package_name), "zip", output_dir, package_name))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "sha256": digest}, ensure_ascii=False))
    return 0


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
