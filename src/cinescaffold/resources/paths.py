"""Resolve CineScaffold's packaged runtime data without relying on the CWD."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class RuntimeResourcePaths:
    semantic_system: Path
    semantic_rules: Path
    semantic_example: Path
    semantic_schema: Path
    translation_rules: Path
    translation_schema: Path
    planning_system: Path

    @classmethod
    def from_package(cls) -> RuntimeResourcePaths:
        """Resolve runtime data from the installed CineScaffold package."""

        root = Path(str(files("cinescaffold.resources")))
        return cls.from_root(root)

    @classmethod
    def from_root(cls, root: Path) -> RuntimeResourcePaths:
        """Resolve an explicit directory containing prompts/ and schemas/."""

        return cls(
            semantic_system=root / "prompts/semantic_parser/system.md",
            semantic_rules=root / "prompts/semantic_parser/rules.md",
            semantic_example=root / "prompts/semantic_parser/format_example.json",
            semantic_schema=root / "schemas/cinematic_brief_model_output.schema.json",
            translation_rules=root / "prompts/semantic_parser/translation_rules.json",
            translation_schema=root / "schemas/semantic_translation_parameters.schema.json",
            planning_system=root / "prompts/scene_planner/system.md",
        )
