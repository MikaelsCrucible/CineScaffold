from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cinescaffold.config import LoadedConfig, merge_config
from cinescaffold.errors import ConfigurationError
from cinescaffold.runtime import RuntimeResourcePaths, build_pipeline_run_config
from tests.helpers import ROOT


class RuntimeConfigTest(unittest.TestCase):
    def test_builds_distinct_semantic_and_planning_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = LoadedConfig(
                None,
                {
                    "semantic_provider": "mock",
                    "planning_provider": "mock",
                    "semantic_input_cost_per_million": "0.1",
                    "semantic_output_cost_per_million": "0.2",
                    "planning_input_cost_per_million": "1.0",
                    "planning_output_cost_per_million": "2.0",
                    "execution_build_backend": "background",
                    "execution_build_timeout_seconds": "240",
                    "execution_render_profile": "control",
                },
            )
            config = build_pipeline_run_config(
                loaded,
                output_dir=root / "run",
                include_semantic=True,
                overwrite=True,
                resources=RuntimeResourcePaths.from_root(ROOT),
                render_profile="preview",
            )

        self.assertEqual(config.semantic_provider.name, "mock")
        self.assertEqual(str(config.semantic_cost_rates.input_per_million), "0.1")
        self.assertEqual(str(config.planning.cost_rates.input_per_million), "1.0")
        self.assertEqual(config.execution.render_profile, "preview")
        self.assertEqual(config.execution.build_backend, "background")
        self.assertEqual(config.execution.build_timeout_seconds, 240.0)

    def test_memory_merge_validates_without_writing(self) -> None:
        original = LoadedConfig(None, {"semantic_provider": "mock"})
        merged = merge_config(
            original,
            {"planning_provider": "openai", "planning_max_requests": "64"},
        )

        self.assertNotIn("planning_provider", original.data)
        self.assertEqual(merged.data["planning_provider"], "openai")
        self.assertEqual(merged.data["planning_max_requests"], "64")

    def test_rejects_negative_price(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "必须是非负数"):
            merge_config(
                LoadedConfig(None, {}),
                {"semantic_input_cost_per_million": "-1"},
            )


if __name__ == "__main__":
    unittest.main()
