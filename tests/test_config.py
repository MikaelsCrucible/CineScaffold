from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cinescaffold.config import load_config, resolve_model_settings, resolve_stage_option
from cinescaffold.errors import ConfigurationError
from cinescaffold.planning.runner import InterpreterRunConfig


class ConfigTest(unittest.TestCase):
    def test_simple_text_config_resolves_provider_model_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".cinescaffold.conf"
            path.write_text(
                "provider = deepseek\nmodel = planner-model\napi_key = secret-value\n",
                encoding="utf-8",
            )
            config = load_config(path)
            settings = resolve_model_settings(
                config,
                "planning",
                cli_provider=None,
                cli_model=None,
                cli_base_url=None,
            )

        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "planner-model")
        self.assertEqual(settings.api_key, "secret-value")

    def test_stage_and_cli_values_override_global_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            path.write_text(
                "\n".join(
                    (
                        "provider = deepseek",
                        "model = shared-model",
                        "semantic_provider = openai",
                        "semantic_model = semantic-model",
                        "openai_api_key = openai-secret",
                        "planning_model_max_tokens = 4096",
                    )
                ),
                encoding="utf-8",
            )
            config = load_config(path)
            semantic = resolve_model_settings(
                config,
                "semantic",
                cli_provider=None,
                cli_model="cli-model",
                cli_base_url=None,
            )

        self.assertEqual(semantic.provider, "openai")
        self.assertEqual(semantic.model, "cli-model")
        self.assertEqual(semantic.api_key, "openai-secret")
        self.assertEqual(
            resolve_stage_option(config, "planning", "model_max_tokens", None),
            4096,
        )

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            path.write_text("modle = typo\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "未知字段"):
                load_config(path)

    def test_low_effort_is_not_accepted_as_thinking_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            path.write_text("planning_thinking_mode = low\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "planning_reasoning_effort"):
                load_config(path)

    def test_explicit_missing_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigurationError, "不存在"):
                load_config(Path(directory) / "missing.conf")

    def test_api_key_is_excluded_from_planning_config_dump(self) -> None:
        config = InterpreterRunConfig(
            provider="deepseek",
            model="test-model",
            api_key="must-not-appear",
            run_dir=Path("run"),
        )

        rendered = config.model_dump(mode="json")

        self.assertNotIn("api_key", rendered)
        self.assertNotIn("must-not-appear", str(rendered))


if __name__ == "__main__":
    unittest.main()
