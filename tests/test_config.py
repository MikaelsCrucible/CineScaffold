from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cinescaffold.config import (
    load_config,
    public_config_values,
    resolve_model_settings,
    resolve_stage_option,
    save_config,
)
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
                        "semantic_thinking_mode = disabled",
                        "semantic_max_tokens = 16384",
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
        self.assertEqual(
            resolve_stage_option(config, "semantic", "thinking_mode", None),
            "disabled",
        )
        self.assertEqual(
            resolve_stage_option(config, "semantic", "max_tokens", None),
            16384,
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

    def test_openai_semantic_stage_accepts_medium_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            path.write_text(
                "semantic_provider = openai\nsemantic_reasoning_effort = medium\n",
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual(config.data["semantic_reasoning_effort"], "medium")

    def test_deepseek_rejects_openai_only_reasoning_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            path.write_text(
                "semantic_provider = deepseek\nsemantic_reasoning_effort = medium\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigurationError, "DeepSeek"):
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

    def test_ui_config_write_is_atomic_validated_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".cinescaffold.conf"
            saved = save_config(
                path,
                {
                    "semantic_provider": "openai",
                    "semantic_model": "semantic-model",
                    "openai_api_key": "secret-value",
                    "planning_max_requests": "72",
                    "execution_render_profile": "preview",
                },
            )
            reloaded = load_config(path)
            public = public_config_values(reloaded)

        self.assertEqual(saved.data, reloaded.data)
        self.assertEqual(reloaded.data["planning_max_requests"], "72")
        self.assertNotIn("openai_api_key", public)
        self.assertTrue(public["openai_api_key_configured"])

    def test_ui_config_empty_value_deletes_but_none_preserves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.conf"
            save_config(path, {"model": "first", "api_key": "secret"})
            saved = save_config(path, {"model": "", "api_key": None})
            backup = load_config(path.with_name("settings.conf.bak"))

        self.assertNotIn("model", saved.data)
        self.assertEqual(saved.data["api_key"], "secret")
        self.assertEqual(backup.data["model"], "first")


if __name__ == "__main__":
    unittest.main()
