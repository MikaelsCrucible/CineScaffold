from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cinescaffold.errors import ConfigurationError


DEFAULT_CONFIG_PATH = Path(".cinescaffold.conf")
PROVIDERS = {"mock", "openai", "deepseek"}
ALLOWED_KEYS = {
    "config_version",
    "provider",
    "model",
    "api_key",
    "base_url",
    "openai_api_key",
    "openai_base_url",
    "deepseek_api_key",
    "deepseek_base_url",
    "semantic_provider",
    "semantic_model",
    "semantic_thinking_mode",
    "semantic_reasoning_effort",
    "semantic_max_tokens",
    "planning_provider",
    "planning_model",
    "planning_thinking_mode",
    "planning_reasoning_effort",
    "planning_model_max_tokens",
}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path | None
    data: dict[str, str]


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str | None
    base_url: str | None
    api_key: str | None


def load_config(path: Path | None) -> LoadedConfig:
    explicit = path is not None
    selected = path or DEFAULT_CONFIG_PATH
    if not selected.is_file():
        if explicit:
            raise ConfigurationError(f"配置文件不存在：{selected}")
        return LoadedConfig(path=None, data={})
    data: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        selected.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"配置文件第 {line_number} 行缺少 =")
        key, value = (item.strip() for item in line.split("=", 1))
        key = key.lower()
        if key not in ALLOWED_KEYS:
            raise ConfigurationError(f"配置文件第 {line_number} 行存在未知字段：{key}")
        if key in data:
            raise ConfigurationError(f"配置字段重复：{key}")
        data[key] = _unquote(value)
    _validate_config(data)
    return LoadedConfig(path=selected.resolve(), data=data)


def resolve_model_settings(
    config: LoadedConfig,
    stage: str,
    *,
    cli_provider: str | None,
    cli_model: str | None,
    cli_base_url: str | None,
) -> ModelSettings:
    provider = cli_provider or config.data.get(f"{stage}_provider") or config.data.get("provider") or "mock"
    if provider not in PROVIDERS:
        raise ConfigurationError(f"不支持的 Provider：{provider}")
    model = cli_model or config.data.get(f"{stage}_model") or config.data.get("model")
    base_url = (
        cli_base_url
        or config.data.get(f"{provider}_base_url")
        or config.data.get("base_url")
    )
    api_key = config.data.get(f"{provider}_api_key") or config.data.get("api_key")
    environment_name = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }.get(provider)
    if api_key is None and environment_name:
        api_key = os.environ.get(environment_name)
    return ModelSettings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def resolve_stage_option(
    config: LoadedConfig,
    stage: str,
    key: str,
    cli_value: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    value = config.data.get(f"{stage}_{key}")
    if key in {"model_max_tokens", "max_tokens"} and value is not None:
        return int(value)
    return value


def _validate_config(data: dict[str, str]) -> None:
    version = data.get("config_version", "1")
    if version != "1":
        raise ConfigurationError(f"不支持的 config_version：{version}")
    for key in ("provider", "semantic_provider", "planning_provider"):
        value = data.get(key)
        if value is not None and value not in PROVIDERS:
            raise ConfigurationError(f"{key} 不支持：{value}")
    for stage in ("semantic", "planning"):
        thinking_mode = data.get(f"{stage}_thinking_mode")
        if thinking_mode not in (None, "enabled", "disabled"):
            raise ConfigurationError(
                f"{stage}_thinking_mode 必须是 enabled 或 disabled；"
                f"low、high、max 请填写 {stage}_reasoning_effort"
            )
        reasoning_effort = data.get(f"{stage}_reasoning_effort")
        if reasoning_effort not in (
            None,
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise ConfigurationError(
                f"{stage}_reasoning_effort 必须是 none、low、medium、high、xhigh 或 max"
            )
        provider = data.get(f"{stage}_provider") or data.get("provider")
        if provider == "deepseek" and reasoning_effort not in (
            None,
            "low",
            "high",
            "max",
        ):
            raise ConfigurationError(
                f"DeepSeek 的 {stage}_reasoning_effort 只支持 low、high 或 max"
            )
    for key in ("semantic_max_tokens", "planning_model_max_tokens"):
        max_tokens = data.get(key)
        if max_tokens is None:
            continue
        try:
            parsed = int(max_tokens)
        except ValueError as error:
            raise ConfigurationError(f"{key} 必须是正整数") from error
        if parsed < 1:
            raise ConfigurationError(f"{key} 必须是正整数")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
