from __future__ import annotations

import json
import os
import shutil
import tempfile
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
    "semantic_input_cost_per_million",
    "semantic_output_cost_per_million",
    "semantic_cache_read_cost_per_million",
    "semantic_cache_write_cost_per_million",
    "semantic_cost_currency",
    "semantic_price_source",
    "planning_provider",
    "planning_model",
    "planning_thinking_mode",
    "planning_reasoning_effort",
    "planning_model_max_tokens",
    "semantic_timeout",
    "planning_max_requests",
    "planning_max_tool_calls",
    "planning_max_input_tokens",
    "planning_max_context_tokens",
    "planning_max_output_tokens",
    "planning_max_total_tokens",
    "planning_max_seconds",
    "planning_max_commit_attempts",
    "planning_trace_max_event_bytes",
    "planning_trace_max_string_chars",
    "planning_input_cost_per_million",
    "planning_output_cost_per_million",
    "planning_cache_read_cost_per_million",
    "planning_cache_write_cost_per_million",
    "planning_cost_currency",
    "planning_price_source",
    "execution_blender_path",
    "execution_mcp_command",
    "execution_render_backend",
    "execution_render_profile",
    "execution_render_timeout_seconds",
}

SECRET_KEYS = {"api_key", "openai_api_key", "deepseek_api_key"}
INTEGER_KEYS = {
    "semantic_max_tokens",
    "planning_model_max_tokens",
    "planning_max_requests",
    "planning_max_tool_calls",
    "planning_max_input_tokens",
    "planning_max_context_tokens",
    "planning_max_output_tokens",
    "planning_max_total_tokens",
    "planning_max_commit_attempts",
    "planning_trace_max_event_bytes",
    "planning_trace_max_string_chars",
}
FLOAT_KEYS = {"semantic_timeout", "planning_max_seconds", "execution_render_timeout_seconds"}
PRICE_KEYS = {
    f"{stage}_{kind}_cost_per_million"
    for stage in ("semantic", "planning")
    for kind in ("input", "output", "cache_read", "cache_write")
}
CONFIG_KEY_ORDER = tuple(sorted(ALLOWED_KEYS, key=lambda key: (key != "config_version", key)))


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
    full_key = f"{stage}_{key}"
    if full_key in INTEGER_KEYS and value is not None:
        return int(value)
    if full_key in FLOAT_KEYS and value is not None:
        return float(value)
    return value


def save_config(path: Path, updates: dict[str, str | None]) -> LoadedConfig:
    """合并并原子写回简单配置；None 表示保留，空字符串表示删除。"""

    existing = load_config(path) if path.is_file() else LoadedConfig(path=None, data={})
    merged = merge_config(existing, updates)
    data = merged.data

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = _render_config(data)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            temporary.chmod(0o600)
        if target.is_file():
            backup = target.with_name(f"{target.name}.bak")
            shutil.copy2(target, backup)
            if os.name != "nt":
                backup.chmod(0o600)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return LoadedConfig(path=target, data=data)


def merge_config(
    config: LoadedConfig,
    updates: dict[str, str | None],
) -> LoadedConfig:
    """校验并合并内存配置，不触碰磁盘。"""

    data = dict(config.data)
    for key, value in updates.items():
        normalized_key = key.lower()
        if normalized_key not in ALLOWED_KEYS:
            raise ConfigurationError(f"配置存在未知字段：{normalized_key}")
        if value is None:
            continue
        normalized_value = str(value).strip()
        if normalized_value:
            data[normalized_key] = normalized_value
        else:
            data.pop(normalized_key, None)
    data.setdefault("config_version", "1")
    _validate_config(data)
    return LoadedConfig(path=config.path, data=data)


def public_config_values(config: LoadedConfig) -> dict[str, str | bool]:
    """UI 只接收非密钥值以及密钥是否存在。"""

    values: dict[str, str | bool] = {
        key: value for key, value in config.data.items() if key not in SECRET_KEYS
    }
    for key in SECRET_KEYS:
        values[f"{key}_configured"] = bool(config.data.get(key))
    return values


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
    for key in INTEGER_KEYS:
        max_tokens = data.get(key)
        if max_tokens is None:
            continue
        try:
            parsed = int(max_tokens)
        except ValueError as error:
            raise ConfigurationError(f"{key} 必须是正整数") from error
        if parsed < 1:
            raise ConfigurationError(f"{key} 必须是正整数")
    for key in FLOAT_KEYS:
        value = data.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError as error:
            raise ConfigurationError(f"{key} 必须是正数") from error
        if parsed <= 0:
            raise ConfigurationError(f"{key} 必须是正数")
    for key in PRICE_KEYS:
        value = data.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError as error:
            raise ConfigurationError(f"{key} 必须是非负数") from error
        if parsed < 0:
            raise ConfigurationError(f"{key} 必须是非负数")
    if data.get("execution_render_backend") not in (None, "background", "mcp"):
        raise ConfigurationError("execution_render_backend 必须是 background 或 mcp")
    if data.get("execution_render_profile") not in (None, "preview", "control"):
        raise ConfigurationError("execution_render_profile 必须是 preview 或 control")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ConfigurationError("配置中的双引号字符串转义无效") from error
        if not isinstance(parsed, str):
            raise ConfigurationError("配置中的双引号值必须是字符串")
        return parsed
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _render_config(data: dict[str, str]) -> str:
    lines = ["# CineScaffold 本地配置；请勿提交包含 API Key 的文件。"]
    for key in CONFIG_KEY_ORDER:
        if key in data:
            lines.append(f"{key} = {_quote_if_needed(data[key])}")
    return "\n".join(lines) + "\n"


def _quote_if_needed(value: str) -> str:
    if not value or value != value.strip() or "#" in value:
        return json.dumps(value, ensure_ascii=False)
    return value
