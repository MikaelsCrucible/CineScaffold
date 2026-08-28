from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cinescaffold.errors import SchemaValidationError


def load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise SchemaValidationError("Schema 根节点必须是对象")
    return value


def validate_model_output(value: Any, schema: dict[str, Any]) -> None:
    """校验本项目使用的 JSON Schema 子集。"""
    _validate(value, schema, schema, "$")


def _validate(value: Any, node: dict[str, Any], root: dict[str, Any], path: str) -> None:
    if "$ref" in node:
        target = _resolve_local_ref(root, node["$ref"])
        _validate(value, target, root, path)
        return

    if "const" in node and value != node["const"]:
        raise SchemaValidationError(f"{path} 必须等于 {node['const']!r}")

    expected = node.get("type")
    if expected is not None and not _matches_type(value, expected):
        raise SchemaValidationError(f"{path} 类型错误，期望 {expected!r}")

    if "enum" in node and value not in node["enum"]:
        raise SchemaValidationError(f"{path} 值不在允许范围内")

    if isinstance(value, dict):
        properties = node.get("properties", {})
        for name in node.get("required", []):
            if name not in value:
                raise SchemaValidationError(f"{path}.{name} 缺失")
        if node.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path} 含额外字段：{', '.join(extras)}")
        for name, child in value.items():
            if name in properties:
                _validate(child, properties[name], root, f"{path}.{name}")

    if isinstance(value, list) and "items" in node:
        for index, item in enumerate(value):
            _validate(item, node["items"], root, f"{path}[{index}]")


def _resolve_local_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"暂不支持外部引用：{reference}")
    current: Any = root
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            raise SchemaValidationError(f"Schema 引用不存在：{reference}")
        current = current[key]
    if not isinstance(current, dict):
        raise SchemaValidationError(f"Schema 引用不是对象：{reference}")
    return current


def _matches_type(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else expected
    return any(_matches_one_type(value, name) for name in names)


def _matches_one_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise SchemaValidationError(f"不支持的 Schema 类型：{name}")
