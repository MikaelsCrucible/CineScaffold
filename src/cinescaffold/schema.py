from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from cinescaffold.errors import SchemaValidationError


SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "default",
        "description",
        "discriminator",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "oneOf",
        "pattern",
        "prefixItems",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)


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

    for branch in node.get("allOf", []):
        _validate(value, branch, root, path)

    if "anyOf" in node:
        _validate_union(
            value,
            node["anyOf"],
            root,
            path,
            require_exactly_one=False,
            description=node.get("description"),
        )
    if "oneOf" in node:
        _validate_union(
            value,
            node["oneOf"],
            root,
            path,
            require_exactly_one=True,
            description=node.get("description"),
        )

    if "const" in node and value != node["const"]:
        raise SchemaValidationError(f"{path} 必须等于 {node['const']!r}")

    expected = node.get("type")
    if expected is not None and not _matches_type(value, expected):
        raise SchemaValidationError(f"{path} 类型错误，期望 {expected!r}")

    if "enum" in node and value not in node["enum"]:
        raise SchemaValidationError(f"{path} 值不在允许范围内")

    if isinstance(value, str):
        if "minLength" in node and len(value) < int(node["minLength"]):
            raise SchemaValidationError(f"{path} 长度小于 {node['minLength']}")
        if "maxLength" in node and len(value) > int(node["maxLength"]):
            raise SchemaValidationError(f"{path} 长度大于 {node['maxLength']}")
        pattern = node.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise SchemaValidationError(f"{path} 不匹配要求格式")

    if _is_number(value):
        _validate_number(value, node, path)

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

    if isinstance(value, list):
        if "minItems" in node and len(value) < int(node["minItems"]):
            raise SchemaValidationError(f"{path} 项目数小于 {node['minItems']}")
        if "maxItems" in node and len(value) > int(node["maxItems"]):
            raise SchemaValidationError(f"{path} 项目数大于 {node['maxItems']}")
        if node.get("uniqueItems") is True and not _items_are_unique(value):
            raise SchemaValidationError(f"{path} 含重复项目")
        for index, child in enumerate(node.get("prefixItems", [])):
            if index < len(value):
                _validate(value[index], child, root, f"{path}[{index}]")
        if "items" in node:
            for index, item in enumerate(value):
                _validate(item, node["items"], root, f"{path}[{index}]")


def _validate_union(
    value: Any,
    branches: Any,
    root: dict[str, Any],
    path: str,
    *,
    require_exactly_one: bool,
    description: Any = None,
) -> None:
    if not isinstance(branches, list) or not branches:
        raise SchemaValidationError(f"{path} 的组合 Schema 必须是非空数组")
    matches = 0
    first_error: SchemaValidationError | None = None
    for branch in branches:
        if not isinstance(branch, dict):
            raise SchemaValidationError(f"{path} 的组合 Schema 分支必须是对象")
        try:
            _validate(value, branch, root, path)
        except SchemaValidationError as error:
            if first_error is None:
                first_error = error
        else:
            matches += 1
    valid = matches == 1 if require_exactly_one else matches >= 1
    if valid:
        return
    kind = "oneOf" if require_exactly_one else "anyOf"
    detail = f"；首个分支错误：{first_error}" if first_error is not None else ""
    contract = _contract_prefix(description)
    raise SchemaValidationError(
        f"{contract}{path} 不满足 {kind}（匹配 {matches} 个分支）{detail}"
    )


def _validate_number(value: int | float, node: dict[str, Any], path: str) -> None:
    number = float(value)
    if not math.isfinite(number):
        raise SchemaValidationError(f"{path} 必须是有限数值")
    if "minimum" in node and number < float(node["minimum"]):
        raise SchemaValidationError(f"{path} 小于最小值 {node['minimum']}")
    if "maximum" in node and number > float(node["maximum"]):
        raise SchemaValidationError(f"{path} 大于最大值 {node['maximum']}")
    if "exclusiveMinimum" in node and number <= float(node["exclusiveMinimum"]):
        raise SchemaValidationError(
            f"{path} 必须大于 {node['exclusiveMinimum']}"
        )
    if "exclusiveMaximum" in node and number >= float(node["exclusiveMaximum"]):
        raise SchemaValidationError(
            f"{path} 必须小于 {node['exclusiveMaximum']}"
        )


def _items_are_unique(value: list[Any]) -> bool:
    serialized = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in value
    ]
    return len(serialized) == len(set(serialized))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _contract_prefix(description: Any) -> str:
    if not isinstance(description, str):
        return ""
    match = re.match(r"(\[SEM-[A-Z-]+\])", description)
    return f"{match.group(1)} " if match is not None else ""


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
