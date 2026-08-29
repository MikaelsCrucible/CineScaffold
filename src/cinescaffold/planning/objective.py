from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


OBJECTIVE_CONTENT_FIELDS = (
    "subjects",
    "subject_motion",
    "scene_design",
    "composition",
    "camera",
    "timeline",
    "uncertainties",
)
SUBJECTIVE_CONTENT_FIELDS = ("summary", "mood")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BriefSourceMetadata(_StrictModel):
    provider: str
    model: str
    parser_prompt_version: str
    rules_sha256: str
    response_id: str | None


class ObjectiveRequirement(_StrictModel):
    path: str
    value: Any
    source_text: str | None = None


class IgnoredSubjectiveField(_StrictModel):
    path: str
    reason: str
    content_sha256: str


class ObjectivePlanningBrief(_StrictModel):
    schema_version: Literal["0.1"] = "0.1"
    source_brief_sha256: str
    subjects: list[dict[str, Any]]
    subject_motion: list[dict[str, Any]]
    scene_design: dict[str, Any]
    composition: dict[str, Any]
    camera: dict[str, Any]
    timeline: dict[str, Any]
    uncertainties: list[dict[str, Any]]
    explicit_requirements: list[ObjectiveRequirement] = Field(default_factory=list)
    source_metadata: BriefSourceMetadata


class ObjectiveProjection(_StrictModel):
    objective_brief: ObjectivePlanningBrief
    ignored_subjective_fields: list[IgnoredSubjectiveField]


def project_objective_brief(brief: dict[str, Any]) -> ObjectiveProjection:
    """在模型调用前剥离主观维度和原始提示词。"""
    if brief.get("schema_version") != "0.1":
        raise ValueError("Agent 1 仅支持 Cinematic Brief v0.1")
    content = brief.get("content")
    provenance = brief.get("provenance")
    if not isinstance(content, dict) or not isinstance(provenance, dict):
        raise ValueError("Cinematic Brief 缺少 content 或 provenance")

    missing = [name for name in OBJECTIVE_CONTENT_FIELDS if name not in content]
    if missing:
        raise ValueError(f"Cinematic Brief 缺少客观字段：{', '.join(missing)}")

    source_metadata = BriefSourceMetadata(
        provider=_required_string(provenance, "provider"),
        model=_required_string(provenance, "model"),
        parser_prompt_version=_required_string(provenance, "parser_prompt_version"),
        rules_sha256=_required_string(provenance, "rules_sha256"),
        response_id=_optional_string(provenance.get("response_id")),
    )
    objective_values = {name: deepcopy(content[name]) for name in OBJECTIVE_CONTENT_FIELDS}
    requirements: list[ObjectiveRequirement] = []
    for name in OBJECTIVE_CONTENT_FIELDS:
        _collect_explicit_requirements(
            objective_values[name],
            f"content.{name}",
            requirements,
        )

    objective_brief = ObjectivePlanningBrief(
        source_brief_sha256=_canonical_sha256(brief),
        explicit_requirements=requirements,
        source_metadata=source_metadata,
        **objective_values,
    )
    ignored = [
        IgnoredSubjectiveField(
            path=f"content.{name}",
            reason="Agent 1 只处理可落为位置、运动或摄影机状态的客观内容",
            content_sha256=_canonical_sha256(content.get(name)),
        )
        for name in SUBJECTIVE_CONTENT_FIELDS
        if name in content
    ]
    return ObjectiveProjection(
        objective_brief=objective_brief,
        ignored_subjective_fields=ignored,
    )


def _collect_explicit_requirements(
    value: Any,
    path: str,
    output: list[ObjectiveRequirement],
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_explicit_requirements(item, f"{path}[{index}]", output)
        return
    if not isinstance(value, dict):
        return

    if value.get("source_status") == "explicit":
        requirement_value = value.get("value", _without_source_fields(value))
        if requirement_value is not None:
            output.append(
                ObjectiveRequirement(
                    path=path,
                    value=deepcopy(requirement_value),
                    source_text=_optional_string(value.get("source_text")),
                )
            )
    if value.get("duration_source_status") == "explicit" and value.get("duration_seconds") is not None:
        output.append(
            ObjectiveRequirement(
                path=f"{path}.duration_seconds",
                value=deepcopy(value["duration_seconds"]),
            )
        )
    for name, child in value.items():
        if name not in {"source_status", "source_text"}:
            _collect_explicit_requirements(child, f"{path}.{name}", output)


def _without_source_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key not in {"source_status", "source_text"}
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"Cinematic Brief provenance.{name} 缺失")
    return item


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None
