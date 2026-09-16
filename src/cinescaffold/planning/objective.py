from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cinescaffold.relationships import (
    CANONICAL_RELATIONSHIP_TYPES,
    normalize_scene_relationships,
)
from cinescaffold.semantic_rules import (
    objective_translation_parameters,
    validate_supported_semantic_fields,
)

OBJECTIVE_CONTENT_FIELDS = (
    "subjects",
    "subject_motion",
    "scene_dynamics",
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
    translation_rules_sha256: str
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
    schema_version: Literal["0.8"]
    source_brief_sha256: str
    subjects: list[dict[str, Any]]
    subject_motion: list[dict[str, Any]]
    scene_dynamics: dict[str, Any]
    scene_design: dict[str, Any]
    composition: dict[str, Any]
    camera: dict[str, Any]
    timeline: dict[str, Any]
    uncertainties: list[dict[str, Any]]
    translation_parameters: dict[str, Any]
    explicit_requirements: list[ObjectiveRequirement] = Field(default_factory=list)
    source_metadata: BriefSourceMetadata


class ObjectiveProjection(_StrictModel):
    objective_brief: ObjectivePlanningBrief
    ignored_subjective_fields: list[IgnoredSubjectiveField]


def has_subject_spatial_motion(objective: ObjectivePlanningBrief) -> bool:
    """Return whether subjects translate in space, independent of camera motion."""

    for motion in objective.subject_motion:
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if isinstance(semantics, dict) and semantics.get("motion_mode") in {
            "self_propelled",
            "carried",
        }:
            return True
    return any(
        isinstance(item, dict)
        and item.get("motion_type") not in {None, "", "static", "interactive"}
        for item in objective.translation_parameters.get("motions", [])
    )


def project_objective_brief(brief: dict[str, Any]) -> ObjectiveProjection:
    """在模型调用前剥离主观维度和原始提示词。"""
    schema_version = brief.get("schema_version")
    if schema_version != "0.8":
        raise ValueError("Agent 1 仅支持当前 Cinematic Brief v0.8")
    content = brief.get("content")
    provenance = brief.get("provenance")
    if not isinstance(content, dict) or not isinstance(provenance, dict):
        raise ValueError("Cinematic Brief 缺少 content 或 provenance")
    translation_parameters = brief.get("translation_parameters")
    if not isinstance(translation_parameters, dict):
        raise ValueError("Cinematic Brief v0.8 缺少 translation_parameters")
    if not _optional_string(provenance.get("translation_rules_sha256")):
        raise ValueError("Cinematic Brief v0.8 缺少 translation_rules_sha256")
    missing = [name for name in OBJECTIVE_CONTENT_FIELDS if name not in content]
    if missing:
        raise ValueError(f"Cinematic Brief 缺少客观字段：{', '.join(missing)}")
    validate_supported_semantic_fields(content)

    source_metadata = BriefSourceMetadata(
        provider=_required_string(provenance, "provider"),
        model=_required_string(provenance, "model"),
        parser_prompt_version=_required_string(provenance, "parser_prompt_version"),
        rules_sha256=_required_string(provenance, "rules_sha256"),
        translation_rules_sha256=_required_string(
            provenance, "translation_rules_sha256"
        ),
        response_id=_optional_string(provenance.get("response_id")),
    )
    objective_values = {
        name: deepcopy(content[name]) for name in OBJECTIVE_CONTENT_FIELDS
    }
    normalize_scene_relationships(objective_values)
    _validate_canonical_relationships(objective_values)
    requirements: list[ObjectiveRequirement] = []
    for name in OBJECTIVE_CONTENT_FIELDS:
        _collect_explicit_requirements(
            objective_values[name],
            f"content.{name}",
            requirements,
        )

    objective_brief = ObjectivePlanningBrief(
        schema_version=schema_version,
        source_brief_sha256=_canonical_sha256(brief),
        translation_parameters=objective_translation_parameters(
            translation_parameters, content
        ),
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
    environmental_motion = content.get("scene_design", {}).get("environmental_motion")
    if environmental_motion:
        ignored.append(
            IgnoredSubjectiveField(
                path="content.scene_design.environmental_motion",
                reason=(
                    "环境特效运动属于最终视频生成层；当前几何白模不把它错误绑定到地面实体"
                ),
                content_sha256=_canonical_sha256(environmental_motion),
            )
        )
    return ObjectiveProjection(
        objective_brief=objective_brief,
        ignored_subjective_fields=ignored,
    )


def _validate_canonical_relationships(content: dict[str, Any]) -> None:
    relationships = content.get("scene_design", {}).get("relationships", [])
    subject_ids = {
        item.get("id")
        for item in content.get("subjects", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    for index, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            raise ValueError(f"v0.8 relationship[{index}] 必须是对象")
        relation_type = relationship.get("type")
        if relation_type not in CANONICAL_RELATIONSHIP_TYPES:
            raise ValueError(
                f"v0.8 relationship[{index}] 使用了 Toolkit 不支持的类型："
                f"{relation_type}"
            )
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        if subject_id not in subject_ids or reference_id not in subject_ids:
            raise ValueError(f"v0.8 relationship[{index}] 必须引用两个已声明实体")
        if subject_id == reference_id:
            raise ValueError(f"v0.8 relationship[{index}] 不得自引用")


def _collect_explicit_requirements(
    value: Any,
    path: str,
    output: list[ObjectiveRequirement],
) -> None:
    if path.startswith("content.scene_design.environmental_motion"):
        return
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
    if (
        value.get("duration_source_status") == "explicit"
        and value.get("duration_seconds") is not None
    ):
        output.append(
            ObjectiveRequirement(
                path=f"{path}.duration_seconds",
                value=deepcopy(value["duration_seconds"]),
            )
        )
    if (
        value.get("duration_source_status") == "explicit"
        and value.get("duration_range_seconds") is not None
    ):
        output.append(
            ObjectiveRequirement(
                path=f"{path}.duration_range_seconds",
                value=deepcopy(value["duration_range_seconds"]),
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
