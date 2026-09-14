from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cinescaffold.relationships import (
    CANONICAL_RELATIONSHIP_TYPES,
    normalize_scene_relationships,
)
from cinescaffold.semantic_rules import objective_translation_parameters

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
    translation_rules_sha256: str | None = None
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
    schema_version: Literal["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"] = "0.1"
    source_brief_sha256: str
    subjects: list[dict[str, Any]]
    subject_motion: list[dict[str, Any]]
    scene_dynamics: dict[str, Any] = Field(
        default_factory=lambda: {
            "mode": "dynamic",
            "source_status": "default",
            "reason": "旧版 Objective 兼容默认",
        }
    )
    scene_design: dict[str, Any]
    composition: dict[str, Any]
    camera: dict[str, Any]
    timeline: dict[str, Any]
    uncertainties: list[dict[str, Any]]
    translation_parameters: dict[str, Any] | None = None
    explicit_requirements: list[ObjectiveRequirement] = Field(default_factory=list)
    source_metadata: BriefSourceMetadata

    @model_validator(mode="before")
    @classmethod
    def restore_legacy_scene_dynamics(cls, value: Any) -> Any:
        """Derive subject-only dynamics when loading pre-field artifacts."""

        if not isinstance(value, dict) or "scene_dynamics" in value:
            return value
        normalized = dict(value)
        legacy_dynamic = _legacy_subject_scene_is_dynamic(
            {"subject_motion": normalized.get("subject_motion", [])},
            normalized.get("translation_parameters"),
        )
        normalized["scene_dynamics"] = {
            "mode": "dynamic" if legacy_dynamic else "static",
            "source_status": "default",
            "reason": "旧版 Objective 产物按主体运动与状态转换兼容恢复",
        }
        return normalized


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
    if any(
        isinstance(item, dict)
        and item.get("motion_type") not in {None, "", "static", "interactive"}
        for item in (objective.translation_parameters or {}).get("motions", [])
    ):
        return True
    # v0.1/v0.2 predate typed motion semantics.  Their structured trajectory
    # field is the least ambiguous compatibility signal; do not reparse action
    # prose or invent a target from it.
    return any(
        isinstance(motion, dict)
        and not isinstance(motion.get("motion_semantics"), dict)
        and _legacy_trajectory_implies_translation(motion.get("trajectory"))
        for motion in objective.subject_motion
    )


def project_objective_brief(brief: dict[str, Any]) -> ObjectiveProjection:
    """在模型调用前剥离主观维度和原始提示词。"""
    schema_version = brief.get("schema_version")
    if schema_version not in {"0.1", "0.2", "0.3", "0.4", "0.5", "0.6"}:
        raise ValueError("Agent 1 仅支持 Cinematic Brief v0.1–v0.6")
    content = brief.get("content")
    provenance = brief.get("provenance")
    if not isinstance(content, dict) or not isinstance(provenance, dict):
        raise ValueError("Cinematic Brief 缺少 content 或 provenance")
    translation_parameters = brief.get("translation_parameters")
    if schema_version in {"0.2", "0.3", "0.4", "0.5", "0.6"} and not isinstance(
        translation_parameters, dict
    ):
        raise ValueError(
            f"Cinematic Brief v{schema_version} 缺少 translation_parameters"
        )
    if schema_version in {"0.2", "0.3", "0.4", "0.5", "0.6"} and not _optional_string(
        provenance.get("translation_rules_sha256")
    ):
        raise ValueError(
            f"Cinematic Brief v{schema_version} 缺少 translation_rules_sha256"
        )

    if "scene_dynamics" not in content and schema_version != "0.6":
        content = deepcopy(content)
        legacy_dynamic = _legacy_subject_scene_is_dynamic(
            content,
            translation_parameters,
        )
        content["scene_dynamics"] = {
            "mode": "dynamic" if legacy_dynamic else "static",
            "source_status": "default",
            "reason": "旧版 Brief 按主体运动与状态转换兼容投影",
        }
    missing = [name for name in OBJECTIVE_CONTENT_FIELDS if name not in content]
    if missing:
        raise ValueError(f"Cinematic Brief 缺少客观字段：{', '.join(missing)}")

    source_metadata = BriefSourceMetadata(
        provider=_required_string(provenance, "provider"),
        model=_required_string(provenance, "model"),
        parser_prompt_version=_required_string(provenance, "parser_prompt_version"),
        rules_sha256=_required_string(provenance, "rules_sha256"),
        translation_rules_sha256=_optional_string(
            provenance.get("translation_rules_sha256")
        ),
        response_id=_optional_string(provenance.get("response_id")),
    )
    objective_values = {
        name: deepcopy(content[name]) for name in OBJECTIVE_CONTENT_FIELDS
    }
    normalize_scene_relationships(objective_values)
    if schema_version == "0.6":
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
        translation_parameters=(
            objective_translation_parameters(translation_parameters, content)
            if schema_version in {"0.2", "0.3", "0.4", "0.5", "0.6"}
            else None
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
            raise ValueError(f"v0.6 relationship[{index}] 必须是对象")
        relation_type = relationship.get("type")
        if relation_type not in CANONICAL_RELATIONSHIP_TYPES:
            raise ValueError(
                f"v0.6 relationship[{index}] 使用了 Toolkit 不支持的类型："
                f"{relation_type}"
            )
        subject_id = relationship.get("subject_id")
        reference_id = relationship.get("reference_id")
        if subject_id not in subject_ids or reference_id not in subject_ids:
            raise ValueError(f"v0.6 relationship[{index}] 必须引用两个已声明实体")
        if subject_id == reference_id:
            raise ValueError(f"v0.6 relationship[{index}] 不得自引用")


def _legacy_subject_scene_is_dynamic(
    content: dict[str, Any],
    translation_parameters: dict[str, Any] | None,
) -> bool:
    """Apply the current subject-only dynamics contract to legacy briefs."""

    for motion in content.get("subject_motion", []):
        if not isinstance(motion, dict):
            continue
        semantics = motion.get("motion_semantics")
        if not isinstance(semantics, dict):
            continue
        postconditions = semantics.get("postconditions")
        changes_state = isinstance(postconditions, dict) and (
            postconditions.get("contained_by_id") is not None
            or postconditions.get("external_visibility") in {"visible", "hidden"}
        )
        if semantics.get("motion_mode") != "stationary" or changes_state:
            return True
    parameters = translation_parameters or {}
    if any(
        isinstance(item, dict)
        and item.get("motion_type") not in {None, "", "static", "interactive"}
        for item in parameters.get("motions", [])
    ):
        return True
    return any(
        isinstance(motion, dict)
        and not isinstance(motion.get("motion_semantics"), dict)
        and _legacy_trajectory_implies_translation(motion.get("trajectory"))
        for motion in content.get("subject_motion", [])
    )


def _legacy_trajectory_implies_translation(value: Any) -> bool:
    """Interpret only the legacy trajectory slot, never free-form action prose."""

    if not isinstance(value, dict):
        return False
    normalized = str(value.get("value") or "").strip().lower()
    if not normalized:
        return False
    return normalized not in {
        "静止",
        "无位移",
        "无",
        "none",
        "static",
        "stationary",
    }


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
