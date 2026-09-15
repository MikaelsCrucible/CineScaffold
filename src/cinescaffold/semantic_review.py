from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REVIEW_SELECTION_POLICY = "attention_only_no_semantic_inference"
_INDEPENDENT_REVIEW_CODE = "independent_semantic_review_required"
_PATH_PATTERN = re.compile(
    r"(?:\$(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*)"
    r"|(?:subject_motion|timeline|scene_design|scene_dynamics|camera|composition)"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*|\[\d+\])*"
)


@dataclass(frozen=True)
class SemanticReviewCatalog:
    version: str
    max_selected_rules: int
    rules: tuple[dict[str, Any], ...]
    sha256: str


def load_semantic_review_catalog(path: Path) -> SemanticReviewCatalog:
    """Load and strictly validate the deterministic review-rule registry."""

    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("语义审查规则目录根节点必须是对象")
    allowed_root = {"version", "max_selected_rules", "rules"}
    if set(value) != allowed_root:
        raise ValueError("语义审查规则目录根字段不完整或包含未知字段")
    version = value["version"]
    maximum = value["max_selected_rules"]
    rules = value["rules"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("语义审查规则目录 version 必须是非空字符串")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError("语义审查规则目录 max_selected_rules 必须是正整数")
    if not isinstance(rules, list) or not rules:
        raise ValueError("语义审查规则目录 rules 必须是非空数组")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, rule in enumerate(rules):
        normalized.append(_validate_review_rule(rule, index, seen))
    return SemanticReviewCatalog(
        version=version,
        max_selected_rules=maximum,
        rules=tuple(normalized),
        sha256=f"sha256:{hashlib.sha256(raw).hexdigest()}",
    )


def build_semantic_review_packet(
    source_text: str,
    draft: dict[str, Any],
    diagnostics: list[dict[str, str]],
    catalog: SemanticReviewCatalog,
) -> dict[str, Any]:
    """Select bounded review guidance without converting cues into semantics."""

    source_folded = source_text.casefold()
    draft_text = "\n".join(_collect_strings(draft)).casefold()
    structural_signals = _draft_structural_signals(draft)
    diagnostic_codes = {
        value
        for item in diagnostics
        if isinstance((value := item.get("code")), str)
    }
    diagnostic_paths = sorted(
        {
            path
            for item in diagnostics
            for path in _extract_paths(item.get("message", ""))
        }
    )

    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for rule in catalog.rules:
        evidence, score = _match_rule(
            rule,
            source_folded=source_folded,
            draft_text=draft_text,
            structural_signals=structural_signals,
            diagnostic_codes=diagnostic_codes,
            diagnostic_paths=diagnostic_paths,
        )
        if score is None:
            continue
        confirmed_codes = set(evidence["diagnostic_codes"]) - {
            _INDEPENDENT_REVIEW_CODE
        }
        finding = {
            "kind": "confirmed_error" if confirmed_codes else "review_risk",
            "rule_id": rule["id"],
            "title": rule["title"],
            "trigger_evidence": evidence,
            "emphasis": rule["emphasis"],
            "audit_questions": rule["audit_questions"],
        }
        ranked.append((score, rule["id"], finding))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    findings = [item[2] for item in ranked[: catalog.max_selected_rules]]
    selected_codes = {
        code
        for finding in findings
        for code in finding["trigger_evidence"]["diagnostic_codes"]
    }
    unmatched = [
        item
        for item in diagnostics
        if item.get("code") not in selected_codes
    ]
    return {
        "review_packet_version": "0.1",
        "catalog_version": catalog.version,
        "selection_policy": REVIEW_SELECTION_POLICY,
        "findings": findings,
        "unmatched_diagnostics": unmatched,
    }


def _validate_review_rule(
    value: object,
    index: int,
    seen: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"语义审查规则 rules[{index}] 必须是对象")
    allowed = {
        "id",
        "title",
        "priority",
        "always",
        "diagnostic_codes",
        "diagnostic_path_prefixes",
        "text_cues",
        "structural_signals",
        "emphasis",
        "audit_questions",
    }
    if set(value) != allowed:
        raise ValueError(f"语义审查规则 rules[{index}] 字段不完整或包含未知字段")
    rule_id = value["id"]
    if not isinstance(rule_id, str) or not rule_id.strip() or rule_id in seen:
        raise ValueError(f"语义审查规则 rules[{index}].id 必须非空且唯一")
    seen.add(rule_id)
    if not isinstance(value["priority"], int) or isinstance(value["priority"], bool):
        raise ValueError(f"语义审查规则 {rule_id} priority 必须是整数")
    if not isinstance(value["always"], bool):
        raise ValueError(f"语义审查规则 {rule_id} always 必须是布尔值")
    for name in (
        "diagnostic_codes",
        "diagnostic_path_prefixes",
        "text_cues",
        "structural_signals",
        "audit_questions",
    ):
        items = value[name]
        if not isinstance(items, list) or any(
            not isinstance(item, str) or not item.strip() for item in items
        ):
            raise ValueError(f"语义审查规则 {rule_id}.{name} 必须是字符串数组")
    for name in ("title", "emphasis"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"语义审查规则 {rule_id}.{name} 必须是非空字符串")
    if not value["audit_questions"]:
        raise ValueError(f"语义审查规则 {rule_id}.audit_questions 不能为空")
    if not (
        value["always"]
        or value["diagnostic_codes"]
        or value["diagnostic_path_prefixes"]
        or value["text_cues"]
        or value["structural_signals"]
    ):
        raise ValueError(f"语义审查规则 {rule_id} 至少需要一种触发条件")
    return value


def _match_rule(
    rule: dict[str, Any],
    *,
    source_folded: str,
    draft_text: str,
    structural_signals: set[str],
    diagnostic_codes: set[str],
    diagnostic_paths: list[str],
) -> tuple[dict[str, list[str]], int | None]:
    matched_codes = sorted(diagnostic_codes & set(rule["diagnostic_codes"]))
    matched_paths = sorted(
        path
        for path in diagnostic_paths
        if any(
            _path_matches_prefix(path, prefix)
            for prefix in rule["diagnostic_path_prefixes"]
        )
    )
    source_cues = sorted(
        cue for cue in rule["text_cues"] if cue.casefold() in source_folded
    )
    draft_cues = sorted(
        cue for cue in rule["text_cues"] if cue.casefold() in draft_text
    )
    matched_signals = sorted(structural_signals & set(rule["structural_signals"]))
    if not (
        rule["always"]
        or matched_codes
        or matched_paths
        or source_cues
        or draft_cues
        or matched_signals
    ):
        return {}, None

    evidence = {
        "diagnostic_codes": matched_codes,
        "diagnostic_paths": matched_paths,
        "source_cues": source_cues,
        "draft_cues": draft_cues,
        "structural_signals": matched_signals,
    }
    score = int(rule["priority"])
    score += 100 * len(matched_codes)
    score += 80 * len(matched_paths)
    score += 30 * len(source_cues)
    score += 20 * len(draft_cues)
    score += 10 * len(matched_signals)
    return evidence, score


def _extract_paths(message: str) -> set[str]:
    if not isinstance(message, str):
        return set()
    return set(_PATH_PATTERN.findall(message))


def _path_matches_prefix(path: str, prefix: str) -> bool:
    if prefix == "$":
        return path == "$" or path.startswith("$.") or path.startswith("$[")
    normalized_path = path.removeprefix("$.")
    normalized_prefix = prefix.removeprefix("$.")
    return normalized_path == normalized_prefix or normalized_path.startswith(
        normalized_prefix + "."
    ) or normalized_path.startswith(normalized_prefix + "[")


def _collect_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _collect_strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _collect_strings(item)]
    return []


def _draft_structural_signals(draft: dict[str, Any]) -> set[str]:
    signals: set[str] = set()
    subjects = draft.get("subjects")
    motions = draft.get("subject_motion")
    relationships = draft.get("scene_design", {}).get("relationships")
    timeline = draft.get("timeline")
    uncertainties = draft.get("uncertainties")
    if isinstance(subjects, list) and len(subjects) >= 2:
        signals.add("multiple_subjects")
    if isinstance(motions, list):
        moving_subjects = {
            motion.get("subject_id")
            for motion in motions
            if isinstance(motion, dict)
            and isinstance(motion.get("motion_semantics"), dict)
            and motion["motion_semantics"].get("motion_mode")
            in {"self_propelled", "carried"}
        }
        if len(moving_subjects) >= 2:
            signals.add("multiple_moving_subjects")
    if isinstance(relationships, list):
        if any(
            isinstance(item, dict) and item.get("timeline_event_id") is not None
            for item in relationships
        ):
            signals.add("event_scoped_relationship")
        if any(
            isinstance(item, dict)
            and item.get("timeline_event_id") is None
            and item.get("temporal_mode") == "throughout"
            for item in relationships
        ):
            signals.add("clip_wide_relationship")
    if isinstance(timeline, dict) and timeline.get("relations"):
        signals.add("temporal_relations_present")
    if isinstance(uncertainties, list) and uncertainties:
        signals.add("uncertainties_present")
    return signals
