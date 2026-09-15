from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

CanonicalRelationshipKind = Literal[
    "far",
    "proximity",
    "relative_position",
    "scale_dominance",
]

CANONICAL_RELATIONSHIP_TYPES = {
    "far_from",
    "proximity",
    "scale_dominance",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
}


@dataclass(frozen=True)
class RelationshipMeaning:
    kind: CanonicalRelationshipKind
    direction: str | None = None


_STATUS_PRIORITY = {
    "unknown": 0,
    "default": 1,
    "inferred": 2,
    "explicit": 3,
}


def classify_relationship(relationship: dict[str, Any]) -> RelationshipMeaning | None:
    """Read only the canonical relationship vocabulary of the current Brief."""

    relation_type = relationship.get("type")
    if relation_type == "far_from":
        return RelationshipMeaning("far")
    if relation_type == "proximity":
        return RelationshipMeaning("proximity")
    if relation_type == "scale_dominance":
        return RelationshipMeaning("scale_dominance")
    direction = {
        "left_of": "left",
        "right_of": "right",
        "front_of": "front",
        "behind": "behind",
        "above": "above",
        "below": "below",
    }.get(relation_type)
    if direction is not None:
        return RelationshipMeaning("relative_position", direction)
    return None


def normalize_scene_relationships(content: dict[str, Any]) -> None:
    """Deduplicate current-contract relationships without reparsing them."""

    scene = content.get("scene_design")
    if not isinstance(scene, dict):
        return
    relationships = scene.setdefault("relationships", [])
    if not isinstance(relationships, list):
        return

    normalized: list[Any] = []
    keys: dict[tuple[Any, ...], int] = {}
    for raw in relationships:
        if not isinstance(raw, dict):
            normalized.append(raw)
            continue
        item = deepcopy(raw)
        key = _relationship_identity(item)
        previous_index = keys.get(key)
        if previous_index is None:
            keys[key] = len(normalized)
            normalized.append(item)
            continue
        normalized[previous_index] = _merge_duplicate_relationship(
            normalized[previous_index],
            item,
        )
    _reject_direct_distance_conflicts(normalized)
    _reject_reversed_far_conflicts(normalized)
    scene["relationships"] = normalized


def _relationship_identity(relationship: dict[str, Any]) -> tuple[Any, ...]:
    meaning = classify_relationship(relationship)
    subject_id = relationship.get("subject_id")
    reference_id = relationship.get("reference_id")
    kind = meaning.kind if meaning is not None else relationship.get("type")
    direction = meaning.direction if meaning is not None else None
    if kind == "proximity":
        pair = tuple(sorted((str(subject_id), str(reference_id))))
    else:
        pair = (subject_id, reference_id)
    return (
        kind,
        direction,
        pair,
        relationship.get("timeline_event_id"),
        relationship.get("temporal_mode", "throughout"),
    )


def _merge_duplicate_relationship(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    first_priority = _STATUS_PRIORITY.get(str(first.get("source_status")), 0)
    second_priority = _STATUS_PRIORITY.get(str(second.get("source_status")), 0)
    use_second = second_priority > first_priority
    preferred = deepcopy(second if use_second else first)
    alternate = first if use_second else second
    if not preferred.get("source_text") and alternate.get("source_text"):
        preferred["source_text"] = alternate["source_text"]
    return preferred


def _reject_direct_distance_conflicts(relationships: list[Any]) -> None:
    meanings: dict[tuple[Any, ...], set[str]] = {}
    for item in relationships:
        if not isinstance(item, dict):
            continue
        meaning = classify_relationship(item)
        if meaning is None or meaning.kind not in {"far", "proximity"}:
            continue
        subject_id = item.get("subject_id")
        reference_id = item.get("reference_id")
        if not isinstance(subject_id, str) or not isinstance(reference_id, str):
            continue
        key = (
            tuple(sorted((subject_id, reference_id))),
            item.get("timeline_event_id"),
            item.get("temporal_mode", "throughout"),
        )
        meanings.setdefault(key, set()).add(meaning.kind)
    if any(value == {"far", "proximity"} for value in meanings.values()):
        raise ValueError("同一实体对在同一时间范围内同时被标记为远离和靠近")


def _reject_reversed_far_conflicts(relationships: list[Any]) -> None:
    orientations: dict[tuple[Any, ...], set[tuple[str, str]]] = {}
    for item in relationships:
        if not isinstance(item, dict):
            continue
        meaning = classify_relationship(item)
        if meaning is None or meaning.kind != "far":
            continue
        subject_id = item.get("subject_id")
        reference_id = item.get("reference_id")
        if not isinstance(subject_id, str) or not isinstance(reference_id, str):
            continue
        key = (
            tuple(sorted((subject_id, reference_id))),
            item.get("timeline_event_id"),
            item.get("temporal_mode", "throughout"),
        )
        orientations.setdefault(key, set()).add((subject_id, reference_id))
    if any(len(value) > 1 for value in orientations.values()):
        raise ValueError("同一实体对在同一时间范围内给出了相反的远景方向")
