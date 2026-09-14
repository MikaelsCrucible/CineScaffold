from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

CanonicalRelationshipKind = Literal[
    "far",
    "proximity",
    "orbit",
    "relative_position",
    "scale_dominance",
    "ground_support",
    "carried_by",
]

CANONICAL_RELATIONSHIP_TYPES = {
    "far_from",
    "proximity",
    "orbit_around",
    "scale_dominance",
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "ground_support",
    "carried_by",
}


@dataclass(frozen=True)
class RelationshipMeaning:
    kind: CanonicalRelationshipKind
    direction: str | None = None
    reverse_subject_reference: bool = False


_STATUS_PRIORITY = {
    "unknown": 0,
    "default": 1,
    "inferred": 2,
    "explicit": 3,
}


def relationship_text(relationship: dict[str, Any]) -> str:
    """Combine every typed relationship field; no field may shadow another."""

    return " ".join(
        str(relationship.get(name) or "").strip().lower()
        for name in ("type", "strength")
    ).strip()


def classify_relationship(relationship: dict[str, Any]) -> RelationshipMeaning | None:
    """Map open provider vocabulary to the closed Planning relationship set."""

    relation_type = str(relationship.get("type") or "").strip().lower()
    strength = str(relationship.get("strength") or "").strip().lower()
    generic_distance_types = {
        "distance",
        "distance_relation",
        "距离",
        "远近",
        "远近关系",
        "纵深关系",
    }
    far_strength = _contains_any(
        strength, ("far", "distant", "background", "远", "后景")
    )
    near_strength = _contains_any(
        strength, ("near", "close", "foreground", "近", "前景")
    )
    if relation_type in generic_distance_types and far_strength:
        return RelationshipMeaning("far")
    if relation_type in generic_distance_types and near_strength:
        return RelationshipMeaning("proximity")
    if relation_type in {
        "far_from",
        "distant_from",
        "distant",
        "background",
        "far",
        "远处",
        "后景",
        "远",
    }:
        return RelationshipMeaning("far")
    if relation_type in {
        "orbit_around",
        "orbits",
        "orbit",
        "公转",
        "环绕",
        "绕行",
    }:
        return RelationshipMeaning("orbit")
    if relation_type in {
        "scale_dominance",
        "larger_than",
        "smaller_than",
        "尺度对比",
        "尺寸对比",
        "体型对比",
    }:
        return RelationshipMeaning(
            "scale_dominance",
            reverse_subject_reference=_contains_any(
                relation_type,
                ("smaller_than", "smaller", "小于", "较小"),
            ),
        )
    if relation_type in {
        "proximity",
        "near",
        "close_to",
        "adjacent",
        "beside",
        "stop_beside",
        "接近",
        "靠近",
        "旁边",
        "身边",
        "接应",
    }:
        return RelationshipMeaning("proximity")
    for direction, markers in (
        ("left", ("left_of", "左侧", "左边")),
        ("right", ("right_of", "右侧", "右边")),
        ("front", ("in_front_of", "front_of", "前方")),
        ("behind", ("behind", "后方")),
        ("above", ("above", "上方")),
        ("below", ("below", "下方")),
    ):
        if relation_type in markers:
            return RelationshipMeaning("relative_position", direction)
    if relation_type in {"ground_support", "on_ground", "supported_by"}:
        return RelationshipMeaning("ground_support")
    if relation_type in {
        "carried_by",
        "contained_by",
        "inside",
        "board_into",
        "enter_into",
    }:
        return RelationshipMeaning("carried_by")
    return None


def normalize_scene_relationships(content: dict[str, Any]) -> None:
    """Canonicalize, orient, and deduplicate relationships before Planning.

    Provider vocabulary remains open at the JSON boundary, but every relation
    recognized by Planning is rewritten to one stable spelling.  Ambiguous
    values stay untouched instead of being guessed as proximity.
    """

    scene = content.get("scene_design")
    if not isinstance(scene, dict):
        return
    relationships = scene.setdefault("relationships", [])
    if not isinstance(relationships, list):
        return

    far_ids, near_ids = _layer_depth_ids(scene.get("spatial_layers"))
    normalized: list[Any] = []
    keys: dict[tuple[Any, ...], int] = {}
    for raw in relationships:
        if not isinstance(raw, dict):
            # Normalization must not erase malformed provider output. Keeping
            # it lets the strict JSON Schema report the real boundary error.
            normalized.append(raw)
            continue
        item = deepcopy(raw)
        item.setdefault("timeline_event_id", None)
        item.setdefault("temporal_mode", "throughout")
        meaning = classify_relationship(item)
        if meaning is not None:
            _apply_canonical_meaning(item, meaning, far_ids, near_ids)
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


def _apply_canonical_meaning(
    relationship: dict[str, Any],
    meaning: RelationshipMeaning,
    far_ids: set[str],
    near_ids: set[str],
) -> None:
    subject_id = relationship.get("subject_id")
    reference_id = relationship.get("reference_id")
    if meaning.kind == "far":
        if isinstance(subject_id, str) and isinstance(reference_id, str):
            if (
                reference_id in far_ids
                and subject_id not in far_ids
                or subject_id in near_ids
                and reference_id not in near_ids
            ):
                relationship["subject_id"], relationship["reference_id"] = (
                    reference_id,
                    subject_id,
                )
        relationship["type"] = "far_from"
        relationship["strength"] = "scene_relative"
    elif meaning.kind == "proximity":
        relationship["type"] = "proximity"
    elif meaning.kind == "orbit":
        relationship["type"] = "orbit_around"
    elif meaning.kind == "scale_dominance":
        if meaning.reverse_subject_reference:
            relationship["subject_id"], relationship["reference_id"] = (
                relationship.get("reference_id"),
                relationship.get("subject_id"),
            )
        relationship["type"] = "scale_dominance"
    elif meaning.kind == "relative_position":
        relationship["type"] = {
            "left": "left_of",
            "right": "right_of",
            "front": "front_of",
            "behind": "behind",
            "above": "above",
            "below": "below",
        }[meaning.direction]
    elif meaning.kind == "ground_support":
        relationship["type"] = "ground_support"
    elif meaning.kind == "carried_by":
        relationship["type"] = "carried_by"


def _layer_depth_ids(layers: Any) -> tuple[set[str], set[str]]:
    far_ids: set[str] = set()
    near_ids: set[str] = set()
    if not isinstance(layers, list):
        return far_ids, near_ids
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        text = str(layer.get("layer") or "").strip().lower()
        ids = {item for item in layer.get("content_ids", []) if isinstance(item, str)}
        if _contains_any(text, ("远", "后景", "far", "background", "distant")):
            far_ids.update(ids)
        if _contains_any(text, ("近", "前景", "near", "foreground")):
            near_ids.update(ids)
    return far_ids, near_ids


def _relationship_identity(relationship: dict[str, Any]) -> tuple[Any, ...]:
    meaning = classify_relationship(relationship)
    subject_id = relationship.get("subject_id")
    reference_id = relationship.get("reference_id")
    kind = meaning.kind if meaning is not None else relationship_text(relationship)
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


def _contains_any(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)
