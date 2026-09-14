from __future__ import annotations

import re
from typing import Any


_CAMERA_MOVEMENT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pan", ("pan", "摇摄", "摇镜", "原地旋转", "固定机位旋转")),
    ("orbit", ("orbit", "环绕", "绕拍")),
    ("push_in", ("push_in", "pushin", "dolly_in", "推近", "推进")),
    ("pull_out", ("pull_out", "pullout", "dolly_out", "拉远", "后拉", "拉开")),
    ("follow", ("follow", "跟随", "跟拍")),
    ("lateral", ("lateral", "truck", "横移", "侧移")),
    ("static", ("static", "fixed", "静止", "固定")),
)

_CAMERA_VIEW_ANGLE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("top_down", ("垂直俯", "正俯", "top_down", "overhead")),
    ("high_angle", ("俯拍", "高角度", "high_angle")),
    ("low_angle", ("仰拍", "低角度", "low_angle")),
    ("eye_level", ("平视", "eye_level")),
)


def classify_camera_movement(value: Any) -> str | None:
    """Return one canonical camera movement without module-specific drift.

    Specific geometric motion wins over generic words such as ``fixed``.  Latin
    aliases are matched as complete normalized tokens so unrelated words do not
    accidentally select a camera program.  A fixed-position rotation phrase is
    pan even when extra timing words separate 固定机位 from 旋转.
    """

    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if "旋转" in normalized and any(
        marker in normalized for marker in ("固定机位", "不平移", "原地")
    ):
        return "pan"
    for kind, aliases in _CAMERA_MOVEMENT_ALIASES:
        for alias in aliases:
            if re.search(r"[\u3400-\u9fff]", alias):
                if alias in normalized:
                    return kind
                continue
            if re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                normalized,
            ):
                return kind
    return None


def classify_camera_view_angle(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    for kind, aliases in _CAMERA_VIEW_ANGLE_ALIASES:
        if _contains_alias(normalized, aliases):
            return kind
    return None


def camera_lens_focal_length(value: Any) -> float | None:
    """Map an explicit numeric or categorical lens intent to millimeters."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    numeric = re.search(r"(\d+(?:\.\d+)?)_?mm(?:\b|_)", normalized)
    if numeric:
        return float(numeric.group(1))
    for focal_length, aliases in (
        (35.0, ("广角", "wide")),
        (85.0, ("长焦", "telephoto")),
        (50.0, ("标准", "normal")),
    ):
        if _contains_alias(normalized, aliases):
            return focal_length
    return None


def _contains_alias(value: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if re.search(r"[\u3400-\u9fff]", alias):
            if alias in value:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", value):
            return True
    return False
