from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable


SECTION_LABELS = (
    "主体",
    "主体运动逻辑",
    "场景设计",
    "影调氛围",
    "构图模式",
    "摄影机视角运动逻辑",
)

_SECTION_PATTERN = re.compile(
    rf"^\s*({'|'.join(re.escape(label) for label in SECTION_LABELS)})\s*[：:]\s*(.*)$"
)


@dataclass(frozen=True)
class TextualSixDimensions:
    """面向人类的六部分电影语义文本。"""

    subject: str
    subject_motion: str
    scene_design: str
    mood: str
    composition: str
    camera: str

    def render(self) -> str:
        values = (
            self.subject,
            self.subject_motion,
            self.scene_design,
            self.mood,
            self.composition,
            self.camera,
        )
        return "\n\n".join(
            f"{label}：\n{value.strip()}" for label, value in zip(SECTION_LABELS, values, strict=True)
        ) + "\n"

    def sha256(self) -> str:
        return f"sha256:{hashlib.sha256(self.render().encode('utf-8')).hexdigest()}"


def parse_textual_six(text: str) -> TextualSixDimensions:
    """解析六个固定标题，正文允许跨越多行。"""

    if not text.strip():
        raise ValueError("文本六维不能为空")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _SECTION_PATTERN.match(line)
        if match:
            current = match.group(1)
            if current in sections:
                raise ValueError(f"文本六维第 {line_number} 行重复定义：{current}")
            sections[current] = [match.group(2)] if match.group(2).strip() else []
            continue
        if current is None:
            if line.strip():
                raise ValueError(f"文本六维第 {line_number} 行不属于任何维度")
            continue
        sections[current].append(line)

    missing = [label for label in SECTION_LABELS if label not in sections]
    if missing:
        raise ValueError("文本六维缺少部分：" + "、".join(missing))
    normalized = {label: "\n".join(sections[label]).strip() for label in SECTION_LABELS}
    empty = [label for label, value in normalized.items() if not value]
    if empty:
        raise ValueError("文本六维存在空白部分：" + "、".join(empty))
    return TextualSixDimensions(*[normalized[label] for label in SECTION_LABELS])


def render_textual_six(content: dict[str, Any]) -> TextualSixDimensions:
    """把权威 Brief 内容投影为简洁、稳定的人类可读视图。"""

    subjects = content.get("subjects", [])
    subject_names = {
        item.get("id"): _first_text(item.get("description"), item.get("category"), fallback="未命名主体")
        for item in subjects
        if isinstance(item, dict)
    }
    subject_lines = [
        _join_unique(
            [
                _first_text(item.get("description"), item.get("category"), fallback="未命名主体"),
                *_attribute_values(item.get("attributes")),
            ]
        )
        for item in subjects
        if isinstance(item, dict)
    ]

    motion_lines: list[str] = []
    for item in content.get("subject_motion", []):
        if not isinstance(item, dict):
            continue
        subject = subject_names.get(item.get("subject_id"), str(item.get("subject_id") or "主体"))
        details = _semantic_values(
            item,
            ignored_keys={"subject_id", "source_status", "source_text", "timeline_event_id"},
        )
        motion_lines.append(f"{subject}：{_join_unique(details) or '未指定'}")

    scene = content.get("scene_design", {})
    scene_lines = _semantic_values(scene) if isinstance(scene, dict) else []
    mood = content.get("mood", {})
    mood_lines = _semantic_values(mood) if isinstance(mood, dict) else []
    composition = content.get("composition", {})
    composition_lines = _semantic_values(composition) if isinstance(composition, dict) else []
    camera = content.get("camera", {})
    camera_lines = _semantic_values(
        camera,
        ignored_keys={"start_time_seconds", "end_time_seconds", "source_status", "source_text"},
    ) if isinstance(camera, dict) else []

    return TextualSixDimensions(
        subject=_lines_or_unspecified(subject_lines),
        subject_motion=_lines_or_unspecified(motion_lines),
        scene_design=_lines_or_unspecified(scene_lines),
        mood=_lines_or_unspecified(mood_lines),
        composition=_lines_or_unspecified(composition_lines),
        camera=_lines_or_unspecified(camera_lines),
    )


def _attribute_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        attribute = item.get("value")
        if isinstance(name, str) and isinstance(attribute, str):
            output.append(f"{name}={attribute}")
    return output


def _semantic_values(value: Any, *, ignored_keys: set[str] | None = None) -> list[str]:
    ignored = ignored_keys or {"source_status", "source_text"}
    output: list[str] = []
    if isinstance(value, list):
        for item in value:
            output.extend(_semantic_values(item, ignored_keys=ignored))
        return _unique(output)
    if not isinstance(value, dict):
        if isinstance(value, str) and value.strip() and value not in {"unknown", "unspecified"}:
            output.append(value.strip())
        return output

    annotated = value.get("value")
    if isinstance(annotated, str) and annotated.strip() and annotated not in {"unknown", "unspecified"}:
        output.append(annotated.strip())
    for key, item in value.items():
        if key in ignored or key == "value" or key.endswith("_id"):
            continue
        output.extend(_semantic_values(item, ignored_keys=ignored))
    return _unique(output)


def _first_text(*values: Any, fallback: str) -> str:
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("value"), str) and value["value"].strip():
            return value["value"].strip()
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _join_unique(values: Iterable[str]) -> str:
    return "、".join(_unique(value for value in values if value))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _lines_or_unspecified(lines: Iterable[str]) -> str:
    values = [line for line in lines if line]
    return "\n".join(f"· {line}" for line in values) if values else "· 未指定"
