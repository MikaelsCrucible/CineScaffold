from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def valid_model_output() -> dict[str, Any]:
    path = (
        ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["summary"] = "测试摘要"
    return value


def valid_planning_brief() -> dict[str, Any]:
    from cinescaffold.semantic_rules import apply_translation_rules, load_translation_rules

    content = valid_model_output()
    content["summary"] = "男人孤独地面对远处飞船"
    content["timeline"].update(
        {"duration_seconds": 6.0, "duration_source_status": "inferred"}
    )
    content["subjects"] = [
        {
            "id": "man_01",
            "category": _annotated("男人", "一个男人"),
            "description": _unknown(),
            "narrative_role": _unknown(),
            "attributes": [],
        },
        {
            "id": "ship_01",
            "category": _annotated("飞船", "巨大的飞船"),
            "description": _unknown(),
            "narrative_role": _unknown(),
            "attributes": [],
        },
    ]
    content["subject_motion"] = []
    content["scene_design"]["relationships"] = [
        {
            "type": "远处",
            "subject_id": "ship_01",
            "reference_id": "man_01",
            "strength": "明显",
            "timeline_event_id": None,
            "temporal_mode": "throughout",
            "source_status": "explicit",
            "source_text": "远处有飞船",
        }
    ]
    content["mood"]["emotional_tones"] = [
        {
            "value": "孤独",
            "source_status": "explicit",
            "source_text": "感觉很孤独",
        }
    ]
    content["camera"]["movement"]["type"] = _annotated("缓慢推近", "镜头慢慢推近")
    rules_path = (
        ROOT
        / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
    )
    normalized, parameters = apply_translation_rules(
        content,
        load_translation_rules(rules_path),
    )
    return {
        "schema_version": "0.7",
        "content": normalized,
        "translation_parameters": parameters,
        "provenance": {
            "source_prompt": "一个男人站在荒漠里，远处有飞船，感觉孤独，镜头慢慢推近。",
            "provider": "mock",
            "model": "mock-cinematic-brief-v0.1",
            "parser_prompt_version": "semantic-parser-v0.1",
            "rules_sha256": "0" * 64,
            "translation_rules_sha256": "1" * 64,
            "response_id": "mock-response-001",
        },
    }


def _annotated(value: str, source_text: str) -> dict[str, Any]:
    return {"value": value, "source_status": "explicit", "source_text": source_text}


def _unknown() -> dict[str, Any]:
    return {"value": None, "source_status": "unknown", "source_text": None}
