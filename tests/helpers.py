from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def valid_model_output() -> dict[str, Any]:
    path = ROOT / "prompts/semantic_parser/format_example.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["summary"] = "测试摘要"
    return value
