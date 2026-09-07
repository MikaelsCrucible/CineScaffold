from __future__ import annotations

import json
from pathlib import Path

from cinescaffold.planning.domain import CandidateState
from cinescaffold.planning.ir import SceneIR


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    schemas = {
        "constraint_plan.schema.json": CandidateState.model_json_schema(
            ref_template="#/$defs/{model}"
        ),
        "scene_ir.schema.json": SceneIR.model_json_schema(ref_template="#/$defs/{model}"),
    }
    for name, schema in schemas.items():
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://cinescaffold.local/schemas/{name}"
        path = ROOT / "src/cinescaffold/resources/schemas" / name
        path.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
