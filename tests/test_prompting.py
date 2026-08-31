from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cinescaffold.prompting import build_prompt
from tests.helpers import ROOT


class PromptingTest(unittest.TestCase):
    def test_rules_are_injected_without_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rules_path = Path(directory) / "rules.md"
            rules_path.write_text("这是外部规则。", encoding="utf-8")
            bundle = build_prompt(
                description="测试描述",
                system_template_path=ROOT / "prompts/semantic_parser/system.md",
                rules_path=rules_path,
                format_example_path=ROOT / "prompts/semantic_parser/format_example.json",
            )
        self.assertIn("这是外部规则。", bundle.system_prompt)
        self.assertIn("测试描述", bundle.user_prompt)
        self.assertIn("仅表示字段形状", bundle.user_prompt)

    def test_scene_planner_prompt_freezes_coordinate_and_visibility_semantics(self) -> None:
        prompt = (ROOT / "prompts/scene_planner/system.md").read_text(encoding="utf-8")

        self.assertIn("右手 `+Z-up`", prompt)
        self.assertIn("`+plane_normal`", prompt)
        self.assertIn("`keep_in_frame` 只表示投影包围盒入框", prompt)

    def test_semantic_rules_freeze_priority_axis_and_lighting_scope(self) -> None:
        rules = (ROOT / "prompts/semantic_parser/rules.md").read_text(encoding="utf-8")

        self.assertIn("明确摄影机、构图或光源要求优先于情绪映射", rules)
        self.assertIn("世界前方为 `(0,-1,0)`", rules)
        self.assertIn("不授权改变 Blender 白模的中性技术照明", rules)


if __name__ == "__main__":
    unittest.main()
