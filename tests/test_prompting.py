from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cinescaffold.errors import PromptTemplateError
from cinescaffold.prompting import build_prompt, build_revision_prompt
from cinescaffold.schema import load_schema
from tests.helpers import ROOT


class PromptingTest(unittest.TestCase):
    def test_rules_are_injected_without_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rules_path = Path(directory) / "rules.md"
            rules_path.write_text("这是外部规则。", encoding="utf-8")
            bundle = build_prompt(
                description="测试描述",
                system_template_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
                rules_path=rules_path,
                format_example_path=ROOT / "src/cinescaffold/resources/prompts/semantic_parser/format_example.json",
            )
        self.assertIn("这是外部规则。", bundle.system_prompt)
        self.assertIn("测试描述", bundle.user_prompt)
        self.assertIn("仅表示字段形状", bundle.user_prompt)

    def test_scene_planner_prompt_freezes_coordinate_and_visibility_semantics(self) -> None:
        prompt = (ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md").read_text(encoding="utf-8")

        self.assertIn("右手 `+Z-up`", prompt)
        self.assertIn("`+plane_normal`", prompt)
        self.assertIn("`keep_in_frame` 只表示投影包围盒入框", prompt)
        self.assertIn("供下游视频生成模型参考的白模控制视频", prompt)
        self.assertIn("soft score 作为记录用的质量指标而非阻断条件", prompt)
        self.assertIn("摄影机自身运动不能作为选择 `maximize_motion_readability` 的理由", prompt)
        self.assertIn("三层互不替代的术语", prompt)
        self.assertIn("不承诺画面中的上下左右", prompt)
        self.assertIn("`inferred` 与系统 `default` 范围只参与评分和选项比较", prompt)
        self.assertIn("一旦应用结果返回 `commit_ready=true`", prompt)

    def test_semantic_rules_define_strict_generic_contract(self) -> None:
        rules = (ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md").read_text(encoding="utf-8")

        self.assertIn("不得用含义不明的“主体”", rules)
        self.assertIn("不得用含义不明的“镜头”", rules)
        self.assertIn("可位于视频片段内任意数值起止点", rules)
        self.assertIn("不得按动作数量机械等分时间", rules)
        self.assertIn("不得发明某个故事专用动作类型或关系类型", rules)
        self.assertIn("不得由情绪词自行推导摄影机位置", rules)
        self.assertIn("各解释共有的可验证事实", rules)
        self.assertIn("独立事件并配合 `at_midpoint`", rules)
        self.assertIn("关系描述的是需要验证的语义事实", rules)
        self.assertNotIn("接到人后", rules)
        self.assertNotIn("等待→上车", rules)

        planning_prompt = (
            ROOT / "src/cinescaffold/resources/prompts/scene_planner/system.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "不得引用 `action.value/source_text`、实体名称或其他自由文本来补造关系",
            planning_prompt,
        )

    def test_semantic_prompt_names_every_structured_output_field(self) -> None:
        prompt = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "src/cinescaffold/resources/prompts/semantic_parser/system.md",
                "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
                "src/cinescaffold/resources/prompts/semantic_parser/revision.md",
            )
        )
        schema = load_schema(
            ROOT
            / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json"
        )

        schema_nodes = [("root", schema), *schema.get("$defs", {}).items()]
        for definition, node in schema_nodes:
            for field_name in node.get("properties", {}):
                with self.subTest(definition=definition, field=field_name):
                    self.assertIn(field_name, prompt)

    def test_revision_prompt_contains_source_draft_and_diagnostics(self) -> None:
        bundle = build_revision_prompt(
            source_text="测试原文",
            draft={"summary": "初稿"},
            diagnostics=[{"code": "test", "message": "需要修正"}],
            review_packet={
                "selection_policy": "attention_only_no_semantic_inference",
                "findings": [{"rule_id": "SOURCE.COVERAGE"}],
            },
            system_template_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
            rules_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
            revision_template_path=ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/revision.md",
        )

        self.assertIn("测试原文", bundle.user_prompt)
        self.assertIn('"summary": "初稿"', bundle.user_prompt)
        self.assertIn('"code": "test"', bundle.user_prompt)
        self.assertIn('"rule_id": "SOURCE.COVERAGE"', bundle.user_prompt)
        self.assertIn("关键词、初稿文字和结构信号只能召回审查规则", bundle.user_prompt)

    def test_revision_prompt_rejects_incomplete_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            template_path = Path(directory) / "revision.md"
            template_path.write_text("{{SOURCE_TEXT}}", encoding="utf-8")

            with self.assertRaises(PromptTemplateError):
                build_revision_prompt(
                    source_text="测试原文",
                    draft={},
                    diagnostics=[],
                    system_template_path=ROOT
                    / "src/cinescaffold/resources/prompts/semantic_parser/system.md",
                    rules_path=ROOT
                    / "src/cinescaffold/resources/prompts/semantic_parser/rules.md",
                    revision_template_path=template_path,
                )


if __name__ == "__main__":
    unittest.main()
