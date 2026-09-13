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

    def test_semantic_rules_freeze_priority_axis_and_lighting_scope(self) -> None:
        rules = (ROOT / "src/cinescaffold/resources/prompts/semantic_parser/rules.md").read_text(encoding="utf-8")

        self.assertIn("明确摄影机、构图或光源要求优先于情绪映射", rules)
        self.assertIn("缺省初始朝向是世界前方 `(0,-1,0)`", rules)
        self.assertIn("普通无目标位移保持 `direction_mode=none`", rules)
        self.assertIn("不授权改变 Blender 白模的中性技术照明", rules)
        self.assertIn("`scene_dynamics` 只描述主体，不描述摄影机", rules)
        self.assertIn("不是开放环境在渲染中的可见硬边界", rules)
        self.assertIn("两者都不得被重新解释为“主要物体最多占画幅多少”", rules)
        self.assertIn("这不等于用户明确指定了该画幅比例", rules)


if __name__ == "__main__":
    unittest.main()
