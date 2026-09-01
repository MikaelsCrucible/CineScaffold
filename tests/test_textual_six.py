from __future__ import annotations

import unittest

from cinescaffold.textual_six import (
    SECTION_FIELDS,
    TextualSixDimensions,
    parse_textual_six,
    render_textual_six,
)
from tests.helpers import valid_model_output


TEXTUAL_SIX = """主体：
男人、巨大飞船

主体运动逻辑：
男人静止、沙尘缓慢运动

场景设计：
荒漠、飞船、前景/中景/后景关系

影调氛围：
冷峻、低饱和、孤独、压迫

构图模式：
人物小比例、巨物主义、负空间

摄影机视角运动逻辑：
远景、低机位、缓慢推近
"""


class TextualSixTest(unittest.TestCase):
    def test_parse_and_render_preserve_all_six_sections(self) -> None:
        parsed = parse_textual_six(TEXTUAL_SIX)

        self.assertEqual(parsed.subject, "男人、巨大飞船")
        self.assertIn("沙尘", parsed.subject_motion)
        self.assertEqual(parse_textual_six(parsed.render()), parsed)
        self.assertRegex(parsed.sha256(), r"^sha256:[0-9a-f]{64}$")

    def test_missing_or_duplicate_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少部分"):
            parse_textual_six("主体：男人")
        with self.assertRaisesRegex(ValueError, "重复定义"):
            parse_textual_six(TEXTUAL_SIX + "\n主体：另一个人\n")

    def test_fixed_ui_fields_render_every_label(self) -> None:
        field_values = {field: f"{label}内容" for field, label in SECTION_FIELDS}

        textual_six = TextualSixDimensions.from_field_values(field_values)

        self.assertEqual(textual_six.field_values(), field_values)
        self.assertEqual(
            [line[:-1] for line in textual_six.render().splitlines() if line.endswith("：")],
            [label for _, label in SECTION_FIELDS],
        )

    def test_fixed_ui_fields_reject_empty_body(self) -> None:
        field_values = {field: label for field, label in SECTION_FIELDS}
        field_values["composition"] = ""

        with self.assertRaisesRegex(ValueError, "构图模式"):
            TextualSixDimensions.from_field_values(field_values)

    def test_brief_projection_uses_human_values_without_internal_ids(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man_01",
                "category": {"value": "男人", "source_status": "explicit", "source_text": "男人"},
                "description": {"value": "孤独的男人", "source_status": "inferred", "source_text": None},
                "narrative_role": {"value": None, "source_status": "unknown", "source_text": None},
                "attributes": [],
            }
        ]
        content["subject_motion"] = []

        rendered = render_textual_six(content).render()

        self.assertIn("孤独的男人", rendered)
        self.assertNotIn("man_01", rendered)
        self.assertEqual(len([label for label in rendered.splitlines() if label.endswith("：")]), 6)


if __name__ == "__main__":
    unittest.main()
