from __future__ import annotations

import unittest

from cinescaffold.semantic_review import (
    REVIEW_SELECTION_POLICY,
    build_semantic_review_packet,
    load_semantic_review_catalog,
)
from tests.helpers import ROOT, valid_model_output


class SemanticReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_semantic_review_catalog(
            ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/review_rules.json"
        )

    def test_local_multi_entity_cue_retrieves_audit_rules_without_mapping(self) -> None:
        packet = build_semantic_review_packet(
            "男生和女生擦肩而过。",
            valid_model_output(),
            [
                {
                    "code": "independent_semantic_review_required",
                    "message": "请逐句复核",
                }
            ],
            self.catalog,
        )

        selected = {item["rule_id"] for item in packet["findings"]}
        self.assertIn("MULTI_ENTITY.SHARED_FACTS", selected)
        self.assertIn("TIMELINE.LOCAL_EVENT_SCOPE", selected)
        self.assertIn("SOURCE.COVERAGE", selected)
        self.assertEqual(packet["selection_policy"], REVIEW_SELECTION_POLICY)
        self.assertLessEqual(len(packet["findings"]), 5)
        self.assertNotIn("proximity", str(packet))
        self.assertTrue(
            all(item["kind"] == "review_risk" for item in packet["findings"])
        )

    def test_cues_from_draft_text_can_retrieve_review_without_source_match(self) -> None:
        draft = valid_model_output()
        draft["summary"] = "两个对象在中段交错"

        packet = build_semantic_review_packet(
            "两个对象各自运动。",
            draft,
            [
                {
                    "code": "independent_semantic_review_required",
                    "message": "请逐句复核",
                }
            ],
            self.catalog,
        )

        finding = next(
            item
            for item in packet["findings"]
            if item["rule_id"] == "MULTI_ENTITY.SHARED_FACTS"
        )
        self.assertEqual(finding["trigger_evidence"]["source_cues"], [])
        self.assertEqual(finding["trigger_evidence"]["draft_cues"], ["交错"])

    def test_deterministic_path_failure_is_marked_confirmed(self) -> None:
        packet = build_semantic_review_packet(
            "一个人移动。",
            valid_model_output(),
            [
                {
                    "code": "semantic_direction_contract_failed",
                    "parent_code": "semantic_contract_failed",
                    "message": "subject_motion[0] 的相对方向缺少有效 target_id",
                }
            ],
            self.catalog,
        )

        finding = next(
            item
            for item in packet["findings"]
            if item["rule_id"] == "MOTION.DIRECTION_EVIDENCE"
        )
        self.assertEqual(finding["kind"], "confirmed_error")
        self.assertIn(
            "subject_motion[0]",
            finding["trigger_evidence"]["diagnostic_paths"],
        )
        self.assertEqual(packet["unmatched_diagnostics"], [])

    def test_selection_is_deterministic(self) -> None:
        arguments = (
            "首先一个人走向汽车，然后上车，同时镜头推近。",
            valid_model_output(),
            [
                {
                    "code": "independent_semantic_review_required",
                    "message": "请逐句复核",
                }
            ],
            self.catalog,
        )

        self.assertEqual(
            build_semantic_review_packet(*arguments),
            build_semantic_review_packet(*arguments),
        )


if __name__ == "__main__":
    unittest.main()
