from __future__ import annotations

import unittest

from cinescaffold.schema import load_schema, validate_model_output
from cinescaffold.semantic_rules import apply_translation_rules, load_translation_rules
from tests.helpers import ROOT, valid_model_output


class SemanticRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_translation_rules(
            ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
        )
        self.schema = load_schema(
            ROOT / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json"
        )
        self.model_schema = load_schema(
            ROOT / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json"
        )

    def test_loneliness_maps_to_fixed_profile_without_preview_lighting(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man_01",
                "category": self._annotated("男人", "一个男人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
        ]
        content["subject_motion"] = [
            self._motion(
                "man_01",
                "站立",
                action_kind="hold",
                motion_type="static",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
            )
        ]
        content["scene_design"]["environment"] = self._annotated("荒漠", "荒漠里")
        content["mood"]["emotional_tones"] = [
            self._statement("孤独", "感觉很孤独")
        ]
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )

        normalized, parameters = apply_translation_rules(content, self.rules)

        validate_model_output(parameters, self.schema)
        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(normalized["camera"]["camera_height"]["value"], "1.2 m")
        self.assertEqual(normalized["camera"]["camera_height"]["source_status"], "inferred")
        self.assertIn(
            "仅供最终视频生成",
            normalized["mood"]["lighting_intent"][-1]["value"],
        )
        self.assertEqual(parameters["emotion_class"]["class_id"], "E1")
        self.assertEqual(parameters["camera"]["height_m"], 1.2)
        self.assertEqual(parameters["camera"]["speed_mps"], 1.0)
        self.assertEqual(normalized["camera"]["movement"]["speed"]["value"], "1 m/s")
        self.assertEqual(parameters["composition"]["subject_frame_ratio"], [0.01, 0.05])
        self.assertEqual(parameters["scene"]["asset_key"], "desert")
        self.assertEqual(parameters["subjects"][0]["reference_height_m"], 1.75)
        self.assertEqual(parameters["subjects"][0]["facing_direction_world"], [0.0, -1.0, 0.0])
        self.assertEqual(parameters["lighting"]["application_scope"], "final_video_generation_only")
        self.assertFalse(parameters["lighting"]["applied_to_blender_preview"])

    def test_radial_camera_speed_uses_actual_duration(self) -> None:
        cases = (
            ("孤独", 15.0, 10.0 / 15.0),
            ("压迫", 10.0, 1.2),
            ("开阔", 10.0, 1.5),
        )
        for feeling, duration, expected_speed in cases:
            with self.subTest(feeling=feeling, duration=duration):
                content = valid_model_output()
                content["mood"]["emotional_tones"] = [
                    self._statement(feeling, f"感觉{feeling}")
                ]
                content["timeline"].update(
                    {
                        "duration_seconds": duration,
                        "duration_source_status": "explicit",
                    }
                )

                _, parameters = apply_translation_rules(content, self.rules)

                self.assertAlmostEqual(
                    parameters["camera"]["speed_mps"],
                    expected_speed,
                )

    def test_invalid_radial_camera_rule_cannot_write_fixed_speed(self) -> None:
        rules = load_translation_rules(
            ROOT / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
        )
        rules["emotion_classes"]["E1"]["camera"]["speed_mps"] = 0.67

        with self.assertRaisesRegex(ValueError, "不得同时写死速度"):
            apply_translation_rules(valid_model_output(), rules)

    def test_missing_slots_use_declared_defaults(self) -> None:
        content = valid_model_output()

        normalized, parameters = apply_translation_rules(content, self.rules)

        self.assertEqual(normalized["subjects"][0]["category"]["value"], "人")
        self.assertEqual(normalized["scene_design"]["environment"]["value"], "空白空间")
        self.assertEqual(normalized["timeline"]["duration_seconds"], 15.0)
        self.assertEqual(parameters["emotion_class"]["class_id"], "E6")
        self.assertEqual(parameters["motions"][0]["motion_type"], "static")
        self.assertEqual(parameters["motions"][0]["speed_range_mps"], [0.0, 0.0])
        default_fields = {item["field"] for item in normalized["uncertainties"]}
        self.assertIn("subjects", default_fields)
        self.assertIn("timeline.duration_seconds", default_fields)

    def test_explicit_camera_is_recorded_as_override(self) -> None:
        content = valid_model_output()
        content["camera"]["view_angle"] = self._annotated("俯拍", "使用俯拍")
        content["mood"]["emotional_tones"] = [
            self._statement("孤独", "感觉孤独")
        ]

        _, parameters = apply_translation_rules(content, self.rules)

        self.assertIn("camera.view_angle", parameters["explicit_override_paths"])
        normalized, _ = apply_translation_rules(content, self.rules)
        self.assertEqual(normalized["camera"]["view_angle"]["value"], "俯拍")

    def test_duration_upper_bound_resolves_to_its_maximum(self) -> None:
        content = valid_model_output()
        content["timeline"]["duration_range_seconds"] = {
            "minimum_seconds": 0.0,
            "maximum_seconds": 10.0,
        }
        content["timeline"]["duration_source_status"] = "explicit"

        normalized, _ = apply_translation_rules(content, self.rules)

        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(normalized["timeline"]["duration_source_status"], "inferred")
        uncertainty = next(
            item
            for item in normalized["uncertainties"]
            if item["field"] == "timeline.duration_seconds"
        )
        self.assertEqual(uncertainty["resolution"], "use_inference")
        self.assertEqual(uncertainty["selected_value"], "10.0")

    def test_target_motion_uses_relative_direction(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man",
                "category": self._annotated("男人", "一个男人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "ship",
                "category": self._annotated("飞船", "一艘飞船"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [
            self._motion(
                "man",
                "走向飞船",
                action_kind="approach",
                motion_type="walking",
                motion_mode="self_propelled",
                direction_mode="toward_target",
                target_id="ship",
                path_type="linear",
            )
        ]
        content["subject_motion"][0]["direction"] = self._annotated(
            "朝向飞船",
            "走向飞船",
        )

        _, parameters = apply_translation_rules(content, self.rules)

        motion = parameters["motions"][0]
        self.assertEqual(motion["motion_type"], "walking")
        self.assertEqual(motion["target_id"], "ship")
        self.assertEqual(motion["direction_mode"], "toward_target")
        self.assertIsNone(motion["direction_vector_world"])
        self.assertEqual(parameters["subjects"][1]["minimum_footprint_m"], [10.0, 10.0])
        self.assertEqual(parameters["subjects"][1]["default_scene_depth_ratio"], 0.5)

    def test_inferred_narrative_verbs_do_not_create_motion_targets(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "passenger",
                "category": self._annotated("乘客", "一名乘客"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "carrier",
                "category": self._annotated("载具", "一辆载具"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [
            self._motion(
                "carrier",
                "接载后离开",
                action_kind="depart",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="away_from_target",
                target_id="passenger",
                path_type="linear",
            )
        ]

        normalized, parameters = apply_translation_rules(content, self.rules)

        semantics = normalized["subject_motion"][0]["motion_semantics"]
        self.assertEqual(semantics["action_kind"], "locomotion")
        self.assertEqual(semantics["direction_mode"], "none")
        self.assertIsNone(semantics["target_id"])
        self.assertEqual(parameters["motions"][0]["action_kind"], "locomotion")
        self.assertEqual(parameters["motions"][0]["direction_mode"], "none")

    def test_dynamic_entity_timelines_preserve_independent_ranges(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "person",
                "category": self._annotated("人", "一个人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "car",
                "category": self._annotated("车辆", "一辆车"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [
            self._motion(
                "person",
                "等待",
                action_kind="hold",
                motion_type="static",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
                timeline_event_id="waiting",
            ),
            self._motion(
                "car",
                "驶来并接走",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="toward_target",
                target_id="person",
                path_type="linear",
                timeline_event_id="pickup",
            ),
        ]
        content["subject_motion"][0].update(
            start_time_seconds=0.0,
            end_time_seconds=7.0,
        )
        content["subject_motion"][1].update(
            start_time_seconds=2.0,
            end_time_seconds=7.0,
        )
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        "id": "waiting",
                        "description": "人物等待",
                        "start_time_seconds": 0.0,
                        "end_time_seconds": 7.0,
                        "reference_ids": ["person"],
                        "source_status": "explicit",
                        "source_text": "一个人在路边等待",
                    },
                    {
                        "id": "pickup",
                        "description": "车辆驶来并接走人物",
                        "start_time_seconds": 2.0,
                        "end_time_seconds": 7.0,
                        "reference_ids": ["car", "person"],
                        "source_status": "explicit",
                        "source_text": "一辆车开了过来把他接走了",
                    },
                ],
                "relations": [
                    {
                        "relation_id": "arrival_ends_wait",
                        "source_event_id": "pickup",
                        "target_event_id": "waiting",
                        "relation": "ends_with",
                        "minimum_gap_seconds": None,
                        "maximum_gap_seconds": None,
                        "source_status": "inferred",
                        "source_text": "等待，然后车辆驶来",
                    },
                    {
                        "relation_id": "wait_precedes_pickup_end",
                        "source_event_id": "waiting",
                        "target_event_id": "pickup",
                        "relation": "overlaps",
                        "minimum_gap_seconds": None,
                        "maximum_gap_seconds": None,
                        "source_status": "inferred",
                        "source_text": "等待期间车辆驶来",
                    },
                ],
            }
        )

        validate_model_output(content, self.model_schema)
        normalized, parameters = apply_translation_rules(
            content,
            self.rules,
            "一个人在路边等待，然后一辆车开了过来把他接走了",
        )

        events = normalized["timeline"]["events"]
        self.assertEqual(
            [(item["start_time_seconds"], item["end_time_seconds"]) for item in events],
            [(0.0, 7.0), (2.0, 7.0)],
        )
        self.assertEqual(
            [
                (item["start_time_seconds"], item["end_time_seconds"])
                for item in normalized["subject_motion"]
            ],
            [(0.0, 7.0), (2.0, 7.0)],
        )
        self.assertEqual(
            [
                (item["start_time_seconds"], item["end_time_seconds"])
                for item in parameters["motions"]
            ],
            [(0.0, 7.0), (2.0, 7.0)],
        )
        self.assertEqual(parameters["scene_dynamics"]["mode"], "dynamic")
        self.assertEqual(len(parameters["temporal_relations"]), 2)

    def test_sequential_events_are_not_mechanically_partitioned(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    self._event("first", "先等待", "person", 10.0),
                    self._event("second", "然后离开", "person", 10.0),
                ],
                "relations": [],
            }
        )

        with self.assertRaisesRegex(ValueError, "不能全部覆盖完整镜头"):
            apply_translation_rules(content, self.rules, "先等待，然后离开")

    def test_camera_only_motion_remains_static_scene(self) -> None:
        content = valid_model_output()
        content["camera"]["movement"]["type"] = self._annotated(
            "缓慢推近",
            "镜头缓慢推近",
        )

        normalized, parameters = apply_translation_rules(content, self.rules)

        self.assertEqual(normalized["scene_dynamics"]["mode"], "static")
        self.assertEqual(parameters["scene_dynamics"]["mode"], "static")

    def test_explicit_push_in_overrides_neutral_static_numeric_profile(self) -> None:
        content = valid_model_output()
        content["camera"]["movement"]["type"] = self._annotated(
            "缓慢推近",
            "镜头慢慢推近",
        )
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )

        normalized, parameters = apply_translation_rules(content, self.rules)

        camera = parameters["camera"]
        self.assertEqual(camera["movement"], "push_in")
        self.assertEqual(camera["source_status"], "explicit")
        self.assertLess(camera["end_distance_m"], camera["start_distance_m"])
        self.assertGreater(camera["speed_mps"], 0.0)
        self.assertEqual(normalized["camera"]["movement"]["trajectory"]["value"], "直线")

    def test_overlapping_before_relation_is_normalized_to_starts_before(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        **self._event("waiting", "人物在路边等待", "person", 7.0),
                        "start_time_seconds": 0.0,
                    },
                    {
                        **self._event("approach", "车辆开过来", "car", 6.0),
                        "start_time_seconds": 2.0,
                    },
                ],
                "relations": [
                    {
                        "relation_id": "rel_wait_before_approach",
                        "source_event_id": "waiting",
                        "target_event_id": "approach",
                        "relation": "before",
                        "minimum_gap_seconds": 0.0,
                        "maximum_gap_seconds": None,
                        "source_status": "inferred",
                        "source_text": "人在路边等待，一辆车开过来",
                    }
                ],
            }
        )

        normalized, parameters = apply_translation_rules(
            content,
            self.rules,
            "一个人在路边等待，一辆车开过来接走他，10 秒。",
        )

        self.assertEqual(normalized["timeline"]["relations"][0]["relation"], "starts_before")
        self.assertEqual(parameters["temporal_relations"][0]["relation"], "starts_before")

    def test_inferred_zero_gap_does_not_block_relation_normalization(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        **self._event("person_wait", "人物等待", "person", 7.0),
                        "start_time_seconds": 0.0,
                    },
                    {
                        **self._event("car_arrival", "车辆到达", "car", 7.0),
                        "start_time_seconds": 2.0,
                    },
                ],
                "relations": [
                    {
                        "relation_id": "car_arrival_meets_person_wait_end",
                        "source_event_id": "car_arrival",
                        "target_event_id": "person_wait",
                        "relation": "meets",
                        "minimum_gap_seconds": 0.0,
                        "maximum_gap_seconds": 0.0,
                        "source_status": "inferred",
                        "source_text": "车辆到达时人物结束等待",
                    }
                ],
            }
        )

        normalized, parameters = apply_translation_rules(content, self.rules)

        relation = normalized["timeline"]["relations"][0]
        self.assertEqual(relation["relation"], "ends_with")
        self.assertIsNone(relation["minimum_gap_seconds"])
        self.assertIsNone(relation["maximum_gap_seconds"])
        self.assertEqual(parameters["temporal_relations"][0]["relation"], "ends_with")

    def test_inconsistent_explicit_temporal_gap_is_rejected(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        **self._event("first", "先等待", "person", 4.0),
                        "start_time_seconds": 0.0,
                    },
                    {
                        **self._event("second", "然后离开", "person", 10.0),
                        "start_time_seconds": 3.0,
                    },
                ],
                "relations": [
                    {
                        "relation_id": "bad_explicit_gap",
                        "source_event_id": "first",
                        "target_event_id": "second",
                        "relation": "before",
                        "minimum_gap_seconds": 1.0,
                        "maximum_gap_seconds": None,
                        "source_status": "explicit",
                        "source_text": "至少间隔一秒",
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "时间不一致"):
            apply_translation_rules(content, self.rules, "先等待，至少一秒后再离开")

    def test_transport_semantics_are_not_reparsed_as_walking(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "person",
                "category": self._annotated("人", "一个人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "car",
                "category": self._annotated("车辆", "一辆车"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [
            self._motion(
                "person",
                "被车接走并随车离开",
                action_kind="transport",
                motion_type="carried",
                motion_mode="carried",
                direction_mode="relative_to_target",
                target_id="car",
                carrier_id="car",
                path_type="stationary",
                contained_by_id="car",
                external_visibility="hidden",
            )
        ]

        _, parameters = apply_translation_rules(content, self.rules)

        motion = parameters["motions"][0]
        self.assertEqual(motion["motion_type"], "carried")
        self.assertEqual(motion["motion_mode"], "carried")
        self.assertEqual(motion["carrier_id"], "car")
        self.assertEqual(motion["action_kind"], "locomotion")
        self.assertEqual(motion["direction_mode"], "none")
        self.assertIsNone(motion["target_id"])
        self.assertEqual(motion["speed_range_mps"], [0.0, 0.0])
        self.assertEqual(motion["postconditions"]["external_visibility"], "hidden")

    def test_inconsistent_carried_semantics_fail_before_planning(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "person",
                "category": self._annotated("人", "一个人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "car",
                "category": self._annotated("车辆", "一辆车"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["subject_motion"] = [
            self._motion(
                "person",
                "被车接走",
                action_kind="transport",
                motion_type="walking",
                motion_mode="carried",
                direction_mode="relative_to_target",
                target_id="car",
                carrier_id="car",
                path_type="stationary",
            )
        ]

        with self.assertRaisesRegex(ValueError, "必须使用 motion_type=carried"):
            apply_translation_rules(content, self.rules)

    def test_simultaneous_full_timeline_events_remain_concurrent(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    self._event("earth_orbit", "地球公转", "earth", 10.0),
                    self._event("moon_orbit", "月球公转", "moon", 10.0),
                ],
            }
        )

        normalized, _ = apply_translation_rules(
            content,
            self.rules,
            "地球围绕太阳公转，同时月球围绕地球公转",
        )

        self.assertEqual(
            [
                (item["start_time_seconds"], item["end_time_seconds"])
                for item in normalized["timeline"]["events"]
            ],
            [(0.0, 10.0), (0.0, 10.0)],
        )

    def test_sequential_prompt_requires_multiple_events(self) -> None:
        content = valid_model_output()
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [self._event("combined", "等待然后离开", "person", 10.0)],
            }
        )

        with self.assertRaisesRegex(ValueError, "没有拆分 timeline.events"):
            apply_translation_rules(
                content,
                self.rules,
                "一个人先等待，然后离开",
            )

    @staticmethod
    def _annotated(value: str, source_text: str) -> dict:
        return {"value": value, "source_status": "explicit", "source_text": source_text}

    @staticmethod
    def _statement(value: str, source_text: str) -> dict:
        return {"value": value, "source_status": "explicit", "source_text": source_text}

    @staticmethod
    def _unknown() -> dict:
        return {"value": None, "source_status": "unknown", "source_text": None}

    @classmethod
    def _motion(
        cls,
        subject_id: str,
        action: str,
        *,
        action_kind: str,
        motion_type: str,
        motion_mode: str,
        direction_mode: str,
        target_id: str | None = None,
        carrier_id: str | None = None,
        path_type: str,
        timeline_event_id: str | None = None,
        contained_by_id: str | None = None,
        external_visibility: str = "unchanged",
    ) -> dict:
        return {
            "motion_id": f"{subject_id}_{action_kind}",
            "subject_id": subject_id,
            "action": cls._annotated(action, action),
            "motion_semantics": {
                "action_kind": action_kind,
                "motion_type": motion_type,
                "motion_mode": motion_mode,
                "direction_mode": direction_mode,
                "target_id": target_id,
                "carrier_id": carrier_id,
                "path_type": path_type,
                "timeline_event_id": timeline_event_id,
                "narrative_required": True,
                "postconditions": {
                    "contained_by_id": contained_by_id,
                    "external_visibility": external_visibility,
                },
                "source_status": "inferred",
                "source_text": action,
            },
            "direction": cls._unknown(),
            "speed": cls._unknown(),
            "trajectory": cls._unknown(),
            "start_time_seconds": None,
            "end_time_seconds": None,
            "secondary_motion": [],
        }

    @staticmethod
    def _event(
        event_id: str,
        description: str,
        subject_id: str,
        duration: float,
    ) -> dict:
        return {
            "id": event_id,
            "description": description,
            "start_time_seconds": 0.0,
            "end_time_seconds": duration,
            "reference_ids": [subject_id],
            "source_status": "explicit",
            "source_text": description,
        }


if __name__ == "__main__":
    unittest.main()
