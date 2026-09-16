from __future__ import annotations

import unittest

from cinescaffold.camera_semantics import (
    camera_lens_focal_length,
    classify_camera_movement,
    classify_camera_view_angle,
)
from cinescaffold.schema import load_schema, validate_model_output
from cinescaffold.semantic_rules import apply_translation_rules, load_translation_rules
from tests.helpers import ROOT, valid_model_output


class SemanticRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_translation_rules(
            ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
        )
        self.schema = load_schema(
            ROOT
            / "src/cinescaffold/resources/schemas/semantic_translation_parameters.schema.json"
        )
        self.model_schema = load_schema(
            ROOT
            / "src/cinescaffold/resources/schemas/cinematic_brief_model_output.schema.json"
        )

    def test_camera_movement_classifier_has_one_unambiguous_priority(self) -> None:
        self.assertEqual(classify_camera_movement("固定机位，先静止后旋转"), "pan")
        self.assertEqual(classify_camera_movement("环绕跟拍"), "orbit")
        self.assertEqual(classify_camera_movement("slow push-in"), "push_in")
        self.assertIsNone(classify_camera_movement("companion relationship"))

    def test_camera_lens_and_angle_aliases_do_not_match_unrelated_words(self) -> None:
        self.assertEqual(camera_lens_focal_length("50 mm"), 50.0)
        self.assertEqual(camera_lens_focal_length("wide-angle lens"), 35.0)
        self.assertIsNone(camera_lens_focal_length("worldwide release"))
        self.assertIsNone(camera_lens_focal_length("normalization pass"))
        self.assertEqual(classify_camera_view_angle("high-angle"), "high_angle")
        self.assertEqual(classify_camera_view_angle("垂直俯拍"), "top_down")

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
        content["mood"]["emotional_tones"] = [self._statement("孤独", "感觉很孤独")]
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )

        normalized, parameters = self._apply(content)

        validate_model_output(parameters, self.schema)
        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(normalized["camera"]["camera_height"]["value"], "1.2 m")
        self.assertEqual(
            normalized["camera"]["camera_height"]["source_status"], "inferred"
        )
        self.assertIn(
            "仅供最终视频生成",
            normalized["mood"]["lighting_intent"][-1]["value"],
        )
        self.assertEqual(parameters["emotion_class"]["class_id"], "E1")
        self.assertEqual(parameters["camera"]["height_m"], 1.2)
        self.assertEqual(parameters["camera"]["speed_mps"], 1.0)
        self.assertEqual(normalized["camera"]["movement"]["speed"]["value"], "1 m/s")
        self.assertEqual(parameters["composition"]["subject_frame_ratio"], [0.01, 0.05])
        self.assertNotIn("pitch_degrees", parameters["camera"])
        self.assertNotIn("horizon_from_bottom_ratio", parameters["composition"])
        self.assertNotIn("subject_horizontal", parameters["composition"])
        self.assertEqual(normalized["camera"]["view_angle"]["source_status"], "unknown")
        self.assertEqual(normalized["composition"]["screen_placements"], [])
        self.assertEqual(parameters["scene"]["asset_key"], "desert")
        self.assertEqual(parameters["subjects"][0]["reference_height_m"], 1.75)
        self.assertEqual(
            parameters["subjects"][0]["facing_direction_world"], [0.0, -1.0, 0.0]
        )
        self.assertEqual(
            parameters["lighting"]["application_scope"], "final_video_generation_only"
        )
        self.assertFalse(parameters["lighting"]["applied_to_blender_preview"])

    def test_camera_only_push_keeps_subject_scene_static(self) -> None:
        content = valid_model_output()
        content["camera"]["movement"].update(
            {
                "type": self._annotated("push_in", "镜头缓慢推近"),
                "speed": self._annotated("缓慢", "缓慢"),
                "start_time_seconds": 0.0,
                "end_time_seconds": 10.0,
            }
        )

        normalized, _ = self._apply(content)

        self.assertEqual(normalized["scene_dynamics"]["mode"], "static")
        self.assertTrue(
            all(
                item["motion_semantics"]["postconditions"]["external_visibility"]
                == "unchanged"
                for item in normalized["subject_motion"]
            )
        )

    def test_unresolved_uncertainty_cannot_claim_a_selected_value(self) -> None:
        content = valid_model_output()
        content["uncertainties"] = [
            {
                "field": "scene_dynamics.mode",
                "reason": "仍未决定",
                "resolution": "unresolved",
                "selected_value": "dynamic",
            }
        ]

        with self.assertRaisesRegex(ValueError, "selected_value 必须为 null"):
            self._apply(content)

    def test_subject_state_timeline_rejects_an_implicit_visible_hold(self) -> None:
        content = valid_model_output()
        content["subject_motion"] = [
            self._motion(
                "person_01",
                "移动",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="none",
                path_type="unspecified",
            )
        ]
        content["subject_motion"][0]["start_time_seconds"] = 3.0
        content["scene_dynamics"]["mode"] = "dynamic"

        with self.assertRaisesRegex(ValueError, "SEM-SUBJECT-STATE-COVERAGE"):
            self._apply(content)

    def test_narrative_carried_requires_overlapping_carrier_motion(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, source),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label, source in (
                ("person_01", "人", "一个人"),
                ("car", "车", "一辆车"),
            )
        ]
        carried = self._motion(
            "person_01",
            "被车带走",
            action_kind="locomotion",
            motion_type="carried",
            motion_mode="carried",
            direction_mode="none",
            carrier_id="car",
            path_type="stationary",
        )
        carried["motion_semantics"]["narrative_required"] = True
        content["subject_motion"] = [carried]
        content["scene_dynamics"]["mode"] = "dynamic"

        with self.assertRaisesRegex(ValueError, "SEM-CARRIED-CARRIER-MOTION"):
            self._apply(content)

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

                _, parameters = self._apply(content)

                self.assertAlmostEqual(
                    parameters["camera"]["speed_mps"],
                    expected_speed,
                )

    def test_invalid_radial_camera_rule_cannot_write_fixed_speed(self) -> None:
        rules = load_translation_rules(
            ROOT
            / "src/cinescaffold/resources/prompts/semantic_parser/translation_rules.json"
        )
        rules["emotion_classes"]["E1"]["camera"]["speed_mps"] = 0.67

        with self.assertRaisesRegex(ValueError, "不得同时写死速度"):
            apply_translation_rules(valid_model_output(), rules)

    def test_missing_slots_use_declared_defaults(self) -> None:
        content = valid_model_output()

        normalized, parameters = self._apply(content)

        self.assertEqual(normalized["subjects"][0]["category"]["value"], "人")
        self.assertEqual(normalized["scene_design"]["environment"]["value"], "空白空间")
        self.assertEqual(normalized["timeline"]["duration_seconds"], 10.0)
        self.assertEqual(parameters["emotion_class"]["class_id"], "E6")
        self.assertEqual(parameters["motions"][0]["motion_type"], "static")
        self.assertEqual(parameters["motions"][0]["speed_range_mps"], [0.0, 0.0])
        default_fields = {item["field"] for item in normalized["uncertainties"]}
        self.assertIn("subjects", default_fields)
        self.assertIn("timeline.duration_seconds", default_fields)

    def test_explicit_camera_is_recorded_as_override(self) -> None:
        content = valid_model_output()
        content["camera"]["view_angle"] = self._annotated(
            "high_angle", "使用俯拍"
        )
        content["mood"]["emotional_tones"] = [self._statement("孤独", "感觉孤独")]

        _, parameters = self._apply(content)

        self.assertIn("camera.view_angle", parameters["explicit_override_paths"])
        normalized, _ = self._apply(content)
        self.assertEqual(normalized["camera"]["view_angle"]["value"], "high_angle")

    def test_duration_upper_bound_resolves_to_its_maximum(self) -> None:
        content = valid_model_output()
        content["timeline"]["duration_range_seconds"] = {
            "minimum_seconds": 0.0,
            "maximum_seconds": 10.0,
        }
        content["timeline"]["duration_source_status"] = "explicit"

        normalized, _ = self._apply(content)

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
                action_kind="locomotion",
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
        content["scene_dynamics"]["mode"] = "dynamic"

        _, parameters = self._apply(content)

        motion = parameters["motions"][0]
        self.assertEqual(motion["motion_type"], "walking")
        self.assertEqual(motion["target_id"], "ship")
        self.assertEqual(motion["direction_mode"], "toward_target")
        self.assertIsNone(motion["direction_vector_world"])
        self.assertEqual(parameters["subjects"][1]["minimum_footprint_m"], [10.0, 10.0])
        self.assertEqual(parameters["subjects"][1]["default_scene_depth_ratio"], 0.5)

    def test_spatial_layer_does_not_invent_a_reference_relationship(self) -> None:
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
                "category": self._annotated("飞船", "巨大的飞船"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        content["scene_design"]["spatial_layers"] = [
            {
                "layer": "远景",
                "content_ids": ["ship"],
                "source_status": "explicit",
                "source_text": "远处有巨大的飞船",
            }
        ]

        normalized, _ = self._apply(content)

        self.assertEqual(normalized["scene_design"]["relationships"], [])

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
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="none",
                path_type="linear",
            )
        ]
        content["scene_dynamics"]["mode"] = "dynamic"

        normalized, parameters = self._apply(content)

        semantics = normalized["subject_motion"][0]["motion_semantics"]
        self.assertEqual(semantics["action_kind"], "locomotion")
        self.assertEqual(semantics["direction_mode"], "none")
        self.assertIsNone(semantics["target_id"])
        self.assertEqual(parameters["motions"][0]["action_kind"], "locomotion")
        self.assertEqual(parameters["motions"][0]["direction_mode"], "none")

    def test_dynamic_entity_timelines_reject_unexplained_independent_gaps(self) -> None:
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
        content["scene_dynamics"]["mode"] = "dynamic"

        validate_model_output(content, self.model_schema)
        with self.assertRaisesRegex(ValueError, "SEM-SUBJECT-STATE-COVERAGE"):
            self._apply(
                content,
                "一个人在路边等待，然后一辆车开了过来把他接走了",
            )

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
            self._apply(content, "先等待，然后离开")

    def test_relationship_throughout_can_target_an_arbitrary_event_interval(
        self,
    ) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, label),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label in (("a", "主体A"), ("b", "主体B"))
        ]
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        "id": "temporary_near",
                        "description": "二者短暂靠近",
                        "start_time_seconds": 4.2,
                        "end_time_seconds": 4.8,
                        "reference_ids": ["a", "b"],
                        "source_status": "explicit",
                        "source_text": "二者在中途短暂靠近",
                    }
                ],
            }
        )
        content["scene_design"]["relationships"] = [
            {
                "type": "proximity",
                "subject_id": "a",
                "reference_id": "b",
                "source_status": "explicit",
                "source_text": "二者在中途短暂靠近",
                "timeline_event_id": "temporary_near",
                "temporal_mode": "throughout",
            }
        ]

        normalized, _ = self._apply(content)

        relation = normalized["scene_design"]["relationships"][0]
        self.assertEqual(relation["timeline_event_id"], "temporary_near")
        self.assertEqual(relation["temporal_mode"], "throughout")
        event = normalized["timeline"]["events"][0]
        self.assertEqual(
            (event["start_time_seconds"], event["end_time_seconds"]),
            (4.2, 4.8),
        )

    def test_relationship_can_target_an_event_midpoint(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, label),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label in (("a", "主体A"), ("b", "主体B"))
        ]
        content["timeline"]["events"] = [
            {
                "id": "closest_moment",
                "description": "二者最接近的时刻",
                "start_time_seconds": 4.0,
                "end_time_seconds": 6.0,
                "reference_ids": ["a", "b"],
                "source_status": "inferred",
                "source_text": None,
            }
        ]
        content["scene_design"]["relationships"] = [
            {
                "type": "proximity",
                "subject_id": "a",
                "reference_id": "b",
                "source_status": "explicit",
                "source_text": "二者短暂交会",
                "timeline_event_id": "closest_moment",
                "temporal_mode": "at_midpoint",
            }
        ]

        normalized, _ = self._apply(content)

        self.assertEqual(
            normalized["scene_design"]["relationships"][0]["temporal_mode"],
            "at_midpoint",
        )

    def test_camera_only_motion_remains_static_scene(self) -> None:
        content = valid_model_output()
        content["camera"]["movement"]["type"] = self._annotated(
            "push_in",
            "镜头缓慢推近",
        )

        normalized, parameters = self._apply(content)

        self.assertEqual(normalized["scene_dynamics"]["mode"], "static")
        self.assertEqual(parameters["scene_dynamics"]["mode"], "static")

    def test_fixed_position_pan_is_isolated_from_subject_motion(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "car_01",
                "category": self._annotated("车辆", "一辆车"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
        ]
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )
        content["subject_motion"] = [
            self._motion(
                "car_01",
                "保持不动",
                action_kind="hold",
                motion_type="static",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
            ),
            self._motion(
                "car_01",
                "开远",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="none",
                path_type="linear",
            )
        ]
        content["subject_motion"][0].update(
            motion_id="car_hold",
            start_time_seconds=0.0,
            end_time_seconds=5.0,
        )
        content["subject_motion"][1].update(
            motion_id="car_move",
            start_time_seconds=5.0,
            end_time_seconds=10.0,
        )
        content["scene_dynamics"]["mode"] = "dynamic"
        camera_movement = content["camera"]["movement"]
        camera_movement.update(
            {
                "type": self._annotated(
                    "pan",
                    "镜头不平移而是旋转地跟着车",
                ),
                "target_id": "car_01",
                "start_time_seconds": 5.0,
                "end_time_seconds": 10.0,
            }
        )

        normalized, parameters = self._apply(content)

        semantics = normalized["subject_motion"][1]["motion_semantics"]
        self.assertEqual(semantics["motion_mode"], "self_propelled")
        self.assertEqual(parameters["camera"]["movement"], "pan")
        self.assertEqual(parameters["camera"]["speed_mps"], 0.0)
        self.assertEqual(normalized["camera"]["movement"]["target_id"], "car_01")
        self.assertIsNone(
            normalized["camera"]["movement"]["trajectory"]["value"]
        )

    def test_environmental_motion_cannot_override_scene_entity_dynamics(self) -> None:
        content = valid_model_output()
        content["scene_design"]["environmental_motion"] = ["风沙流动"]
        content["scene_dynamics"] = {
            "mode": "dynamic",
            "source_status": "inferred",
            "reason": "环境中有风沙",
        }

        with self.assertRaisesRegex(ValueError, "scene_dynamics.mode"):
            self._apply(content)

    def test_explicit_push_in_overrides_neutral_static_numeric_profile(self) -> None:
        content = valid_model_output()
        content["camera"]["movement"]["type"] = self._annotated(
            "push_in",
            "镜头慢慢推近",
        )
        content["timeline"].update(
            {"duration_seconds": 10.0, "duration_source_status": "explicit"}
        )

        normalized, parameters = self._apply(content)

        camera = parameters["camera"]
        self.assertEqual(camera["movement"], "push_in")
        self.assertEqual(camera["source_status"], "explicit")
        self.assertLess(camera["end_distance_m"], camera["start_distance_m"])
        self.assertGreater(camera["speed_mps"], 0.0)
        self.assertIsNone(
            normalized["camera"]["movement"]["trajectory"]["value"]
        )

    def test_overlapping_before_relation_is_rejected_for_revision(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "时间不一致"):
            self._apply(content)

    def test_inferred_zero_gap_relation_is_rejected_for_revision(self) -> None:
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

        with self.assertRaisesRegex(ValueError, "时间不一致"):
            self._apply(content)

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
            self._apply(content, "先等待，至少一秒后再离开")

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
                action_kind="locomotion",
                motion_type="carried",
                motion_mode="carried",
                direction_mode="none",
                carrier_id="car",
                path_type="stationary",
                contained_by_id="car",
                external_visibility="becomes_hidden",
            ),
            self._motion(
                "car",
                "载着人离开",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="none",
                path_type="unspecified",
            ),
        ]
        content["scene_dynamics"]["mode"] = "dynamic"

        _, parameters = self._apply(content)

        motion = next(
            item for item in parameters["motions"] if item["subject_id"] == "person"
        )
        self.assertEqual(motion["motion_type"], "carried")
        self.assertEqual(motion["motion_mode"], "carried")
        self.assertEqual(motion["carrier_id"], "car")
        self.assertEqual(motion["action_kind"], "locomotion")
        self.assertEqual(motion["direction_mode"], "none")
        self.assertIsNone(motion["target_id"])
        self.assertEqual(motion["speed_range_mps"], [0.0, 0.0])
        self.assertEqual(
            motion["postconditions"]["external_visibility"], "becomes_hidden"
        )

    def test_redundant_carried_semantics_are_rejected_for_revision(
        self,
    ) -> None:
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
        content["scene_dynamics"]["mode"] = "dynamic"

        with self.assertRaisesRegex(ValueError, "motion_mode=carried"):
            self._apply(content)

    def test_locomotion_conflict_is_rejected_for_revision(
        self,
    ) -> None:
        content = valid_model_output()
        content["camera"]["movement"]["type"] = self._annotated(
            "pan",
            "镜头不平移而是旋转地跟着车",
        )
        content["camera"]["movement"]["target_id"] = "person_01"
        content["subject_motion"] = [
            self._motion(
                "person_01",
                "开走",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="stationary",
                direction_mode="none",
                path_type="linear",
            )
        ]

        with self.assertRaisesRegex(ValueError, "action_kind=hold"):
            self._apply(content)

    def test_hold_motion_type_conflict_is_rejected_for_revision(
        self,
    ) -> None:
        content = valid_model_output()
        content["subject_motion"] = [
            self._motion(
                "person_01",
                "等待",
                action_kind="hold",
                motion_type="moving",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
            )
        ]

        with self.assertRaisesRegex(ValueError, "motion_type=static"):
            self._apply(content)

    def test_tied_motion_evidence_still_fails_closed(self) -> None:
        content = valid_model_output()
        content["subject_motion"] = [
            self._motion(
                "person_01",
                "含义不明",
                action_kind="other",
                motion_type="moving",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
            )
        ]

        with self.assertRaisesRegex(ValueError, "必须使用 action_kind=hold"):
            self._apply(content)

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

        normalized, _ = self._apply(
            content,
            "地球围绕太阳公转，同时月球围绕地球公转",
        )

        self.assertEqual(
            {
                item["id"]: (item["start_time_seconds"], item["end_time_seconds"])
                for item in normalized["timeline"]["events"]
                if item["id"] in {"earth_orbit", "moon_orbit"}
            },
            {"earth_orbit": (0.0, 10.0), "moon_orbit": (0.0, 10.0)},
        )

    def test_sequential_prompt_requires_multiple_events(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "person_01",
                "category": self._annotated("人", "一个人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
        ]
        content["subject_motion"] = [
            self._motion(
                "person_01",
                "等待然后离开",
                action_kind="hold",
                motion_type="static",
                motion_mode="stationary",
                direction_mode="none",
                path_type="stationary",
                timeline_event_id="combined",
            )
        ]
        content["timeline"].update(
            {
                "duration_seconds": 10.0,
                "duration_source_status": "explicit",
                "events": [
                    {
                        **self._event("combined", "等待然后离开", "person", 10.0),
                        "reference_ids": ["person_01"],
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "没有拆分 timeline.events"):
            self._apply(content, "一个人先等待，然后离开")

    def test_relative_target_linear_path_is_rejected_before_planning(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": "man_01",
                "category": self._annotated("男人", "男人"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
            {
                "id": "ship_01",
                "category": self._annotated("飞船", "飞船"),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            },
        ]
        motion = self._motion(
            "man_01",
            "相对飞船运动",
            action_kind="locomotion",
            motion_type="moving",
            motion_mode="self_propelled",
            direction_mode="relative_to_target",
            target_id="ship_01",
            path_type="linear",
        )
        motion["direction"] = self._annotated("相对飞船", "相对飞船运动")
        content["subject_motion"] = [motion]
        content["scene_dynamics"]["mode"] = "dynamic"

        with self.assertRaisesRegex(ValueError, "relative_to_target"):
            self._apply(content)

    def test_closed_target_relative_paths_survive_unknown_direction_annotation(
        self,
    ) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, label),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label in (
                ("sun", "太阳"),
                ("earth", "地球"),
                ("moon", "月亮"),
            )
        ]
        content["subject_motion"] = [
            self._motion(
                "earth",
                "地球绕太阳公转",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="relative_to_target",
                target_id="sun",
                path_type="circular",
            ),
            self._motion(
                "moon",
                "月亮绕地球公转",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="relative_to_target",
                target_id="earth",
                path_type="circular",
            ),
        ]
        content["scene_dynamics"]["mode"] = "dynamic"

        normalized, parameters = self._apply(content)

        self.assertEqual(
            [
                (
                    item["motion_semantics"]["direction_mode"],
                    item["motion_semantics"]["target_id"],
                )
                for item in normalized["subject_motion"]
                if item["subject_id"] in {"earth", "moon"}
            ],
            [("relative_to_target", "sun"), ("relative_to_target", "earth")],
        )
        self.assertEqual(
            [
                item["target_id"]
                for item in parameters["motions"]
                if item["subject_id"] in {"earth", "moon"}
            ],
            ["sun", "earth"],
        )

    def test_closed_path_requires_its_own_typed_target(self) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, label),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label in (("man_01", "人物"), ("sun", "太阳"))
        ]
        content["subject_motion"] = [
            self._motion(
                "man_01",
                "围绕太阳运动",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="none",
                path_type="circular",
            )
        ]
        content["scene_dynamics"]["mode"] = "dynamic"
        content["scene_design"]["relationships"] = []

        with self.assertRaisesRegex(ValueError, "相对闭合路径缺少有效几何目标"):
            self._apply(content)

    def test_orbit_relation_does_not_hide_conflicting_closed_path_direction(
        self,
    ) -> None:
        content = valid_model_output()
        content["subjects"] = [
            {
                "id": entity_id,
                "category": self._annotated(label, label),
                "description": self._unknown(),
                "narrative_role": self._unknown(),
                "attributes": [],
            }
            for entity_id, label in (("moon", "月亮"), ("earth", "地球"))
        ]
        content["subject_motion"] = [
            self._motion(
                "moon",
                "月亮绕地球公转",
                action_kind="locomotion",
                motion_type="moving",
                motion_mode="self_propelled",
                direction_mode="away_from_target",
                target_id="earth",
                path_type="circular",
            )
        ]
        content["scene_dynamics"]["mode"] = "dynamic"
        content["scene_design"]["relationships"] = []

        with self.assertRaisesRegex(ValueError, "相对闭合路径"):
            self._apply(content)

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
        local_components: list[str] | None = None,
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
                "local_components": (
                    local_components
                    if local_components is not None
                    else (["rotation"] if motion_mode == "local_interaction" else [])
                ),
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
            "start_time_seconds": 0.0,
            "end_time_seconds": 10.0,
            "secondary_motion": [],
        }

    def _apply(
        self,
        content: dict,
        source_prompt: str | None = None,
    ) -> tuple[dict, dict]:
        """Give legacy unit fixtures the event records required by v0.21 input."""

        events = content.setdefault("timeline", {}).setdefault("events", [])
        for index, motion in enumerate(content.get("subject_motion", [])):
            semantics = motion["motion_semantics"]
            if semantics.get("timeline_event_id") is not None:
                continue
            event_id = f"test_motion_{index}"
            semantics["timeline_event_id"] = event_id
            events.append(
                {
                    "id": event_id,
                    "description": motion["action"]["value"],
                    "start_time_seconds": motion["start_time_seconds"],
                    "end_time_seconds": motion["end_time_seconds"],
                    "reference_ids": [motion["subject_id"]],
                    "source_status": "inferred",
                    "source_text": motion["action"]["source_text"],
                }
            )
        return apply_translation_rules(content, self.rules, source_prompt)

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
            "reference_ids": ["person_01"] if subject_id == "person" else [],
            "source_status": "explicit",
            "source_text": description,
        }


if __name__ == "__main__":
    unittest.main()
