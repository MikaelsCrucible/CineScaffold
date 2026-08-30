from __future__ import annotations

import unittest

from cinescaffold.planning.domain import TrackSpec, TransformValue
from cinescaffold.planning.geometry import sample_path_track, sample_transform_track


class PlanningGeometryTest(unittest.TestCase):
    def test_closed_path_returns_to_first_point(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "orbit",
                "target_entity_id": "earth",
                "type": "path_follow",
                "time_range_seconds": [0.0, 10.0],
                "path": {
                    "representation": "polyline",
                    "space": "target_relative",
                    "target_id": "sun",
                    "control_points": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
                    "closed": True,
                },
            }
        )

        sampled = sample_path_track(track, 10.0, TransformValue())

        self.assertEqual(sampled.translation_m, (10.0, 0.0, 0.0))
        self.assertEqual(sampled.space, "target_relative")
        self.assertEqual(sampled.target_id, "sun")

    def test_closed_path_cycle_count_repeats_without_duplicating_points(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "fast_orbit",
                "target_entity_id": "moon",
                "type": "path_follow",
                "time_range_seconds": [0.0, 6.0],
                "path": {
                    "space": "target_relative",
                    "target_id": "earth",
                    "control_points": [
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [-2.0, 0.0, 0.0],
                        [0.0, -2.0, 0.0],
                    ],
                    "closed": True,
                    "cycle_count": 3.0,
                },
            }
        )

        halfway_first_cycle = sample_path_track(track, 1.0, TransformValue())
        start_second_cycle = sample_path_track(track, 2.0, TransformValue())

        self.assertEqual(halfway_first_cycle.translation_m, (-2.0, 0.0, 0.0))
        self.assertEqual(start_second_cycle.translation_m, (2.0, 0.0, 0.0))

    def test_open_path_rejects_multiple_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "只有闭合 Path"):
            TrackSpec.model_validate(
                {
                    "track_id": "invalid_repeat",
                    "target_entity_id": "moon",
                    "type": "path_follow",
                    "time_range_seconds": [0.0, 6.0],
                    "path": {
                        "control_points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "cycle_count": 2.0,
                    },
                }
            )

    def test_transform_interpolation_preserves_reference_frame(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "relative_move",
                "target_entity_id": "moon",
                "type": "transform",
                "time_range_seconds": [0.0, 10.0],
                "keyframes": [
                    {
                        "time_seconds": 0.0,
                        "value": {
                            "translation_m": [2.0, 0.0, 0.0],
                            "space": "target_relative",
                            "target_id": "earth",
                        },
                    },
                    {
                        "time_seconds": 10.0,
                        "value": {
                            "translation_m": [0.0, 2.0, 0.0],
                            "space": "target_relative",
                            "target_id": "earth",
                        },
                    },
                ],
            }
        )

        sampled = sample_transform_track(track, 5.0, TransformValue())

        self.assertEqual(sampled.translation_m, (1.0, 1.0, 0.0))
        self.assertEqual(sampled.space, "target_relative")
        self.assertEqual(sampled.target_id, "earth")

    def test_transform_track_rejects_mixed_reference_frames(self) -> None:
        with self.assertRaisesRegex(ValueError, "相同参考系"):
            TrackSpec.model_validate(
                {
                    "track_id": "invalid",
                    "target_entity_id": "moon",
                    "type": "transform",
                    "time_range_seconds": [0.0, 10.0],
                    "keyframes": [
                        {"time_seconds": 0.0, "value": {"translation_m": [0, 0, 0]}},
                        {
                            "time_seconds": 10.0,
                            "value": {
                                "translation_m": [1, 0, 0],
                                "space": "target_relative",
                                "target_id": "earth",
                            },
                        },
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
