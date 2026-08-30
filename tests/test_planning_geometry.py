from __future__ import annotations

import unittest

from cinescaffold.planning.domain import TrackSpec, TransformValue
from cinescaffold.planning.geometry import sample_path_track, sample_transform_track


class PlanningGeometryTest(unittest.TestCase):
    def test_circle_path_is_analytic_at_quarter_turn(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "circle",
                "target_entity_id": "earth",
                "type": "path_follow",
                "time_range_seconds": [0.0, 10.0],
                "path": {
                    "representation": "circle",
                    "space": "target_relative",
                    "target_id": "sun",
                    "radius_m": 10.0,
                },
            }
        )

        sampled = sample_path_track(track, 2.5, TransformValue())

        self.assertAlmostEqual(sampled.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(sampled.translation_m[1], 10.0, places=8)
        self.assertEqual(sampled.translation_m[2], 0.0)

    def test_circle_arc_length_keeps_exact_radius(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "circle_arc_length",
                "target_entity_id": "earth",
                "type": "path_follow",
                "time_range_seconds": [0.0, 10.0],
                "path": {
                    "representation": "circle",
                    "radius_m": 12.0,
                    "parameterization": "arc_length",
                },
            }
        )

        for time_seconds in (0.37, 1.91, 4.73, 8.88):
            sampled = sample_path_track(track, time_seconds, TransformValue())
            radius = sum(item * item for item in sampled.translation_m) ** 0.5
            self.assertAlmostEqual(radius, 12.0, places=10)

    def test_ellipse_path_uses_distinct_axes(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "ellipse",
                "target_entity_id": "earth",
                "type": "path_follow",
                "time_range_seconds": [0.0, 8.0],
                "path": {
                    "representation": "ellipse",
                    "semi_major_axis_m": 8.0,
                    "semi_minor_axis_m": 2.0,
                },
            }
        )

        sampled = sample_path_track(track, 2.0, TransformValue())

        self.assertAlmostEqual(sampled.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(sampled.translation_m[1], 2.0, places=8)

    def test_catmull_rom_path_passes_through_waypoints(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "s_curve",
                "target_entity_id": "subject",
                "type": "path_follow",
                "time_range_seconds": [0.0, 6.0],
                "path": {
                    "representation": "catmull_rom",
                    "control_points": [
                        [-6.0, 0.0, 0.0],
                        [-2.0, 3.0, 0.0],
                        [2.0, -3.0, 0.0],
                        [6.0, 0.0, 0.0],
                    ],
                },
            }
        )

        sampled = sample_path_track(track, 2.0, TransformValue())

        self.assertEqual(sampled.translation_m, (-2.0, 3.0, 0.0))

    def test_lemniscate_path_crosses_its_center(self) -> None:
        track = TrackSpec.model_validate(
            {
                "track_id": "figure_eight",
                "target_entity_id": "subject",
                "type": "path_follow",
                "time_range_seconds": [0.0, 8.0],
                "path": {
                    "representation": "lemniscate",
                    "width_m": 8.0,
                    "height_m": 4.0,
                },
            }
        )

        start = sample_path_track(track, 0.0, TransformValue())
        halfway = sample_path_track(track, 4.0, TransformValue())

        self.assertEqual(start.translation_m, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(halfway.translation_m[0], 0.0, places=8)
        self.assertAlmostEqual(halfway.translation_m[1], 0.0, places=8)

    def test_analytic_path_rejects_parallel_plane_basis(self) -> None:
        with self.assertRaisesRegex(ValueError, "不得与 plane_normal 平行"):
            TrackSpec.model_validate(
                {
                    "track_id": "invalid_circle",
                    "target_entity_id": "earth",
                    "type": "path_follow",
                    "time_range_seconds": [0.0, 10.0],
                    "path": {
                        "representation": "circle",
                        "radius_m": 10.0,
                        "plane_normal": [0.0, 0.0, 1.0],
                        "axis_direction": [0.0, 0.0, 2.0],
                    },
                }
            )

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
