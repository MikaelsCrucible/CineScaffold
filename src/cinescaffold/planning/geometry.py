from __future__ import annotations

import math
from typing import Any

from cinescaffold.planning.domain import (
    ProxyGeometry,
    Quaternion,
    TrackSpec,
    TransformValue,
    Vec3,
)


IDENTITY_QUATERNION: Quaternion = (1.0, 0.0, 0.0, 0.0)
UNIT_SCALE: Vec3 = (1.0, 1.0, 1.0)


def sample_transform_track(
    track: TrackSpec | None,
    time_seconds: float,
    fallback: TransformValue,
) -> TransformValue:
    if track is None or not track.keyframes:
        return _complete_transform(fallback)
    keyframes = sorted(track.keyframes, key=lambda item: item.time_seconds)
    if time_seconds <= keyframes[0].time_seconds:
        return _transform_from_value(keyframes[0].value, fallback)
    if time_seconds >= keyframes[-1].time_seconds:
        return _transform_from_value(keyframes[-1].value, fallback)

    for left, right in zip(keyframes, keyframes[1:]):
        if left.time_seconds <= time_seconds <= right.time_seconds:
            start = _transform_from_value(left.value, fallback)
            end = _transform_from_value(right.value, fallback)
            if left.interpolation == "step":
                return start
            ratio = (time_seconds - left.time_seconds) / (
                right.time_seconds - left.time_seconds
            )
            if left.interpolation == "smooth":
                ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            return TransformValue(
                translation_m=_lerp_vec3(start.translation_m, end.translation_m, ratio),
                rotation_quaternion_wxyz=_nlerp_quaternion(
                    start.rotation_quaternion_wxyz,
                    end.rotation_quaternion_wxyz,
                    ratio,
                ),
                scale=_lerp_vec3(start.scale, end.scale, ratio),
                space="world",
            )
    return _complete_transform(fallback)


def sample_path_track(
    track: TrackSpec,
    time_seconds: float,
    fallback: TransformValue,
) -> TransformValue:
    if track.path is None:
        return _complete_transform(fallback)
    start, end = track.time_range_seconds
    ratio = min(1.0, max(0.0, (time_seconds - start) / (end - start)))
    points = track.path.control_points
    if track.path.parameterization == "arc_length":
        position = _sample_arc_length(points, ratio)
    else:
        position = _sample_polyline(points, ratio)
    completed = _complete_transform(fallback)
    return TransformValue(
        translation_m=position,
        rotation_quaternion_wxyz=completed.rotation_quaternion_wxyz,
        scale=completed.scale,
        space="world",
    )


def sample_scalar_track(track: TrackSpec | None, time_seconds: float, fallback: float) -> float:
    if track is None or not track.keyframes:
        return fallback
    keyframes = sorted(track.keyframes, key=lambda item: item.time_seconds)
    if time_seconds <= keyframes[0].time_seconds:
        return float(keyframes[0].value)
    if time_seconds >= keyframes[-1].time_seconds:
        return float(keyframes[-1].value)
    for left, right in zip(keyframes, keyframes[1:]):
        if left.time_seconds <= time_seconds <= right.time_seconds:
            if left.interpolation == "step":
                return float(left.value)
            ratio = (time_seconds - left.time_seconds) / (
                right.time_seconds - left.time_seconds
            )
            if left.interpolation == "smooth":
                ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            return _lerp(float(left.value), float(right.value), ratio)
    return fallback


def look_at_camera_quaternion(position: Vec3, target: Vec3) -> Quaternion:
    forward = normalize(subtract(target, position))
    if length(forward) < 1e-8:
        raise ValueError("摄影机位置不能与观察目标重合")
    world_up: Vec3 = (0.0, 0.0, 1.0)
    if abs(dot(forward, world_up)) > 0.999:
        world_up = (0.0, 1.0, 0.0)
    axis_x = normalize(cross(forward, world_up))
    axis_z = multiply(forward, -1.0)
    axis_y = normalize(cross(axis_z, axis_x))
    matrix = (
        (axis_x[0], axis_y[0], axis_z[0]),
        (axis_x[1], axis_y[1], axis_z[1]),
        (axis_x[2], axis_y[2], axis_z[2]),
    )
    return matrix_to_quaternion(matrix)


def project_point(
    point: Vec3,
    camera_position: Vec3,
    camera_rotation: Quaternion,
    focal_length_mm: float,
    sensor_width_mm: float,
    aspect_ratio: float,
) -> tuple[float, float, float]:
    local = rotate_vector(quaternion_conjugate(camera_rotation), subtract(point, camera_position))
    depth = -local[2]
    if depth <= 1e-8:
        return (math.nan, math.nan, depth)
    sensor_height_mm = sensor_width_mm / aspect_ratio
    screen_x = 0.5 + focal_length_mm * local[0] / (depth * sensor_width_mm)
    screen_y = 0.5 - focal_length_mm * local[1] / (depth * sensor_height_mm)
    return (screen_x, screen_y, depth)


def projected_radius(
    geometry: ProxyGeometry,
    scale: Vec3,
    depth: float,
    focal_length_mm: float,
    sensor_width_mm: float,
) -> float:
    if depth <= 0:
        return math.inf
    radius = geometry_bounding_radius(geometry) * max(scale)
    return focal_length_mm * radius / (depth * sensor_width_mm)


def geometry_bounding_radius(geometry: ProxyGeometry) -> float:
    if geometry.type == "box":
        return 0.5 * math.sqrt(sum(item * item for item in geometry.size_xyz_m))
    if geometry.type == "sphere":
        return geometry.radius_m
    if geometry.type == "capsule":
        return geometry.radius_m + geometry.segment_length_m / 2.0
    if geometry.type == "cylinder":
        return math.sqrt(geometry.radius_m**2 + (geometry.depth_m / 2.0) ** 2)
    if geometry.type == "cone":
        radius = max(geometry.radius_bottom_m, geometry.radius_top_m)
        return math.sqrt(radius**2 + (geometry.depth_m / 2.0) ** 2)
    return 0.5 * math.sqrt(sum(item * item for item in geometry.size_xy_m))


def rotate_vector(quaternion: Quaternion, vector: Vec3) -> Vec3:
    pure: Quaternion = (0.0, *vector)
    rotated = quaternion_multiply(quaternion_multiply(quaternion, pure), quaternion_conjugate(quaternion))
    return (rotated[1], rotated[2], rotated[3])


def quaternion_multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quaternion_conjugate(value: Quaternion) -> Quaternion:
    return (value[0], -value[1], -value[2], -value[3])


def matrix_to_quaternion(matrix: tuple[tuple[float, ...], ...]) -> Quaternion:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        factor = math.sqrt(trace + 1.0) * 2.0
        result = (
            0.25 * factor,
            (matrix[2][1] - matrix[1][2]) / factor,
            (matrix[0][2] - matrix[2][0]) / factor,
            (matrix[1][0] - matrix[0][1]) / factor,
        )
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            factor = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            result = (
                (matrix[2][1] - matrix[1][2]) / factor,
                0.25 * factor,
                (matrix[0][1] + matrix[1][0]) / factor,
                (matrix[0][2] + matrix[2][0]) / factor,
            )
        elif index == 1:
            factor = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            result = (
                (matrix[0][2] - matrix[2][0]) / factor,
                (matrix[0][1] + matrix[1][0]) / factor,
                0.25 * factor,
                (matrix[1][2] + matrix[2][1]) / factor,
            )
        else:
            factor = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            result = (
                (matrix[1][0] - matrix[0][1]) / factor,
                (matrix[0][2] + matrix[2][0]) / factor,
                (matrix[1][2] + matrix[2][1]) / factor,
                0.25 * factor,
            )
    norm = math.sqrt(sum(item * item for item in result))
    normalized = tuple(item / norm for item in result)
    return normalized  # type: ignore[return-value]


def subtract(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def multiply(value: Vec3, scalar: float) -> Vec3:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)


def dot(left: Vec3, right: Vec3) -> float:
    return sum(a * b for a, b in zip(left, right))


def cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def length(value: Vec3) -> float:
    return math.sqrt(dot(value, value))


def normalize(value: Vec3) -> Vec3:
    norm = length(value)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return multiply(value, 1.0 / norm)


def _complete_transform(value: TransformValue) -> TransformValue:
    return TransformValue(
        translation_m=value.translation_m or (0.0, 0.0, 0.0),
        rotation_quaternion_wxyz=value.rotation_quaternion_wxyz or IDENTITY_QUATERNION,
        scale=value.scale or UNIT_SCALE,
        space="world",
    )


def _transform_from_value(value: Any, fallback: TransformValue) -> TransformValue:
    if isinstance(value, TransformValue):
        parsed = value
    elif isinstance(value, dict):
        parsed = TransformValue.model_validate(value)
    else:
        raise ValueError("transform 关键帧值必须是对象")
    completed = _complete_transform(fallback)
    return TransformValue(
        translation_m=parsed.translation_m or completed.translation_m,
        rotation_quaternion_wxyz=(
            parsed.rotation_quaternion_wxyz or completed.rotation_quaternion_wxyz
        ),
        scale=parsed.scale or completed.scale,
        space="world",
    )


def _sample_polyline(points: list[Vec3], ratio: float) -> Vec3:
    position = ratio * (len(points) - 1)
    index = min(len(points) - 2, int(position))
    local_ratio = position - index
    return _lerp_vec3(points[index], points[index + 1], local_ratio)


def _sample_arc_length(points: list[Vec3], ratio: float) -> Vec3:
    lengths = [length(subtract(right, left)) for left, right in zip(points, points[1:])]
    total = sum(lengths)
    if total <= 1e-12:
        return points[0]
    target = ratio * total
    elapsed = 0.0
    for index, segment in enumerate(lengths):
        if elapsed + segment >= target:
            return _lerp_vec3(points[index], points[index + 1], (target - elapsed) / segment)
        elapsed += segment
    return points[-1]


def _lerp(left: float, right: float, ratio: float) -> float:
    return left + (right - left) * ratio


def _lerp_vec3(left: Vec3 | None, right: Vec3 | None, ratio: float) -> Vec3:
    assert left is not None and right is not None
    return tuple(_lerp(a, b, ratio) for a, b in zip(left, right))  # type: ignore[return-value]


def _nlerp_quaternion(
    left: Quaternion | None,
    right: Quaternion | None,
    ratio: float,
) -> Quaternion:
    assert left is not None and right is not None
    if dot(left[1:], right[1:]) + left[0] * right[0] < 0:
        right = tuple(-item for item in right)  # type: ignore[assignment]
    value = tuple(_lerp(a, b, ratio) for a, b in zip(left, right))
    norm = math.sqrt(sum(item * item for item in value))
    return tuple(item / norm for item in value)  # type: ignore[return-value]
