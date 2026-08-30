from __future__ import annotations

import math
from typing import Any, Callable

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
    if time_seconds < track.time_range_seconds[0]:
        # 晚开始的轨道不得把首关键帧提前施加到整个镜头。
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
                space=start.space,
                target_id=start.target_id,
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
    if time_seconds < start:
        return _complete_transform(fallback)
    ratio = min(1.0, max(0.0, (time_seconds - start) / (end - start)))
    if track.interpolation == "smooth":
        ratio = ratio * ratio * (3.0 - 2.0 * ratio)
    if track.path.closed:
        # 闭合路径可在同一 Track 时间段内重复或只走部分圈数。
        ratio = (ratio * track.path.cycle_count) % 1.0
    position = _sample_path_position(track.path, ratio)
    completed = _complete_transform(fallback)
    return TransformValue(
        translation_m=position,
        rotation_quaternion_wxyz=completed.rotation_quaternion_wxyz,
        scale=completed.scale,
        space=track.path.space,
        target_id=track.path.target_id,
    )


def _sample_path_position(path: Any, ratio: float) -> Vec3:
    if path.representation in {"polyline", "sampled"}:
        points = list(path.control_points)
        if path.closed and points[-1] != points[0]:
            points.append(points[0])
        if path.parameterization == "arc_length":
            return _sample_arc_length(points, ratio)
        return _sample_polyline(points, ratio)

    if path.representation == "catmull_rom":
        evaluator = lambda value: _sample_catmull_rom(
            list(path.control_points),
            value,
            closed=path.closed,
        )
    else:
        evaluator = lambda value: _sample_analytic_path(path, value)
    if path.representation == "circle":
        # 圆的参数角天然等弧长，直接求值以保持精确半径。
        return evaluator(ratio)
    if path.parameterization == "arc_length":
        return _sample_parametric_arc_length(evaluator, ratio)
    return evaluator(ratio)


def _sample_analytic_path(path: Any, ratio: float) -> Vec3:
    axis_u, axis_v = _path_plane_basis(path.plane_normal, path.axis_direction)
    sign = 1.0 if path.direction == "counterclockwise" else -1.0
    angle = math.radians(path.initial_phase_degrees) + sign * math.tau * ratio
    if path.representation == "circle":
        local_u = path.radius_m * math.cos(angle)
        local_v = path.radius_m * math.sin(angle)
    elif path.representation == "ellipse":
        local_u = path.semi_major_axis_m * math.cos(angle)
        local_v = path.semi_minor_axis_m * math.sin(angle)
    else:
        # Gerono 双纽线使用完整 width/height 作为包围盒尺寸。
        local_u = path.width_m * 0.5 * math.sin(angle)
        local_v = path.height_m * math.sin(angle) * math.cos(angle)
    return add(
        path.center_offset_m,
        add(multiply(axis_u, local_u), multiply(axis_v, local_v)),
    )


def _path_plane_basis(plane_normal: Vec3, axis_direction: Vec3) -> tuple[Vec3, Vec3]:
    normal = normalize(plane_normal)
    projected = subtract(axis_direction, multiply(normal, dot(axis_direction, normal)))
    axis_u = normalize(projected)
    axis_v = normalize(cross(normal, axis_u))
    return axis_u, axis_v


def _sample_catmull_rom(points: list[Vec3], ratio: float, *, closed: bool) -> Vec3:
    if closed:
        ratio %= 1.0
        segment_count = len(points)
        position = ratio * segment_count
        index = int(position) % segment_count
        local_ratio = position - math.floor(position)
        p0 = points[(index - 1) % segment_count]
        p1 = points[index]
        p2 = points[(index + 1) % segment_count]
        p3 = points[(index + 2) % segment_count]
    else:
        if ratio >= 1.0:
            return points[-1]
        segment_count = len(points) - 1
        position = max(0.0, ratio) * segment_count
        index = min(segment_count - 1, int(position))
        local_ratio = position - index
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
    t2 = local_ratio * local_ratio
    t3 = t2 * local_ratio
    values = []
    for axis in range(3):
        value = 0.5 * (
            2.0 * p1[axis]
            + (-p0[axis] + p2[axis]) * local_ratio
            + (2.0 * p0[axis] - 5.0 * p1[axis] + 4.0 * p2[axis] - p3[axis]) * t2
            + (-p0[axis] + 3.0 * p1[axis] - 3.0 * p2[axis] + p3[axis]) * t3
        )
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def _sample_parametric_arc_length(
    evaluator: Callable[[float], Vec3],
    ratio: float,
) -> Vec3:
    # 固定采样密度保证曲线重参数化可复现。
    points = [evaluator(index / 512.0) for index in range(513)]
    return _sample_arc_length(points, ratio)


def sample_scalar_track(track: TrackSpec | None, time_seconds: float, fallback: float) -> float:
    if track is None or not track.keyframes:
        return fallback
    if time_seconds < track.time_range_seconds[0]:
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


def project_geometry_bounds(
    geometry: ProxyGeometry,
    transform: TransformValue,
    camera_position: Vec3,
    camera_rotation: Quaternion,
    focal_length_mm: float,
    sensor_width_mm: float,
    aspect_ratio: float,
) -> tuple[float, float, float, float, float]:
    """投影代理体的局部包围盒，返回屏幕矩形与中心深度。"""
    center = transform.translation_m or (0.0, 0.0, 0.0)
    rotation = transform.rotation_quaternion_wxyz or IDENTITY_QUATERNION
    scale = transform.scale or UNIT_SCALE
    projected: list[tuple[float, float, float]] = []
    for point in geometry_local_bounds_points(geometry):
        scaled = tuple(point[index] * scale[index] for index in range(3))
        world = add(center, rotate_vector(rotation, scaled))
        value = project_point(
            world,
            camera_position,
            camera_rotation,
            focal_length_mm,
            sensor_width_mm,
            aspect_ratio,
        )
        if not all(math.isfinite(item) for item in value) or value[2] <= 0:
            return (math.nan, math.nan, math.nan, math.nan, value[2])
        projected.append(value)
    center_depth = project_point(
        center,
        camera_position,
        camera_rotation,
        focal_length_mm,
        sensor_width_mm,
        aspect_ratio,
    )[2]
    return (
        min(item[0] for item in projected),
        min(item[1] for item in projected),
        max(item[0] for item in projected),
        max(item[1] for item in projected),
        center_depth,
    )


def projected_box_axis_lengths(
    geometry: ProxyGeometry,
    transform: TransformValue,
    camera_position: Vec3,
    camera_rotation: Quaternion,
    focal_length_mm: float,
    sensor_width_mm: float,
    aspect_ratio: float,
) -> tuple[float, float, float] | None:
    """测量 box 三条局部轴在屏幕上的投影长度。"""
    if geometry.type != "box":
        return None
    center = transform.translation_m or (0.0, 0.0, 0.0)
    rotation = transform.rotation_quaternion_wxyz or IDENTITY_QUATERNION
    scale = transform.scale or UNIT_SCALE
    lengths: list[float] = []
    for axis, size in enumerate(geometry.size_xyz_m):
        half = size * scale[axis] / 2.0
        local_start = [0.0, 0.0, 0.0]
        local_end = [0.0, 0.0, 0.0]
        local_start[axis] = -half
        local_end[axis] = half
        start = add(center, rotate_vector(rotation, tuple(local_start)))
        end = add(center, rotate_vector(rotation, tuple(local_end)))
        projected_start = project_point(
            start,
            camera_position,
            camera_rotation,
            focal_length_mm,
            sensor_width_mm,
            aspect_ratio,
        )
        projected_end = project_point(
            end,
            camera_position,
            camera_rotation,
            focal_length_mm,
            sensor_width_mm,
            aspect_ratio,
        )
        if projected_start[2] <= 0 or projected_end[2] <= 0:
            return None
        lengths.append(
            math.hypot(
                projected_end[0] - projected_start[0],
                projected_end[1] - projected_start[1],
            )
        )
    return tuple(lengths)  # type: ignore[return-value]


def geometry_local_bounds_points(geometry: ProxyGeometry) -> list[Vec3]:
    """返回能覆盖代理体的确定性局部包围盒顶点。"""
    half = geometry_half_extents(geometry)
    if geometry.type == "plane":
        return [
            (x * half[0], y * half[1], 0.0)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
        ]
    return [
        (x * half[0], y * half[1], z * half[2])
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]


def geometry_half_extents(geometry: ProxyGeometry) -> Vec3:
    if geometry.type == "box":
        return tuple(item / 2.0 for item in geometry.size_xyz_m)  # type: ignore[return-value]
    if geometry.type == "sphere":
        return (geometry.radius_m, geometry.radius_m, geometry.radius_m)
    if geometry.type == "capsule":
        values = [geometry.radius_m, geometry.radius_m, geometry.radius_m]
        values[{"+X": 0, "+Y": 1, "+Z": 2}[geometry.axis]] += geometry.segment_length_m / 2.0
        return tuple(values)  # type: ignore[return-value]
    if geometry.type in {"cylinder", "cone"}:
        radius = (
            geometry.radius_m
            if geometry.type == "cylinder"
            else max(geometry.radius_bottom_m, geometry.radius_top_m)
        )
        values = [radius, radius, radius]
        values[{"+X": 0, "+Y": 1, "+Z": 2}[geometry.axis]] = geometry.depth_m / 2.0
        return tuple(values)  # type: ignore[return-value]
    return (geometry.size_xy_m[0] / 2.0, geometry.size_xy_m[1] / 2.0, 0.0)


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


def add(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


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
        space=value.space,
        target_id=value.target_id,
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
        space=parsed.space,
        target_id=parsed.target_id,
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
