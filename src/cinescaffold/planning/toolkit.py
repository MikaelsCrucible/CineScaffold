from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from cinescaffold.planning.domain import (
    CameraCandidate,
    CameraStatic,
    CandidateState,
    ConstraintSpec,
    EntitySpec,
    PlanningProfile,
    TimelineSpec,
    TrackSpec,
    TransformValue,
    ValidationReport,
    Violation,
)
from cinescaffold.planning.geometry import (
    dot,
    geometry_bounding_radius,
    length,
    look_at_camera_quaternion,
    normalize,
    project_point,
    projected_radius,
    rotate_vector,
    sample_path_track,
    sample_scalar_track,
    sample_transform_track,
    subtract,
)
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.planning.store import CandidateStore, MutationResult, canonical_hash


TOOLKIT_VERSION = "0.1"
CONSTRAINT_CATALOG_VERSION = "0.1"
SUPPORTED_CONSTRAINTS = {
    "relative_position",
    "distance_range",
    "depth_order",
    "screen_region",
    "projected_size",
    "projected_scale_ratio",
    "keep_in_frame",
    "look_at",
    "camera_distance",
    "focal_length_range",
    "camera_motion_direction",
    "position_at_time",
    "motion_direction",
    "hold",
}
FULL_VALIDATION_CHECKS = [
    "schema",
    "references",
    "timeline",
    "hierarchy",
    "transforms",
    "projection",
    "composition",
    "visibility",
    "motion",
    "camera",
    "hard_semantics",
    "rebuildability",
]


class ScenePlanningToolkit:
    def __init__(
        self,
        objective_brief: ObjectivePlanningBrief,
        profile: PlanningProfile | None = None,
        scene_id: str | None = None,
    ) -> None:
        self.objective_brief = objective_brief.model_copy(deep=True)
        self.profile = profile or PlanningProfile()
        self.store = CandidateStore(
            _initial_candidate(self.objective_brief, self.profile, scene_id)
        )

    def get_capabilities(self, sections: list[str] | None = None) -> dict[str, Any]:
        state = self.store.get()
        requested = sections or [
            "entities",
            "constraints",
            "tracks",
            "camera",
            "validators",
            "limits",
        ]
        data: dict[str, Any] = {
            "versions": {
                "toolkit": TOOLKIT_VERSION,
                "constraint_catalog": CONSTRAINT_CATALOG_VERSION,
                "scene_ir": "0.1",
                "profile": self.profile.profile_id,
            },
            "sections": requested,
            "supported_geometry": ["box", "sphere", "capsule", "cylinder", "cone", "plane"],
            "supported_tracks": ["transform", "path_follow", "visibility", "look_at", "focal_length"],
            "supported_constraints": sorted(SUPPORTED_CONSTRAINTS),
            "validators": FULL_VALIDATION_CHECKS,
            "current_counts": {
                "entities": len(state.entities),
                "tracks": len(state.motion_tracks),
                "constraints": len(state.constraints),
                "revision": state.revision,
            },
        }
        gaps = [
            "compound_proxy_geometry",
            "occlusion_fraction_validator",
            "collision_clearance_solver",
            "negative_space_validator",
            "event_synchronization_solver",
        ]
        return _envelope(state.revision, state.revision, data=data, capability_gaps=gaps)

    def inspect_candidate(
        self,
        revision: int | None = None,
        view: str = "summary",
        entity_ids: list[str] | None = None,
        camera_ids: list[str] | None = None,
        constraint_ids: list[str] | None = None,
        time_range_seconds: tuple[float, float] | None = None,
        compare_to_revision: int | None = None,
    ) -> dict[str, Any]:
        del camera_ids, time_range_seconds
        state = self.store.get(revision)
        if view == "summary":
            data: Any = {
                "scene_id": state.scene_id,
                "revision": state.revision,
                "entity_count": len(state.entities),
                "motion_track_count": len(state.motion_tracks),
                "constraint_count": len(state.constraints),
                "camera_ready": state.camera is not None,
                "validation": state.validation.model_dump(mode="json") if state.validation else None,
                "candidate_hash": canonical_hash(state),
            }
        elif view == "entities":
            selected = entity_ids or sorted(state.entities)
            data = {
                name: state.entities[name].model_dump(mode="json")
                for name in selected
                if name in state.entities
            }
        elif view == "camera":
            data = state.camera.model_dump(mode="json") if state.camera else None
        elif view == "constraints":
            selected = constraint_ids or sorted(state.constraints)
            data = {
                name: state.constraints[name].model_dump(mode="json")
                for name in selected
                if name in state.constraints
            }
        elif view == "violations":
            data = state.validation.model_dump(mode="json") if state.validation else None
        elif view == "timeline":
            data = state.timeline.model_dump(mode="json") | {"frame_end": state.timeline.frame_end}
        elif view == "diff":
            if compare_to_revision is None:
                return _rejected(state.revision, "diff 视图必须提供 compare_to_revision")
            compared = self.store.get(compare_to_revision)
            data = {
                "from_revision": compared.revision,
                "to_revision": state.revision,
                "from_hash": canonical_hash(compared),
                "to_hash": canonical_hash(state),
            }
        elif view == "full_ir":
            data = state.model_dump(mode="json")
        else:
            return _rejected(state.revision, f"未知 inspect view：{view}")
        return _envelope(state.revision, state.revision, data=data)

    def apply_entity_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [EntitySpec.model_validate(item) for item in upserts]
            if set(item.entity_id for item in parsed) & set(remove_ids):
                return _rejected(self.store.current_revision, "同一 Entity 不能同时 upsert 和删除")

            def mutate(state: CandidateState):
                referenced = _referenced_entity_ids(state)
                blocked = sorted(set(remove_ids) & referenced)
                if blocked:
                    raise ValueError(f"Entity 仍被引用，不能删除：{', '.join(blocked)}")
                changes: list[dict[str, Any]] = []
                for entity_id in remove_ids:
                    if state.entities.pop(entity_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"entities.{entity_id}"})
                for entity in parsed:
                    operation = "replace" if entity.entity_id in state.entities else "add"
                    state.entities[entity.entity_id] = entity
                    changes.append({"operation": operation, "path": f"entities.{entity.entity_id}"})
                _validate_parent_references(state)
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, str(error))

    def apply_constraint_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [ConstraintSpec.model_validate(item) for item in upserts]
            unsupported = sorted({item.type for item in parsed} - SUPPORTED_CONSTRAINTS)
            if unsupported:
                return _envelope(
                    self.store.current_revision,
                    self.store.current_revision,
                    status="unsupported",
                    capability_gaps=[f"constraint:{name}" for name in unsupported],
                    next_actions=["改用已列出的等价约束，或返回 UnsupportedResult"],
                )
            if set(item.constraint_id for item in parsed) & set(remove_ids):
                return _rejected(self.store.current_revision, "同一 Constraint 不能同时 upsert 和删除")

            def mutate(state: CandidateState):
                changes: list[dict[str, Any]] = []
                for constraint_id in remove_ids:
                    existing = state.constraints.get(constraint_id)
                    if existing and existing.strength == "hard" and existing.source_status == "explicit":
                        raise ValueError("不得删除 explicit hard constraint")
                    if state.constraints.pop(constraint_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"constraints.{constraint_id}"})
                for constraint in parsed:
                    existing = state.constraints.get(constraint.constraint_id)
                    if existing and existing.strength == "hard" and existing.source_status == "explicit":
                        if constraint.strength != "hard" or constraint.source_ref != existing.source_ref:
                            raise ValueError("不得降级或改写 explicit hard constraint 来源")
                    _validate_constraint_time(constraint, state.timeline.duration_seconds)
                    state.constraints[constraint.constraint_id] = constraint
                    changes.append({
                        "operation": "replace" if existing else "add",
                        "path": f"constraints.{constraint.constraint_id}",
                    })
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, str(error))

    def apply_motion_patch(
        self,
        upserts: list[dict[str, Any]],
        remove_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_ids = remove_ids or []
        try:
            parsed = [TrackSpec.model_validate(item) for item in upserts]
            if set(item.track_id for item in parsed) & set(remove_ids):
                return _rejected(self.store.current_revision, "同一 Track 不能同时 upsert 和删除")

            def mutate(state: CandidateState):
                changes: list[dict[str, Any]] = []
                for track_id in remove_ids:
                    if state.motion_tracks.pop(track_id, None) is not None:
                        changes.append({"operation": "remove", "path": f"motion_tracks.{track_id}"})
                for track in parsed:
                    if not track.target_entity_id or track.target_entity_id not in state.entities:
                        raise ValueError(f"Track 目标 Entity 不存在：{track.target_entity_id}")
                    _validate_track_time(track, state.timeline.duration_seconds)
                    existing = track.track_id in state.motion_tracks
                    state.motion_tracks[track.track_id] = track
                    changes.append({
                        "operation": "replace" if existing else "add",
                        "path": f"motion_tracks.{track.track_id}",
                    })
                return changes, []

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, str(error))

    def apply_camera_patch(
        self,
        camera_id: str,
        projection: str,
        active: bool,
        static: dict[str, Any],
        tracks: list[dict[str, Any]],
        remove_track_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        remove_track_ids = remove_track_ids or []
        try:
            if projection != "perspective":
                return _envelope(
                    self.store.current_revision,
                    self.store.current_revision,
                    status="unsupported",
                    capability_gaps=[f"camera_projection:{projection}"],
                )
            parsed_static = CameraStatic.model_validate(static)
            parsed_tracks = [TrackSpec.model_validate(item) for item in tracks]

            def mutate(state: CandidateState):
                current_tracks = deepcopy(state.camera.tracks) if state.camera else {}
                for track_id in remove_track_ids:
                    current_tracks.pop(track_id, None)
                for track in parsed_tracks:
                    if track.target_entity_id is not None:
                        raise ValueError("Camera Track 不应填写 target_entity_id")
                    _validate_track_time(track, state.timeline.duration_seconds)
                    current_tracks[track.track_id] = track
                previous = state.camera
                state.camera = CameraCandidate(
                    camera_id=camera_id,
                    projection="perspective",
                    active=active,
                    static=parsed_static,
                    tracks=current_tracks,
                    solved_transform=(
                        previous.solved_transform if previous else TransformValue()
                    ),
                )
                return ([{"operation": "replace" if previous else "add", "path": "camera"}], [])

            return _mutation_envelope(self.store.apply(mutate))
        except (ValidationError, ValueError) as error:
            return _rejected(self.store.current_revision, str(error))

    def solve_candidate(
        self,
        scope: str = "all",
        constraint_ids: list[str] | None = None,
        allowed_variables: list[str] | None = None,
        locked_variables: list[str] | None = None,
        profile: str = "research_default",
        strategy: str = "auto",
    ) -> dict[str, Any]:
        del allowed_variables, profile
        if scope not in {"layout", "camera", "motion", "all"}:
            return _rejected(self.store.current_revision, f"未知 solve scope：{scope}")
        if strategy not in {"auto", "heuristic", "numeric", "hybrid"}:
            return _rejected(self.store.current_revision, f"未知 solve strategy：{strategy}")
        selected_ids = set(constraint_ids or [])
        locked = set(locked_variables or [])

        def mutate(state: CandidateState):
            changes: list[dict[str, Any]] = []
            if scope in {"layout", "motion", "all"}:
                for index, entity in enumerate(state.entities.values()):
                    if f"{entity.entity_id}.translation" in locked:
                        continue
                    current = entity.solved_transform
                    if current.translation_m is None:
                        radius = _geometry_half_height(entity.proxy)
                        entity.solved_transform = TransformValue(
                            translation_m=(float(index * 2), float(index * 5), radius),
                            rotation_quaternion_wxyz=current.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
                            scale=current.scale or (1.0, 1.0, 1.0),
                            space="world",
                        )
                        changes.append({"operation": "solve", "path": f"entities.{entity.entity_id}.solved_transform"})
                    else:
                        entity.solved_transform = _complete_transform(entity.solved_transform)

                for constraint in state.constraints.values():
                    if selected_ids and constraint.constraint_id not in selected_ids:
                        continue
                    if constraint.type in {"relative_position", "depth_order", "distance_range"}:
                        if _apply_layout_constraint(state, constraint, self.profile):
                            changes.append({"operation": "solve", "path": f"constraints.{constraint.constraint_id}"})

            if scope in {"camera", "all"}:
                if state.camera is None:
                    focus = next(iter(state.entities), None)
                    state.camera = CameraCandidate(static=CameraStatic(focus_target_id=focus))
                    changes.append({"operation": "solve", "path": "camera"})
                camera = state.camera
                if camera.solved_transform.translation_m is None:
                    camera.solved_transform = TransformValue(
                        translation_m=(0.0, -self.profile.default_camera_distance_m, 2.0),
                        rotation_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                        scale=(1.0, 1.0, 1.0),
                        space="world",
                    )
                    changes.append({"operation": "solve", "path": "camera.solved_transform"})
                else:
                    camera.solved_transform = _complete_transform(camera.solved_transform)
                if camera.static.focal_length_mm is None:
                    camera.static.focal_length_mm = self.profile.default_focal_length_mm
                    changes.append({"operation": "solve", "path": "camera.static.focal_length_mm"})

            return changes, []

        try:
            mutation = self.store.apply(mutate)
        except ValueError as error:
            return _rejected(self.store.current_revision, str(error))
        report = self._validate(self.store.get(), FULL_VALIDATION_CHECKS)
        self.store.save_validation(report)
        data = {
            "solve_status": "solved" if report.hard_pass else "partial",
            "solver": "deterministic_heuristic_v0.1",
            "strategy_requested": strategy,
            "random_seed": self.profile.random_seed,
            "hard_pass": report.hard_pass,
            "soft_score": report.soft_score,
        }
        return _mutation_envelope(
            mutation,
            data=data,
            violations=[item.model_dump(mode="json") for item in report.violations],
            capability_gaps=report.capability_gaps,
        )

    def validate_candidate(
        self,
        revision: int | None = None,
        checks: list[str] | None = None,
        sampling_profile: str = "research_default",
    ) -> dict[str, Any]:
        del sampling_profile
        state = self.store.get(revision)
        selected = checks or FULL_VALIDATION_CHECKS
        unknown = sorted(set(selected) - set(FULL_VALIDATION_CHECKS))
        if unknown:
            return _rejected(state.revision, f"未知 Validator：{', '.join(unknown)}")
        report = self._validate(state, selected)
        if revision is None or revision == self.store.current_revision:
            self.store.save_validation(report)
        return _envelope(
            state.revision,
            state.revision,
            data=report.model_dump(mode="json"),
            violations=[item.model_dump(mode="json") for item in report.violations],
            capability_gaps=report.capability_gaps,
        )

    def restore_candidate(self, source_revision: int, reason: str) -> dict[str, Any]:
        if not reason.strip():
            return _rejected(self.store.current_revision, "restore 必须记录 reason")
        try:
            result = self.store.restore(source_revision)
            return _mutation_envelope(result, data={"reason": reason})
        except ValueError as error:
            return _rejected(self.store.current_revision, str(error))

    def _validate(self, state: CandidateState, checks: list[str]) -> ValidationReport:
        violations: list[Violation] = []
        gaps: list[str] = []
        if "references" in checks or "hierarchy" in checks:
            violations.extend(_reference_violations(state))
        if "timeline" in checks:
            violations.extend(_timeline_violations(state))
        if "transforms" in checks or "rebuildability" in checks:
            violations.extend(_transform_violations(state))
        if "camera" in checks or "rebuildability" in checks:
            violations.extend(_camera_violations(state))
        if "hard_semantics" in checks:
            violations.extend(_hard_semantic_violations(state))

        soft_total = 0.0
        soft_passed = 0.0
        for constraint in state.constraints.values():
            if constraint.type not in SUPPORTED_CONSTRAINTS:
                gaps.append(f"constraint:{constraint.type}")
                continue
            result = _constraint_violation(state, constraint, self.profile)
            if constraint.strength == "soft":
                soft_total += constraint.weight
                if result is None:
                    soft_passed += constraint.weight
            if result is not None:
                violations.append(result)
        soft_score = soft_passed / soft_total if soft_total else 1.0
        hard_pass = not any(item.severity == "hard" for item in violations) and not gaps
        return ValidationReport(
            revision=state.revision,
            hard_pass=hard_pass,
            soft_score=soft_score,
            checks=checks,
            violations=violations,
            capability_gaps=sorted(set(gaps)),
        )


def _initial_candidate(
    brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
    scene_id: str | None,
) -> CandidateState:
    duration = brief.timeline.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = profile.default_duration_seconds
    frame_count = max(1, round(duration * profile.fps_numerator / profile.fps_denominator))
    suffix = brief.source_brief_sha256.removeprefix("sha256:")[:12]
    return CandidateState(
        scene_id=scene_id or f"scene_{suffix}",
        timeline=TimelineSpec(
            fps_numerator=profile.fps_numerator,
            fps_denominator=profile.fps_denominator,
            frame_count=frame_count,
            duration_seconds=frame_count * profile.fps_denominator / profile.fps_numerator,
        ),
        required_source_refs=[item.path for item in brief.explicit_requirements],
        runner_mapped_source_refs=[
            item.path
            for item in brief.explicit_requirements
            if item.path == "content.timeline.duration_seconds"
        ],
    )


def _referenced_entity_ids(state: CandidateState) -> set[str]:
    result = {entity.parent_id for entity in state.entities.values() if entity.parent_id}
    result.update(
        track.target_entity_id for track in state.motion_tracks.values() if track.target_entity_id
    )
    for constraint in state.constraints.values():
        result.update(constraint.subjects)
        for key, value in _parameters(constraint).items():
            if key.endswith("_id") and isinstance(value, str) and value != "camera_main":
                result.add(value)
    return result


def _validate_parent_references(state: CandidateState) -> None:
    for entity in state.entities.values():
        if entity.parent_id and entity.parent_id not in state.entities:
            raise ValueError(f"Entity parent 不存在：{entity.parent_id}")
    for entity_id in state.entities:
        seen: set[str] = set()
        current: str | None = entity_id
        while current:
            if current in seen:
                raise ValueError("Entity 父子层级存在循环")
            seen.add(current)
            parent = state.entities.get(current)
            current = parent.parent_id if parent else None


def _validate_constraint_time(constraint: ConstraintSpec, duration: float) -> None:
    if constraint.time_range_seconds[1] > duration + 1e-9:
        raise ValueError(f"Constraint 超出 timeline：{constraint.constraint_id}")


def _validate_track_time(track: TrackSpec, duration: float) -> None:
    if track.time_range_seconds[1] > duration + 1e-9:
        raise ValueError(f"Track 超出 timeline：{track.track_id}")
    for keyframe in track.keyframes:
        if not 0 <= keyframe.time_seconds < duration:
            raise ValueError(f"Track keyframe 超出半开 timeline：{track.track_id}")


def _complete_transform(value: TransformValue) -> TransformValue:
    return TransformValue(
        translation_m=value.translation_m or (0.0, 0.0, 0.0),
        rotation_quaternion_wxyz=value.rotation_quaternion_wxyz or (1.0, 0.0, 0.0, 0.0),
        scale=value.scale or (1.0, 1.0, 1.0),
        space="world",
    )


def _geometry_half_height(geometry) -> float:
    if geometry.type == "box":
        return geometry.size_xyz_m[2] / 2.0
    if geometry.type == "sphere":
        return geometry.radius_m
    if geometry.type == "capsule":
        return geometry.radius_m + geometry.segment_length_m / 2.0
    if geometry.type in {"cylinder", "cone"}:
        return geometry.depth_m / 2.0
    return 0.01


def _apply_layout_constraint(
    state: CandidateState,
    constraint: ConstraintSpec,
    profile: PlanningProfile,
) -> bool:
    params = _parameters(constraint)
    if constraint.type == "depth_order":
        near_id = params.get("near_entity_id")
        far_id = params.get("far_entity_id")
        if near_id not in state.entities or far_id not in state.entities:
            return False
        near = _complete_transform(state.entities[near_id].solved_transform)
        far = _complete_transform(state.entities[far_id].solved_transform)
        gap = float(params.get("minimum_depth_gap_meters") or profile.default_depth_gap_m)
        # 世界 Y 间隔需给倾斜视线留出确定性余量。
        state.entities[far_id].solved_transform = far.model_copy(
            update={"translation_m": (far.translation_m[0], near.translation_m[1] + gap * 1.1, far.translation_m[2])}
        )
        return True
    if constraint.type == "relative_position":
        subject_id = params.get("subject_id")
        reference_id = params.get("reference_id")
        if subject_id not in state.entities or reference_id not in state.entities:
            return False
        subject = _complete_transform(state.entities[subject_id].solved_transform)
        reference = _complete_transform(state.entities[reference_id].solved_transform)
        gap = float(params.get("minimum_gap") or 2.0)
        position = list(subject.translation_m)
        relation = params.get("relation")
        axis_sign = {
            "left": (0, -1), "right": (0, 1),
            "front": (1, -1), "behind": (1, 1),
            "below": (2, -1), "above": (2, 1),
        }.get(relation)
        if axis_sign is None:
            return False
        axis, sign = axis_sign
        position[axis] = reference.translation_m[axis] + sign * gap
        state.entities[subject_id].solved_transform = subject.model_copy(
            update={"translation_m": tuple(position)}
        )
        return True
    if constraint.type == "distance_range":
        ids = params.get("entity_ids") or constraint.subjects
        if not isinstance(ids, (list, tuple)) or len(ids) != 2 or any(item not in state.entities for item in ids):
            return False
        first = _complete_transform(state.entities[ids[0]].solved_transform)
        second = _complete_transform(state.entities[ids[1]].solved_transform)
        minimum = float(params.get("minimum_meters", 0.0))
        maximum = float(params.get("maximum_meters", max(minimum, profile.default_depth_gap_m)))
        distance = (minimum + maximum) / 2.0
        state.entities[ids[1]].solved_transform = second.model_copy(
            update={"translation_m": (first.translation_m[0], first.translation_m[1] + distance, second.translation_m[2])}
        )
        return True
    return False


def _reference_violations(state: CandidateState) -> list[Violation]:
    violations: list[Violation] = []
    try:
        _validate_parent_references(state)
    except ValueError as error:
        violations.append(_violation("HIERARCHY_INVALID", str(error)))
    for track in state.motion_tracks.values():
        if track.target_entity_id not in state.entities:
            violations.append(_violation("TRACK_TARGET_MISSING", f"Track 目标不存在：{track.track_id}"))
    for constraint in state.constraints.values():
        missing = sorted(item for item in constraint.subjects if item not in state.entities)
        if missing:
            violations.append(_constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", {"missing": missing}))
    return violations


def _timeline_violations(state: CandidateState) -> list[Violation]:
    violations: list[Violation] = []
    for track in list(state.motion_tracks.values()) + (
        list(state.camera.tracks.values()) if state.camera else []
    ):
        try:
            _validate_track_time(track, state.timeline.duration_seconds)
        except ValueError as error:
            violations.append(_violation("TRACK_TIME_INVALID", str(error)))
    return violations


def _transform_violations(state: CandidateState) -> list[Violation]:
    violations: list[Violation] = []
    for entity in state.entities.values():
        transform = entity.solved_transform
        if transform.translation_m is None or transform.rotation_quaternion_wxyz is None or transform.scale is None:
            violations.append(_violation(
                "UNRESOLVED_ENTITY_TRANSFORM",
                f"Entity 仍含未求解 Transform：{entity.entity_id}",
                entity_ids=[entity.entity_id],
            ))
        if entity.parent_id:
            parent_scale = state.entities[entity.parent_id].solved_transform.scale
            if parent_scale and max(parent_scale) - min(parent_scale) > 1e-8:
                violations.append(_violation(
                    "PARENT_NONUNIFORM_SCALE_UNSUPPORTED",
                    f"父级非均匀缩放无法无损编译：{entity.parent_id}",
                    entity_ids=[entity.parent_id, entity.entity_id],
                ))
    return violations


def _camera_violations(state: CandidateState) -> list[Violation]:
    if state.camera is None:
        return [_violation("CAMERA_MISSING", "缺少活动透视摄影机")]
    transform = state.camera.solved_transform
    violations: list[Violation] = []
    if transform.translation_m is None or transform.rotation_quaternion_wxyz is None:
        violations.append(_violation("UNRESOLVED_CAMERA_TRANSFORM", "摄影机 Transform 尚未求解"))
    if state.camera.static.focal_length_mm is None:
        violations.append(_violation("UNRESOLVED_FOCAL_LENGTH", "摄影机焦距尚未求解"))
    if state.camera.static.focus_target_id and state.camera.static.focus_target_id not in state.entities:
        violations.append(_violation("CAMERA_TARGET_MISSING", "摄影机观察目标不存在"))
    return violations


def _hard_semantic_violations(state: CandidateState) -> list[Violation]:
    mapped: set[str] = set(state.runner_mapped_source_refs)
    for entity in state.entities.values():
        mapped.update(entity.source_refs)
    for track in state.motion_tracks.values():
        if track.source_ref:
            mapped.add(track.source_ref)
    for constraint in state.constraints.values():
        mapped.add(constraint.source_ref)
    if state.camera:
        mapped.update(state.camera.static.source_refs)
        mapped.update(track.source_ref for track in state.camera.tracks.values() if track.source_ref)
    return [
        _violation(
            "UNMAPPED_EXPLICIT_REQUIREMENT",
            f"明确客观要求未映射：{source_ref}",
            expected={"source_ref": source_ref},
        )
        for source_ref in state.required_source_refs
        if source_ref not in mapped
    ]


def _constraint_violation(
    state: CandidateState,
    constraint: ConstraintSpec,
    profile: PlanningProfile,
) -> Violation | None:
    params = _parameters(constraint)
    sample_times = _constraint_sample_times(constraint, state.timeline)
    for time_seconds in sample_times:
        transforms = {
            entity_id: _entity_transform_at(state, entity_id, time_seconds)
            for entity_id in state.entities
        }
        camera = _camera_state_at(state, time_seconds, profile)
        if camera is None:
            return _constraint_error(constraint, "CAMERA_MISSING", None)
        camera_transform, focal_length = camera

        if constraint.type == "depth_order":
            near_id = params.get("near_entity_id")
            far_id = params.get("far_entity_id")
            if near_id not in transforms or far_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            near_depth = _projection(state, near_id, transforms, camera_transform, focal_length, profile)[2]
            far_depth = _projection(state, far_id, transforms, camera_transform, focal_length, profile)[2]
            gap = float(params.get("minimum_depth_gap_meters") or 0.0)
            if not near_depth + gap <= far_depth:
                return _constraint_error(constraint, "DEPTH_ORDER_VIOLATED", {"near_depth": near_depth, "far_depth": far_depth})

        elif constraint.type == "relative_position":
            subject_id = params.get("subject_id")
            reference_id = params.get("reference_id")
            if subject_id not in transforms or reference_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            subject = transforms[subject_id].translation_m
            reference = transforms[reference_id].translation_m
            relation = params.get("relation")
            gap = float(params.get("minimum_gap") or 0.0)
            valid = {
                "left": subject[0] <= reference[0] - gap,
                "right": subject[0] >= reference[0] + gap,
                "front": subject[1] <= reference[1] - gap,
                "behind": subject[1] >= reference[1] + gap,
                "below": subject[2] <= reference[2] - gap,
                "above": subject[2] >= reference[2] + gap,
            }.get(relation, False)
            if not valid:
                return _constraint_error(constraint, "RELATIVE_POSITION_VIOLATED", {"subject": subject, "reference": reference})

        elif constraint.type == "distance_range":
            ids = params.get("entity_ids") or constraint.subjects
            if not isinstance(ids, (list, tuple)) or len(ids) != 2 or any(item not in transforms for item in ids):
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            actual = length(subtract(transforms[ids[0]].translation_m, transforms[ids[1]].translation_m))
            if not float(params.get("minimum_meters", 0.0)) <= actual <= float(params.get("maximum_meters", math.inf)):
                return _constraint_error(constraint, "DISTANCE_RANGE_VIOLATED", {"distance_m": actual})

        elif constraint.type in {"screen_region", "projected_size", "keep_in_frame"}:
            entity_id = params.get("entity_id")
            if entity_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            x, y, depth = _projection(state, entity_id, transforms, camera_transform, focal_length, profile)
            radius = projected_radius(state.entities[entity_id].proxy, transforms[entity_id].scale, depth, focal_length, state.camera.static.sensor_width_mm)
            if constraint.type == "screen_region":
                region = params.get("region")
                if not _valid_region(region) or not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                    return _constraint_error(constraint, "SCREEN_REGION_VIOLATED", {"screen_center": [x, y]})
            elif constraint.type == "projected_size":
                size = radius * 2.0
                if not float(params.get("minimum", 0.0)) <= size <= float(params.get("maximum", math.inf)):
                    return _constraint_error(constraint, "PROJECTED_SIZE_VIOLATED", {"projected_size": size})
            else:
                inside = _inside_fraction(x, y, radius)
                if inside < float(params.get("minimum_inside_fraction", 1.0)):
                    return _constraint_error(constraint, "ENTITY_OUT_OF_FRAME", {"inside_fraction": inside})

        elif constraint.type == "projected_scale_ratio":
            first_id = params.get("numerator_entity_id") or params.get("subject_id")
            second_id = params.get("denominator_entity_id") or params.get("reference_id")
            if first_id not in transforms or second_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            first_projection = _projection(state, first_id, transforms, camera_transform, focal_length, profile)
            second_projection = _projection(state, second_id, transforms, camera_transform, focal_length, profile)
            first_radius = projected_radius(state.entities[first_id].proxy, transforms[first_id].scale, first_projection[2], focal_length, state.camera.static.sensor_width_mm)
            second_radius = projected_radius(state.entities[second_id].proxy, transforms[second_id].scale, second_projection[2], focal_length, state.camera.static.sensor_width_mm)
            ratio = first_radius / second_radius if second_radius > 0 else math.inf
            if not float(params.get("minimum_ratio", 0.0)) <= ratio <= float(params.get("maximum_ratio", math.inf)):
                return _constraint_error(constraint, "PROJECTED_SCALE_RATIO_VIOLATED", {"ratio": ratio})

        elif constraint.type == "camera_distance":
            target_id = params.get("target_id")
            if target_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            actual = length(subtract(camera_transform.translation_m, transforms[target_id].translation_m))
            if not float(params.get("minimum_meters", 0.0)) <= actual <= float(params.get("maximum_meters", math.inf)):
                return _constraint_error(constraint, "CAMERA_DISTANCE_VIOLATED", {"distance_m": actual})

        elif constraint.type == "focal_length_range":
            if not float(params.get("minimum_mm", 0.0)) <= focal_length <= float(params.get("maximum_mm", math.inf)):
                return _constraint_error(constraint, "FOCAL_LENGTH_RANGE_VIOLATED", {"focal_length_mm": focal_length})

        elif constraint.type == "look_at":
            observer_id = params.get("observer_id")
            target_id = params.get("target_id")
            if target_id not in transforms:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            target_position = transforms[target_id].translation_m
            if observer_id == state.camera.camera_id:
                observer_position = camera_transform.translation_m
                forward = rotate_vector(camera_transform.rotation_quaternion_wxyz, (0.0, 0.0, -1.0))
            elif observer_id in transforms:
                observer_position = transforms[observer_id].translation_m
                forward = rotate_vector(transforms[observer_id].rotation_quaternion_wxyz, (0.0, 1.0, 0.0))
            else:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            desired = normalize(subtract(target_position, observer_position))
            cosine = max(-1.0, min(1.0, dot(normalize(forward), desired)))
            angle = math.degrees(math.acos(cosine))
            if angle > float(params.get("maximum_angle_error_degrees", 1.0)):
                return _constraint_error(constraint, "LOOK_AT_VIOLATED", {"angle_error_degrees": angle})

        elif constraint.type == "position_at_time":
            target_id = params.get("target_id")
            expected = params.get("position_m")
            if target_id not in transforms or not isinstance(expected, (list, tuple)) or len(expected) != 3:
                return _constraint_error(constraint, "CONSTRAINT_PARAMETER_INVALID", None)
            tolerance = float(params.get("tolerance_m", 0.01))
            actual = transforms[target_id].translation_m
            if length(subtract(actual, tuple(expected))) > tolerance:
                return _constraint_error(constraint, "POSITION_AT_TIME_VIOLATED", {"position_m": actual})

    if constraint.type in {"camera_motion_direction", "motion_direction", "hold"}:
        start, end = constraint.time_range_seconds
        end_sample = min(end - 1e-6, state.timeline.duration_seconds - 1e-6)
        if constraint.type == "camera_motion_direction":
            start_state = _camera_state_at(state, start, profile)
            end_state = _camera_state_at(state, end_sample, profile)
            if start_state is None or end_state is None:
                return _constraint_error(constraint, "CAMERA_MISSING", None)
            delta = subtract(end_state[0].translation_m, start_state[0].translation_m)
        else:
            target_id = params.get("target_id")
            if target_id not in state.entities:
                return _constraint_error(constraint, "CONSTRAINT_REFERENCE_MISSING", None)
            start_transform = _entity_transform_at(state, target_id, start)
            end_transform = _entity_transform_at(state, target_id, end_sample)
            delta = subtract(end_transform.translation_m, start_transform.translation_m)
        if constraint.type == "hold":
            tolerance = float(params.get("tolerance_m", 1e-4))
            if length(delta) > tolerance:
                return _constraint_error(constraint, "HOLD_VIOLATED", {"translation_delta_m": delta})
        else:
            direction = params.get("direction")
            minimum = float(params.get("minimum_displacement_m", 0.01))
            if not _direction_matches(delta, direction, minimum):
                return _constraint_error(constraint, "MOTION_DIRECTION_VIOLATED", {"delta_m": delta})
    return None


def _constraint_sample_times(constraint: ConstraintSpec, timeline: TimelineSpec) -> list[float]:
    start, end = constraint.time_range_seconds
    return [start, min((start + end) / 2.0, timeline.duration_seconds - 1e-6), min(end - 1e-6, timeline.duration_seconds - 1e-6)]


def _entity_transform_at(state: CandidateState, entity_id: str, time_seconds: float) -> TransformValue:
    entity = state.entities[entity_id]
    tracks = [track for track in state.motion_tracks.values() if track.target_entity_id == entity_id]
    transform = next((track for track in tracks if track.type == "transform"), None)
    path = next((track for track in tracks if track.type == "path_follow"), None)
    result = sample_transform_track(transform, time_seconds, entity.solved_transform)
    if path is not None:
        result = sample_path_track(path, time_seconds, result)
    return result


def _camera_state_at(
    state: CandidateState,
    time_seconds: float,
    profile: PlanningProfile,
) -> tuple[TransformValue, float] | None:
    if state.camera is None:
        return None
    tracks = list(state.camera.tracks.values())
    transform_track = next((track for track in tracks if track.type == "transform"), None)
    path_track = next((track for track in tracks if track.type == "path_follow"), None)
    transform = sample_transform_track(transform_track, time_seconds, state.camera.solved_transform)
    if path_track is not None:
        transform = sample_path_track(path_track, time_seconds, transform)
    focus_id = state.camera.static.focus_target_id
    look_track = next((track for track in tracks if track.type == "look_at"), None)
    if look_track and look_track.target_id:
        focus_id = look_track.target_id
    if focus_id in state.entities:
        target = _entity_transform_at(state, focus_id, time_seconds).translation_m
        transform = transform.model_copy(
            update={"rotation_quaternion_wxyz": look_at_camera_quaternion(transform.translation_m, target)}
        )
    focal_track = next((track for track in tracks if track.type == "focal_length"), None)
    focal = sample_scalar_track(
        focal_track,
        time_seconds,
        state.camera.static.focal_length_mm or profile.default_focal_length_mm,
    )
    return transform, focal


def _projection(state, entity_id, transforms, camera_transform, focal_length, profile):
    del profile
    return project_point(
        transforms[entity_id].translation_m,
        camera_transform.translation_m,
        camera_transform.rotation_quaternion_wxyz,
        focal_length,
        state.camera.static.sensor_width_mm,
        1280 / 720,
    )


def _valid_region(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 4 and value[0] <= value[2] and value[1] <= value[3]


def _inside_fraction(x: float, y: float, radius: float) -> float:
    if not all(math.isfinite(item) for item in (x, y, radius)) or radius <= 0:
        return 0.0
    left, right = x - radius, x + radius
    top, bottom = y - radius, y + radius
    inside_width = max(0.0, min(1.0, right) - max(0.0, left))
    inside_height = max(0.0, min(1.0, bottom) - max(0.0, top))
    return min(1.0, inside_width * inside_height / ((2 * radius) ** 2))


def _direction_matches(delta, direction, minimum):
    mapping = {
        "left": delta[0] <= -minimum,
        "right": delta[0] >= minimum,
        "forward": delta[1] >= minimum,
        "backward": delta[1] <= -minimum,
        "up": delta[2] >= minimum,
        "down": delta[2] <= -minimum,
        "push_in": delta[1] >= minimum,
        "pull_out": delta[1] <= -minimum,
    }
    return mapping.get(direction, False)


def _constraint_error(constraint: ConstraintSpec, code: str, actual: Any) -> Violation:
    return Violation(
        id=f"violation_{constraint.constraint_id}_{code.lower()}",
        code=code,
        severity=constraint.strength,
        constraint_id=constraint.constraint_id,
        entity_ids=constraint.subjects,
        time_range_seconds=constraint.time_range_seconds,
        expected=_parameters(constraint),
        actual=actual,
        adjustable_variables=["entity transforms", "camera transform", "camera focal length"],
        message=f"约束 {constraint.constraint_id} 未满足：{code}",
    )


def _violation(code: str, message: str, **kwargs) -> Violation:
    return Violation(
        id=f"violation_{code.lower()}_{canonical_hash([message, kwargs])[-8:]}",
        code=code,
        severity="hard",
        message=message,
        **kwargs,
    )


def _parameters(constraint: ConstraintSpec) -> dict[str, Any]:
    return constraint.parameters.model_dump(mode="json", exclude_none=True)


def _envelope(
    before: int,
    after: int,
    *,
    status: str = "ok",
    changes: list[dict[str, Any]] | None = None,
    data: Any = None,
    violations: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    capability_gaps: list[str] | None = None,
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tool_version": TOOLKIT_VERSION,
        "status": status,
        "revision_before": before,
        "revision_after": after,
        "changes": changes or [],
        "data": data if data is not None else {},
        "violations": violations or [],
        "warnings": warnings or [],
        "capability_gaps": capability_gaps or [],
        "next_actions": next_actions or [],
    }


def _mutation_envelope(
    result: MutationResult,
    *,
    data: Any = None,
    violations: list[dict[str, Any]] | None = None,
    capability_gaps: list[str] | None = None,
) -> dict[str, Any]:
    return _envelope(
        result.revision_before,
        result.revision_after,
        status=result.status,
        changes=result.changes,
        data=data,
        violations=violations,
        warnings=result.warnings,
        capability_gaps=capability_gaps,
    )


def _rejected(revision: int, message: str) -> dict[str, Any]:
    return _envelope(
        revision,
        revision,
        status="rejected",
        warnings=[message],
        next_actions=["修正参数后重试"],
    )
