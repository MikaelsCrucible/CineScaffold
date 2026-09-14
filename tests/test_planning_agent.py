from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from cinescaffold.planning.agent import (
    ConstraintPatchInput,
    DesignSearchStalled,
    EntityPatchInput,
    EscalatedEntityPatchInput,
    NON_TOOL_TEXT_MARKER,
    PathPatchInput,
    PlanningDeps,
    TrackPatchInput,
    _compact_tool_call_history,
    _prepare_candidate_patch_tool,
    _prepare_candidate_tool,
    _prepare_escalated_candidate_tool,
    _prepare_design_apply_tool,
    _prepare_design_options_tool,
    _prepare_manual_mutation_tool,
    _prepare_repair_apply_tool,
    _prepare_repair_suggestion_tool,
    _prepare_scene_skeleton_tool,
)
from cinescaffold.planning.domain import (
    ConstraintSpec,
    TrackSpec,
    ValidationReport,
    Violation,
)
from cinescaffold.planning.models import create_planning_model
from cinescaffold.planning.objective import project_objective_brief
from cinescaffold.planning.trace import TraceRecorder
from tests.helpers import valid_planning_brief
from tests.test_planning_design import _desert_skeleton
from tests.test_planning_toolkit import (
    _man_entity,
    _projected_motion_toolkit,
    _solved_toolkit,
    _toolkit,
)


class PlanningProtocolTest(unittest.TestCase):
    def test_agent_patch_schemas_are_compact_but_domain_validation_stays_strict(
        self,
    ) -> None:
        compact_chars = len(json.dumps(TrackPatchInput.model_json_schema()))
        domain_chars = len(json.dumps(TrackSpec.model_json_schema()))
        constraint_chars = len(json.dumps(ConstraintPatchInput.model_json_schema()))
        domain_constraint_chars = len(json.dumps(ConstraintSpec.model_json_schema()))

        # 坐标契约增加少量说明后仍需显著小于完整领域联合。
        self.assertLess(compact_chars, domain_chars * 0.7)
        self.assertLess(constraint_chars, domain_constraint_chars * 0.4)
        constraint_types = ConstraintPatchInput.model_json_schema()["properties"][
            "type"
        ]["enum"]
        self.assertIn("projected_size", constraint_types)
        self.assertNotIn("event_order", constraint_types)
        self.assertNotIn(
            "solved_transform", EntityPatchInput.model_json_schema()["properties"]
        )
        self.assertIn(
            "solved_transform",
            EscalatedEntityPatchInput.model_json_schema()["properties"],
        )

    def test_path_tool_schema_exposes_coordinate_and_direction_conventions(
        self,
    ) -> None:
        properties = PathPatchInput.model_json_schema()["properties"]

        self.assertIn("+Z", properties["plane_normal"]["description"])
        self.assertIn("+X", properties["axis_direction"]["description"])
        self.assertIn("+plane_normal", properties["direction"]["description"])
        self.assertIn("右手 +Z-up", properties["space"]["description"])

    def test_tool_history_drops_text_but_keeps_thinking_and_calls(self) -> None:
        response = ModelResponse(
            parts=[
                TextPart("冗长分析"),
                ThinkingPart(
                    "内部推理",
                    id="reasoning_content",
                    provider_name="deepseek",
                ),
                ToolCallPart("inspect_candidate", {"view": "summary"}, "call_1"),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            compacted = _compact_tool_call_history(_context(deps), [response])

        self.assertEqual(len(compacted[0].parts), 2)
        self.assertIsInstance(compacted[0].parts[0], ThinkingPart)
        self.assertIsInstance(compacted[0].parts[1], ToolCallPart)

    def test_deepseek_thinking_history_is_not_windowed(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart("原始 Brief")])]
        for index in range(10):
            call_id = f"call_{index}"
            messages.append(
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            f"reasoning-{index}",
                            id="reasoning_content",
                            provider_name="deepseek",
                        ),
                        ToolCallPart("inspect_candidate", {}, call_id),
                    ]
                )
            )
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "inspect_candidate", {"revision": index}, call_id
                        )
                    ]
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                preserve_complete_thinking_history=True,
            )
            compacted = _compact_tool_call_history(_context(deps), messages)

        self.assertEqual(len(compacted), len(messages))
        thinking_parts = [
            part
            for message in compacted
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ThinkingPart)
        ]
        self.assertEqual(
            [part.content for part in thinking_parts],
            [f"reasoning-{index}" for index in range(10)],
        )

    def test_deepseek_payload_keeps_reasoning_content_with_tool_call(self) -> None:
        objective = project_objective_brief(valid_planning_brief()).objective_brief
        model = create_planning_model(
            "deepseek",
            "deepseek-v4-pro",
            objective,
            api_key="test-key",
        )
        messages = [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        "reasoning-sentinel",
                        id="reasoning_content",
                        provider_name="deepseek",
                    ),
                    ToolCallPart("inspect_candidate", {}, "call_1"),
                ]
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                preserve_complete_thinking_history=True,
            )
            compacted = _compact_tool_call_history(_context(deps), messages)
            payload = asyncio.run(
                model._map_messages(compacted, ModelRequestParameters())
            )

        self.assertEqual(payload[0]["reasoning_content"], "reasoning-sentinel")
        self.assertEqual(payload[0]["tool_calls"][0]["id"], "call_1")

    def test_repeated_design_failure_starts_a_fresh_recovery_round(self) -> None:
        failure = {
            "status": "no_change",
            "data": {
                "failure_signature": "sha256:stable",
                "failure_codes": ["SPEED_RANGE_VIOLATED"],
                "failure_reasons": ["Design Option 未通过全部硬约束"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            deps._observe_design_search(failure)
            deps._observe_design_search(failure)

            with self.assertRaises(DesignSearchStalled) as raised:
                _compact_tool_call_history(_context(deps), [])

        self.assertEqual(raised.exception.failure["repeat_count"], 2)
        self.assertEqual(
            raised.exception.failure["failure_codes"],
            ["SPEED_RANGE_VIOLATED"],
        )
        self.assertIsNone(deps.pending_design_stall)

    def test_identical_rejected_design_retry_starts_recovery_round(self) -> None:
        failure = {
            "status": "no_change",
            "data": {
                "failure_signature": "sha256:stable",
                "failure_codes": ["SPEED_RANGE_VIOLATED"],
                "failure_reasons": ["Design Option 未通过全部硬约束"],
            },
        }
        duplicate = {
            "status": "rejected",
            "data": {},
            "warnings": ["不得重复完全相同的 Design Options 请求"],
        }
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            deps._observe_design_search(failure)
            deps._observe_design_search(duplicate)

            with self.assertRaises(DesignSearchStalled):
                _compact_tool_call_history(_context(deps), [])

    def test_tool_history_compacts_pure_text_response_once(self) -> None:
        response = ModelResponse(
            parts=[TextPart("无工具正文" * 2_000)],
            provider_response_id="response_long_text",
        )

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            first = _compact_tool_call_history(_context(deps), [response])
            second = _compact_tool_call_history(_context(deps), [response])
            events = [
                json.loads(line)
                for line in (Path(directory) / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(first[0].parts[0].content, NON_TOOL_TEXT_MARKER)
        self.assertEqual(second[0].parts[0].content, NON_TOOL_TEXT_MARKER)
        self.assertLess(len(first[0].parts[0].content), 100)
        compact_events = [
            event
            for event in events
            if event["event_type"] == "model_non_tool_text_compacted"
        ]
        self.assertEqual(len(compact_events), 1)
        self.assertEqual(compact_events[0]["payload"]["omitted_chars"], 10_000)

    def test_tool_history_keeps_objective_and_recent_paired_window(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart("原始 Brief")])]
        for index in range(10):
            call_id = f"call_{index}"
            messages.append(
                ModelResponse(parts=[ToolCallPart("inspect_candidate", {}, call_id)])
            )
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            "inspect_candidate", {"revision": index}, call_id
                        )
                    ]
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            compacted = _compact_tool_call_history(_context(deps), messages)
            trace = (Path(directory) / "trace.jsonl").read_text(encoding="utf-8")

        self.assertEqual(compacted[0].parts[0].content, "原始 Brief")
        self.assertLessEqual(len(compacted), 14)
        self.assertIsInstance(compacted[1], ModelResponse)
        self.assertIsInstance(compacted[-1], ModelRequest)
        self.assertIn("model_history_compacted", trace)

    def test_tool_result_coalesces_adjacent_frame_violations_for_agent_only(
        self,
    ) -> None:
        sampled = [
            _sampled_ground_violation("frame_0", 0.0, 0.5),
            _sampled_ground_violation("frame_1", 1 / 24, 2.0),
            _sampled_ground_violation("frame_2", 2 / 24, 1.0),
            _sampled_ground_violation("frame_12", 12 / 24, 0.75),
        ]
        envelope = {
            "status": "ok",
            "revision_after": 0,
            "data": {"violations": sampled},
            "violations": sampled,
        }

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            _read_capabilities(deps)
            result = deps.call_tool(
                "validate_candidate",
                {"checks": ["transforms"]},
                lambda: envelope,
            )
            events = [
                json.loads(line)
                for line in (Path(directory) / "trace.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        agent_violations = result["violations"]
        self.assertEqual(len(agent_violations), 2)
        self.assertEqual(agent_violations[0]["id"], "frame_1")
        self.assertEqual(agent_violations[0]["time_range_seconds"], [0.0, 2 / 24])
        self.assertEqual(agent_violations[0]["actual"]["sample_count"], 3)
        self.assertEqual(
            agent_violations[0]["actual"]["worst_sample"]["actual"]["penetration_m"],
            2.0,
        )
        self.assertNotIn("violations", result["data"])
        self.assertEqual(result["data"]["violation_count"], 2)
        self.assertEqual(
            result["data"]["violations_location"],
            "top_level_violations",
        )
        self.assertEqual(
            result["data"]["repair_focus"]["primary_codes"],
            ["ENTITY_INTERSECTS_GROUND"],
        )
        self.assertEqual(
            result["data"]["repair_focus"]["secondary_violation_count"],
            1,
        )
        self.assertEqual(
            result["data"]["repair_focus"]["distinct_secondary_cause_count"],
            0,
        )
        self.assertTrue(
            result["data"]["repair_focus"]["all_compacted_violations_retained"]
        )
        completed = next(
            event
            for event in events
            if event["event_type"] == "tool_call_completed"
            and event["payload"].get("tool_name") == "validate_candidate"
        )
        self.assertEqual(len(completed["payload"]["result"]["violations"]), 4)
        self.assertEqual(len(envelope["violations"]), 4)

    def test_scene_skeleton_must_precede_candidate_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            result = deps.call_tool(
                "inspect_candidate",
                {"view": "summary", "revision": None},
                lambda: deps.toolkit.inspect_candidate(view="summary"),
            )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("Scene Skeleton", result["warnings"][0])

    def test_design_tools_are_exposed_in_symbolic_then_numeric_order(self) -> None:
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            context = _context(deps)

            self.assertIs(
                asyncio.run(_prepare_scene_skeleton_tool(context, sentinel)),
                sentinel,
            )
            self.assertIsNone(
                asyncio.run(_prepare_design_options_tool(context, sentinel))
            )
            deps.call_tool(
                "submit_scene_skeleton",
                {"skeleton": _desert_skeleton()},
                lambda: deps.toolkit.submit_scene_skeleton(_desert_skeleton()),
            )
            self.assertIsNone(
                asyncio.run(_prepare_scene_skeleton_tool(context, sentinel))
            )
            self.assertIs(
                asyncio.run(_prepare_design_options_tool(context, sentinel)),
                sentinel,
            )

            options = deps.call_tool(
                "request_design_options",
                {"preference": "balanced", "max_options": 1},
                lambda: deps.toolkit.request_design_options(max_options=1),
            )
            option = options["data"]["options"][0]
            self.assertIs(
                asyncio.run(_prepare_design_apply_tool(context, sentinel)),
                sentinel,
            )
            applied = deps.call_tool(
                "apply_design_option",
                {
                    "base_revision": option["base_revision"],
                    "option_id": option["option_id"],
                },
                lambda: deps.toolkit.apply_design_option(
                    option["base_revision"], option["option_id"]
                ),
            )

            self.assertTrue(applied["data"]["commit_ready"])
            self.assertIsNone(
                asyncio.run(_prepare_design_apply_tool(context, sentinel))
            )
            self.assertIsNone(asyncio.run(_prepare_candidate_tool(context, sentinel)))

    def test_failed_design_search_reopens_symbolic_skeleton(self) -> None:
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            context = _context(deps)
            deps.toolkit.submit_scene_skeleton(_desert_skeleton())
            result = deps.toolkit.request_design_options(
                max_options=1,
                custom_size_requests=[
                    {
                        "entity_id": "man_01",
                        "minimum_xyz_m": [1.0, 1.0, 1.0],
                        "maximum_xyz_m": [1.0, 1.0, 1.0],
                        "preferred_xyz_m": [1.0, 1.0, 1.0],
                        "rationale": "触发无合法候选",
                    }
                ],
            )

            prepared = asyncio.run(_prepare_scene_skeleton_tool(context, sentinel))
            resubmitted = deps.call_tool(
                "submit_scene_skeleton",
                {"skeleton": _desert_skeleton()},
                lambda: deps.toolkit.submit_scene_skeleton(_desert_skeleton()),
            )

        self.assertEqual(result["status"], "no_change")
        self.assertIs(prepared, sentinel)
        self.assertEqual(resubmitted["status"], "ok")

    def test_identical_inspect_is_rejected_without_checkpoint(self) -> None:
        checkpoints: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                checkpoint_writer=lambda candidate: _checkpoint(
                    candidate.revision, checkpoints
                ),
            )
            _read_capabilities(deps)
            arguments = {"view": "summary", "revision": None}
            first = deps.call_tool(
                "inspect_candidate",
                arguments,
                lambda: deps.toolkit.inspect_candidate(view="summary"),
            )
            second = deps.call_tool(
                "inspect_candidate",
                arguments,
                lambda: deps.toolkit.inspect_candidate(view="summary"),
            )

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "rejected")
        self.assertEqual(checkpoints, [])

    def test_commit_ready_revision_rejects_further_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _solved_toolkit())
            _read_capabilities(deps)
            result = deps.call_tool(
                "inspect_candidate",
                {"view": "summary", "revision": None},
                lambda: deps.toolkit.inspect_candidate(view="summary"),
            )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("CommitRequest", result["next_actions"][0])

    def test_partial_hard_pass_does_not_close_candidate_tools(self) -> None:
        toolkit = _solved_toolkit()
        partial = toolkit.validate_candidate(checks=["references"])
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), toolkit)
            _read_capabilities(deps)
            prepared = asyncio.run(_prepare_candidate_tool(_context(deps), sentinel))
            inspected = deps.call_tool(
                "inspect_candidate",
                {"view": "summary", "revision": None},
                lambda: toolkit.inspect_candidate(view="summary"),
            )

        self.assertTrue(partial["data"]["hard_pass"])
        self.assertFalse(partial["data"]["commit_ready"])
        self.assertIs(prepared, sentinel)
        self.assertEqual(inspected["status"], "ok")

    def test_historical_inspect_does_not_write_checkpoint(self) -> None:
        checkpoints: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                checkpoint_writer=lambda candidate: _checkpoint(
                    candidate.revision, checkpoints
                ),
            )
            _read_capabilities(deps)
            deps.call_tool(
                "apply_entity_patch",
                {"upserts": [_man_entity()], "remove_ids": []},
                lambda: deps.toolkit.apply_entity_patch([_man_entity()], []),
            )
            result = deps.call_tool(
                "inspect_candidate",
                {"view": "summary", "revision": 0},
                lambda: deps.toolkit.inspect_candidate(view="summary", revision=0),
            )

        self.assertEqual(result["revision_after"], 0)
        self.assertEqual(deps.toolkit.store.current_revision, 1)
        self.assertEqual(checkpoints, [1])

    def test_repair_tools_are_exposed_only_when_domain_state_needs_them(self) -> None:
        toolkit = _projected_motion_toolkit(
            end_position=(0.0, 0.0, 0.9),
            camera_position=(0.0, -10.0, 1.5),
        )
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), toolkit)
            _read_capabilities(deps)
            context = _context(deps)

            before = asyncio.run(_prepare_repair_suggestion_tool(context, sentinel))
            deps.call_tool(
                "validate_candidate",
                {"checks": ["motion"]},
                lambda: toolkit.validate_candidate(checks=["motion"]),
            )
            suggestion_ready = asyncio.run(
                _prepare_repair_suggestion_tool(context, sentinel)
            )
            toolkit.suggest_repairs(max_options=1)
            suggestion_after_search = asyncio.run(
                _prepare_repair_suggestion_tool(context, sentinel)
            )
            apply_ready = asyncio.run(_prepare_repair_apply_tool(context, sentinel))

        self.assertIsNone(before)
        self.assertIs(suggestion_ready, sentinel)
        self.assertIsNone(suggestion_after_search)
        self.assertIs(apply_ready, sentinel)

    def test_manual_mutation_reopens_after_deterministic_repair_is_exhausted(
        self,
    ) -> None:
        toolkit = _projected_motion_toolkit(
            end_position=(0.0, 0.0, 0.9),
            camera_position=(0.0, -10.0, 1.5),
        )
        toolkit.validate_candidate(checks=["motion"])
        toolkit._repair_search_exhausted_revision = toolkit.store.current_revision
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            context = _context(_deps(Path(directory), toolkit))

            legacy_prepared = asyncio.run(
                _prepare_manual_mutation_tool(context, sentinel)
            )
            prepared = asyncio.run(_prepare_escalated_candidate_tool(context, sentinel))
            repair_tool = ToolDefinition(
                name="apply_candidate_patch",
                description="组合修复。",
            )
            candidate_patch = asyncio.run(
                _prepare_candidate_patch_tool(context, repair_tool)
            )

        self.assertIsNone(legacy_prepared)
        self.assertIs(prepared, sentinel)
        self.assertIsNotNone(candidate_patch)
        assert candidate_patch is not None
        self.assertIn("这些是建议而非硬隐藏", candidate_patch.description or "")

    def test_unrepairable_hard_error_exposes_patch_on_first_failure(self) -> None:
        toolkit = _projected_motion_toolkit(
            end_position=(0.0, 0.0, 0.9),
            camera_position=(0.0, -10.0, 1.5),
        )
        revision = toolkit.store.current_revision
        toolkit.store.save_validation(
            ValidationReport(
                revision=revision,
                hard_pass=False,
                soft_score=1.0,
                checks=["motion"],
                violations=[
                    Violation(
                        id="unrepairable_motion",
                        code="MOTION_DIRECTION_SEMANTICS_UNMET",
                        severity="hard",
                        entity_ids=["man_01"],
                        message="动作主体没有完成要求的方向位移",
                    ),
                    Violation(
                        id="repairable_camera",
                        code="PROJECTED_MOTION_UNREADABLE",
                        severity="hard",
                        entity_ids=["man_01"],
                        message="屏幕投影不清晰",
                    ),
                ],
            )
        )
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            context = _context(_deps(Path(directory), toolkit))

            prepared = asyncio.run(_prepare_escalated_candidate_tool(context, sentinel))

        self.assertIs(prepared, sentinel)
        self.assertFalse(toolkit.has_exhausted_repair_search)


def _deps(
    path: Path,
    toolkit,
    checkpoint_writer=None,
    *,
    preserve_complete_thinking_history: bool = False,
) -> PlanningDeps:
    return PlanningDeps(
        toolkit=toolkit,
        trace=TraceRecorder(path / "trace.jsonl", "protocol_test"),
        deadline_monotonic=time.monotonic() + 30.0,
        checkpoint_writer=checkpoint_writer,
        preserve_complete_thinking_history=preserve_complete_thinking_history,
    )


def _context(deps: PlanningDeps):
    from pydantic_ai.tools import RunContext

    return RunContext(deps=deps, model=None, usage=None)


def _read_capabilities(deps: PlanningDeps) -> None:
    deps.call_tool(
        "get_capabilities",
        {"sections": ["limits"]},
        lambda: deps.toolkit.get_capabilities(["limits"]),
    )


def _checkpoint(revision: int, checkpoints: list[int]) -> Path:
    checkpoints.append(revision)
    return Path(f"revision_{revision:04d}.json")


def _sampled_ground_violation(
    violation_id: str,
    time_seconds: float,
    penetration_m: float,
) -> dict:
    return {
        "id": violation_id,
        "code": "ENTITY_INTERSECTS_GROUND",
        "severity": "hard",
        "constraint_id": None,
        "entity_ids": ["person", "ground"],
        "time_range_seconds": [time_seconds, time_seconds],
        "expected": {"maximum_penetration_m": 0.01},
        "actual": {
            "penetration_m": penetration_m,
            "clearance_m": -penetration_m,
            "ground_z": 0.0,
            "ground_entity_id": "ground",
            "mode": "must_be_above",
        },
        "adjustable_variables": ["entity transforms", "ground_interaction"],
        "message": "Entity 地面交互不符合 must_be_above：person",
    }


if __name__ == "__main__":
    unittest.main()
