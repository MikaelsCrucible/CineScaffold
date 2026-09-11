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

from cinescaffold.planning.agent import (
    ConstraintPatchInput,
    NON_TOOL_TEXT_MARKER,
    PathPatchInput,
    PlanningDeps,
    TrackPatchInput,
    _compact_tool_call_history,
    _prepare_candidate_tool,
    _prepare_design_apply_tool,
    _prepare_design_options_tool,
    _prepare_manual_mutation_tool,
    _prepare_repair_apply_tool,
    _prepare_repair_suggestion_tool,
    _prepare_scene_skeleton_tool,
)
from cinescaffold.planning.domain import ConstraintSpec, TrackSpec
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
    def test_agent_patch_schemas_are_compact_but_domain_validation_stays_strict(self) -> None:
        compact_chars = len(json.dumps(TrackPatchInput.model_json_schema()))
        domain_chars = len(json.dumps(TrackSpec.model_json_schema()))
        constraint_chars = len(json.dumps(ConstraintPatchInput.model_json_schema()))
        domain_constraint_chars = len(json.dumps(ConstraintSpec.model_json_schema()))

        # 坐标契约增加少量说明后仍需显著小于完整领域联合。
        self.assertLess(compact_chars, domain_chars * 0.7)
        self.assertLess(constraint_chars, domain_constraint_chars * 0.4)

    def test_path_tool_schema_exposes_coordinate_and_direction_conventions(self) -> None:
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
                    parts=[ToolReturnPart("inspect_candidate", {"revision": index}, call_id)]
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
                for line in (Path(directory) / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(first[0].parts[0].content, NON_TOOL_TEXT_MARKER)
        self.assertEqual(second[0].parts[0].content, NON_TOOL_TEXT_MARKER)
        self.assertLess(len(first[0].parts[0].content), 100)
        compact_events = [
            event for event in events if event["event_type"] == "model_non_tool_text_compacted"
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
                    parts=[ToolReturnPart("inspect_candidate", {"revision": index}, call_id)]
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
            self.assertIsNone(
                asyncio.run(_prepare_candidate_tool(context, sentinel))
            )

    def test_identical_inspect_is_rejected_without_checkpoint(self) -> None:
        checkpoints: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                checkpoint_writer=lambda candidate: _checkpoint(candidate.revision, checkpoints),
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

    def test_historical_inspect_does_not_write_checkpoint(self) -> None:
        checkpoints: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(
                Path(directory),
                _toolkit(),
                checkpoint_writer=lambda candidate: _checkpoint(candidate.revision, checkpoints),
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

            before = asyncio.run(
                _prepare_repair_suggestion_tool(context, sentinel)
            )
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
            apply_ready = asyncio.run(
                _prepare_repair_apply_tool(context, sentinel)
            )

        self.assertIsNone(before)
        self.assertIs(suggestion_ready, sentinel)
        self.assertIsNone(suggestion_after_search)
        self.assertIs(apply_ready, sentinel)

    def test_manual_mutation_reopens_after_deterministic_repair_is_exhausted(self) -> None:
        toolkit = _projected_motion_toolkit(
            end_position=(0.0, 0.0, 0.9),
            camera_position=(0.0, -10.0, 1.5),
        )
        toolkit.validate_candidate(checks=["motion"])
        toolkit._repair_search_exhausted_revision = toolkit.store.current_revision
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory:
            context = _context(_deps(Path(directory), toolkit))

            prepared = asyncio.run(_prepare_manual_mutation_tool(context, sentinel))

        self.assertIs(prepared, sentinel)


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


if __name__ == "__main__":
    unittest.main()
