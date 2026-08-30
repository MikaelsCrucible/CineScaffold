from __future__ import annotations

import tempfile
import time
import unittest
import json
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

from cinescaffold.planning.agent import (
    ConstraintPatchInput,
    NON_TOOL_TEXT_MARKER,
    PlanningDeps,
    TrackPatchInput,
    _compact_tool_call_history,
)
from cinescaffold.planning.domain import ConstraintSpec, TrackSpec
from cinescaffold.planning.trace import TraceRecorder
from tests.test_planning_toolkit import _man_entity, _solved_toolkit, _toolkit


class PlanningProtocolTest(unittest.TestCase):
    def test_agent_patch_schemas_are_compact_but_domain_validation_stays_strict(self) -> None:
        compact_chars = len(json.dumps(TrackPatchInput.model_json_schema()))
        domain_chars = len(json.dumps(TrackSpec.model_json_schema()))
        constraint_chars = len(json.dumps(ConstraintPatchInput.model_json_schema()))
        domain_constraint_chars = len(json.dumps(ConstraintSpec.model_json_schema()))

        self.assertLess(compact_chars, domain_chars * 0.65)
        self.assertLess(constraint_chars, domain_constraint_chars * 0.4)

    def test_tool_history_drops_text_and_thinking_but_keeps_calls(self) -> None:
        response = ModelResponse(
            parts=[
                TextPart("冗长分析"),
                ThinkingPart("内部推理"),
                ToolCallPart("inspect_candidate", {"view": "summary"}, "call_1"),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            compacted = _compact_tool_call_history(_context(deps), [response])

        self.assertEqual(len(compacted[0].parts), 1)
        self.assertIsInstance(compacted[0].parts[0], ToolCallPart)

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

    def test_capabilities_must_be_read_before_other_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deps = _deps(Path(directory), _toolkit())
            result = deps.call_tool(
                "inspect_candidate",
                {"view": "summary", "revision": None},
                lambda: deps.toolkit.inspect_candidate(view="summary"),
            )

        self.assertEqual(result["status"], "rejected")
        self.assertIn("get_capabilities", result["warnings"][0])

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


def _deps(path: Path, toolkit, checkpoint_writer=None) -> PlanningDeps:
    return PlanningDeps(
        toolkit=toolkit,
        trace=TraceRecorder(path / "trace.jsonl", "protocol_test"),
        deadline_monotonic=time.monotonic() + 30.0,
        checkpoint_writer=checkpoint_writer,
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
