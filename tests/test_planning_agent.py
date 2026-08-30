from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from cinescaffold.planning.agent import PlanningDeps
from cinescaffold.planning.trace import TraceRecorder
from tests.test_planning_toolkit import _man_entity, _solved_toolkit, _toolkit


class PlanningProtocolTest(unittest.TestCase):
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
