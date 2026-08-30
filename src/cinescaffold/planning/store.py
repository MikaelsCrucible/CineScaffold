from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cinescaffold.planning.domain import CandidateState


Mutator = Callable[[CandidateState], tuple[list[dict[str, Any]], list[str]]]


@dataclass(frozen=True)
class MutationResult:
    status: str
    revision_before: int
    revision_after: int
    changes: list[dict[str, Any]]
    warnings: list[str]


class CandidateStore:
    """保存不可变 revision，调用方只能获得深拷贝。"""

    def __init__(self, initial: CandidateState) -> None:
        if initial.revision < 0:
            raise ValueError("初始 Candidate revision 不能为负数")
        self._history: dict[int, CandidateState] = {
            initial.revision: initial.model_copy(deep=True)
        }
        self._current_revision = initial.revision
        self._committed_revision: int | None = None

    @property
    def current_revision(self) -> int:
        return self._current_revision

    @property
    def revisions(self) -> list[int]:
        return sorted(self._history)

    @property
    def committed_revision(self) -> int | None:
        return self._committed_revision

    def get(self, revision: int | None = None) -> CandidateState:
        selected = self._current_revision if revision is None else revision
        if selected not in self._history:
            raise ValueError(f"Candidate revision 不存在：{selected}")
        return self._history[selected].model_copy(deep=True)

    def apply(self, mutator: Mutator) -> MutationResult:
        if self._committed_revision is not None:
            raise ValueError("已提交的 Candidate Store 不允许继续修改")
        before = self.get()
        after = before.model_copy(deep=True)
        changes, warnings = mutator(after)
        if _state_hash(before) == _state_hash(after):
            return MutationResult("no_change", before.revision, before.revision, [], warnings)

        next_revision = max(self._history) + 1
        after.revision = next_revision
        after.validation = None
        self._history[next_revision] = after.model_copy(deep=True)
        self._current_revision = next_revision
        return MutationResult("ok", before.revision, next_revision, changes, warnings)

    def restore(self, source_revision: int) -> MutationResult:
        source = self.get(source_revision)

        def mutator(target: CandidateState) -> tuple[list[dict[str, Any]], list[str]]:
            current_revision = target.revision
            restored = source.model_dump()
            restored["revision"] = current_revision
            replacement = CandidateState.model_validate(restored)
            target.__dict__.update(replacement.__dict__)
            return ([{"operation": "restore", "source_revision": source_revision}], [])

        return self.apply(mutator)

    def save_validation(self, report: Any) -> None:
        state = self._history[self._current_revision].model_copy(deep=True)
        state.validation = report.model_copy(deep=True)
        self._history[self._current_revision] = state

    def commit(self, revision: int) -> None:
        if revision not in self._history:
            raise ValueError(f"Candidate revision 不存在：{revision}")
        self._committed_revision = revision


def canonical_hash(value: Any) -> str:
    if isinstance(value, BaseException):
        value = str(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _state_hash(state: CandidateState) -> str:
    value = state.model_dump(mode="json", exclude={"revision", "validation"})
    return canonical_hash(value)
