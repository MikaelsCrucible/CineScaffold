from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from cinescaffold.planning.domain import CandidateState, PlanningProfile, StrictModel
from cinescaffold.planning.objective import ObjectivePlanningBrief
from cinescaffold.planning.store import canonical_hash


CHECKPOINT_VERSION = "0.1"


class CandidateCheckpoint(StrictModel):
    checkpoint_version: Literal["0.1"] = "0.1"
    run_id: str
    toolkit_version: str
    source_brief_sha256: str
    profile_id: str
    candidate_hash: str
    candidate: CandidateState


def write_candidate_checkpoint(
    directory: Path,
    *,
    run_id: str,
    toolkit_version: str,
    source_brief_sha256: str,
    profile_id: str,
    candidate: CandidateState,
) -> Path:
    """原子写入 revision checkpoint 与 latest 指针。"""
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = CandidateCheckpoint(
        run_id=run_id,
        toolkit_version=toolkit_version,
        source_brief_sha256=source_brief_sha256,
        profile_id=profile_id,
        candidate_hash=canonical_hash(candidate),
        candidate=candidate,
    )
    payload = json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    revision_path = directory / f"revision_{candidate.revision:04d}.json"
    _atomic_write(revision_path, payload)
    _atomic_write(directory.parent / "checkpoint_latest.json", payload)
    return revision_path


def load_candidate_checkpoint(
    path: Path,
    *,
    objective_brief: ObjectivePlanningBrief,
    profile: PlanningProfile,
    toolkit_version: str,
) -> CandidateCheckpoint:
    checkpoint = CandidateCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.toolkit_version != toolkit_version:
        raise ValueError(
            f"Checkpoint Toolkit 版本不兼容：{checkpoint.toolkit_version} != {toolkit_version}"
        )
    if checkpoint.source_brief_sha256 != objective_brief.source_brief_sha256:
        raise ValueError("Checkpoint 与当前 Cinematic Brief hash 不一致")
    if checkpoint.profile_id != profile.profile_id:
        raise ValueError(f"Checkpoint Profile 不兼容：{checkpoint.profile_id} != {profile.profile_id}")
    if checkpoint.candidate_hash != canonical_hash(checkpoint.candidate):
        raise ValueError("Checkpoint Candidate hash 校验失败")
    return checkpoint


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
