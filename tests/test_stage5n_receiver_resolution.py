from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.fathi_benchmark.run_certified_external_parent_forward import (
    resolve_stage5n_current_receiver,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage5n_receiver_resolves_nested_artifact(tmp_path: Path):
    receiver = tmp_path / "current_external_receiver.npy"
    receiver.write_bytes(b"certified-current")
    summary = tmp_path / "bundle_summary_v3.json"
    summary.write_text(
        json.dumps(
            {
                "objective": {"J_external": 1.0},
                "artifacts": {
                    "external_forward": {
                        "current": {
                            "path": str(receiver),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved, evidence = resolve_stage5n_current_receiver(
        tmp_path,
        summary,
        expected_sha256=_sha(receiver),
    )

    assert resolved == receiver.resolve()
    assert evidence["matching_candidate_count"] == 1
    assert evidence["selected_sha256"] == _sha(receiver)


def test_stage5n_receiver_deduplicates_same_path(tmp_path: Path):
    receiver = tmp_path / "current_external_receiver.npy"
    receiver.write_bytes(b"same-current")
    summary = tmp_path / "bundle_summary_v3.json"
    summary.write_text(
        json.dumps(
            {
                "a": str(receiver),
                "b": {"path": str(receiver)},
            }
        ),
        encoding="utf-8",
    )

    resolved, evidence = resolve_stage5n_current_receiver(
        tmp_path,
        summary,
        expected_sha256=_sha(receiver),
    )

    assert resolved == receiver.resolve()
    assert evidence["candidate_count"] == 1
    assert evidence["matching_candidate_count"] == 1


def test_stage5n_receiver_blocks_wrong_hash(tmp_path: Path):
    receiver = tmp_path / "current_external_receiver.npy"
    receiver.write_bytes(b"wrong-current")
    summary = tmp_path / "bundle_summary_v3.json"
    summary.write_text(
        json.dumps({"current": str(receiver)}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        resolve_stage5n_current_receiver(
            tmp_path,
            summary,
            expected_sha256="0" * 64,
        )
