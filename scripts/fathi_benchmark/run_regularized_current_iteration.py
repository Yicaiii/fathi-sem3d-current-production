"""Run the first production Fathi-TV transition on the CURRENT 917c721 lineage.

This is deliberately additive: the frozen data-only CURRENT driver is not
modified. Gate 4C initially authorizes only iter000 -> iter001. Once that
transition closes, later iterations can be generalized using the accepted
regularized history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    artifact_record,
    atomic_json,
    exact_reverse_result,
    gradient_bridge_result,
    optimizer_direction_result,
    retained_primal_result,
)
from scripts.fathi_benchmark.current_pipeline_artifacts import (
    generate_raw_alpha_candidate,
)
from scripts.fathi_benchmark.external_armijo import ArmijoParameters
from scripts.fathi_benchmark.generic_iteration_runner import GenericIterationRunner
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.regularization.regularized_pipeline_artifacts import (
    REGULARIZED_ARMIJO_ACCEPTED_RESULT,
    execute_regularized_armijo,
    persist_parent_regularized_objective,
    promote_regularized_accepted_trial,
    register_regularized_gradient,
)
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
    runtime_resolve_path,
)


SOURCE_BASELINE = "917c721"
DEFAULT_RUNTIME = "configs/fathi_s43_repro_tv_p20_t052_runtime.json"
DEFAULT_ENGINE = "configs/fathi_s43_repro_tv_p20_t052_iteration_engine.json"
DEFAULT_REGULARIZATION = "configs/fathi_s43_repro_tv_p20_t052_regularization.json"

STOPS = (
    "preflight",
    "parent-forward",
    "reverse",
    "gradient",
    "direction",
    "armijo",
    "promote",
)


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {source}")
    return value


def _pass(path: Path, result: str) -> bool:
    if not path.is_file():
        return False
    try:
        return _json(path).get("result") == result
    except Exception:
        return False


def _run(command: list[str], *, cwd: Path) -> None:
    print("COMMAND =", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess failed with rc={completed.returncode}: "
            + " ".join(command)
        )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def source_guard(repo: Path) -> dict[str, str]:
    module_repo = Path(__file__).resolve().parents[2]
    if module_repo != repo.resolve():
        raise RuntimeError(
            f"source guard failed: module repo {module_repo} != requested repo {repo}"
        )

    top = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve()
    if top != repo.resolve():
        raise RuntimeError("source guard failed: git top-level differs from repo")

    branch = _git(repo, "branch", "--show-current")

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_BASELINE, "HEAD"],
        cwd=str(repo),
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"source guard failed: HEAD is not descended from {SOURCE_BASELINE}"
        )

    remote = _git(repo, "remote", "get-url", "origin")

    return {
        "repo": str(repo),
        "branch": branch,
        "head": _git(repo, "rev-parse", "HEAD"),
        "baseline_ancestor": SOURCE_BASELINE,
        "origin": remote,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--parent-iteration",
        "--k",
        dest="parent_iteration",
        type=int,
        required=True,
    )
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME)
    parser.add_argument("--engine-config", default=DEFAULT_ENGINE)
    parser.add_argument(
        "--regularization-config",
        default=DEFAULT_REGULARIZATION,
    )
    parser.add_argument("--reference-manifest")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--replay-stride", type=int, default=10)
    parser.add_argument(
        "--reverse-checkpoint-interval",
        type=int,
        default=50,
    )
    parser.add_argument("--stop-after", choices=STOPS, default="promote")
    args = parser.parse_args()

    if args.parent_iteration != 0:
        parser.error(
            "Gate 4C initially authorizes only regularized iter000 -> iter001. "
            "Do not generalize history until this transition closes."
        )

    repo = Path(args.repo).expanduser().resolve()
    guard = source_guard(repo)

    runtime_path = resolve_path(args.runtime_config, base=repo)
    engine_path = resolve_path(args.engine_config, base=repo)
    regularization_path = resolve_path(
        args.regularization_config,
        base=repo,
    )
    runtime = _json(runtime_path)
    engine = _json(engine_path)
    run = str(engine["run_id"])
    parent = int(args.parent_iteration)

    if str(runtime.get("benchmark_name")) != run:
        raise RuntimeError("runtime/engine run identity mismatch")
    contract = engine.get("regularization_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("regularized engine lacks regularization_contract")
    if contract.get("weights_frozen_within_parent_line_search") is not True:
        raise RuntimeError("regularized engine does not freeze parent Eq.24 weights")

    runtime_paths = iteration_runtime_paths(
        runtime,
        parent,
        repo_root=repo,
    )
    paths = build_iteration_paths(
        engine,
        parent,
        child_iteration=parent + 1,
        repository_root=repo,
        runtime_root=runtime_paths["runtime_root"],
    )
    runner = GenericIterationRunner.from_config_files(
        run_id=run,
        parent_iteration=parent,
        child_iteration=parent + 1,
        repository_root=repo,
        runtime_config_path=runtime_path,
        engine_config_path=engine_path,
    )
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else (
            paths.results_run_root
            / "certified_external_reference.json"
        ).resolve()
    )

    accepted_parent = paths.parent_accepted / "accepted_summary.json"
    if not _pass(accepted_parent, accepted_model_result(parent)):
        raise RuntimeError(
            f"regularized bootstrap parent is not PASS: {accepted_parent}"
        )
    if not reference_path.is_file():
        raise RuntimeError(
            f"regularized certified reference missing: {reference_path}"
        )

    if paths.optimizer_history.exists():
        history_files = [
            p
            for p in paths.optimizer_history.rglob("*")
            if p.is_file()
        ]
        if history_files:
            raise RuntimeError(
                "Gate 4C iter000 must start with empty regularized L-BFGS history"
            )

    preflight = {
        "result": "PASS_FATHI_TV_GATE4C_PRODUCTION_PREFLIGHT",
        "source_guard": guard,
        "run_id": run,
        "parent_iteration": parent,
        "child_iteration": parent + 1,
        "transition": paths.identity.transition_id,
        "runtime_config": str(runtime_path),
        "engine_config": str(engine_path),
        "regularization_config": str(regularization_path),
        "reference_manifest": str(reference_path),
        "accepted_parent": str(accepted_parent),
        "optimizer_history_empty": True,
        "numerical_runs": {
            "sem3d": 0,
            "external_forward": 0,
            "exact_reverse": 0,
            "optimizer": 0,
        },
    }
    preflight_path = (
        paths.transition_root / "regularized_gate4c_preflight.json"
    )
    atomic_json(preflight_path, preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.stop_after == "preflight":
        return

    primal_summary = (
        paths.exact_reverse / "primal_forward" / "summary.json"
    )
    if _pass(primal_summary, retained_primal_result(parent)):
        print(f"PARENT_FORWARD = REUSE {primal_summary}")
    else:
        parent_forward_command = [
            sys.executable,
            str(
                repo
                / "scripts/fathi_benchmark/"
                "run_certified_external_parent_forward.py"
            ),
            "--repo",
            str(repo),
            "--config",
            str(runtime_path),
            "--iter-k",
            str(parent),
            "--reference-manifest",
            str(reference_path),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
        ]
        reference = _json(reference_path)
        sample_count = int(reference["contract"]["sample_count"])
        primal_dir = paths.exact_reverse / "primal_forward"
        completed_existing = (
            (primal_dir / "current_external_receiver.npy").is_file()
            and (primal_dir / "checkpoint" / "current_latest.npz").is_file()
            and (
                primal_dir
                / "checkpoint"
                / "current_primal_retained"
                / f"primal_{sample_count:06d}.npz"
            ).is_file()
        )
        if completed_existing:
            print(
                "PARENT_FORWARD = RECERTIFY_EXISTING_COMPLETED_ARTIFACTS "
                "(no numerical rerun)"
            )
            parent_forward_command.append("--certify-existing")
        _run(parent_forward_command, cwd=repo)
    if args.stop_after == "parent-forward":
        return

    reverse_summary = (
        paths.exact_reverse / "production_reverse" / "summary.json"
    )
    if _pass(reverse_summary, exact_reverse_result(parent)):
        print(f"EXACT_REVERSE = REUSE {reverse_summary}")
    else:
        common = [
            sys.executable,
            str(
                repo
                / "scripts/fathi_benchmark/"
                "run_exact_reverse_gradient_generic.py"
            ),
            "--repo",
            str(repo),
            "--config",
            str(runtime_path),
            "--engine-config",
            str(engine_path),
            "--iter-k",
            str(parent),
            "--reference-manifest",
            str(reference_path),
            "--batch-size",
            str(args.batch_size),
            "--replay-stride",
            str(args.replay_stride),
            "--reverse-checkpoint-interval",
            str(args.reverse_checkpoint_interval),
        ]
        _run(common + ["--action", "preflight"], cwd=repo)
        _run(common + ["--action", "reverse"], cwd=repo)
    if args.stop_after == "reverse":
        return

    gradient_summary = paths.gradient_root / "summary.json"
    if _pass(gradient_summary, gradient_bridge_result(parent)):
        print(f"DATA_GRADIENT_BRIDGE = REUSE {gradient_summary}")
    else:
        _run(
            [
                sys.executable,
                str(
                    repo
                    / "scripts/fathi_benchmark/"
                    "bridge_certified_external_gradient.py"
                ),
                "--repo",
                str(repo),
                "--config",
                str(runtime_path),
                "--iteration",
                str(parent),
                "--reference-manifest",
                str(reference_path),
            ],
            cwd=repo,
        )

    eq21_dir = paths.gradient_root / "regularized_eq21"
    eq21_summary = eq21_dir / "gate3_eq21_summary.json"
    if not (
        eq21_summary.is_file()
        and _json(eq21_summary).get("result")
        == "PASS_FATHI_TV_GATE3_EQ21_CONTROL_ASSEMBLY"
    ):
        _run(
            [
                sys.executable,
                "-m",
                "scripts.fathi_benchmark.regularization.run_tv_eq21_gate",
                "--runtime-config",
                str(runtime_path),
                "--regularization-config",
                str(regularization_path),
                "--material-dir",
                str(
                    paths.parent_accepted
                    / str(engine["material"]["directory"])
                ),
                "--corrected-gradient-dir",
                str(paths.gradient_root),
                "--frequency-stage",
                "p20",
                "--output-dir",
                str(eq21_dir),
            ],
            cwd=repo,
        )

    registered = register_regularized_gradient(
        repo=repo,
        paths=paths,
        eq21_directory=eq21_dir,
    )
    parent_regularized_objective = (
        persist_parent_regularized_objective(
            repo=repo,
            paths=paths,
            eq21_directory=eq21_dir,
            registered_gradient_path=registered,
        )
    )
    print(f"REGISTERED_TOTAL_GRADIENT = {registered}")
    print(
        "PARENT_REGULARIZED_OBJECTIVE = "
        f"{parent_regularized_objective}"
    )
    if args.stop_after == "gradient":
        return

    direction_summary = (
        paths.optimizer_root
        / f"iter_{parent:03d}_lbfgs_eq25_direction"
        / "direction_summary.json"
    )
    if _pass(direction_summary, optimizer_direction_result(parent)):
        print(f"DIRECTION = REUSE {direction_summary}")
    else:
        direction_request = {
            "run_id": run,
            "parent_iteration": parent,
            "child_iteration": parent + 1,
            "transition": paths.identity.transition_id,
            "registered_gradient_manifest": artifact_record(
                registered, repo=repo
            ),
            "accepted_parent_summary": artifact_record(
                accepted_parent, repo=repo
            ),
            "history_outcomes": [],
        }
        request_path = (
            paths.optimizer_root / "direction_request.json"
        )
        atomic_json(request_path, direction_request)
        result = runner.compute_optimizer_direction(
            direction_request
        )
        direction_summary = runner.persist_optimizer_direction(
            direction_request,
            result,
        )
        print(f"DIRECTION = BUILT {direction_summary}")
    if args.stop_after == "direction":
        return

    reference = _json(reference_path)
    true_path = runtime_resolve_path(
        reference["certification_assets"]["true_external"],
        repo_root=repo,
    )

    armijo_summary = execute_regularized_armijo(
        repo=repo,
        paths=paths,
        engine_config=engine,
        runtime_config=runtime,
        runtime_config_path=runtime_path,
        regularization_config_path=regularization_path,
        reference_manifest=reference_path,
        accepted_parent_record=artifact_record(
            accepted_parent, repo=repo
        ),
        registered_gradient_record=artifact_record(
            registered, repo=repo
        ),
        direction_record=artifact_record(
            direction_summary, repo=repo
        ),
        parent_objective_record=artifact_record(
            parent_regularized_objective, repo=repo
        ),
        batch_size=int(args.batch_size),
        checkpoint_interval=int(args.checkpoint_interval),
    )
    armijo_payload = _json(armijo_summary)
    if (
        armijo_payload.get("result")
        != REGULARIZED_ARMIJO_ACCEPTED_RESULT
        or armijo_payload.get("accepted") is not True
    ):
        raise RuntimeError(
            "regularized Armijo completed without an accepted trial"
        )
    print(f"REGULARIZED_ARMIJO = {armijo_summary}")
    if args.stop_after == "armijo":
        return

    accepted_child = promote_regularized_accepted_trial(
        repo=repo,
        paths=paths,
        material_config=engine["material"],
        armijo_summary_path=armijo_summary,
    )

    print("=" * 76)
    print(
        "RESULT = "
        f"PASS_FATHI_TV_REGULARIZED_ITERATION_{parent:03d}_"
        f"TO_{parent + 1:03d}_CLOSED"
    )
    print(f"ACCEPTED_CHILD = {accepted_child}")
    print(f"CHILD_STATE = {paths.child_state}")
    print("=" * 76)


if __name__ == "__main__":
    main()
