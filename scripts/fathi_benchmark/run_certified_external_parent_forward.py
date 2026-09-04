"""Run the certified external physical forward for an iteration parent model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from scripts.fathi_benchmark.certified_data_objective import (
    certified_data_objective,
    float64_diagnostic,
)
from scripts.exact_adjoint.s43_external_forward import (
    ExternalForwardDriver,
    load_certified_reference,
    run_external_forward,
    sha256_arrays,
    sha256_file,
)
from scripts.fathi_benchmark.runtime_paths import (
    runtime_resolve_path,
    iteration_runtime_paths,
    resolve_path,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    artifact_record,
    retained_primal_result,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.path_consistency import (
    validate_path_config_consistency,
)


HISTORICAL_PASS_RESULT = "PASS_CERTIFIED_PARENT_EXTERNAL_FORWARD"


def current_parent_forward_contract(paths) -> dict:
    """Return the exact schema/path contract without executing a forward."""

    iteration = paths.identity.parent_iteration
    return {
        "schema_version": 1,
        "result": retained_primal_result(iteration),
        "run_id": paths.identity.run_id,
        "parent_iteration": iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "output_path": str(paths.exact_reverse / "primal_forward"),
        "current_receiver_filename": "current_external_receiver.npy",
        "summary_filename": "summary.json",
        "required_summary_fields": [
            "material_sha256",
            "driver_signature_sha256",
            "forward_run_signature_sha256",
            "current_external_receiver",
            "true_external_receiver",
            "objective",
            "retained_primal",
            "immutable_operator_identity",
        ],
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_asset(repo: Path, value: str) -> Path:
    return runtime_resolve_path(
        value,
        repo_root=repo,
    )



def _json_string_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _json_string_leaves(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _json_string_leaves(child, path + (str(index),))
    elif isinstance(value, str):
        yield path, value


def resolve_stage5n_current_receiver(
    repo: Path,
    stage5n_summary_path: Path,
    *,
    expected_sha256: str,
) -> tuple[Path, dict]:
    payload = json.loads(stage5n_summary_path.read_text(encoding="utf-8"))
    candidates = []
    seen = set()

    for json_path, raw_value in _json_string_leaves(payload):
        if Path(raw_value).name != "current_external_receiver.npy":
            continue
        try:
            resolved = manifest_asset(repo, raw_value).resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_file():
            continue
        seen.add(key)
        digest = sha256_file(resolved)
        candidates.append(
            {
                "json_path": ".".join(json_path),
                "raw_value": raw_value,
                "resolved_path": key,
                "sha256": digest,
                "matches_expected_sha256": digest == expected_sha256,
            }
        )

    matching = [
        item for item in candidates
        if item["matches_expected_sha256"]
    ]
    if not matching:
        raise RuntimeError(
            "iter000 Stage5N summary has no current_external_receiver.npy "
            "whose SHA256 matches the completed regularized parent forward; "
            f"candidate_count={len(candidates)}"
        )

    matching.sort(
        key=lambda item: (
            len(item["json_path"]),
            item["json_path"],
            item["resolved_path"],
        )
    )
    selected = matching[0]
    evidence = {
        "classification": (
            "STAGE5N_CURRENT_RECEIVER_RESOLVED_BY_REFERENCED_ARTIFACT_SHA256"
        ),
        "stage5n_summary": str(stage5n_summary_path.resolve()),
        "expected_sha256": expected_sha256,
        "candidate_count": len(candidates),
        "matching_candidate_count": len(matching),
        "selected_json_path": selected["json_path"],
        "selected_path": selected["resolved_path"],
        "selected_sha256": selected["sha256"],
        "candidates": candidates,
    }
    return Path(selected["resolved_path"]), evidence


def expected_parent_objective(
    repo: Path,
    reference: dict,
    runtime: dict,
    iteration: int,
) -> tuple[float, Path, str]:
    if int(iteration) == 0:
        path = manifest_asset(
            repo, reference["certification_assets"]["stage5n_summary"]
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            float(payload["objective"]["J_external"]),
            path,
            str(payload["result"]),
        )

    path = Path(runtime["parent_workspace"]) / "accepted_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("result") != accepted_model_result(iteration):
        raise RuntimeError(f"parent accepted summary is not certified PASS: {path}")
    if int(payload["iter"]) != int(iteration):
        raise RuntimeError("parent accepted-summary iteration mismatch")
    return float(payload["objective"]["accepted"]), path.resolve(), payload["result"]


def load_external(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    value = np.load(path)
    if value.dtype != np.float64 or value.shape != expected_shape:
        raise RuntimeError(
            f"external receiver contract mismatch: {path}: "
            f"dtype={value.dtype}, shape={value.shape}, expected={expected_shape}"
        )
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite external receiver array: {path}")
    return np.asarray(value, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--iter-k", type=int, required=True)
    parser.add_argument("--reference-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--certify-existing",
        action="store_true",
        help=(
            "Certify already-completed parent-forward artifacts without "
            "launching another numerical forward."
        ),
    )
    args = parser.parse_args()

    if args.iter_k < 0:
        parser.error("--iter-k must be nonnegative")
    if args.batch_size < 1 or args.checkpoint_interval < 1:
        parser.error("batch and checkpoint intervals must be positive")

    repo = Path(args.repo).expanduser().resolve()
    config_path = resolve_path(args.config, base=repo)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = iteration_runtime_paths(config, args.iter_k, repo_root=repo)
    run = str(config["benchmark_name"])
    engine_path = (
        repo / "configs" / f"{run}_iteration_engine.json"
    ).resolve()
    engine = json.loads(engine_path.read_text(encoding="utf-8"))
    validate_path_config_consistency(
        config,
        engine,
        repository_root=repo,
    )
    paths = build_iteration_paths(
        engine,
        args.iter_k,
        child_iteration=args.iter_k + 1,
        repository_root=repo,
        runtime_root=runtime["runtime_root"],
    )
    if paths.transition_root != Path(runtime["transition_root"]):
        raise RuntimeError("runtime/iteration-engine transition path mismatch")
    if paths.parent_accepted != Path(runtime["parent_workspace"]):
        raise RuntimeError("runtime/iteration-engine parent path mismatch")
    pass_result = retained_primal_result(args.iter_k)
    reference_path = (
        resolve_path(args.reference_manifest, base=repo)
        if args.reference_manifest
        else repo / "results" / run / "certified_external_reference.json"
    ).resolve()
    _, reference = load_certified_reference(repo, run, reference_path)
    contract = reference["contract"]
    output_dir = (
        resolve_path(args.output_dir, base=repo)
        if args.output_dir
        else paths.exact_reverse / "primal_forward"
    ).resolve()
    if output_dir != (paths.exact_reverse / "primal_forward").resolve():
        raise RuntimeError("CURRENT parent-forward output is not canonical")
    material_dir = Path(runtime["parent_workspace"]) / "mat" / "h5"
    material_files = {
        component: material_dir / str(engine["material"]["files"][component])
        for component in ("kappa", "mu", "density")
    }
    missing = [str(path) for path in material_files.values() if not path.is_file()]
    if missing:
        raise RuntimeError("missing parent material: " + ", ".join(missing))

    initial = iteration_runtime_paths(config, 0, repo_root=repo)
    initial_density = (
        Path(initial["parent_workspace"]) / "mat" / "h5" / "Mat_0_Density.h5"
    )
    if sha256_file(material_files["density"]) != sha256_file(
        initial_density
    ):
        raise RuntimeError("parent density differs from frozen coupled-mass density")

    driver = ExternalForwardDriver(
        repo,
        run,
        material_dir,
        batch_size=args.batch_size,
        reference_manifest=reference_path,
    )
    sample_count = int(contract["sample_count"])
    expected_shape = (
        sample_count,
        int(contract["receiver_count"]),
        int(contract["component_count"]),
    )
    certified_dt = float(contract["dt"])
    if not math.isclose(
        driver.dt, certified_dt, rel_tol=0.0, abs_tol=1.0e-18
    ):
        raise RuntimeError("driver dt differs from certified reference")
    if driver.receiver_count != expected_shape[1]:
        raise RuntimeError("driver receiver count differs from certified reference")
    if sha256_file(driver.paths["stf"]) != reference["immutable_input_assets"][
        "reference_stf_sha256"
    ]:
        raise RuntimeError("reference STF hash mismatch")
    if sha256_file(driver.paths["true_external"]) != reference["hashes"][
        "true_external_sha256"
    ]:
        raise RuntimeError("true external hash mismatch")
    receiver_nodes_path = Path(driver.paths["receiver"]) / "receiver_nodes.npy"
    receiver_weights_path = Path(driver.paths["receiver"]) / "receiver_weights.npy"
    if (
        sha256_file(receiver_nodes_path)
        != reference["hashes"]["receiver_nodes_sha256"]
        or sha256_file(receiver_weights_path)
        != reference["hashes"]["receiver_weights_sha256"]
    ):
        raise RuntimeError("physical receiver operator hash mismatch")
    receiver_hash = sha256_arrays(driver.receiver_nodes, driver.receiver_weights)

    expected_j, expected_j_source, expected_j_result = expected_parent_objective(
        repo, reference, runtime, args.iter_k
    )
    material_hashes = {
        name: sha256_file(path) for name, path in material_files.items()
    }
    signature_payload = {
        "schema_version": 1,
        "iteration": int(args.iter_k),
        "transition": runtime["transition"],
        "reference_manifest_sha256": sha256_file(reference_path),
        "driver_signature_sha256": driver.signature,
        "material_sha256": material_hashes,
        "expected_parent_objective": expected_j,
        "sample_count": sample_count,
        "receiver_operator_sha256": receiver_hash,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    summary_path = output_dir / "summary.json"
    current_path = output_dir / "current_external_receiver.npy"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("result") == pass_result
            and existing.get("input_signature_sha256") == signature
            and current_path.is_file()
            and existing["files"]["current_external_sha256"]
            == sha256_file(current_path)
        ):
            print(f"RESULT = {pass_result}")
            print(f"OUTPUT = {output_dir}")
            print("IDEMPOTENT_REUSE = true")
            return
        raise RuntimeError(f"refusing non-matching existing parent forward: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint" / "current_latest.npz"
    retained_dir = output_dir / "checkpoint" / "current_primal_retained"
    retained_last = retained_dir / f"primal_{sample_count:06d}.npz"

    if args.certify_existing:
        missing_existing = [
            str(path)
            for path in (current_path, checkpoint_path, retained_last)
            if not path.is_file()
        ]
        if missing_existing:
            raise RuntimeError(
                "cannot certify incomplete existing parent forward: "
                + ", ".join(missing_existing)
            )
        run_summary = {
            "classification": (
                "RECERTIFIED_EXISTING_COMPLETED_EXTERNAL_FORWARD_NO_RERUN"
            ),
            "signature_sha256": signature,
            "reused_existing_artifacts": True,
            "current_invocation_external_forwards": 0,
        }
    else:
        run_summary = run_external_forward(
            driver,
            sample_count,
            {"primal": current_path},
            checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            retained_primal_dir=retained_dir,
        )

    current = load_external(current_path, expected_shape)
    truth = load_external(Path(driver.paths["true_external"]), expected_shape)
    residual = current - truth
    objective_eval = certified_data_objective(
        current,
        truth,
        certified_dt=certified_dt,
        driver_dt=float(driver.dt),
    )
    objective = float(objective_eval.value)
    legacy_driver_dt_eval = certified_data_objective(
        current,
        truth,
        certified_dt=float(driver.dt),
        driver_dt=float(driver.dt),
    )
    relative = abs(objective - expected_j) / max(
        abs(expected_j), np.finfo(np.float64).tiny
    )
    current_receiver_hash = sha256_file(current_path)
    accepted_receiver_hash = None
    accepted_receiver_path = None
    accepted_receiver_resolution = None
    if int(args.iter_k) == 0:
        (
            accepted_receiver_path,
            accepted_receiver_resolution,
        ) = resolve_stage5n_current_receiver(
            repo,
            expected_j_source,
            expected_sha256=current_receiver_hash,
        )
        accepted_receiver_hash = sha256_file(accepted_receiver_path)
    else:
        accepted_payload = json.loads(expected_j_source.read_text(encoding="utf-8"))
        accepted_receiver = accepted_payload.get("accepted_external_receiver")
        if isinstance(accepted_receiver, dict):
            accepted_receiver_path = manifest_asset(
                repo, str(accepted_receiver["path"])
            )
        else:
            trial_path = manifest_asset(
                repo, str(accepted_payload["external_armijo_trial"])
            )
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            accepted_receiver_path = manifest_asset(
                repo, str(trial["candidate_external_receiver"])
            )
        accepted_receiver_hash = sha256_file(accepted_receiver_path)
    gates = {
        "reference_manifest_pass": True,
        "reference_true_external_hash": True,
        "reference_receiver_operator": True,
        "fixed_dt": True,
        "sample_shape": True,
        "density_matches_frozen_coupled_mass": True,
        "retained_endpoint_present": retained_last.is_file(),
        "objective_bitwise_equal_to_accepted": objective == expected_j,
        "current_receiver_bitwise_equal_to_accepted": (
            accepted_receiver_hash is None
            or accepted_receiver_hash == current_receiver_hash
        ),
        "all_finite": bool(np.all(np.isfinite(residual))),
    }
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        failure_diagnostic = {
            "schema_version": 1,
            "result": "BLOCK_CERTIFIED_PARENT_EXTERNAL_FORWARD_GATE",
            "run_id": run,
            "iteration": int(args.iter_k),
            "transition": runtime["transition"],
            "failed_gates": failed,
            "gates": gates,
            "objective_diagnostic": objective_eval.diagnostic(
                expected=expected_j
            ),
            "legacy_driver_dt_objective": {
                **float64_diagnostic(legacy_driver_dt_eval.value),
                "equal_expected": (
                    legacy_driver_dt_eval.value == expected_j
                ),
            },
            "expected_j_source": str(expected_j_source),
            "reference_manifest": str(reference_path),
            "current_receiver_sha256": current_receiver_hash,
            "accepted_receiver_sha256": accepted_receiver_hash,
            "accepted_receiver_resolution": accepted_receiver_resolution,
        }
        atomic_json(output_dir / "failure_diagnostic.json", failure_diagnostic)
        raise RuntimeError("parent-forward gates failed: " + ", ".join(failed))

    recertification = {
        "performed": bool(args.certify_existing),
        "classification": (
            "RECERTIFIED_EXISTING_COMPLETED_PARENT_FORWARD_AFTER_DT_CONTRACT_ROOT_CAUSE"
            if args.certify_existing
            else "LIVE_CERTIFIED_PARENT_FORWARD"
        ),
        "root_cause": (
            "DRIVER_DT_ONE_ULP_BELOW_CERTIFIED_OBJECTIVE_DT"
            if objective_eval.dt_ulp_distance == 1
            and legacy_driver_dt_eval.value != expected_j
            and objective == expected_j
            else "NONE_OBSERVED"
        ),
        "driver_dt": float64_diagnostic(float(driver.dt)),
        "certified_objective_dt": float64_diagnostic(certified_dt),
        "dt_ulp_distance": objective_eval.dt_ulp_distance,
        "legacy_driver_dt_J": float64_diagnostic(legacy_driver_dt_eval.value),
        "certified_dt_J": float64_diagnostic(objective),
        "expected_J": float64_diagnostic(expected_j),
        "numerical_reruns_after_initial_completed_forward": 0,
    }

    summary = {
        "schema_version": 1,
        "result": pass_result,
        "run_id": run,
        "parent_iteration": int(args.iter_k),
        "child_iteration": int(args.iter_k) + 1,
        "iteration": int(args.iter_k),
        "transition": runtime["transition"],
        "input_signature_sha256": signature,
        "reference_manifest": str(reference_path),
        "reference_manifest_sha256": sha256_file(reference_path),
        "material_dir": str(material_dir.resolve()),
        "material_sha256": material_hashes,
        "runtime_config": artifact_record(config_path, repo=repo),
        "iteration_engine_config": artifact_record(engine_path, repo=repo),
        "accepted_parent_summary": (
            None
            if int(args.iter_k) == 0
            else artifact_record(expected_j_source, repo=repo)
        ),
        "immutable_operator_identity": {
            "reference_manifest_sha256": sha256_file(reference_path),
            "driver_signature_sha256": driver.signature,
            "receiver_operator_sha256": receiver_hash,
        },
        "objective": {
            "J_external": objective,
            "accepted_J": expected_j,
            "bitwise_equal": objective == expected_j,
            "relative_error": relative,
            "expected_source": str(expected_j_source),
            "expected_source_result": expected_j_result,
            "residual_sign": "current_external - true_external",
            "time_weighting": "native fixed-dt trapezoidal quadrature",
            "sample_count": sample_count,
            "receiver_count": expected_shape[1],
            "component_count": expected_shape[2],
            "dt": certified_dt,
            "driver_dt": float(driver.dt),
            "dt_ulp_distance": objective_eval.dt_ulp_distance,
            "certified_quadrature_dt_source": "reference.contract.dt",
            "residual_l2": float(np.linalg.norm(residual.reshape(-1))),
        },
        "external_forward": run_summary,
        "recertification": recertification,
        "driver_signature_sha256": driver.signature,
        "forward_run_signature_sha256": str(
            run_summary.get("signature_sha256", signature)
        ),
        "certified_parent_receiver_reference": {
            "path": (
                str(accepted_receiver_path)
                if accepted_receiver_path is not None
                else None
            ),
            "sha256": accepted_receiver_hash,
            "resolution": accepted_receiver_resolution,
        },
        "current_external_receiver": {
            **artifact_record(current_path, repo=repo),
            "bitwise_equal_to_accepted_parent": gates[
                "current_receiver_bitwise_equal_to_accepted"
            ],
        },
        "true_external_receiver": {
            **artifact_record(driver.paths["true_external"], repo=repo),
            "rerun": False,
        },
        "retained_primal": {
            "directory": str(retained_dir),
            "positions": [
                int(path.stem.split("_")[-1])
                for path in sorted(retained_dir.glob("primal_*.npz"))
            ],
            "sha256": {
                path.name: sha256_file(path)
                for path in sorted(retained_dir.glob("primal_*.npz"))
            },
            "latest_resume_checkpoint": str(checkpoint_path),
            "latest_resume_checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "receiver_operator_sha256": receiver_hash,
        "gates": gates,
        "sem3d_runs": 0,
        "full_external_forwards": 1,
        "current_invocation_external_forwards": (
            0 if args.certify_existing else 1
        ),
    }
    atomic_json(summary_path, summary)
    print(f"RESULT = {pass_result}")
    print(f"J_EXTERNAL = {objective:.17e}")
    print(f"OBJECTIVE_RELATIVE_ERROR = {relative:.17e}")
    print(f"OUTPUT = {output_dir}")


if __name__ == "__main__":
    main()
