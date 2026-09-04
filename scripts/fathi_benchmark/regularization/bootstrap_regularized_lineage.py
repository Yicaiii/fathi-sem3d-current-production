from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from scripts.exact_adjoint.s43_external_forward import load_certified_reference, sha256_file
from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    atomic_json,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.runtime_paths import resolve_path


DEFAULT_BASELINE_RUN = "fathi_s43_repro_p20_t052"
DEFAULT_NEW_RUN = "fathi_s43_repro_tv_p20_t052"
DEFAULT_RUNTIME = "configs/fathi_s43_repro_tv_p20_t052_runtime.json"
DEFAULT_ENGINE = "configs/fathi_s43_repro_tv_p20_t052_iteration_engine.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _material_hashes(material_dir: Path, material_cfg: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(material_cfg["files"][component]): sha256_file(
            material_dir / str(material_cfg["files"][component])
        )
        for component in ("kappa", "mu", "density")
    }


def _copy_material_only(
    source_material: Path,
    destination_material: Path,
    material_cfg: Mapping[str, Any],
) -> None:
    destination_material.mkdir(parents=True, exist_ok=False)
    for component in ("kappa", "mu", "density"):
        name = str(material_cfg["files"][component])
        shutil.copy2(source_material / name, destination_material / name)


def _derived_reference(
    *,
    baseline_reference: dict[str, Any],
    baseline_reference_path: Path,
    baseline_run: str,
    new_run: str,
    stage5n_summary_rel: str,
    stage5n_summary_path: Path,
    regularized_runtime_path: Path,
) -> dict[str, Any]:
    reference = json.loads(json.dumps(baseline_reference))
    if reference.get("result") != "PASS_CERTIFIED_EXTERNAL_REFERENCE_CONTRACT":
        raise RuntimeError("baseline reference is not PASS")
    if str(reference.get("run")) != baseline_run:
        raise RuntimeError("baseline reference run mismatch")

    reference["run"] = new_run
    # Keep reference_root/operator/TRUE paths anchored to the already-certified baseline.
    certification_assets = reference.setdefault("certification_assets", {})
    certification_assets["stage5n_summary"] = stage5n_summary_rel

    provenance = reference.setdefault("provenance", {})
    provenance["classification"] = (
        "DERIVED_REGULARIZED_LINEAGE_FROM_CERTIFIED_BASELINE_REFERENCE_NO_NUMERICAL_RERUN"
    )
    provenance["source_reference_manifest"] = str(baseline_reference_path)
    provenance["source_reference_manifest_sha256"] = sha256_file(
        baseline_reference_path
    )
    provenance["derived_runtime_config"] = str(regularized_runtime_path)
    provenance["derived_runtime_config_sha256"] = sha256_file(
        regularized_runtime_path
    )
    provenance["stage5n_summary"] = str(stage5n_summary_path)
    provenance["stage5n_summary_sha256"] = sha256_file(stage5n_summary_path)
    provenance["immutable_operator_and_true_assets_reused"] = True
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--runtime-root")
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME)
    parser.add_argument("--engine-config", default=DEFAULT_ENGINE)
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--new-run", default=DEFAULT_NEW_RUN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    runtime_root = Path(
        args.runtime_root
        or os.environ.get("FATHI_RUNTIME_ROOT", str(repo))
    ).expanduser().resolve()

    runtime_path = resolve_path(args.runtime_config, base=repo)
    engine_path = resolve_path(args.engine_config, base=repo)
    runtime = _json(runtime_path)
    engine = _json(engine_path)

    if str(runtime.get("benchmark_name")) != args.new_run:
        raise RuntimeError("regularized runtime benchmark_name mismatch")
    if str(engine.get("run_id")) != args.new_run:
        raise RuntimeError("regularized engine run_id mismatch")

    paths = build_iteration_paths(
        engine,
        0,
        child_iteration=1,
        repository_root=repo,
        runtime_root=runtime_root,
    )

    baseline_parent = (
        runtime_root
        / "data"
        / "reproduction"
        / args.baseline_run
        / "iterations"
        / "iter_000"
        / "accepted"
    ).resolve()
    source_material = baseline_parent / str(engine["material"]["directory"])
    if not source_material.is_dir():
        raise RuntimeError(f"baseline iter000 material missing: {source_material}")

    destination_parent = paths.parent_accepted
    destination_material = destination_parent / str(engine["material"]["directory"])

    j0 = float(engine["optimizer"]["fixed_reproduction_scaling"]["J_ref"])
    baseline_reference_path = (
        runtime_root / "results" / args.baseline_run / "certified_external_reference.json"
    ).resolve()
    baseline_reference = _json(baseline_reference_path)

    stage5n_summary_path = (
        runtime_root
        / "results"
        / args.baseline_run
        / "iter_000_to_iter_001"
        / "current_t052_external_bridge"
        / "bundle_summary_v3.json"
    ).resolve()
    if not stage5n_summary_path.is_file():
        raise RuntimeError(f"certified iter000 objective summary missing: {stage5n_summary_path}")
    stage5n = _json(stage5n_summary_path)
    stage5n_j0 = float(stage5n["objective"]["J_external"])
    if stage5n_j0 != j0:
        raise RuntimeError(
            f"certified iter000 J0 mismatch: stage5n={stage5n_j0:.17e} J_ref={j0:.17e}"
        )

    source_hashes = _material_hashes(source_material, engine["material"])
    reference_output = (
        runtime_root / "results" / args.new_run / "certified_external_reference.json"
    ).resolve()
    # Keep this asset absolute: the iter000 parent-forward special case resolves
    # certification_assets.stage5n_summary directly from the source repo unless
    # an absolute path is supplied. This preserves source/runtime separation.
    stage5n_rel = str(stage5n_summary_path)

    report = {
        "schema_version": 1,
        "result": "PASS_REGULARIZED_LINEAGE_BOOTSTRAP_PREFLIGHT",
        "mode": "execute" if args.execute else "dry-run",
        "baseline_run": args.baseline_run,
        "run_id": args.new_run,
        "certified_data_J0": j0,
        "baseline_parent": str(baseline_parent),
        "regularized_parent": str(destination_parent),
        "regularized_results_root": str(paths.results_run_root),
        "regularized_optimizer_history": str(paths.optimizer_history),
        "optimizer_history_exists_before_bootstrap": paths.optimizer_history.exists(),
        "source_material_sha256": source_hashes,
        "baseline_reference": {
            "path": str(baseline_reference_path),
            "sha256": sha256_file(baseline_reference_path),
        },
        "stage5n_summary": {
            "path": str(stage5n_summary_path),
            "sha256": sha256_file(stage5n_summary_path),
            "J_external": stage5n_j0,
        },
        "derived_reference_output": str(reference_output),
        "numerical_runs": {
            "sem3d": 0,
            "external_forward": 0,
            "exact_reverse": 0,
            "optimizer": 0,
        },
    }

    if not args.execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if destination_parent.exists():
        accepted_summary_path = destination_parent / "accepted_summary.json"
        if not accepted_summary_path.is_file():
            raise RuntimeError("existing regularized iter000 is incomplete")
        existing = _json(accepted_summary_path)
        if existing.get("result") != accepted_model_result(0):
            raise RuntimeError("existing regularized iter000 summary is not PASS")
        if str(existing.get("run")) != args.new_run or int(existing.get("iter", -1)) != 0:
            raise RuntimeError("existing regularized iter000 identity mismatch")
        existing_hashes = _material_hashes(destination_material, engine["material"])
        if existing_hashes != source_hashes:
            raise RuntimeError("existing regularized iter000 material differs from baseline")
    else:
        destination_parent.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_parent.with_name(
            f".{destination_parent.name}.tmp.{os.getpid()}"
        )
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            temporary_material = temporary / str(engine["material"]["directory"])
            _copy_material_only(source_material, temporary_material, engine["material"])
            copied_hashes = _material_hashes(temporary_material, engine["material"])
            if copied_hashes != source_hashes:
                raise RuntimeError("copied regularized iter000 material hash mismatch")
            accepted = {
                "schema_version": 1,
                "result": accepted_model_result(0),
                "run": args.new_run,
                "run_id": args.new_run,
                "iter": 0,
                "objective": {
                    "accepted": j0,
                    "classification": (
                        "CERTIFIED_DATA_OBJECTIVE_FOR_FORWARD_REVERSE_COMPATIBILITY"
                    ),
                },
                "regularized_parent_objective": {
                    "status": "PENDING_CURRENT_PARENT_GRADIENT_AND_EQ24_WEIGHTS",
                    "statement": (
                        "The frozen-weight J_total is constructed after the iter000 "
                        "data and TV covectors determine beta_lambda and beta_mu."
                    ),
                },
                "material_sha256": copied_hashes,
                "lineage": {
                    "classification": "REGULARIZED_ITER000_BOOTSTRAP_FROM_CERTIFIED_BASELINE_MATERIAL",
                    "baseline_run": args.baseline_run,
                    "baseline_parent": str(baseline_parent),
                    "baseline_material_sha256": source_hashes,
                    "optimizer_history_reused": False,
                },
            }
            atomic_json(temporary / "accepted_summary.json", accepted)
            temporary.replace(destination_parent)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    reference_output.parent.mkdir(parents=True, exist_ok=True)
    derived = _derived_reference(
        baseline_reference=baseline_reference,
        baseline_reference_path=baseline_reference_path,
        baseline_run=args.baseline_run,
        new_run=args.new_run,
        stage5n_summary_rel=stage5n_rel,
        stage5n_summary_path=stage5n_summary_path,
        regularized_runtime_path=runtime_path,
    )
    if reference_output.exists():
        existing_reference = _json(reference_output)
        if existing_reference != derived:
            raise RuntimeError("existing derived reference conflicts")
    else:
        atomic_json(reference_output, derived)

    load_certified_reference(repo, args.new_run, reference_output)

    destination_hashes = _material_hashes(destination_material, engine["material"])
    if destination_hashes != source_hashes:
        raise RuntimeError("final regularized iter000 material differs from baseline")
    if paths.optimizer_history.exists() and any(paths.optimizer_history.iterdir()):
        raise RuntimeError("regularized optimizer history is not empty at iter000")

    report["result"] = "PASS_REGULARIZED_LINEAGE_BOOTSTRAP"
    report["accepted_summary"] = str(destination_parent / "accepted_summary.json")
    report["derived_reference_sha256"] = sha256_file(reference_output)
    report["optimizer_history_empty"] = (
        not paths.optimizer_history.exists()
        or not any(paths.optimizer_history.iterdir())
    )

    audit_path = (
        runtime_root / "results" / args.new_run / "bootstrap_regularized_lineage_audit.json"
    )
    atomic_json(audit_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
