from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.exact_adjoint.s43_external_forward import sha256_file


DEFAULT_BASELINE_RUNTIME = "configs/fathi_s43_repro_p20_t052_runtime.json"
DEFAULT_BASELINE_ENGINE = "configs/fathi_s43_repro_p20_t052_iteration_engine.json"
DEFAULT_REGULARIZATION = "configs/fathi_s43_repro_tv_p20_t052_regularization.json"
DEFAULT_OUTPUT_RUNTIME = "configs/fathi_s43_repro_tv_p20_t052_runtime.json"
DEFAULT_OUTPUT_ENGINE = "configs/fathi_s43_repro_tv_p20_t052_iteration_engine.json"
DEFAULT_NEW_RUN = "fathi_s43_repro_tv_p20_t052"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive_engine(
    baseline: Mapping[str, Any],
    *,
    new_run: str,
    regularization_config: str,
) -> dict[str, Any]:
    engine = copy.deepcopy(dict(baseline))
    baseline_run = str(engine["run_id"])
    if baseline_run == new_run:
        raise ValueError("new run_id must differ from the baseline run_id")

    namespace = engine.get("namespace")
    if not isinstance(namespace, Mapping):
        raise ValueError("iteration engine requires namespace")
    for key in ("data_run_pattern", "results_run_pattern"):
        value = str(namespace.get(key, ""))
        if "{run_id}" not in value:
            raise ValueError(f"namespace.{key} must remain run_id-driven")

    engine["run_id"] = new_run
    engine["regularization_contract"] = {
        "classification": "FATHI_TV_REGULARIZED_PARENT_FROZEN_WEIGHT_CONTRACT",
        "config": regularization_config,
        "scalar_objective": (
            "J_total = J_data + beta_lambda*Q_lambda + beta_mu*Q_mu"
        ),
        "gradient_rhs": (
            "b_total = b_data + beta_lambda*b_tv_lambda + beta_mu*b_tv_mu"
        ),
        "eq24_symbol": "beta",
        "eq9_equivalent_mapping": "R_eq9 = 2*beta",
        "weights_frozen_within_parent_line_search": True,
        "history_policy": "new run namespace; iter000 starts with empty L-BFGS history",
    }

    armijo = engine.get("external_armijo")
    if not isinstance(armijo, dict):
        raise ValueError("iteration engine requires external_armijo")
    armijo["objective"] = (
        "frozen-parent-weight regularized objective "
        "J_data + beta_lambda*Q_lambda + beta_mu*Q_mu"
    )

    return engine


def _replace_run(value: Any, baseline_run: str, new_run: str) -> Any:
    if isinstance(value, str):
        return value.replace(baseline_run, new_run)
    return value


def derive_runtime(
    baseline: Mapping[str, Any],
    *,
    baseline_run: str,
    new_run: str,
    certified_data_j0: float,
    regularization_config: str,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(baseline))
    configured_run = str(runtime.get("benchmark_name", baseline_run))
    if configured_run != baseline_run:
        raise ValueError(
            f"baseline runtime benchmark_name mismatch: {configured_run} != {baseline_run}"
        )
    runtime["benchmark_name"] = new_run

    layout = runtime.get("runtime_layout")
    if isinstance(layout, dict):
        mutable_keys = (
            "initial_parent_workspace",
            "iteration_pattern",
            "state_pattern",
            "transition_result_pattern",
        )
        for key in mutable_keys:
            if key in layout:
                before = str(layout[key])
                after = before.replace(baseline_run, new_run)
                if before == after:
                    raise ValueError(
                        f"runtime_layout.{key} does not contain the baseline run_id"
                    )
                layout[key] = after
        # TRUE observations are immutable and deliberately shared with the certified baseline.
        if "true_observed_workspace" in layout:
            layout["true_observed_workspace"] = str(layout["true_observed_workspace"])

    production = runtime.setdefault("production_objective", {})
    if not isinstance(production, dict):
        raise ValueError("production_objective must be an object")
    stale_j0 = production.get("J0")
    production.update(
        {
            "classification": "FATHI_TV_REGULARIZED_T052_PARENT_FROZEN_WEIGHTS",
            "J0": float(certified_data_j0),
            "data_J0_certified": float(certified_data_j0),
            "baseline_runtime_J0_before_derivation": stale_j0,
            "regularization_config": regularization_config,
            "total_objective": (
                "J_total = J_data + beta_lambda*Q_lambda + beta_mu*Q_mu"
            ),
            "weight_policy": "Fathi Eq.24; beta frozen within each parent line search",
        }
    )

    objective = runtime.setdefault("objective", {})
    if isinstance(objective, dict):
        objective["total"] = (
            "J_total = J_data + beta_lambda*Q_lambda + beta_mu*Q_mu"
        )
        objective["regularization_scalar"] = (
            "J_reg = beta_lambda*Q_lambda + beta_mu*Q_mu"
        )
        objective["eq9_equivalent_mapping"] = "R_eq9 = 2*beta"

    new_only = runtime.get("new_only_runtime_contract")
    if isinstance(new_only, dict):
        if "active_run_namespace" in new_only:
            new_only["active_run_namespace"] = new_run
        if "J0" in new_only:
            new_only["J0"] = float(certified_data_j0)

    runtime["regularized_lineage"] = {
        "classification": "NEW_REGULARIZED_LINEAGE_DO_NOT_OVERWRITE_BASELINE",
        "baseline_run": baseline_run,
        "run_id": new_run,
        "certified_data_J0": float(certified_data_j0),
        "regularization_config": regularization_config,
        "baseline_data_results_are_immutable": True,
        "optimizer_history_reuse_from_baseline": False,
    }
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--baseline-runtime", default=DEFAULT_BASELINE_RUNTIME)
    parser.add_argument("--baseline-engine", default=DEFAULT_BASELINE_ENGINE)
    parser.add_argument("--regularization-config", default=DEFAULT_REGULARIZATION)
    parser.add_argument("--output-runtime", default=DEFAULT_OUTPUT_RUNTIME)
    parser.add_argument("--output-engine", default=DEFAULT_OUTPUT_ENGINE)
    parser.add_argument("--new-run", default=DEFAULT_NEW_RUN)
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    baseline_runtime_path = (repo / args.baseline_runtime).resolve()
    baseline_engine_path = (repo / args.baseline_engine).resolve()
    regularization_path = (repo / args.regularization_config).resolve()
    output_runtime_path = (repo / args.output_runtime).resolve()
    output_engine_path = (repo / args.output_engine).resolve()

    baseline_runtime = _json(baseline_runtime_path)
    baseline_engine = _json(baseline_engine_path)
    baseline_run = str(baseline_engine["run_id"])
    if str(baseline_runtime.get("benchmark_name")) != baseline_run:
        raise RuntimeError("baseline runtime/engine run identity mismatch")
    if not regularization_path.is_file():
        raise RuntimeError(f"regularization config missing: {regularization_path}")

    scaling = baseline_engine["optimizer"]["fixed_reproduction_scaling"]
    certified_j0 = float(scaling["J_ref"])
    if int(scaling["J_ref_iteration"]) != 0:
        raise RuntimeError("fixed reproduction J_ref must belong to iter000")

    relative_reg = regularization_path.relative_to(repo).as_posix()
    engine = derive_engine(
        baseline_engine,
        new_run=args.new_run,
        regularization_config=relative_reg,
    )
    runtime = derive_runtime(
        baseline_runtime,
        baseline_run=baseline_run,
        new_run=args.new_run,
        certified_data_j0=certified_j0,
        regularization_config=relative_reg,
    )

    if output_runtime_path.exists() or output_engine_path.exists():
        raise FileExistsError(
            "derived config already exists; remove it explicitly only after inspecting provenance"
        )

    _atomic_json(output_runtime_path, runtime)
    _atomic_json(output_engine_path, engine)

    report = {
        "result": "PASS_DERIVED_REGULARIZED_CONFIGS",
        "baseline_run": baseline_run,
        "new_run": args.new_run,
        "certified_data_J0": certified_j0,
        "baseline_runtime_config_sha256": sha256_file(baseline_runtime_path),
        "baseline_engine_config_sha256": sha256_file(baseline_engine_path),
        "regularization_config_sha256": sha256_file(regularization_path),
        "output_runtime": str(output_runtime_path),
        "output_engine": str(output_engine_path),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
