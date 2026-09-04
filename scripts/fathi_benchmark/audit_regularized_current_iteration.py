from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.fathi_benchmark.current_pipeline_contracts import (
    accepted_model_result,
    optimizer_direction_result,
    registered_gradient_result,
    retained_primal_result,
    sha256_file,
)
from scripts.fathi_benchmark.iteration_context import build_iteration_paths
from scripts.fathi_benchmark.runtime_paths import (
    iteration_runtime_paths,
    resolve_path,
)


DEFAULT_RUNTIME = "configs/fathi_s43_repro_tv_p20_t052_runtime.json"
DEFAULT_ENGINE = "configs/fathi_s43_repro_tv_p20_t052_iteration_engine.json"


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {source}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--parent-iteration", "--k", type=int, required=True)
    parser.add_argument("--runtime-config", default=DEFAULT_RUNTIME)
    parser.add_argument("--engine-config", default=DEFAULT_ENGINE)
    args = parser.parse_args()

    if args.parent_iteration != 0:
        parser.error("Gate 4C closure audit currently certifies only iter000 -> iter001")

    repo = Path(args.repo).expanduser().resolve()
    runtime_path = resolve_path(args.runtime_config, base=repo)
    engine_path = resolve_path(args.engine_config, base=repo)
    runtime = _json(runtime_path)
    engine = _json(engine_path)
    parent = int(args.parent_iteration)
    child = parent + 1

    runtime_paths = iteration_runtime_paths(
        runtime,
        parent,
        repo_root=repo,
    )
    paths = build_iteration_paths(
        engine,
        parent,
        child_iteration=child,
        repository_root=repo,
        runtime_root=runtime_paths["runtime_root"],
    )

    parent_forward_path = (
        paths.exact_reverse / "primal_forward" / "summary.json"
    )
    registered_path = paths.gradient_root / "registered_gradient.json"
    direction_path = (
        paths.optimizer_root
        / f"iter_{parent:03d}_lbfgs_eq25_direction"
        / "direction_summary.json"
    )
    parent_obj_path = (
        paths.gradient_root / "regularized_parent_objective.json"
    )
    armijo_path = paths.line_search_root / "armijo_summary.json"
    child_path = paths.child_accepted / "accepted_summary.json"

    parent_forward = _json(parent_forward_path)
    registered = _json(registered_path)
    direction = _json(direction_path)
    parent_obj = _json(parent_obj_path)
    armijo = _json(armijo_path)
    accepted = _json(child_path)

    _require(
        parent_forward.get("result") == retained_primal_result(parent),
        "regularized parent forward result mismatch",
    )
    _require(
        parent_forward.get("objective", {}).get(
            "certified_quadrature_dt_source"
        )
        == "reference.contract.dt",
        "parent data objective did not use certified reference dt",
    )
    _require(
        parent_forward.get("recertification", {}).get("root_cause")
        == "DRIVER_DT_ONE_ULP_BELOW_CERTIFIED_OBJECTIVE_DT",
        "Gate4C dt root cause was not preserved in parent-forward provenance",
    )

    _require(
        registered.get("result") == registered_gradient_result(parent),
        "registered total gradient result mismatch",
    )
    _require(
        registered.get("gradient_classification")
        == "FATHI_TV_TOTAL_PHYSICAL_GRADIENT",
        "registered gradient is not TV-total",
    )
    _require(
        direction.get("result") == optimizer_direction_result(parent),
        "optimizer direction result mismatch",
    )
    _require(
        float(direction["mtilde_slope"]) < 0.0,
        "regularized direction is not a descent direction",
    )
    _require(
        parent_obj.get("result")
        == "PASS_FATHI_TV_REGULARIZED_PARENT_OBJECTIVE",
        "parent regularized objective is not PASS",
    )
    _require(
        armijo.get("result")
        == "PASS_FATHI_TV_REGULARIZED_ARMIJO_ACCEPTED",
        "regularized Armijo result mismatch",
    )
    _require(
        armijo.get("accepted") is True,
        "regularized Armijo accepted flag is false",
    )
    _require(
        accepted.get("result") == accepted_model_result(child),
        "accepted child result mismatch",
    )

    trial_record = armijo["accepted_trial"]
    trial_path = Path(str(trial_record["path"])).expanduser()
    if not trial_path.is_absolute():
        trial_path = repo / trial_path
    trial_path = trial_path.resolve()
    _require(
        sha256_file(trial_path) == str(trial_record["sha256"]),
        "accepted trial SHA mismatch",
    )
    trial = _json(trial_path)

    _require(
        trial.get("accepted") is True,
        "accepted trial flag is false",
    )
    _require(
        float(trial["candidate_total_objective"])
        <= float(trial["armijo_rhs_total"]),
        "accepted total objective violates Armijo",
    )
    _require(
        float(trial["candidate_total_objective"])
        < float(trial["parent_total_objective"]),
        "accepted total objective is not strict descent",
    )
    _require(
        math.isclose(
            float(trial["beta_lambda_frozen"]),
            float(parent_obj["beta_lambda"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "lambda beta was not frozen",
    )
    _require(
        math.isclose(
            float(trial["beta_mu_frozen"]),
            float(parent_obj["beta_mu"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "mu beta was not frozen",
    )
    _require(
        math.isclose(
            float(accepted["objective"]["accepted"]),
            float(trial["candidate_data_objective"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "accepted child data objective differs from certified candidate data objective",
    )
    _require(
        math.isclose(
            float(
                accepted[
                    "regularized_acceptance_under_parent_weights"
                ]["accepted_total"]
            ),
            float(trial["candidate_total_objective"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "accepted child regularized total differs from accepted trial",
    )

    history_files = []
    if paths.optimizer_history.exists():
        history_files = [
            str(p)
            for p in paths.optimizer_history.rglob("*")
            if p.is_file()
        ]
    _require(
        not history_files,
        "iter000 regularized transition consumed/persisted unexpected old history",
    )

    report = {
        "schema_version": 1,
        "result": "PASS_FATHI_TV_REGULARIZED_ITER000_TO_ITER001_FINAL_CLOSURE_AUDIT",
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": child,
        "transition": paths.identity.transition_id,
        "accepted_alpha": float(trial["alpha"]),
        "parent_forward_dt_root_cause": parent_forward[
            "recertification"
        ],
        "parent_data_J": float(parent_obj["J_data"]),
        "parent_total_J": float(parent_obj["J_total"]),
        "accepted_data_J": float(trial["candidate_data_objective"]),
        "accepted_total_J": float(trial["candidate_total_objective"]),
        "beta_lambda": float(parent_obj["beta_lambda"]),
        "beta_mu": float(parent_obj["beta_mu"]),
        "mtilde_slope": float(direction["mtilde_slope"]),
        "history_count": 0,
        "hashes": {
            "parent_forward_summary_sha256": sha256_file(
                parent_forward_path
            ),
            "registered_gradient_sha256": sha256_file(registered_path),
            "direction_summary_sha256": sha256_file(direction_path),
            "parent_regularized_objective_sha256": sha256_file(parent_obj_path),
            "armijo_summary_sha256": sha256_file(armijo_path),
            "accepted_trial_sha256": sha256_file(trial_path),
            "accepted_summary_sha256": sha256_file(child_path),
        },
    }

    audit_dir = paths.transition_root / "regularized_closure_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    output = audit_dir / "final_closure_audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "RESULT = "
        "PASS_FATHI_TV_REGULARIZED_ITER000_TO_ITER001_FINAL_CLOSURE_AUDIT"
    )
    print(f"PARENT_DATA_J = {report['parent_data_J']:.17e}")
    print(f"PARENT_TOTAL_J = {report['parent_total_J']:.17e}")
    print(f"ACCEPTED_DATA_J = {report['accepted_data_J']:.17e}")
    print(f"ACCEPTED_TOTAL_J = {report['accepted_total_J']:.17e}")
    print(f"ACCEPTED_ALPHA = {report['accepted_alpha']:.17g}")
    print("HISTORY_COUNT = 0")
    print(f"OUTPUT = {output}")


if __name__ == "__main__":
    main()
