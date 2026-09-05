from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from scripts.fathi_benchmark.current_pipeline_artifacts import (
    CandidateEvaluation,
    generate_raw_alpha_candidate,
    evaluate_candidate_external,
)
from scripts.fathi_benchmark.certified_data_objective import (
    certified_data_objective,
    float64_diagnostic,
)
from scripts.fathi_benchmark.current_pipeline_contracts import (
    SCHEMA_VERSION,
    accepted_model_result,
    artifact_record,
    atomic_json,
    canonical_sha256,
    candidate_generated_result,
    optimizer_direction_result,
    promotion_result,
    registered_gradient_result,
    require_identity,
    require_result,
    sha256_file,
    verify_artifact_record,
)
from scripts.fathi_benchmark.external_armijo import (
    ArmijoParameters,
    armijo_decision,
)
from scripts.fathi_benchmark.iteration_context import IterationPaths
from scripts.fathi_benchmark.regularization.tv_weight import (
    EQ24_ACTIVE_STATUS,
    EQ24_DEFERRED_FLAT_STATUS,
    EQ24_FLAT_PARENT_POLICY_VERSION,
    SECANT_PAIR_CURVATURE_POLICY_VERSION,
    evaluate_secant_pair_curvature,
)


REGULARIZED_EQ21_RESULT = "PASS_FATHI_TV_GATE3_EQ21_CONTROL_ASSEMBLY"
REGULARIZED_PARENT_OBJECTIVE_RESULT = "PASS_FATHI_TV_REGULARIZED_PARENT_OBJECTIVE"
REGULARIZED_ARMIJO_READY_RESULT = "PASS_FATHI_TV_REGULARIZED_ARMIJO_READY"
REGULARIZED_ARMIJO_ACCEPTED_RESULT = "PASS_FATHI_TV_REGULARIZED_ARMIJO_ACCEPTED"
REGULARIZED_ARMIJO_REJECTED_RESULT = "BLOCK_FATHI_TV_REGULARIZED_ARMIJO_NO_ACCEPTED_TRIAL"


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {source}")
    return value


def _record_for_final(
    temporary_path: Path,
    final_path: Path,
    *,
    repo: Path,
) -> dict[str, str]:
    try:
        recorded = str(final_path.resolve().relative_to(repo.resolve()))
    except ValueError:
        recorded = str(final_path.resolve())
    return {
        "path": recorded,
        "sha256": sha256_file(temporary_path),
    }


def compose_frozen_regularized_objective(
    *,
    data_objective: float,
    q_lambda: float,
    q_mu: float,
    beta_lambda: float,
    beta_mu: float,
) -> dict[str, float]:
    values = (
        data_objective,
        q_lambda,
        q_mu,
        beta_lambda,
        beta_mu,
    )
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("regularized objective inputs must be finite")
    if data_objective < 0.0 or q_lambda < 0.0 or q_mu < 0.0:
        raise ValueError("objective/TV values must be non-negative")
    if beta_lambda < 0.0 or beta_mu < 0.0:
        raise ValueError("effective Eq.24 beta weights must be non-negative")

    j_reg_lambda = float(beta_lambda * q_lambda)
    j_reg_mu = float(beta_mu * q_mu)
    j_reg = float(j_reg_lambda + j_reg_mu)
    j_total = float(data_objective + j_reg)
    return {
        "J_data": float(data_objective),
        "Q_lambda": float(q_lambda),
        "Q_mu": float(q_mu),
        "beta_lambda": float(beta_lambda),
        "beta_mu": float(beta_mu),
        "J_reg_lambda": j_reg_lambda,
        "J_reg_mu": j_reg_mu,
        "J_reg": j_reg,
        "J_total": j_total,
    }



def require_eq24_status_beta_contract(
    eq21: Mapping[str, Any],
) -> dict[str, str]:
    """Second-line validation of Eq24 status, beta, and flatness evidence."""

    if eq21.get("eq24_policy_version") != EQ24_FLAT_PARENT_POLICY_VERSION:
        raise ValueError("Eq.24 policy version mismatch")

    varrho = float(eq21["varrho"])
    statuses: dict[str, str] = {}
    for field in ("lambda", "mu"):
        rec = eq21["eq24"][field]
        status = str(rec.get("status"))
        beta = float(rec["beta_eq21"])
        ratio = float(rec["weighted_reg_over_data_l2"])
        grad_gate = bool(rec.get("gradient_gate_pass"))
        q_gate = bool(rec.get("Q_floor_gate_pass"))

        if status == EQ24_DEFERRED_FLAT_STATUS:
            if not (grad_gate and q_gate):
                raise ValueError(
                    f"{field} deferred Eq.24 lacks dual flat-parent gates"
                )
            if beta != 0.0 or ratio != 0.0:
                raise ValueError(
                    f"{field} deferred Eq.24 must have beta=ratio=0"
                )
        elif status == EQ24_ACTIVE_STATUS:
            if grad_gate and q_gate:
                raise ValueError(
                    f"{field} active Eq.24 conflicts with dual flat-parent gates"
                )
            if beta <= 0.0:
                raise ValueError(f"{field} active Eq.24 beta must be positive")
            if abs(ratio - varrho) > 1.0e-12:
                raise ValueError(
                    f"{field} active Eq.24 norm ratio differs from varrho"
                )
        else:
            raise ValueError(f"unsupported Eq.24 status for {field}: {status}")
        statuses[field] = status

    return statuses


def regularized_secant_pair_outcome(
    s: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Durable history outcome: use iff s^T y > 0, otherwise explicit skip."""

    outcome = evaluate_secant_pair_curvature(s, y)
    return {
        "policy_version": outcome.policy_version,
        "s_dot_y": outcome.s_dot_y,
        "s_l2": outcome.s_l2,
        "y_l2": outcome.y_l2,
        "normalized_curvature": outcome.normalized_curvature,
        "use_in_lbfgs_history": outcome.use_in_lbfgs_history,
        "status": outcome.status,
    }




BOOTSTRAP_PARENT_CLASSIFICATION = (
    "REGULARIZED_ITER000_BOOTSTRAP_FROM_CERTIFIED_BASELINE_MATERIAL"
)


def require_regularized_parent_identity(
    manifest: Mapping[str, Any],
    paths: IterationPaths,
    *,
    label: str,
) -> str:
    """Validate either a promoted transition identity or the certified iter000 bootstrap."""

    identity_fields = ("parent_iteration", "child_iteration", "transition")
    present = [field in manifest for field in identity_fields]

    if all(present):
        require_identity(manifest, paths, label=label)
        return "TRANSITION_IDENTITY"

    if any(present):
        raise ValueError(
            f"{label} has partial transition identity; bootstrap summaries "
            "must omit all transition fields"
        )

    if paths.identity.parent_iteration != 0:
        raise ValueError(
            f"{label} lacks transition identity outside iter000 bootstrap"
        )

    if manifest.get("run_id") != paths.identity.run_id:
        raise ValueError(f"{label} bootstrap run_id mismatch")

    run_alias = manifest.get("run")
    if run_alias is not None and run_alias != paths.identity.run_id:
        raise ValueError(f"{label} bootstrap run alias mismatch")

    if int(manifest.get("iter", -1)) != 0:
        raise ValueError(f"{label} bootstrap iter mismatch")

    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(f"{label} bootstrap lineage missing")

    if lineage.get("classification") != BOOTSTRAP_PARENT_CLASSIFICATION:
        raise ValueError(f"{label} bootstrap lineage classification mismatch")

    if lineage.get("optimizer_history_reused") is not False:
        raise ValueError(f"{label} bootstrap optimizer-history provenance mismatch")

    material_sha = manifest.get("material_sha256")
    if not isinstance(material_sha, Mapping) or not material_sha:
        raise ValueError(f"{label} bootstrap material SHA provenance missing")

    return "CERTIFIED_ITER000_BOOTSTRAP_IDENTITY"


def register_regularized_gradient(
    *,
    repo: str | Path,
    paths: IterationPaths,
    eq21_directory: str | Path,
) -> Path:
    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    eq21_dir = Path(eq21_directory).expanduser().resolve()
    eq21_summary_path = eq21_dir / "gate3_eq21_summary.json"
    eq21 = _json(eq21_summary_path)
    if eq21.get("result") != REGULARIZED_EQ21_RESULT:
        raise RuntimeError("Eq.21 regularized assembly is not PASS")
    eq24_statuses = require_eq24_status_beta_contract(eq21)

    parent_summary_path = paths.parent_accepted / "accepted_summary.json"
    parent_summary = _json(parent_summary_path)
    require_result(
        parent_summary,
        accepted_model_result(parent),
        label="regularized accepted parent",
    )
    parent_identity_mode = require_regularized_parent_identity(
        parent_summary,
        paths,
        label="regularized accepted parent",
    )

    bridge_summary_path = paths.gradient_root / "summary.json"
    reverse_summary_path = (
        paths.exact_reverse / "production_reverse" / "summary.json"
    )
    mtilde_summary_path = (
        paths.gradient_root / "mtilde_solve" / "mtilde_gradient_summary.json"
    )

    files = {
        "lambda": eq21_dir / "g_total_lambda.npy",
        "mu": eq21_dir / "g_total_mu.npy",
        "coordinates": eq21_dir / "active_coords.npy",
        "active_indices": paths.gradient_root / "mtilde_active_full_indices.npy",
        "active_h5_indices": paths.gradient_root / "active_h5_indices.npy",
        "mtilde": (
            paths.gradient_root
            / "mtilde_solve"
            / "Mtilde_interior_sparse.npz"
        ),
    }
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"regularized gradient {name}: {path}")

    records = {
        name: artifact_record(path, repo=root)
        for name, path in files.items()
    }

    material_sha = parent_summary.get("material_sha256")
    if not isinstance(material_sha, Mapping) or not material_sha:
        raise ValueError("regularized accepted parent lacks material SHA provenance")

    signature_payload = {
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "parent_identity_mode": parent_identity_mode,
        "parent_summary": artifact_record(parent_summary_path, repo=root),
        "bridge_summary": artifact_record(bridge_summary_path, repo=root),
        "reverse_summary": artifact_record(reverse_summary_path, repo=root),
        "mtilde_summary": artifact_record(mtilde_summary_path, repo=root),
        "eq21_summary": artifact_record(eq21_summary_path, repo=root),
        "lambda": records["lambda"],
        "mu": records["mu"],
        "active_indices": records["active_indices"],
        "active_h5_indices": records["active_h5_indices"],
        "coordinates": records["coordinates"],
        "mtilde": records["mtilde"],
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": registered_gradient_result(parent),
        "run_id": paths.identity.run_id,
        "iteration": parent,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "lambda": records["lambda"],
        "mu": records["mu"],
        "active_indices": records["active_indices"],
        "active_h5_indices": records["active_h5_indices"],
        "coordinates": records["coordinates"],
        "mtilde": records["mtilde"],
        "ordering": "canonical active-control order",
        "units": "physical Pa-space Riesz gradient",
        "gradient_classification": "FATHI_TV_TOTAL_PHYSICAL_GRADIENT",
        "parent_accepted_model": {
            "summary": artifact_record(parent_summary_path, repo=root),
            "result": accepted_model_result(parent),
            "identity_mode": parent_identity_mode,
            "material_sha256": dict(material_sha),
        },
        "source_bridge_summary": artifact_record(
            bridge_summary_path, repo=root
        ),
        "source_reverse_summary": artifact_record(
            reverse_summary_path, repo=root
        ),
        "source_mtilde_summary": artifact_record(
            mtilde_summary_path, repo=root
        ),
        "regularization": {
            "eq21_summary": artifact_record(
                eq21_summary_path, repo=root
            ),
            "varrho": float(eq21["varrho"]),
            "beta_lambda_eq21": float(
                eq21["eq24"]["lambda"]["beta_eq21"]
            ),
            "beta_mu_eq21": float(
                eq21["eq24"]["mu"]["beta_eq21"]
            ),
            "Q_lambda": float(eq21["tv_chain_rule"]["lambda_Q"]),
            "Q_mu": float(eq21["tv_chain_rule"]["mu_Q"]),
            "scalar_convention": (
                "J_reg = beta_lambda*Q_lambda + beta_mu*Q_mu"
            ),
            "eq9_equivalent_mapping": "R_eq9 = 2*beta",
            "eq24_policy_version": eq21["eq24_policy_version"],
            "eq24_lambda_status": eq24_statuses["lambda"],
            "eq24_mu_status": eq24_statuses["mu"],
            "next_secant_pair_policy": SECANT_PAIR_CURVATURE_POLICY_VERSION,
        },
        "registration_signature_sha256": canonical_sha256(
            signature_payload
        ),
    }

    output = paths.gradient_root / "registered_gradient.json"
    if output.is_file():
        existing = _json(output)
        if existing != manifest:
            raise ValueError("existing regularized registered gradient conflicts")
        return output
    atomic_json(output, manifest)
    return output


def persist_parent_regularized_objective(
    *,
    repo: str | Path,
    paths: IterationPaths,
    eq21_directory: str | Path,
    registered_gradient_path: str | Path,
) -> Path:
    root = Path(repo).expanduser().resolve()
    eq21_dir = Path(eq21_directory).expanduser().resolve()
    eq21_summary_path = eq21_dir / "gate3_eq21_summary.json"
    eq21 = _json(eq21_summary_path)
    if eq21.get("result") != REGULARIZED_EQ21_RESULT:
        raise RuntimeError("Eq.21 regularized assembly is not PASS")
    eq24_statuses = require_eq24_status_beta_contract(eq21)

    accepted_path = paths.parent_accepted / "accepted_summary.json"
    accepted = _json(accepted_path)
    require_result(
        accepted,
        accepted_model_result(paths.identity.parent_iteration),
        label="regularized objective accepted parent",
    )
    parent_identity_mode = require_regularized_parent_identity(
        accepted,
        paths,
        label="regularized objective accepted parent",
    )
    data_objective = float(accepted["objective"]["accepted"])

    values = compose_frozen_regularized_objective(
        data_objective=data_objective,
        q_lambda=float(eq21["tv_chain_rule"]["lambda_Q"]),
        q_mu=float(eq21["tv_chain_rule"]["mu_Q"]),
        beta_lambda=float(eq21["eq24"]["lambda"]["beta_eq21"]),
        beta_mu=float(eq21["eq24"]["mu"]["beta_eq21"]),
    )

    payload = {
        "schema_version": 1,
        "result": REGULARIZED_PARENT_OBJECTIVE_RESULT,
        "run_id": paths.identity.run_id,
        "parent_iteration": paths.identity.parent_iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        **values,
        "weights_frozen_within_line_search": True,
        "accepted_parent_identity_mode": parent_identity_mode,
        "scalar_convention": (
            "J_total = J_data + beta_lambda*Q_lambda + beta_mu*Q_mu"
        ),
        "gradient_convention": (
            "b_total = b_data + beta_lambda*b_tv_lambda + beta_mu*b_tv_mu"
        ),
        "eq9_equivalent_mapping": "R_eq9 = 2*beta",
        "eq24_policy_version": eq21["eq24_policy_version"],
        "eq24_lambda_status": eq24_statuses["lambda"],
        "eq24_mu_status": eq24_statuses["mu"],
        "next_secant_pair_policy": SECANT_PAIR_CURVATURE_POLICY_VERSION,
        "accepted_parent_summary": artifact_record(
            accepted_path, repo=root
        ),
        "registered_total_gradient": artifact_record(
            registered_gradient_path, repo=root
        ),
        "eq21_summary": artifact_record(eq21_summary_path, repo=root),
    }
    payload["input_signature_sha256"] = canonical_sha256(
        {
            k: payload[k]
            for k in (
                "run_id",
                "parent_iteration",
                "transition",
                "J_data",
                "Q_lambda",
                "Q_mu",
                "beta_lambda",
                "beta_mu",
                "accepted_parent_summary",
                "registered_total_gradient",
                "eq21_summary",
            )
        }
    )

    output = paths.gradient_root / "regularized_parent_objective.json"
    if output.is_file():
        existing = _json(output)
        if existing != payload:
            raise ValueError("existing parent regularized objective conflicts")
        return output
    atomic_json(output, payload)
    return output


@dataclass(frozen=True)
class RegularizedCandidateEvaluation:
    external: CandidateEvaluation
    tv_gate_record: Mapping[str, Any]
    data_objective_record: Mapping[str, Any]
    objective: Mapping[str, float]


def evaluate_candidate_regularized(
    *,
    repo: str | Path,
    paths: IterationPaths,
    runtime_config: Mapping[str, Any],
    runtime_config_path: str | Path,
    regularization_config_path: str | Path,
    reference_manifest: str | Path,
    candidate_summary_path: str | Path,
    trial_directory: str | Path,
    beta_lambda: float,
    beta_mu: float,
    batch_size: int,
    checkpoint_interval: int,
) -> RegularizedCandidateEvaluation:
    root = Path(repo).expanduser().resolve()
    candidate_path = Path(candidate_summary_path).expanduser().resolve()
    trial_dir = Path(trial_directory).expanduser().resolve()

    external = evaluate_candidate_external(
        repo=root,
        paths=paths,
        runtime_config=runtime_config,
        reference_manifest=reference_manifest,
        candidate_summary_path=candidate_path,
        trial_directory=trial_dir,
        batch_size=int(batch_size),
        checkpoint_interval=int(checkpoint_interval),
    )

    material_dir = candidate_path.parent / str(
        runtime_config.get("material_directory", "mat/h5")
    )
    if not material_dir.is_dir():
        material_dir = candidate_path.parent / "mat" / "h5"
    if not material_dir.is_dir():
        raise RuntimeError(f"candidate material directory missing: {material_dir}")

    tv_gate_path = trial_dir / "candidate_tv_gate.json"
    if not tv_gate_path.is_file():
        command = [
            sys.executable,
            "-m",
            "scripts.fathi_benchmark.regularization.run_tv_gate",
            "--runtime-config",
            str(Path(runtime_config_path).expanduser().resolve()),
            "--regularization-config",
            str(Path(regularization_config_path).expanduser().resolve()),
            "--material-dir",
            str(material_dir),
            "--output",
            str(tv_gate_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(root),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"candidate TV evaluation failed with rc={completed.returncode}"
            )

    tv_gate = _json(tv_gate_path)
    if tv_gate.get("result") != "PASS_FATHI_TV_PHASE1_GATE":
        raise RuntimeError("candidate TV gate is not PASS")

    reference_payload = _json(reference_manifest)
    certified_dt = float(reference_payload["contract"]["dt"])
    current_receiver_path = verify_artifact_record(
        root,
        external.current_receiver,
        label="regularized candidate current receiver",
    )
    true_receiver_path = verify_artifact_record(
        root,
        external.true_receiver,
        label="regularized candidate TRUE receiver",
    )
    current_receiver = np.asarray(
        np.load(current_receiver_path), dtype=np.float64
    )
    true_receiver = np.asarray(
        np.load(true_receiver_path), dtype=np.float64
    )
    data_eval = certified_data_objective(
        current_receiver,
        true_receiver,
        certified_dt=certified_dt,
        driver_dt=float(external.dt),
    )
    data_diag_path = trial_dir / "candidate_data_objective.json"
    data_diag = {
        "schema_version": 1,
        "result": "PASS_CERTIFIED_CANDIDATE_DATA_OBJECTIVE",
        "run_id": paths.identity.run_id,
        "parent_iteration": paths.identity.parent_iteration,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "candidate_summary": artifact_record(candidate_path, repo=root),
        "current_receiver": dict(external.current_receiver),
        "true_receiver": dict(external.true_receiver),
        "objective": data_eval.diagnostic(),
        "legacy_driver_dt_objective": float64_diagnostic(
            float(external.objective)
        ),
        "canonical_quadrature_dt_source": "reference.contract.dt",
    }
    atomic_json(data_diag_path, data_diag)

    q_lambda = float(
        tv_gate["parent_material_audit"]["lambda"]["value"]
    )
    q_mu = float(
        tv_gate["parent_material_audit"]["mu"]["value"]
    )
    objective = compose_frozen_regularized_objective(
        data_objective=float(data_eval.value),
        q_lambda=q_lambda,
        q_mu=q_mu,
        beta_lambda=float(beta_lambda),
        beta_mu=float(beta_mu),
    )
    return RegularizedCandidateEvaluation(
        external=external,
        tv_gate_record=artifact_record(tv_gate_path, repo=root),
        data_objective_record=artifact_record(data_diag_path, repo=root),
        objective=objective,
    )


def execute_regularized_armijo(
    *,
    repo: str | Path,
    paths: IterationPaths,
    engine_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    runtime_config_path: str | Path,
    regularization_config_path: str | Path,
    reference_manifest: str | Path,
    accepted_parent_record: Mapping[str, Any],
    registered_gradient_record: Mapping[str, Any],
    direction_record: Mapping[str, Any],
    parent_objective_record: Mapping[str, Any],
    batch_size: int,
    checkpoint_interval: int,
) -> Path:
    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    parent_obj_path = verify_artifact_record(
        root,
        parent_objective_record,
        label="regularized parent objective",
    )
    parent_obj = _json(parent_obj_path)
    if parent_obj.get("result") != REGULARIZED_PARENT_OBJECTIVE_RESULT:
        raise ValueError("regularized parent objective is not PASS")

    direction_path = verify_artifact_record(
        root,
        direction_record,
        label="regularized optimizer direction",
    )
    direction = _json(direction_path)
    require_result(
        direction,
        optimizer_direction_result(parent),
        label="regularized optimizer direction",
    )
    require_identity(direction, paths, label="regularized optimizer direction")

    parameters = ArmijoParameters.from_config(engine_config)
    armijo_input = {
        "schema_version": 1,
        "result": REGULARIZED_ARMIJO_READY_RESULT,
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "parent_total_objective": float(parent_obj["J_total"]),
        "parent_data_objective": float(parent_obj["J_data"]),
        "slope": float(direction["mtilde_slope"]),
        "beta_lambda": float(parent_obj["beta_lambda"]),
        "beta_mu": float(parent_obj["beta_mu"]),
        "weights_frozen_within_line_search": True,
        "parameters": {
            "c1": float(parameters.c1),
            "rho": float(parameters.rho),
            "alpha0": float(parameters.alpha0),
            "maximum_backtracks": int(parameters.maximum_backtracks),
        },
        "parent_accepted_artifact": dict(accepted_parent_record),
        "gradient_artifact": dict(registered_gradient_record),
        "direction_artifact": dict(direction_record),
        "parent_regularized_objective": dict(parent_objective_record),
    }
    armijo_input["input_signature_sha256"] = canonical_sha256(
        armijo_input
    )
    armijo_input_path = paths.line_search_root / "armijo_input.json"
    if armijo_input_path.is_file():
        existing = _json(armijo_input_path)
        if existing != armijo_input:
            raise ValueError("existing regularized Armijo input conflicts")
    else:
        atomic_json(armijo_input_path, armijo_input)

    summary_path = paths.line_search_root / "armijo_summary.json"
    if summary_path.is_file():
        existing = _json(summary_path)
        if (
            existing.get("input_signature_sha256")
            == armijo_input["input_signature_sha256"]
            and existing.get("accepted") is True
        ):
            return summary_path
        raise ValueError("existing regularized Armijo summary conflicts")

    trial_records: list[dict[str, Any]] = []
    accepted_trial: dict[str, Any] | None = None

    for trial_index, alpha in parameters.schedule():
        candidate_path = generate_raw_alpha_candidate(
            repo=root,
            paths=paths,
            material_config=engine_config["material"],
            accepted_parent_record=accepted_parent_record,
            direction_record=direction_record,
            parameters=parameters,
            trial_index=int(trial_index),
            alpha=float(alpha),
        )
        candidate = _json(candidate_path)
        require_result(
            candidate,
            candidate_generated_result(parent, int(trial_index)),
            label="regularized candidate",
        )
        require_identity(candidate, paths, label="regularized candidate")

        trial_dir = (
            paths.line_search_root
            / "trials"
            / candidate_path.parent.name
        ).resolve()
        trial_dir.mkdir(parents=True, exist_ok=True)

        evaluation = evaluate_candidate_regularized(
            repo=root,
            paths=paths,
            runtime_config=runtime_config,
            runtime_config_path=runtime_config_path,
            regularization_config_path=regularization_config_path,
            reference_manifest=reference_manifest,
            candidate_summary_path=candidate_path,
            trial_directory=trial_dir,
            beta_lambda=float(parent_obj["beta_lambda"]),
            beta_mu=float(parent_obj["beta_mu"]),
            batch_size=int(batch_size),
            checkpoint_interval=int(checkpoint_interval),
        )

        decision = armijo_decision(
            parent_objective=float(parent_obj["J_total"]),
            candidate_objective=float(
                evaluation.objective["J_total"]
            ),
            slope=float(direction["mtilde_slope"]),
            alpha=float(alpha),
            c1=float(parameters.c1),
        )

        trial_payload = {
            "schema_version": 1,
            "result": (
                f"PASS_ITER{parent:03d}_FATHI_TV_ARMIJO_TRIAL_"
                f"{int(trial_index):03d}_"
                f"{'ACCEPTED' if decision['accepted'] else 'REJECTED'}"
            ),
            "run_id": paths.identity.run_id,
            "parent_iteration": parent,
            "child_iteration": paths.identity.child_iteration,
            "transition": paths.identity.transition_id,
            "trial_index": int(trial_index),
            "alpha": float(alpha),
            "accepted": bool(decision["accepted"]),
            "strict_descent": bool(decision["strict_descent"]),
            "armijo": bool(decision["armijo"]),
            "armijo_rhs_total": float(decision["armijo_rhs"]),
            "slope_total": float(direction["mtilde_slope"]),
            "parent_data_objective": float(parent_obj["J_data"]),
            "parent_total_objective": float(parent_obj["J_total"]),
            "candidate_data_objective": float(
                evaluation.objective["J_data"]
            ),
            "candidate_Q_lambda": float(
                evaluation.objective["Q_lambda"]
            ),
            "candidate_Q_mu": float(
                evaluation.objective["Q_mu"]
            ),
            "candidate_J_reg_lambda": float(
                evaluation.objective["J_reg_lambda"]
            ),
            "candidate_J_reg_mu": float(
                evaluation.objective["J_reg_mu"]
            ),
            "candidate_J_reg": float(
                evaluation.objective["J_reg"]
            ),
            "candidate_total_objective": float(
                evaluation.objective["J_total"]
            ),
            "beta_lambda_frozen": float(parent_obj["beta_lambda"]),
            "beta_mu_frozen": float(parent_obj["beta_mu"]),
            "candidate_summary": artifact_record(
                candidate_path, repo=root
            ),
            "candidate_material_signature_sha256": str(
                candidate["candidate_material_signature_sha256"]
            ),
            "candidate_receiver": dict(
                evaluation.external.current_receiver
            ),
            "true_receiver": dict(
                evaluation.external.true_receiver
            ),
            "candidate_tv_gate": dict(evaluation.tv_gate_record),
            "candidate_data_objective": dict(
                evaluation.data_objective_record
            ),
            "registered_gradient": dict(registered_gradient_record),
            "durable_direction": dict(direction_record),
            "parent_accepted_model": dict(accepted_parent_record),
            "parent_regularized_objective": dict(
                parent_objective_record
            ),
        }
        trial_payload["input_signature_sha256"] = canonical_sha256(
            trial_payload
        )
        trial_path = trial_dir / "trial_summary.json"
        atomic_json(trial_path, trial_payload)
        trial_record = artifact_record(trial_path, repo=root)
        trial_records.append(trial_record)

        if decision["accepted"]:
            accepted_trial = trial_record
            break

    accepted = accepted_trial is not None
    summary = {
        "schema_version": 1,
        "result": (
            REGULARIZED_ARMIJO_ACCEPTED_RESULT
            if accepted
            else REGULARIZED_ARMIJO_REJECTED_RESULT
        ),
        "run_id": paths.identity.run_id,
        "parent_iteration": parent,
        "child_iteration": paths.identity.child_iteration,
        "transition": paths.identity.transition_id,
        "accepted": accepted,
        "accepted_trial": accepted_trial,
        "trials": trial_records,
        "input_signature_sha256": armijo_input[
            "input_signature_sha256"
        ],
        "parent_regularized_objective": dict(
            parent_objective_record
        ),
        "registered_gradient": dict(registered_gradient_record),
        "durable_direction": dict(direction_record),
    }
    atomic_json(summary_path, summary)
    return summary_path


def promote_regularized_accepted_trial(
    *,
    repo: str | Path,
    paths: IterationPaths,
    material_config: Mapping[str, Any],
    armijo_summary_path: str | Path,
) -> Path:
    root = Path(repo).expanduser().resolve()
    parent = paths.identity.parent_iteration
    child = paths.identity.child_iteration

    armijo_path = Path(armijo_summary_path).expanduser().resolve()
    armijo = _json(armijo_path)
    if (
        armijo.get("result") != REGULARIZED_ARMIJO_ACCEPTED_RESULT
        or armijo.get("accepted") is not True
        or not armijo.get("accepted_trial")
    ):
        raise RuntimeError("regularized Armijo did not accept a candidate")

    trial_path = verify_artifact_record(
        root,
        armijo["accepted_trial"],
        label="regularized accepted trial",
    )
    trial = _json(trial_path)
    if trial.get("accepted") is not True:
        raise RuntimeError("regularized accepted trial is not accepted")

    candidate_path = verify_artifact_record(
        root,
        trial["candidate_summary"],
        label="regularized accepted candidate",
    )
    candidate = _json(candidate_path)
    require_result(
        candidate,
        candidate_generated_result(parent, int(trial["trial_index"])),
        label="regularized accepted candidate",
    )
    require_identity(candidate, paths, label="regularized accepted candidate")

    candidate_state_path = verify_artifact_record(
        root,
        candidate["candidate_state"],
        label="regularized candidate state",
    )
    source_dir = candidate_path.parent

    input_signature = canonical_sha256(
        {
            "armijo_summary": artifact_record(armijo_path, repo=root),
            "accepted_trial": dict(armijo["accepted_trial"]),
            "candidate_summary": dict(trial["candidate_summary"]),
            "candidate_material_signature_sha256": candidate[
                "candidate_material_signature_sha256"
            ],
        }
    )

    accepted_summary_path = (
        paths.child_accepted / "accepted_summary.json"
    )
    if paths.child_accepted.exists():
        existing = _json(accepted_summary_path)
        if (
            existing.get("result") != accepted_model_result(child)
            or existing.get("promotion_input_signature_sha256")
            != input_signature
        ):
            raise ValueError("existing regularized accepted child conflicts")
        return accepted_summary_path

    paths.child_accepted.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.child_accepted.with_name(
        f".{paths.child_accepted.name}.tmp.{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(temporary)

    try:
        shutil.copytree(source_dir, temporary)
        (temporary / "candidate_summary.json").unlink(
            missing_ok=True
        )

        material_records: dict[str, Any] = {}
        material_hashes: dict[str, str] = {}
        for component in ("kappa", "mu", "density"):
            filename = str(material_config["files"][component])
            source_material = (
                temporary
                / str(material_config["directory"])
                / filename
            )
            final_material = (
                paths.child_accepted
                / str(material_config["directory"])
                / filename
            )
            record = _record_for_final(
                source_material,
                final_material,
                repo=root,
            )
            expected = str(
                candidate["candidate_material"][component]["sha256"]
            )
            if record["sha256"] != expected:
                raise ValueError(
                    f"promoted regularized {component} differs from candidate"
                )
            material_records[component] = record
            material_hashes[filename] = record["sha256"]

        summary = {
            "schema_version": SCHEMA_VERSION,
            "result": accepted_model_result(child),
            "promotion_result": promotion_result(parent, child),
            "run": paths.identity.run_id,
            "run_id": paths.identity.run_id,
            "iter": child,
            "parent_iteration": parent,
            "child_iteration": child,
            "transition": paths.identity.transition_id,
            "accepted_alpha": float(trial["alpha"]),
            "objective": {
                "parent": float(trial["parent_data_objective"]),
                "accepted": float(
                    trial["candidate_data_objective"]
                ),
                "classification": (
                    "CERTIFIED_DATA_OBJECTIVE_FOR_FORWARD_REVERSE_COMPATIBILITY"
                ),
            },
            "regularized_acceptance_under_parent_weights": {
                "beta_lambda": float(
                    trial["beta_lambda_frozen"]
                ),
                "beta_mu": float(trial["beta_mu_frozen"]),
                "parent_total": float(
                    trial["parent_total_objective"]
                ),
                "accepted_Q_lambda": float(
                    trial["candidate_Q_lambda"]
                ),
                "accepted_Q_mu": float(
                    trial["candidate_Q_mu"]
                ),
                "accepted_J_reg": float(
                    trial["candidate_J_reg"]
                ),
                "accepted_total": float(
                    trial["candidate_total_objective"]
                ),
                "armijo_rhs_total": float(
                    trial["armijo_rhs_total"]
                ),
                "slope_total": float(trial["slope_total"]),
                "weights_frozen_within_line_search": True,
            },
            "parent_accepted_model": dict(
                trial["parent_accepted_model"]
            ),
            "registered_parent_gradient": dict(
                trial["registered_gradient"]
            ),
            "direction": dict(trial["durable_direction"]),
            "candidate": dict(trial["candidate_summary"]),
            "candidate_objective_trial": dict(
                armijo["accepted_trial"]
            ),
            "armijo_summary": artifact_record(
                armijo_path, repo=root
            ),
            "parent_regularized_objective": dict(
                trial["parent_regularized_objective"]
            ),
            "accepted_external_receiver": dict(
                trial["candidate_receiver"]
            ),
            "true_external_receiver": dict(
                trial["true_receiver"]
            ),
            "external_receiver_sha256": str(
                trial["candidate_receiver"]["sha256"]
            ),
            "true_external_sha256": str(
                trial["true_receiver"]["sha256"]
            ),
            "material": material_records,
            "material_sha256": material_hashes,
            "candidate_material_signature_sha256": str(
                candidate["candidate_material_signature_sha256"]
            ),
            "promotion_input_signature_sha256": input_signature,
        }
        atomic_json(temporary / "accepted_summary.json", summary)
        temporary.replace(paths.child_accepted)

        paths.child_state.parent.mkdir(parents=True, exist_ok=True)
        if paths.child_state.exists():
            if sha256_file(paths.child_state) != sha256_file(
                candidate_state_path
            ):
                raise ValueError(
                    "existing regularized child state conflicts"
                )
        else:
            state_tmp = paths.child_state.with_name(
                paths.child_state.name + ".tmp"
            )
            shutil.copy2(candidate_state_path, state_tmp)
            os.replace(state_tmp, paths.child_state)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return accepted_summary_path
