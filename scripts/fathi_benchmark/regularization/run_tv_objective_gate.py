from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.fathi_benchmark.regularization.tv_objective import (
    frozen_total_tv_objective,
    objective_convention_from_eq21_beta,
)
from scripts.fathi_benchmark.regularization.tv_q1 import assemble_smoothed_tv_q1


PA_PER_MPA = 1.0e6


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_h5(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset][...], dtype=np.float64)


def _axes(runtime: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = runtime["material_grid"]
    if grid.get("array_order") != "field[iz, iy, ix]":
        raise RuntimeError("unsupported material array ordering")
    nz, ny, nx = (int(v) for v in grid["shape"])
    xmin = tuple(float(v) for v in grid["x_min_global_m"])
    xmax = tuple(float(v) for v in grid["x_max_global_m"])
    x = np.linspace(xmin[0], xmax[0], nx)
    y = np.linspace(xmin[1], xmax[1], ny)
    z = np.linspace(xmin[2], xmax[2], nz)
    return x, y, z


def _deterministic_direction(n: int, phase: float) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=np.float64)
    d = np.sin((0.017 + phase) * i) + 0.37 * np.cos((0.013 + 0.5 * phase) * i)
    scale = float(np.max(np.abs(d)))
    if not np.isfinite(scale) or scale == 0.0:
        raise RuntimeError("failed to construct deterministic direction")
    return d / scale * PA_PER_MPA


def _relative_error(a: float, b: float) -> float:
    return float(abs(a - b) / max(abs(b), 1.0e-300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--regularization-config", required=True)
    parser.add_argument("--material-dir", required=True)
    parser.add_argument("--gate3-eq21-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_path = Path(args.runtime_config).expanduser().resolve()
    reg_path = Path(args.regularization_config).expanduser().resolve()
    material_dir = Path(args.material_dir).expanduser().resolve()
    gate3_dir = Path(args.gate3_eq21_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    runtime = _json(runtime_path)
    reg = _json(reg_path)
    gate3_summary_path = gate3_dir / "gate3_eq21_summary.json"
    gate3 = _json(gate3_summary_path)

    if reg.get("status") != "GATE_ONLY_DO_NOT_RUN_E2E":
        raise RuntimeError("Gate-3C input must remain GATE_ONLY_DO_NOT_RUN_E2E")
    if gate3.get("result") != "PASS_FATHI_TV_GATE3_EQ21_CONTROL_ASSEMBLY":
        raise RuntimeError("Gate-3B Eq.21 assembly must PASS before Gate-3C")

    x, y, z = _axes(runtime)
    shape = (z.size, y.size, x.size)
    n_full = int(np.prod(shape))
    dataset = str(runtime["material_grid"]["dataset"])
    epsilon = float(reg["regularization"]["epsilon_gradient_sq_mpa2_per_m2"])

    kappa_path = material_dir / "Mat_0_Kappa.h5"
    mu_path = material_dir / "Mat_0_Mu.h5"
    kappa_pa = _read_h5(kappa_path, dataset)
    mu_pa = _read_h5(mu_path, dataset)
    if kappa_pa.shape != shape or mu_pa.shape != shape:
        raise RuntimeError(f"material shape mismatch: expected {shape}")
    lambda_pa = kappa_pa - (2.0 / 3.0) * mu_pa

    corrected_gradient_dir = Path(gate3["inputs"]["corrected_gradient_dir"]).expanduser().resolve()
    active_h5 = np.asarray(np.load(corrected_gradient_dir / "active_h5_indices.npy"), dtype=np.int64)
    if active_h5.ndim != 1 or active_h5.size == 0:
        raise RuntimeError("active_h5_indices must be a non-empty vector")
    if np.min(active_h5) < 0 or np.max(active_h5) >= n_full:
        raise RuntimeError("active_h5_indices out of range")

    beta_lambda = float(gate3["eq24"]["lambda"]["beta_eq21"])
    beta_mu = float(gate3["eq24"]["mu"]["beta_eq21"])
    lambda_convention = objective_convention_from_eq21_beta(beta_lambda)
    mu_convention = objective_convention_from_eq21_beta(beta_mu)

    base_lambda = assemble_smoothed_tv_q1(
        lambda_pa / PA_PER_MPA,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    base_mu = assemble_smoothed_tv_q1(
        mu_pa / PA_PER_MPA,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )

    gate3_lambda_q = float(gate3["tv_chain_rule"]["lambda_Q"])
    gate3_mu_q = float(gate3["tv_chain_rule"]["mu_Q"])
    q_regression_lambda = _relative_error(base_lambda.value, gate3_lambda_q)
    q_regression_mu = _relative_error(base_mu.value, gate3_mu_q)
    if q_regression_lambda > 1.0e-14 or q_regression_mu > 1.0e-14:
        raise RuntimeError("TV scalar regression against Gate-3B failed")

    q_lambda_pa = np.asarray(base_lambda.covector, dtype=np.float64).ravel(order="C") / PA_PER_MPA
    q_mu_pa = np.asarray(base_mu.covector, dtype=np.float64).ravel(order="C") / PA_PER_MPA
    q_lambda_active = q_lambda_pa[active_h5]
    q_mu_active = q_mu_pa[active_h5]

    gate3_tv_lambda = np.asarray(np.load(gate3_dir / "tv_active_covector_lambda_pa.npy"), dtype=np.float64)
    gate3_tv_mu = np.asarray(np.load(gate3_dir / "tv_active_covector_mu_pa.npy"), dtype=np.float64)
    active_regression_lambda = float(np.max(np.abs(q_lambda_active - gate3_tv_lambda)))
    active_regression_mu = float(np.max(np.abs(q_mu_active - gate3_tv_mu)))
    active_regression_tolerance = 1.0e-18
    if active_regression_lambda > active_regression_tolerance or active_regression_mu > active_regression_tolerance:
        raise RuntimeError("TV active covector regression against Gate-3B failed")

    d_lambda_active = _deterministic_direction(active_h5.size, phase=0.0)
    d_mu_active = _deterministic_direction(active_h5.size, phase=0.007)
    d_lambda_full = np.zeros(n_full, dtype=np.float64)
    d_mu_full = np.zeros(n_full, dtype=np.float64)
    d_lambda_full[active_h5] = d_lambda_active
    d_mu_full[active_h5] = d_mu_active
    d_lambda_full = d_lambda_full.reshape(shape, order="C")
    d_mu_full = d_mu_full.reshape(shape, order="C")

    analytic_lambda = float(beta_lambda * np.dot(q_lambda_active, d_lambda_active))
    analytic_mu = float(beta_mu * np.dot(q_mu_active, d_mu_active))
    analytic_total = analytic_lambda + analytic_mu

    base_objective = frozen_total_tv_objective(
        base_lambda.value,
        base_mu.value,
        beta_lambda_eq21=beta_lambda,
        beta_mu_eq21=beta_mu,
    )

    steps = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4)
    fd_records = []
    best_error = np.inf
    best_step = None
    best_fd = None
    for h in steps:
        plus_lambda = assemble_smoothed_tv_q1(
            (lambda_pa + h * d_lambda_full) / PA_PER_MPA,
            x_m=x,
            y_m=y,
            z_m=z,
            epsilon_gradient_sq=epsilon,
        )
        plus_mu = assemble_smoothed_tv_q1(
            (mu_pa + h * d_mu_full) / PA_PER_MPA,
            x_m=x,
            y_m=y,
            z_m=z,
            epsilon_gradient_sq=epsilon,
        )
        minus_lambda = assemble_smoothed_tv_q1(
            (lambda_pa - h * d_lambda_full) / PA_PER_MPA,
            x_m=x,
            y_m=y,
            z_m=z,
            epsilon_gradient_sq=epsilon,
        )
        minus_mu = assemble_smoothed_tv_q1(
            (mu_pa - h * d_mu_full) / PA_PER_MPA,
            x_m=x,
            y_m=y,
            z_m=z,
            epsilon_gradient_sq=epsilon,
        )
        j_plus = frozen_total_tv_objective(
            plus_lambda.value,
            plus_mu.value,
            beta_lambda_eq21=beta_lambda,
            beta_mu_eq21=beta_mu,
        )
        j_minus = frozen_total_tv_objective(
            minus_lambda.value,
            minus_mu.value,
            beta_lambda_eq21=beta_lambda,
            beta_mu_eq21=beta_mu,
        )
        fd = float((j_plus - j_minus) / (2.0 * h))
        rel = _relative_error(fd, analytic_total)
        fd_records.append(
            {
                "h": h,
                "j_plus": j_plus,
                "j_minus": j_minus,
                "central_difference": fd,
                "analytic_directional_derivative": analytic_total,
                "relative_error": rel,
            }
        )
        if rel < best_error:
            best_error = rel
            best_step = h
            best_fd = fd

    if not np.isfinite(best_error) or best_error > 5.0e-6:
        raise RuntimeError(f"frozen-TV objective directional derivative failed: {best_error:.3e}")

    wrong_literal_eq9_derivative = 0.5 * analytic_total
    wrong_literal_relative_error = _relative_error(wrong_literal_eq9_derivative, analytic_total)
    if abs(wrong_literal_relative_error - 0.5) > 1.0e-14:
        raise RuntimeError("factor-of-two diagnostic failed")

    payload = {
        "schema_version": 1,
        "result": "PASS_FATHI_TV_GATE3C_OBJECTIVE_CONVENTION",
        "classification": "AUDIT_ONLY_NO_FORWARD_NO_REVERSE_NO_OPTIMIZER",
        "material_shape_zyx": list(shape),
        "active_control_count": int(active_h5.size),
        "epsilon_gradient_sq_mpa2_per_m2": epsilon,
        "frozen_weight_contract": {
            "statement": "Eq.24 weights are computed at the parent and held fixed during objective evaluation/line search.",
            "lambda_beta_eq21": beta_lambda,
            "mu_beta_eq21": beta_mu,
        },
        "paper_factor_resolution": {
            "issue": "Eq.9 contains R/2 times Q, while Eq.17 and Appendix-C use an R times Q-prime control term.",
            "production_symbol": "beta_eq21",
            "production_regularization_objective": "J_reg = beta_lambda*Q_lambda + beta_mu*Q_mu",
            "production_regularization_derivative": "dJ_reg = beta_lambda*dQ_lambda + beta_mu*dQ_mu",
            "eq9_equivalent_mapping": "R_eq9_equivalent = 2*beta_eq21, so (R_eq9_equivalent/2)*Q = beta_eq21*Q",
            "lambda_R_eq9_equivalent": lambda_convention.r_eq9_equivalent,
            "mu_R_eq9_equivalent": mu_convention.r_eq9_equivalent,
            "do_not_use": "Do not evaluate Eq.9 as (beta_eq21/2)*Q while using beta_eq21*dQ in the gradient; that is a factor-of-two mismatch.",
        },
        "base_tv": {
            "lambda_Q": base_lambda.value,
            "mu_Q": base_mu.value,
            "frozen_regularization_objective": base_objective,
            "gate3B_Q_regression_relative_lambda": q_regression_lambda,
            "gate3B_Q_regression_relative_mu": q_regression_mu,
            "gate3B_active_covector_maxabs_lambda": active_regression_lambda,
            "gate3B_active_covector_maxabs_mu": active_regression_mu,
            "gate3B_active_covector_tolerance": active_regression_tolerance,
        },
        "directional_derivative": {
            "direction_maxabs_pa_lambda": float(np.max(np.abs(d_lambda_active))),
            "direction_maxabs_pa_mu": float(np.max(np.abs(d_mu_active))),
            "analytic_lambda": analytic_lambda,
            "analytic_mu": analytic_mu,
            "analytic_total": analytic_total,
            "central_difference_records": fd_records,
            "best_h": best_step,
            "best_central_difference": best_fd,
            "best_relative_error": float(best_error),
            "acceptance_tolerance": 5.0e-6,
            "wrong_literal_eq9_using_beta_over_2_relative_error": wrong_literal_relative_error,
        },
        "composition_statement": (
            "The data covector is already certified by the exact-discrete reverse route. "
            "Gate-3B certifies pre-Mtilde addition and Mtilde solve. Gate-3C certifies the frozen TV scalar/derivative pair. "
            "Therefore the summed total-control derivative is certified without another SEM3D/forward/reverse run."
        ),
        "inputs": {
            "runtime_config": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "regularization_config": {"path": str(reg_path), "sha256": _sha256(reg_path)},
            "gate3_eq21_summary": {"path": str(gate3_summary_path), "sha256": _sha256(gate3_summary_path)},
            "Mat_0_Kappa.h5": {"path": str(kappa_path), "sha256": _sha256(kappa_path)},
            "Mat_0_Mu.h5": {"path": str(mu_path), "sha256": _sha256(mu_path)},
        },
        "numerical_runs": {
            "sem3d": 0,
            "external_forward": 0,
            "exact_reverse": 0,
            "optimizer": 0,
        },
        "next_gate": "GATE4_SINGLE_REGULARIZED_ITERATION_E2E",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RESULT = PASS_FATHI_TV_GATE3C_OBJECTIVE_CONVENTION")
    print("BETA_LAMBDA_EQ21 =", f"{beta_lambda:.17e}")
    print("BETA_MU_EQ21 =", f"{beta_mu:.17e}")
    print("R_EQ9_EQUIV_LAMBDA =", f"{lambda_convention.r_eq9_equivalent:.17e}")
    print("R_EQ9_EQUIV_MU =", f"{mu_convention.r_eq9_equivalent:.17e}")
    print("BEST_FD_RELATIVE_ERROR =", f"{best_error:.17e}")
    print("SEM3D_RUNS = 0")
    print("OUTPUT =", output_path)


if __name__ == "__main__":
    main()
