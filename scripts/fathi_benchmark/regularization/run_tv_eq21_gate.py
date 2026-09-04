from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import spsolve

from scripts.fathi_benchmark.regularization.tv_q1 import assemble_smoothed_tv_q1
from scripts.fathi_benchmark.regularization.tv_weight import fathi_eq24_weight


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


def _expected_full_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack(
        (
            xx.ravel(order="C"),
            yy.ravel(order="C"),
            zz.ravel(order="C"),
        )
    )


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1.0e-300))


def _maxabs(a: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64))))


def _require_same(label: str, a: np.ndarray, b: np.ndarray, atol: float = 0.0) -> dict:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        raise RuntimeError(f"{label}: shape mismatch {aa.shape} != {bb.shape}")
    err = _maxabs(aa - bb)
    if err > atol:
        raise RuntimeError(f"{label}: maxabs mismatch {err:.17e} > {atol:.17e}")
    return {"shape": list(aa.shape), "max_abs_difference": err, "tolerance": atol}


def _varrho(reg: dict, stage: str) -> float:
    policy = reg["regularization"]["varrho_policy"]
    if policy.get("type") != "frequency_stage_piecewise_constant":
        raise RuntimeError("unsupported varrho policy")
    if stage not in ("p20", "p30", "p40"):
        raise RuntimeError(f"unsupported frequency stage: {stage}")
    value = float(policy[stage])
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise RuntimeError("varrho must lie in (0, 1]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--regularization-config", required=True)
    parser.add_argument("--material-dir", required=True)
    parser.add_argument("--corrected-gradient-dir", required=True)
    parser.add_argument("--frequency-stage", default="p20")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runtime_path = Path(args.runtime_config).expanduser().resolve()
    reg_path = Path(args.regularization_config).expanduser().resolve()
    material_dir = Path(args.material_dir).expanduser().resolve()
    gradient_dir = Path(args.corrected_gradient_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    runtime = _json(runtime_path)
    reg = _json(reg_path)
    if reg.get("status") != "GATE_ONLY_DO_NOT_RUN_E2E":
        raise RuntimeError("Gate-3 input must remain GATE_ONLY_DO_NOT_RUN_E2E")

    x, y, z = _axes(runtime)
    shape = (z.size, y.size, x.size)
    n_full = int(np.prod(shape))
    dataset = str(runtime["material_grid"]["dataset"])
    epsilon = float(reg["regularization"]["epsilon_gradient_sq_mpa2_per_m2"])
    varrho = _varrho(reg, args.frequency_stage)

    kappa_path = material_dir / "Mat_0_Kappa.h5"
    mu_path = material_dir / "Mat_0_Mu.h5"
    kappa_pa = _read_h5(kappa_path, dataset)
    mu_pa = _read_h5(mu_path, dataset)
    if kappa_pa.shape != shape or mu_pa.shape != shape:
        raise RuntimeError(f"material shape mismatch: expected {shape}")
    lambda_pa = kappa_pa - (2.0 / 3.0) * mu_pa

    tv_lambda = assemble_smoothed_tv_q1(
        lambda_pa / PA_PER_MPA,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    tv_mu = assemble_smoothed_tv_q1(
        mu_pa / PA_PER_MPA,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )

    # Q is assembled in the MPa material coordinate.  By the chain rule,
    # dQ/dm_Pa = 1e-6 * dQ/dm_MPa.
    tv_full_lambda_pa = np.asarray(tv_lambda.covector, dtype=np.float64).ravel(order="C") / PA_PER_MPA
    tv_full_mu_pa = np.asarray(tv_mu.covector, dtype=np.float64).ravel(order="C") / PA_PER_MPA
    if tv_full_lambda_pa.shape != (n_full,) or tv_full_mu_pa.shape != (n_full,):
        raise RuntimeError("TV full-control covector shape mismatch")

    full_coords = np.asarray(np.load(gradient_dir / "full_control_coords.npy"), dtype=np.float64)
    full_data_lambda = np.asarray(np.load(gradient_dir / "full_control_covector_lambda.npy"), dtype=np.float64)
    full_data_mu = np.asarray(np.load(gradient_dir / "full_control_covector_mu.npy"), dtype=np.float64)
    full_h5_kappa = np.asarray(np.load(gradient_dir / "full_h5_dataset_covector_kappa.npy"), dtype=np.float64)
    full_h5_mu = np.asarray(np.load(gradient_dir / "full_h5_dataset_covector_mu.npy"), dtype=np.float64)
    active_h5 = np.asarray(np.load(gradient_dir / "active_h5_indices.npy"), dtype=np.int64)
    active_coords = np.asarray(np.load(gradient_dir / "mtilde_active_coords.npy"), dtype=np.float64)

    rhs_dir = gradient_dir / "rhs"
    rhs_coords = np.asarray(np.load(rhs_dir / "full_grid_trace_RHS_total_coords.npy"), dtype=np.float64)
    rhs_data_lambda = np.asarray(np.load(rhs_dir / "full_grid_trace_RHS_total_lambda.npy"), dtype=np.float64)
    rhs_data_mu = np.asarray(np.load(rhs_dir / "full_grid_trace_RHS_total_mu.npy"), dtype=np.float64)

    solve_dir = gradient_dir / "mtilde_solve"
    matrix_path = solve_dir / "Mtilde_interior_sparse.npz"
    matrix = load_npz(matrix_path).tocsr()
    gradient_coords = np.asarray(np.load(solve_dir / "gradient_coords.npy"), dtype=np.float64)
    existing_g_lambda = np.asarray(np.load(solve_dir / "g_lambda.npy"), dtype=np.float64)
    existing_g_mu = np.asarray(np.load(solve_dir / "g_mu.npy"), dtype=np.float64)

    expected_full_coords = _expected_full_coords(x, y, z)
    if full_coords.shape != (n_full, 3):
        raise RuntimeError(f"full-control coordinate shape mismatch: {full_coords.shape}")
    coordinate_contract = {
        "full_control_matches_runtime_C_order": _require_same(
            "full control/runtime C-order coordinates", full_coords, expected_full_coords, atol=1.0e-12
        ),
    }

    for name, arr in (
        ("full_data_lambda", full_data_lambda),
        ("full_data_mu", full_data_mu),
        ("full_h5_kappa", full_h5_kappa),
        ("full_h5_mu", full_h5_mu),
    ):
        if arr.shape != (n_full,):
            raise RuntimeError(f"{name} shape mismatch: {arr.shape}")
    if active_h5.ndim != 1:
        raise RuntimeError("active_h5_indices must be one-dimensional")
    if active_h5.size == 0 or np.min(active_h5) < 0 or np.max(active_h5) >= n_full:
        raise RuntimeError("active_h5_indices out of range")
    if np.unique(active_h5).size != active_h5.size:
        raise RuntimeError("active_h5_indices contain duplicates")
    n_active = int(active_h5.size)

    coordinate_contract["active_coords_from_full_C_order"] = _require_same(
        "active coordinates", active_coords, full_coords[active_h5], atol=1.0e-12
    )
    coordinate_contract["rhs_coords_equal_active_coords"] = _require_same(
        "RHS coordinates", rhs_coords, active_coords, atol=1.0e-12
    )
    coordinate_contract["gradient_coords_equal_active_coords"] = _require_same(
        "gradient coordinates", gradient_coords, active_coords, atol=1.0e-12
    )

    parameterization_contract = {
        "h5_kappa_equals_physical_lambda": _require_same(
            "H5 kappa/physical lambda covector", full_h5_kappa, full_data_lambda, atol=0.0
        ),
        "h5_mu_chain_rule": _require_same(
            "H5 mu chain rule",
            full_h5_mu,
            full_data_mu - (2.0 / 3.0) * full_data_lambda,
            atol=0.0,
        ),
    }

    data_restriction_contract = {
        "lambda": _require_same(
            "data lambda active restriction", rhs_data_lambda, full_data_lambda[active_h5], atol=0.0
        ),
        "mu": _require_same(
            "data mu active restriction", rhs_data_mu, full_data_mu[active_h5], atol=0.0
        ),
    }

    tv_active_lambda_pa = tv_full_lambda_pa[active_h5]
    tv_active_mu_pa = tv_full_mu_pa[active_h5]
    if rhs_data_lambda.shape != (n_active,) or rhs_data_mu.shape != (n_active,):
        raise RuntimeError("data RHS shape does not match active-control count")
    if matrix.shape != (n_active, n_active):
        raise RuntimeError(f"Mtilde shape mismatch: {matrix.shape} != {(n_active, n_active)}")

    weight_lambda = fathi_eq24_weight(rhs_data_lambda, tv_active_lambda_pa, varrho=varrho)
    weight_mu = fathi_eq24_weight(rhs_data_mu, tv_active_mu_pa, varrho=varrho)

    weighted_tv_lambda = weight_lambda.weight * tv_active_lambda_pa
    weighted_tv_mu = weight_mu.weight * tv_active_mu_pa
    ratio_lambda = float(np.linalg.norm(weighted_tv_lambda) / max(np.linalg.norm(rhs_data_lambda), 1.0e-300))
    ratio_mu = float(np.linalg.norm(weighted_tv_mu) / max(np.linalg.norm(rhs_data_mu), 1.0e-300))
    if abs(ratio_lambda - varrho) > 1.0e-12:
        raise RuntimeError("Eq.24 lambda norm-ratio contract failed")
    if abs(ratio_mu - varrho) > 1.0e-12:
        raise RuntimeError("Eq.24 mu norm-ratio contract failed")

    # Zero-regularization regression: the canonical Mtilde solve must be reproduced
    # before any total-RHS solve is trusted.
    reproduced_data_g_lambda = np.asarray(spsolve(matrix, rhs_data_lambda), dtype=np.float64)
    reproduced_data_g_mu = np.asarray(spsolve(matrix, rhs_data_mu), dtype=np.float64)
    regression_lambda = _relative_l2(reproduced_data_g_lambda, existing_g_lambda)
    regression_mu = _relative_l2(reproduced_data_g_mu, existing_g_mu)
    if regression_lambda > 1.0e-12 or regression_mu > 1.0e-12:
        raise RuntimeError(
            f"data-only Mtilde regression failed: lambda={regression_lambda:.3e}, mu={regression_mu:.3e}"
        )

    rhs_total_lambda = rhs_data_lambda + weighted_tv_lambda
    rhs_total_mu = rhs_data_mu + weighted_tv_mu
    g_total_lambda = np.asarray(spsolve(matrix, rhs_total_lambda), dtype=np.float64)
    g_total_mu = np.asarray(spsolve(matrix, rhs_total_mu), dtype=np.float64)
    if not np.all(np.isfinite(g_total_lambda)) or not np.all(np.isfinite(g_total_mu)):
        raise RuntimeError("non-finite total physical gradient")

    residual_lambda = matrix @ g_total_lambda - rhs_total_lambda
    residual_mu = matrix @ g_total_mu - rhs_total_mu
    relative_residual_lambda = float(
        np.linalg.norm(residual_lambda) / max(np.linalg.norm(rhs_total_lambda), 1.0e-300)
    )
    relative_residual_mu = float(
        np.linalg.norm(residual_mu) / max(np.linalg.norm(rhs_total_mu), 1.0e-300)
    )
    if relative_residual_lambda > 1.0e-10 or relative_residual_mu > 1.0e-10:
        raise RuntimeError("total Mtilde residual exceeds tolerance")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "tv_full_covector_lambda_pa.npy": tv_full_lambda_pa,
        "tv_full_covector_mu_pa.npy": tv_full_mu_pa,
        "tv_active_covector_lambda_pa.npy": tv_active_lambda_pa,
        "tv_active_covector_mu_pa.npy": tv_active_mu_pa,
        "weighted_tv_active_lambda.npy": weighted_tv_lambda,
        "weighted_tv_active_mu.npy": weighted_tv_mu,
        "rhs_data_lambda.npy": rhs_data_lambda,
        "rhs_data_mu.npy": rhs_data_mu,
        "rhs_total_lambda.npy": rhs_total_lambda,
        "rhs_total_mu.npy": rhs_total_mu,
        "active_coords.npy": active_coords,
        "g_total_lambda.npy": g_total_lambda,
        "g_total_mu.npy": g_total_mu,
    }
    for name, array in arrays.items():
        np.save(output_dir / name, np.asarray(array))

    payload = {
        "schema_version": 1,
        "result": "PASS_FATHI_TV_GATE3_EQ21_CONTROL_ASSEMBLY",
        "classification": "AUDIT_ONLY_NO_FORWARD_NO_REVERSE_NO_OPTIMIZER",
        "frequency_stage": args.frequency_stage,
        "varrho": varrho,
        "material_coordinate_tv_kernel": "MPa",
        "assembled_control_coordinate": "physical Pa",
        "array_order": "field[iz, iy, ix] flattened in C order",
        "material_shape_zyx": list(shape),
        "full_control_count": n_full,
        "active_control_count": n_active,
        "coordinate_contract": coordinate_contract,
        "parameterization_contract": parameterization_contract,
        "data_restriction_contract": data_restriction_contract,
        "tv_chain_rule": {
            "formula": "dQ/dm_Pa = 1e-6 * dQ/dm_MPa",
            "scale": 1.0 / PA_PER_MPA,
            "lambda_Q": tv_lambda.value,
            "mu_Q": tv_mu.value,
            "lambda_full_covector_l2_mpa": float(np.linalg.norm(tv_lambda.covector.ravel(order="C"))),
            "mu_full_covector_l2_mpa": float(np.linalg.norm(tv_mu.covector.ravel(order="C"))),
            "lambda_active_covector_l2_pa": float(np.linalg.norm(tv_active_lambda_pa)),
            "mu_active_covector_l2_pa": float(np.linalg.norm(tv_active_mu_pa)),
        },
        "eq24": {
            "formula": "beta = varrho * ||g_mis||_2 / ||g_reg||_2",
            "norm_space": "same pre-Mtilde active physical-Pa control covectors",
            "lambda": {
                "data_l2": weight_lambda.misfit_l2,
                "regularization_l2": weight_lambda.regularization_l2,
                "beta_eq21": weight_lambda.weight,
                "weighted_reg_over_data_l2": ratio_lambda,
            },
            "mu": {
                "data_l2": weight_mu.misfit_l2,
                "regularization_l2": weight_mu.regularization_l2,
                "beta_eq21": weight_mu.weight,
                "weighted_reg_over_data_l2": ratio_mu,
            },
        },
        "mtilde": {
            "matrix_shape": list(matrix.shape),
            "data_only_regression_relative_lambda": regression_lambda,
            "data_only_regression_relative_mu": regression_mu,
            "total_relative_residual_lambda": relative_residual_lambda,
            "total_relative_residual_mu": relative_residual_mu,
            "existing_data_gradient_l2_lambda": float(np.linalg.norm(existing_g_lambda)),
            "existing_data_gradient_l2_mu": float(np.linalg.norm(existing_g_mu)),
            "total_gradient_l2_lambda": float(np.linalg.norm(g_total_lambda)),
            "total_gradient_l2_mu": float(np.linalg.norm(g_total_mu)),
        },
        "objective_convention_status": (
            "NOT_YET_PROMOTED_TO_ARMIJO; Gate 3 certifies Eq.21 RHS assembly only. "
            "The Eq.9 1/2 versus Appendix-C normalization is intentionally kept out of production until "
            "a total-objective directional derivative gate fixes one scalar/gradient convention."
        ),
        "inputs": {
            "runtime_config": {"path": str(runtime_path), "sha256": _sha256(runtime_path)},
            "regularization_config": {"path": str(reg_path), "sha256": _sha256(reg_path)},
            "Mat_0_Kappa.h5": {"path": str(kappa_path), "sha256": _sha256(kappa_path)},
            "Mat_0_Mu.h5": {"path": str(mu_path), "sha256": _sha256(mu_path)},
            "corrected_gradient_dir": str(gradient_dir),
            "Mtilde": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        },
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": _sha256(output_dir / name)}
            for name in arrays
        },
        "numerical_runs": {
            "sem3d": 0,
            "external_forward": 0,
            "exact_reverse": 0,
            "optimizer": 0,
        },
    }

    summary_path = output_dir / "gate3_eq21_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RESULT = PASS_FATHI_TV_GATE3_EQ21_CONTROL_ASSEMBLY")
    print("VARrho =", f"{varrho:.17e}")
    print("RATIO_LAMBDA =", f"{ratio_lambda:.17e}")
    print("RATIO_MU =", f"{ratio_mu:.17e}")
    print("DATA_REGRESSION_LAMBDA =", f"{regression_lambda:.17e}")
    print("DATA_REGRESSION_MU =", f"{regression_mu:.17e}")
    print("MTILDE_RESIDUAL_LAMBDA =", f"{relative_residual_lambda:.17e}")
    print("MTILDE_RESIDUAL_MU =", f"{relative_residual_mu:.17e}")
    print("OUTPUT =", summary_path)


if __name__ == "__main__":
    main()
