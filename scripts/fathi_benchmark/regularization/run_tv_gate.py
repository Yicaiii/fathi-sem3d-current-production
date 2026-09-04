from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.fathi_benchmark.regularization.tv_q1 import assemble_smoothed_tv_q1


PA_PER_MPA = 1.0e6


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


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


def _layered(runtime: dict, z: np.ndarray, component: str) -> np.ndarray:
    layers = runtime["materials"]["true"]["layers"]
    if len(layers) != 3:
        raise RuntimeError("this gate currently expects the certified three-layer S43 target")
    top = float(layers[0][f"{component}_pa"]) / PA_PER_MPA
    middle = float(layers[1][f"{component}_pa"]) / PA_PER_MPA
    bottom = float(layers[2][f"{component}_pa"]) / PA_PER_MPA
    profile = np.where(z < -27.0, bottom, np.where(z < -12.0, middle, top))
    return profile


def _read_h5(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset][...], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--regularization-config", required=True)
    parser.add_argument("--material-dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    runtime_path = Path(args.runtime_config).expanduser().resolve()
    reg_path = Path(args.regularization_config).expanduser().resolve()
    runtime = _json(runtime_path)
    regularization = _json(reg_path)

    if regularization.get("status") != "GATE_ONLY_DO_NOT_RUN_E2E":
        raise RuntimeError("phase-1 gate config must remain GATE_ONLY_DO_NOT_RUN_E2E")

    tv = regularization["regularization"]
    epsilon = float(tv["epsilon_gradient_sq_mpa2_per_m2"])
    x, y, z = _axes(runtime)
    shape = (z.size, y.size, x.size)

    lambda_true_z = _layered(runtime, z, "lambda")
    mu_true_z = _layered(runtime, z, "mu")
    lambda_true = np.broadcast_to(lambda_true_z[:, None, None], shape).copy()
    mu_true = np.broadcast_to(mu_true_z[:, None, None], shape).copy()
    lambda_constant = np.full(shape, float(runtime["materials"]["initial"]["lambda_pa"]) / PA_PER_MPA)
    mu_constant = np.full(shape, float(runtime["materials"]["initial"]["mu_pa"]) / PA_PER_MPA)

    synthetic = {}
    for name, field in (
        ("lambda_constant", lambda_constant),
        ("mu_constant", mu_constant),
        ("lambda_layered", lambda_true),
        ("mu_layered", mu_true),
    ):
        result = assemble_smoothed_tv_q1(
            field,
            x_m=x,
            y_m=y,
            z_m=z,
            epsilon_gradient_sq=epsilon,
        )
        synthetic[name] = {
            "value": result.value,
            "covector_l2": float(np.linalg.norm(result.covector.ravel())),
            "covector_maxabs": float(np.max(np.abs(result.covector))),
            "min_gradient_sq": result.min_gradient_sq,
            "max_gradient_sq": result.max_gradient_sq,
            "min_denominator": result.min_denominator,
            "max_denominator": result.max_denominator,
            "quadrature_evaluations": result.quadrature_evaluations,
        }

    constant_maxabs_tol = 1.0e-12
    if synthetic["lambda_constant"]["covector_maxabs"] > constant_maxabs_tol:
        raise RuntimeError("constant lambda produced a non-zero TV covector")
    if synthetic["mu_constant"]["covector_maxabs"] > constant_maxabs_tol:
        raise RuntimeError("constant mu produced a non-zero TV covector")
    if synthetic["lambda_layered"]["max_denominator"] <= synthetic["lambda_layered"]["min_denominator"]:
        raise RuntimeError("layered lambda did not produce pointwise-varying TV denominators")
    if synthetic["mu_layered"]["max_denominator"] <= synthetic["mu_layered"]["min_denominator"]:
        raise RuntimeError("layered mu did not produce pointwise-varying TV denominators")


    # Piecewise-constant layered target: Q1 TV support must be localized to
    # nodal planes adjacent to an interface.  This is a direct discriminator
    # against implementations that collapse the TV denominator to a global scalar.
    def interface_support_check(profile: np.ndarray, field: np.ndarray) -> dict:
        result = assemble_smoothed_tv_q1(
            field, x_m=x, y_m=y, z_m=z, epsilon_gradient_sq=epsilon
        )
        plane_maxabs = np.max(np.abs(result.covector), axis=(1, 2))
        support = np.zeros(profile.size, dtype=bool)
        jumps = np.flatnonzero(np.diff(profile) != 0.0)
        for j in jumps:
            support[j] = True
            support[j + 1] = True
        inside = float(np.max(plane_maxabs[support])) if np.any(support) else 0.0
        outside = float(np.max(plane_maxabs[~support])) if np.any(~support) else 0.0
        if inside <= 0.0:
            raise RuntimeError("layered TV covector is zero at all interface-adjacent planes")
        if outside > max(1.0e-10, 1.0e-9 * inside):
            raise RuntimeError(
                f"TV covector leaked into layer interiors: outside={outside:.3e}, inside={inside:.3e}"
            )
        return {
            "interface_jump_indices": [int(v) for v in jumps],
            "interface_adjacent_z_indices": [int(v) for v in np.flatnonzero(support)],
            "interface_maxabs": inside,
            "layer_interior_maxabs": outside,
        }

    interface_localization = {
        "lambda": interface_support_check(lambda_true_z, lambda_true),
        "mu": interface_support_check(mu_true_z, mu_true),
    }

    interface_ratio_lambda = synthetic["lambda_layered"]["max_gradient_sq"] / epsilon
    interface_ratio_mu = synthetic["mu_layered"]["max_gradient_sq"] / epsilon
    if min(interface_ratio_lambda, interface_ratio_mu) < 100.0:
        raise RuntimeError("epsilon is not at least two orders below the squared interface-gradient scale")

    parent = None
    if args.material_dir:
        material_dir = Path(args.material_dir).expanduser().resolve()
        dataset = str(runtime["material_grid"]["dataset"])
        kappa_pa = _read_h5(material_dir / "Mat_0_Kappa.h5", dataset)
        mu_pa = _read_h5(material_dir / "Mat_0_Mu.h5", dataset)
        if kappa_pa.shape != shape or mu_pa.shape != shape:
            raise RuntimeError(f"material shape mismatch: expected {shape}")
        lambda_pa = kappa_pa - (2.0 / 3.0) * mu_pa
        parent = {}
        for name, field in (
            ("lambda", lambda_pa / PA_PER_MPA),
            ("mu", mu_pa / PA_PER_MPA),
        ):
            result = assemble_smoothed_tv_q1(
                field,
                x_m=x,
                y_m=y,
                z_m=z,
                epsilon_gradient_sq=epsilon,
            )
            parent[name] = {
                "value": result.value,
                "covector_l2_mpa_coordinate": float(np.linalg.norm(result.covector.ravel())),
                "covector_maxabs_mpa_coordinate": float(np.max(np.abs(result.covector))),
                "max_gradient_sq_mpa2_per_m2": result.max_gradient_sq,
            }

    payload = {
        "schema_version": 1,
        "result": "PASS_FATHI_TV_PHASE1_GATE",
        "classification": "AUDIT_ONLY_NO_FORWARD_NO_REVERSE_NO_OPTIMIZER",
        "runtime_config": str(runtime_path),
        "regularization_config": str(reg_path),
        "material_shape_zyx": list(shape),
        "array_order": "field[iz, iy, ix]",
        "material_coordinate": "MPa",
        "spatial_coordinate": "m",
        "epsilon_form": "sqrt(|grad m|^2 + epsilon), not epsilon^2",
        "epsilon_gradient_sq_mpa2_per_m2": epsilon,
        "synthetic": synthetic,
        "interface_localization": interface_localization,
        "epsilon_scale_check": {
            "lambda_max_gradient_sq_over_epsilon": interface_ratio_lambda,
            "mu_max_gradient_sq_over_epsilon": interface_ratio_mu,
            "requirement": ">= 100",
        },
        "parent_material_audit": parent,
        "numerical_runs": {
            "sem3d": 0,
            "external_forward": 0,
            "exact_reverse": 0,
        },
    }

    output = Path(args.output).expanduser().resolve() if args.output else Path("tv_phase1_gate.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RESULT = PASS_FATHI_TV_PHASE1_GATE")
    print("OUTPUT =", output)


if __name__ == "__main__":
    main()
