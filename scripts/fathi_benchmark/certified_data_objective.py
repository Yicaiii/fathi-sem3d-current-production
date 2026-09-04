from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping

import numpy as np

from scripts.exact_adjoint.certify_exact_adjoint_with_fixed_dt_fd import (
    trapezoid_weights,
)


DT_ABS_TOL = 1.0e-18


def float64_bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def float64_diagnostic(value: float) -> dict[str, Any]:
    scalar = float(value)
    return {
        "value": scalar,
        "repr": repr(scalar),
        "hex": scalar.hex(),
        "uint64_bits": float64_bits(scalar),
        "type": type(scalar).__name__,
    }


@dataclass(frozen=True)
class CertifiedDataObjective:
    value: float
    objective_dt: float
    driver_dt: float
    dt_abs_difference: float
    dt_ulp_distance: int
    sample_count: int
    receiver_count: int
    component_count: int

    def diagnostic(self, *, expected: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "objective": float64_diagnostic(self.value),
            "objective_dt": float64_diagnostic(self.objective_dt),
            "driver_dt": float64_diagnostic(self.driver_dt),
            "dt_abs_difference": self.dt_abs_difference,
            "dt_ulp_distance": self.dt_ulp_distance,
            "sample_count": self.sample_count,
            "receiver_count": self.receiver_count,
            "component_count": self.component_count,
            "quadrature_dt_source": "certified_reference.contract.dt",
        }
        if expected is not None:
            target = float(expected)
            payload["expected"] = float64_diagnostic(target)
            payload["objective_equal_expected"] = self.value == target
            payload["objective_expected_ulp_distance"] = abs(
                float64_bits(self.value) - float64_bits(target)
            )
        return payload


def certified_data_objective(
    current: np.ndarray,
    truth: np.ndarray,
    *,
    certified_dt: float,
    driver_dt: float,
) -> CertifiedDataObjective:
    current64 = np.asarray(current, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    if current64.ndim != 3 or truth64.ndim != 3:
        raise ValueError("certified data objective requires rank-3 receiver arrays")
    if current64.shape != truth64.shape:
        raise ValueError("current/TRUE receiver shape mismatch")
    if not np.all(np.isfinite(current64)) or not np.all(np.isfinite(truth64)):
        raise ValueError("current/TRUE receiver contains non-finite values")

    objective_dt = float(certified_dt)
    runtime_dt = float(driver_dt)
    if not math.isfinite(objective_dt) or objective_dt <= 0.0:
        raise ValueError("certified objective dt must be finite and positive")
    if not math.isfinite(runtime_dt) or runtime_dt <= 0.0:
        raise ValueError("driver dt must be finite and positive")
    if not math.isclose(
        runtime_dt,
        objective_dt,
        rel_tol=0.0,
        abs_tol=DT_ABS_TOL,
    ):
        raise ValueError(
            "driver dt differs from certified objective dt beyond contract tolerance"
        )

    sample_count, receiver_count, component_count = current64.shape
    time_grid = np.arange(sample_count, dtype=np.float64) * objective_dt
    weights = trapezoid_weights(time_grid)
    residual = current64 - truth64
    value = 0.5 * float(
        np.sum(weights[:, None, None] * residual * residual)
    )
    return CertifiedDataObjective(
        value=value,
        objective_dt=objective_dt,
        driver_dt=runtime_dt,
        dt_abs_difference=float(runtime_dt - objective_dt),
        dt_ulp_distance=abs(float64_bits(runtime_dt) - float64_bits(objective_dt)),
        sample_count=sample_count,
        receiver_count=receiver_count,
        component_count=component_count,
    )
