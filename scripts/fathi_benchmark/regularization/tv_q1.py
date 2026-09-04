from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


_GAUSS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
_LOCAL_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        (-1.0,  1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        (-1.0,  1.0,  1.0),
        ( 1.0,  1.0,  1.0),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class TVQ1Result:
    """Unweighted smoothed-TV component and its control-space covector.

    The scalar component is

        Q(m) = int sqrt(|grad m_h|^2 + epsilon) dOmega.

    Therefore dQ/dm is exactly the Fathi Appendix-C regularization vector
    before multiplication by the parameter-specific factor R_lambda/R_mu.
    """

    value: float
    covector: np.ndarray
    epsilon_gradient_sq: float
    quadrature_evaluations: int
    min_gradient_sq: float
    max_gradient_sq: float
    min_denominator: float
    max_denominator: float


def _axis(values: Iterable[float], expected: int, name: str) -> np.ndarray:
    axis = np.asarray(tuple(values), dtype=np.float64)
    if axis.ndim != 1 or axis.size != expected:
        raise ValueError(f"{name} axis must have length {expected}")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} axis contains non-finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} axis must be strictly increasing")
    return axis


def _shape_gradients_reference(xi: float, eta: float, zeta: float) -> np.ndarray:
    sx = _LOCAL_SIGNS[:, 0]
    sy = _LOCAL_SIGNS[:, 1]
    sz = _LOCAL_SIGNS[:, 2]
    dxi = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta)
    deta = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta)
    dzeta = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta)
    return np.column_stack((dxi, deta, dzeta))


def _element_values(field: np.ndarray, iz: int, iy: int, ix: int) -> np.ndarray:
    return np.asarray(
        [
            field[iz,     iy,     ix],
            field[iz,     iy,     ix + 1],
            field[iz,     iy + 1, ix],
            field[iz,     iy + 1, ix + 1],
            field[iz + 1, iy,     ix],
            field[iz + 1, iy,     ix + 1],
            field[iz + 1, iy + 1, ix],
            field[iz + 1, iy + 1, ix + 1],
        ],
        dtype=np.float64,
    )


def _scatter_local(target: np.ndarray, local: np.ndarray, iz: int, iy: int, ix: int) -> None:
    target[iz,     iy,     ix]     += local[0]
    target[iz,     iy,     ix + 1] += local[1]
    target[iz,     iy + 1, ix]     += local[2]
    target[iz,     iy + 1, ix + 1] += local[3]
    target[iz + 1, iy,     ix]     += local[4]
    target[iz + 1, iy,     ix + 1] += local[5]
    target[iz + 1, iy + 1, ix]     += local[6]
    target[iz + 1, iy + 1, ix + 1] += local[7]


def assemble_smoothed_tv_q1(
    field: np.ndarray,
    *,
    x_m: Iterable[float],
    y_m: Iterable[float],
    z_m: Iterable[float],
    epsilon_gradient_sq: float,
) -> TVQ1Result:
    """Assemble Fathi Eq. (9)/(17) and Appendix C on a rectilinear Q1 grid.

    The array contract is ``field[iz, iy, ix]``.  The denominator
    ``sqrt(|grad m_h|^2 + epsilon)`` is evaluated independently at every
    2x2x2 Gauss point.  The implementation uses the literal ``+ epsilon``;
    it never substitutes ``epsilon**2``.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("field must be a 3-D array in field[iz, iy, ix] order")
    if not np.all(np.isfinite(values)):
        raise ValueError("field contains non-finite values")
    if not np.isfinite(epsilon_gradient_sq) or epsilon_gradient_sq <= 0.0:
        raise ValueError("epsilon_gradient_sq must be finite and strictly positive")

    nz, ny, nx = values.shape
    if min(nx, ny, nz) < 2:
        raise ValueError("Q1 TV requires at least two nodes on every axis")

    x = _axis(x_m, nx, "x")
    y = _axis(y_m, ny, "y")
    z = _axis(z_m, nz, "z")

    dx = np.diff(x)[None, None, :]
    dy = np.diff(y)[None, :, None]
    dz = np.diff(z)[:, None, None]
    inv_x = 2.0 / dx
    inv_y = 2.0 / dy
    inv_z = 2.0 / dz
    det_j = dx * dy * dz / 8.0

    nodal = np.stack(
        (
            values[:-1, :-1, :-1],
            values[:-1, :-1, 1:],
            values[:-1, 1:, :-1],
            values[:-1, 1:, 1:],
            values[1:, :-1, :-1],
            values[1:, :-1, 1:],
            values[1:, 1:, :-1],
            values[1:, 1:, 1:],
        ),
        axis=-1,
    )

    covector = np.zeros_like(values)
    total = 0.0
    qcount = 0
    min_grad_sq = np.inf
    max_grad_sq = 0.0
    min_denom = np.inf
    max_denom = 0.0

    target_slices = (
        (slice(None, -1), slice(None, -1), slice(None, -1)),
        (slice(None, -1), slice(None, -1), slice(1, None)),
        (slice(None, -1), slice(1, None), slice(None, -1)),
        (slice(None, -1), slice(1, None), slice(1, None)),
        (slice(1, None), slice(None, -1), slice(None, -1)),
        (slice(1, None), slice(None, -1), slice(1, None)),
        (slice(1, None), slice(1, None), slice(None, -1)),
        (slice(1, None), slice(1, None), slice(1, None)),
    )

    for xi, eta, zeta in product(_GAUSS, repeat=3):
        grad_ref = _shape_gradients_reference(xi, eta, zeta)
        gx = np.tensordot(nodal, grad_ref[:, 0], axes=([-1], [0])) * inv_x
        gy = np.tensordot(nodal, grad_ref[:, 1], axes=([-1], [0])) * inv_y
        gz = np.tensordot(nodal, grad_ref[:, 2], axes=([-1], [0])) * inv_z
        grad_sq = gx * gx + gy * gy + gz * gz
        denom = np.sqrt(grad_sq + epsilon_gradient_sq)

        total += float(np.sum(denom * det_j))
        common = det_j / denom

        for a, target in enumerate(target_slices):
            contribution = (
                grad_ref[a, 0] * inv_x * gx
                + grad_ref[a, 1] * inv_y * gy
                + grad_ref[a, 2] * inv_z * gz
            ) * common
            covector[target] += contribution

        qcount += int(grad_sq.size)
        min_grad_sq = min(min_grad_sq, float(np.min(grad_sq)))
        max_grad_sq = max(max_grad_sq, float(np.max(grad_sq)))
        min_denom = min(min_denom, float(np.min(denom)))
        max_denom = max(max_denom, float(np.max(denom)))

    return TVQ1Result(
        value=float(total),
        covector=covector,
        epsilon_gradient_sq=float(epsilon_gradient_sq),
        quadrature_evaluations=int(qcount),
        min_gradient_sq=float(min_grad_sq),
        max_gradient_sq=float(max_grad_sq),
        min_denominator=float(min_denom),
        max_denominator=float(max_denom),
    )
