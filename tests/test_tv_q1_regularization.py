from __future__ import annotations

import numpy as np

from scripts.fathi_benchmark.regularization.tv_q1 import assemble_smoothed_tv_q1
from scripts.fathi_benchmark.regularization.tv_weight import fathi_eq24_weight


def _grid():
    x = np.linspace(-2.0, 2.0, 5)
    y = np.linspace(-1.5, 1.5, 4)
    z = np.linspace(-4.0, 0.0, 5)
    return x, y, z


def _assemble(field: np.ndarray, epsilon: float = 0.01):
    x, y, z = _grid()
    return assemble_smoothed_tv_q1(
        field,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )


def test_constant_material_has_zero_regularization_covector():
    field = np.full((5, 4, 5), 80.0)
    result = _assemble(field)
    assert np.linalg.norm(result.covector.ravel()) < 1.0e-12
    assert result.max_gradient_sq < 1.0e-24


def test_layered_material_uses_pointwise_quadrature_denominator():
    _, _, z = _grid()
    profile = np.where(z < -2.0, 125.0, 80.0)
    field = np.broadcast_to(profile[:, None, None], (5, 4, 5)).copy()
    result = _assemble(field)
    assert np.linalg.norm(result.covector.ravel()) > 0.0
    assert result.max_gradient_sq > result.min_gradient_sq
    assert result.max_denominator > result.min_denominator


def test_tv_directional_derivative_matches_covector():
    rng = np.random.default_rng(20260904)
    _, _, z = _grid()
    profile = np.where(z < -2.0, 125.0, np.where(z < -1.0, 101.25, 80.0))
    field = np.broadcast_to(profile[:, None, None], (5, 4, 5)).copy()
    direction = rng.normal(size=field.shape)
    direction /= np.linalg.norm(direction.ravel())

    base = _assemble(field)
    analytic = float(np.sum(base.covector * direction))

    h = 1.0e-5
    plus = _assemble(field + h * direction).value
    minus = _assemble(field - h * direction).value
    finite_difference = float((plus - minus) / (2.0 * h))

    scale = max(1.0, abs(analytic), abs(finite_difference))
    assert abs(analytic - finite_difference) / scale < 2.0e-7


def test_literal_epsilon_not_epsilon_squared():
    field = np.full((5, 4, 5), 80.0)
    epsilon = 0.01
    result = _assemble(field, epsilon)
    volume = 4.0 * 3.0 * 4.0
    expected = np.sqrt(epsilon) * volume
    assert np.isclose(result.value, expected, rtol=1.0e-12, atol=1.0e-12)


def test_eq24_weight_matches_fathi_formula():
    gmis = np.asarray([3.0, 4.0])
    greg = np.asarray([0.0, 2.0])
    result = fathi_eq24_weight(gmis, greg, varrho=0.5)
    assert np.isclose(result.misfit_l2, 5.0)
    assert np.isclose(result.regularization_l2, 2.0)
    assert np.isclose(result.weight, 1.25)
