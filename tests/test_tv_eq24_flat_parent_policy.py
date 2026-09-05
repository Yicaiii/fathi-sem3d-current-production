import math

import numpy as np

from scripts.fathi_benchmark.regularization.regularized_pipeline_artifacts import (
    compose_frozen_regularized_objective,
    regularized_secant_pair_outcome,
)
from scripts.fathi_benchmark.regularization.tv_weight import (
    EQ24_ACTIVE_STATUS,
    EQ24_DEFERRED_FLAT_STATUS,
    SECANT_PAIR_ACCEPTED_STATUS,
    SECANT_PAIR_SKIPPED_STATUS,
    floating_sum_relative_bound,
    fathi_eq24_weight_with_flat_parent_policy,
    q1_constant_gradient_roundoff_bound_sq,
)


def test_flat_parent_requires_both_independent_gates():
    gmis = np.array([3.0, 4.0], dtype=np.float64)
    greg = np.array([1.0e-15, -2.0e-15], dtype=np.float64)

    w = fathi_eq24_weight_with_flat_parent_policy(
        gmis,
        greg,
        varrho=0.5,
        max_gradient_sq=1.0e-30,
        epsilon_gradient_sq=1.0e-2,
        gradient_roundoff_bound_sq=1.0e-26,
        q_value=7200.0,
        q_floor_value=7200.0,
        q_floor_relative_tolerance=1.0e-11,
    )
    assert w.status == EQ24_DEFERRED_FLAT_STATUS
    assert w.gradient_gate_pass is True
    assert w.q_floor_gate_pass is True
    assert w.weight == 0.0


def test_gradient_gate_alone_does_not_defer():
    gmis = np.array([3.0, 4.0], dtype=np.float64)
    greg = np.array([1.0, 2.0], dtype=np.float64)

    w = fathi_eq24_weight_with_flat_parent_policy(
        gmis,
        greg,
        varrho=0.5,
        max_gradient_sq=1.0e-30,
        epsilon_gradient_sq=1.0e-2,
        gradient_roundoff_bound_sq=1.0e-26,
        q_value=7200.1,
        q_floor_value=7200.0,
        q_floor_relative_tolerance=1.0e-12,
    )
    assert w.status == EQ24_ACTIVE_STATUS
    assert w.gradient_gate_pass is True
    assert w.q_floor_gate_pass is False


def test_q_floor_gate_alone_does_not_defer():
    gmis = np.array([3.0, 4.0], dtype=np.float64)
    greg = np.array([1.0, 2.0], dtype=np.float64)

    w = fathi_eq24_weight_with_flat_parent_policy(
        gmis,
        greg,
        varrho=0.5,
        max_gradient_sq=1.0,
        epsilon_gradient_sq=1.0e-2,
        gradient_roundoff_bound_sq=1.0e-26,
        q_value=7200.0,
        q_floor_value=7200.0,
        q_floor_relative_tolerance=1.0e-12,
    )
    assert w.status == EQ24_ACTIVE_STATUS
    assert w.gradient_gate_pass is False
    assert w.q_floor_gate_pass is True


def test_nonflat_parent_retains_literal_eq24_ratio():
    gmis = np.array([3.0, 4.0], dtype=np.float64)
    greg = np.array([1.0, 2.0], dtype=np.float64)

    w = fathi_eq24_weight_with_flat_parent_policy(
        gmis,
        greg,
        varrho=0.5,
        max_gradient_sq=1.0,
        epsilon_gradient_sq=1.0e-2,
        gradient_roundoff_bound_sq=1.0e-26,
        q_value=8000.0,
        q_floor_value=7200.0,
        q_floor_relative_tolerance=1.0e-11,
    )
    assert w.status == EQ24_ACTIVE_STATUS
    ratio = np.linalg.norm(w.weight * greg) / np.linalg.norm(gmis)
    assert np.isclose(ratio, 0.5, rtol=0.0, atol=1.0e-15)


def test_q1_roundoff_bound_scales_with_field_and_mesh():
    coarse = q1_constant_gradient_roundoff_bound_sq(
        max_abs_field_mpa=80.0,
        min_dx_m=1.25,
        min_dy_m=1.25,
        min_dz_m=1.25,
    )
    fine = q1_constant_gradient_roundoff_bound_sq(
        max_abs_field_mpa=80.0,
        min_dx_m=0.5,
        min_dy_m=0.5,
        min_dz_m=0.5,
    )
    larger_material = q1_constant_gradient_roundoff_bound_sq(
        max_abs_field_mpa=160.0,
        min_dx_m=1.25,
        min_dy_m=1.25,
        min_dz_m=1.25,
    )
    assert fine > coarse
    assert larger_material > coarse
    assert np.isclose(larger_material / coarse, 4.0)


def test_gamma_n_sum_bound_grows_with_quadrature_count():
    assert floating_sum_relative_bound(1000) > floating_sum_relative_bound(100)


def test_zero_effective_beta_keeps_parent_total_equal_to_data():
    out = compose_frozen_regularized_objective(
        data_objective=3.0e-8,
        q_lambda=7200.0,
        q_mu=7200.0,
        beta_lambda=0.0,
        beta_mu=0.0,
    )
    assert out["J_reg"] == 0.0
    assert out["J_total"] == out["J_data"]


def test_positive_secant_curvature_is_accepted():
    out = regularized_secant_pair_outcome(
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
    )
    assert out["s_dot_y"] > 0.0
    assert out["use_in_lbfgs_history"] is True
    assert out["status"] == SECANT_PAIR_ACCEPTED_STATUS


def test_nonpositive_secant_curvature_is_explicitly_skipped():
    out = regularized_secant_pair_outcome(
        np.array([1.0, 0.0]),
        np.array([-2.0, 0.0]),
    )
    assert out["s_dot_y"] < 0.0
    assert out["use_in_lbfgs_history"] is False
    assert out["status"] == SECANT_PAIR_SKIPPED_STATUS
