import numpy as np

from scripts.fathi_benchmark.regularization.tv_objective import (
    frozen_total_tv_objective,
    objective_convention_from_eq21_beta,
)
from scripts.fathi_benchmark.regularization.tv_q1 import assemble_smoothed_tv_q1


def test_eq9_equivalent_factor_maps_to_eq21_beta() -> None:
    beta = 3.7e-6
    q = 12.5
    c = objective_convention_from_eq21_beta(beta)
    assert c.r_eq9_equivalent == 2.0 * beta
    assert np.isclose(0.5 * c.r_eq9_equivalent * q, beta * q, rtol=0.0, atol=0.0)


def test_literal_beta_over_2_would_be_factor_two_wrong() -> None:
    beta = 2.0e-4
    qprime_dot_d = -8.0
    desired = beta * qprime_dot_d
    wrong = 0.5 * beta * qprime_dot_d
    assert np.isclose(wrong / desired, 0.5, rtol=0.0, atol=0.0)


def test_frozen_tv_objective_matches_directional_derivative_in_pa_coordinate() -> None:
    x = np.linspace(-2.0, 2.0, 5)
    y = np.linspace(-1.5, 1.5, 4)
    z = np.linspace(-3.0, 0.0, 6)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    lambda_mpa = 80.0 + 0.7 * xx + 0.2 * yy * yy + 0.13 * zz * zz
    mu_mpa = 75.0 + 0.1 * xx * xx - 0.4 * yy + 0.09 * zz * zz
    epsilon = 0.01
    beta_lambda = 1.3e-5
    beta_mu = 0.9e-5

    base_lambda = assemble_smoothed_tv_q1(
        lambda_mpa,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    base_mu = assemble_smoothed_tv_q1(
        mu_mpa,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )

    i = np.arange(lambda_mpa.size, dtype=np.float64)
    d_lambda_pa = (np.sin(0.37 * (i + 1.0))).reshape(lambda_mpa.shape) * 1.0e6
    d_mu_pa = (np.cos(0.29 * (i + 1.0))).reshape(mu_mpa.shape) * 1.0e6
    q_lambda_pa = base_lambda.covector / 1.0e6
    q_mu_pa = base_mu.covector / 1.0e6
    analytic = float(
        beta_lambda * np.sum(q_lambda_pa * d_lambda_pa)
        + beta_mu * np.sum(q_mu_pa * d_mu_pa)
    )

    h = 1.0e-4
    plus_lambda = assemble_smoothed_tv_q1(
        lambda_mpa + h * d_lambda_pa / 1.0e6,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    plus_mu = assemble_smoothed_tv_q1(
        mu_mpa + h * d_mu_pa / 1.0e6,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    minus_lambda = assemble_smoothed_tv_q1(
        lambda_mpa - h * d_lambda_pa / 1.0e6,
        x_m=x,
        y_m=y,
        z_m=z,
        epsilon_gradient_sq=epsilon,
    )
    minus_mu = assemble_smoothed_tv_q1(
        mu_mpa - h * d_mu_pa / 1.0e6,
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
    finite_difference = (j_plus - j_minus) / (2.0 * h)
    assert np.isclose(finite_difference, analytic, rtol=2.0e-7, atol=1.0e-12)
