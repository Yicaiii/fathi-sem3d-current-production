from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TVObjectiveConvention:
    beta_eq21: float
    r_eq9_equivalent: float


def objective_convention_from_eq21_beta(beta_eq21: float) -> TVObjectiveConvention:
    beta = float(beta_eq21)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta_eq21 must be finite and positive")
    return TVObjectiveConvention(
        beta_eq21=beta,
        r_eq9_equivalent=2.0 * beta,
    )


def frozen_tv_component(q_value: float, *, beta_eq21: float) -> float:
    q = float(q_value)
    beta = float(beta_eq21)
    if not math.isfinite(q) or not math.isfinite(beta):
        raise ValueError("TV objective inputs must be finite")
    return beta * q


def frozen_total_tv_objective(
    q_lambda: float,
    q_mu: float,
    *,
    beta_lambda_eq21: float,
    beta_mu_eq21: float,
) -> float:
    return frozen_tv_component(q_lambda, beta_eq21=beta_lambda_eq21) + frozen_tv_component(
        q_mu, beta_eq21=beta_mu_eq21
    )
