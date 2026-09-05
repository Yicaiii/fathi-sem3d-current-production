from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EQ24_FLAT_PARENT_POLICY_VERSION = "FLAT_PARENT_DUAL_GATE_DEFER_V2"
EQ24_ACTIVE_STATUS = "ACTIVE_EQ24"
EQ24_DEFERRED_FLAT_STATUS = "DEFERRED_FLAT_TV_PARENT"

SECANT_PAIR_CURVATURE_POLICY_VERSION = "POSITIVE_S_DOT_Y_ELSE_SKIP_V1"
SECANT_PAIR_ACCEPTED_STATUS = "ACCEPTED_POSITIVE_CURVATURE"
SECANT_PAIR_SKIPPED_STATUS = "SKIPPED_NONPOSITIVE_CURVATURE"


@dataclass(frozen=True)
class Eq24Weight:
    varrho: float
    misfit_l2: float
    regularization_l2: float
    weight: float
    status: str = EQ24_ACTIVE_STATUS

    max_gradient_sq: float | None = None
    epsilon_gradient_sq: float | None = None
    max_gradient_sq_over_epsilon: float | None = None

    gradient_roundoff_bound_sq: float | None = None
    gradient_roundoff_bound_sq_over_epsilon: float | None = None
    gradient_gate_pass: bool | None = None
    gradient_gate_margin: float | None = None
    gradient_gate_margin_orders: float | None = None

    q_value: float | None = None
    q_floor_value: float | None = None
    q_floor_relative_error: float | None = None
    q_floor_relative_tolerance: float | None = None
    q_floor_gate_pass: bool | None = None
    q_floor_gate_margin: float | None = None
    q_floor_exact: bool | None = None


@dataclass(frozen=True)
class SecantCurvatureOutcome:
    policy_version: str
    s_dot_y: float
    s_l2: float
    y_l2: float
    normalized_curvature: float
    use_in_lbfgs_history: bool
    status: str


def _unit_roundoff() -> float:
    # IEEE-754 unit roundoff u = eps / 2.
    return float(np.finfo(np.float64).eps / 2.0)


def floating_sum_relative_bound(term_count: int) -> float:
    """Higham gamma_n bound for n-term floating-point accumulation."""

    n = int(term_count)
    if n <= 0:
        raise ValueError("term_count must be positive")
    u = _unit_roundoff()
    nu = n * u
    if not nu < 1.0:
        raise ValueError("term_count too large for gamma_n bound")
    return float(nu / (1.0 - nu))


def q1_constant_gradient_roundoff_bound_sq(
    *,
    max_abs_field_mpa: float,
    min_dx_m: float,
    min_dy_m: float,
    min_dz_m: float,
) -> float:
    """Conservative Q1 constant-field gradient roundoff bound.

    At a Q1 Gauss point, each reference-coordinate derivative is an
    eight-term dot product.  For a constant nodal field the exact dot product
    is zero, while the floating-point residual is bounded by gamma_8 times
    ``max_abs_field * sum(abs(dN/dxi))``.  For the trilinear Q1 basis at the
    2x2x2 Gauss points, that L1 coefficient sum is exactly 1 for each axis.

    Multiplication by the physical inverse-Jacobian gives 2/h per axis.
    This makes the bound scale explicitly with material magnitude and mesh
    spacing instead of using a fixed multiple of machine epsilon.
    """

    M = float(max_abs_field_mpa)
    hx = float(min_dx_m)
    hy = float(min_dy_m)
    hz = float(min_dz_m)
    if not math.isfinite(M) or M < 0.0:
        raise ValueError("max_abs_field_mpa must be finite and non-negative")
    if not all(math.isfinite(v) and v > 0.0 for v in (hx, hy, hz)):
        raise ValueError("minimum mesh spacings must be finite and positive")

    gamma8 = floating_sum_relative_bound(8)
    bx = gamma8 * M * (2.0 / hx)
    by = gamma8 * M * (2.0 / hy)
    bz = gamma8 * M * (2.0 / hz)
    return float(bx * bx + by * by + bz * bz)


def _safe_margin(limit: float, observed: float) -> tuple[float | None, float | None]:
    if observed == 0.0:
        return None, None
    margin = float(limit / observed)
    if margin <= 0.0 or not math.isfinite(margin):
        return margin, None
    return margin, float(math.log10(margin))


def fathi_eq24_weight(
    misfit_covector: np.ndarray,
    regularization_covector: np.ndarray,
    *,
    varrho: float,
) -> Eq24Weight:
    """Return the non-degenerate Fathi Eq. (24) regularization factor.

    beta = varrho * ||g_mis||_2 / ||g_reg||_2.

    Both vectors must already use the same control coordinate and units.
    """

    if not math.isfinite(varrho) or varrho <= 0.0:
        raise ValueError("varrho must be finite and positive")
    gmis = np.asarray(misfit_covector, dtype=np.float64).reshape(-1)
    greg = np.asarray(regularization_covector, dtype=np.float64).reshape(-1)
    if gmis.shape != greg.shape:
        raise ValueError("misfit and regularization covectors must have identical shape")
    if not np.all(np.isfinite(gmis)) or not np.all(np.isfinite(greg)):
        raise ValueError("Eq.24 covectors must be finite")

    misfit_l2 = float(np.linalg.norm(gmis))
    regularization_l2 = float(np.linalg.norm(greg))
    if regularization_l2 == 0.0:
        raise ZeroDivisionError("Eq.24 is undefined for a zero regularization covector")

    weight = float(varrho * misfit_l2 / regularization_l2)
    return Eq24Weight(
        varrho=float(varrho),
        misfit_l2=misfit_l2,
        regularization_l2=regularization_l2,
        weight=weight,
        status=EQ24_ACTIVE_STATUS,
    )


def fathi_eq24_weight_with_flat_parent_policy(
    misfit_covector: np.ndarray,
    regularization_covector: np.ndarray,
    *,
    varrho: float,
    max_gradient_sq: float,
    epsilon_gradient_sq: float,
    gradient_roundoff_bound_sq: float,
    q_value: float,
    q_floor_value: float,
    q_floor_relative_tolerance: float,
) -> Eq24Weight:
    """Apply Eq. (24) with an explicit dual-gate singular-start policy.

    For a spatially constant smoothed-TV parent, dQ/dm is analytically zero
    and Eq. (24) is undefined because ||g_reg|| = 0.  Floating-point Q1
    assembly can leave a cancellation residual.  That residual must not be
    normalized to ``varrho * ||g_mis||``.

    A parent is classified as *numerically TV-flat* only when BOTH independent
    checks pass:

      1. the observed max |grad m_h|^2 is below a scale- and mesh-aware Q1
         constant-field roundoff bound;
      2. Q itself is at the discrete epsilon-floor within a gamma_n
         accumulation bound supplied by the caller.

    Only in that dual-gate state is the effective Eq.24 weight set to zero for
    the current outer iteration.  At every non-flat parent the literal Fathi
    Eq.24 formula is retained unchanged.
    """

    scalars = (
        varrho,
        max_gradient_sq,
        epsilon_gradient_sq,
        gradient_roundoff_bound_sq,
        q_value,
        q_floor_value,
        q_floor_relative_tolerance,
    )
    if not all(math.isfinite(float(v)) for v in scalars):
        raise ValueError("Eq.24 flat-parent evidence must be finite")
    if varrho <= 0.0:
        raise ValueError("varrho must be positive")
    if max_gradient_sq < 0.0:
        raise ValueError("max_gradient_sq must be non-negative")
    if epsilon_gradient_sq <= 0.0:
        raise ValueError("epsilon_gradient_sq must be positive")
    if gradient_roundoff_bound_sq < 0.0:
        raise ValueError("gradient_roundoff_bound_sq must be non-negative")
    if q_floor_value <= 0.0:
        raise ValueError("q_floor_value must be positive")
    if q_floor_relative_tolerance < 0.0:
        raise ValueError("q_floor_relative_tolerance must be non-negative")

    gmis = np.asarray(misfit_covector, dtype=np.float64).reshape(-1)
    greg = np.asarray(regularization_covector, dtype=np.float64).reshape(-1)
    if gmis.shape != greg.shape:
        raise ValueError("misfit and regularization covectors must have identical shape")
    if not np.all(np.isfinite(gmis)) or not np.all(np.isfinite(greg)):
        raise ValueError("Eq.24 covectors must be finite")

    grad_ratio = float(max_gradient_sq / epsilon_gradient_sq)
    grad_bound_ratio = float(gradient_roundoff_bound_sq / epsilon_gradient_sq)
    grad_pass = bool(max_gradient_sq <= gradient_roundoff_bound_sq)
    grad_margin, grad_orders = _safe_margin(
        gradient_roundoff_bound_sq,
        max_gradient_sq,
    )

    q_rel_error = float(abs(q_value - q_floor_value) / q_floor_value)
    q_pass = bool(q_rel_error <= q_floor_relative_tolerance)
    q_margin, _ = _safe_margin(q_floor_relative_tolerance, q_rel_error)
    q_exact = bool(q_value == q_floor_value)

    common = {
        "max_gradient_sq": float(max_gradient_sq),
        "epsilon_gradient_sq": float(epsilon_gradient_sq),
        "max_gradient_sq_over_epsilon": grad_ratio,
        "gradient_roundoff_bound_sq": float(gradient_roundoff_bound_sq),
        "gradient_roundoff_bound_sq_over_epsilon": grad_bound_ratio,
        "gradient_gate_pass": grad_pass,
        "gradient_gate_margin": grad_margin,
        "gradient_gate_margin_orders": grad_orders,
        "q_value": float(q_value),
        "q_floor_value": float(q_floor_value),
        "q_floor_relative_error": q_rel_error,
        "q_floor_relative_tolerance": float(q_floor_relative_tolerance),
        "q_floor_gate_pass": q_pass,
        "q_floor_gate_margin": q_margin,
        "q_floor_exact": q_exact,
    }

    if grad_pass and q_pass:
        return Eq24Weight(
            varrho=float(varrho),
            misfit_l2=float(np.linalg.norm(gmis)),
            regularization_l2=float(np.linalg.norm(greg)),
            weight=0.0,
            status=EQ24_DEFERRED_FLAT_STATUS,
            **common,
        )

    active = fathi_eq24_weight(
        gmis,
        greg,
        varrho=varrho,
    )
    return Eq24Weight(
        varrho=active.varrho,
        misfit_l2=active.misfit_l2,
        regularization_l2=active.regularization_l2,
        weight=active.weight,
        status=active.status,
        **common,
    )


def evaluate_secant_pair_curvature(
    s: np.ndarray,
    y: np.ndarray,
) -> SecantCurvatureOutcome:
    """Evaluate the L-BFGS curvature condition and explicitly accept/skip.

    The pair is eligible for L-BFGS history iff s^T y > 0.  Non-positive
    curvature is not silently ingested; it is recorded as a skipped pair.
    This is especially important for the first pair that spans the
    flat-parent Eq.24 activation transition.
    """

    ss = np.asarray(s, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    if ss.shape != yy.shape:
        raise ValueError("secant s and y must have identical shape")
    if ss.size == 0:
        raise ValueError("secant pair must be non-empty")
    if not np.all(np.isfinite(ss)) or not np.all(np.isfinite(yy)):
        raise ValueError("secant pair must be finite")

    s_l2 = float(np.linalg.norm(ss))
    y_l2 = float(np.linalg.norm(yy))
    sty = float(np.dot(ss, yy))
    denom = s_l2 * y_l2
    normalized = float(sty / denom) if denom > 0.0 else 0.0
    use = bool(sty > 0.0)

    return SecantCurvatureOutcome(
        policy_version=SECANT_PAIR_CURVATURE_POLICY_VERSION,
        s_dot_y=sty,
        s_l2=s_l2,
        y_l2=y_l2,
        normalized_curvature=normalized,
        use_in_lbfgs_history=use,
        status=(
            SECANT_PAIR_ACCEPTED_STATUS
            if use
            else SECANT_PAIR_SKIPPED_STATUS
        ),
    )
