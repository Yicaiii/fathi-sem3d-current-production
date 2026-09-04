from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Eq24Weight:
    varrho: float
    misfit_l2: float
    regularization_l2: float
    weight: float


def fathi_eq24_weight(
    misfit_covector: np.ndarray,
    regularization_covector: np.ndarray,
    *,
    varrho: float,
) -> Eq24Weight:
    """Return the Fathi Eq. (24) parameter-specific regularization factor.

    R = varrho * ||g_mis||_2 / ||g_reg||_2.

    Both input vectors must be expressed in the same control coordinate and
    units before this routine is called.
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
    )
