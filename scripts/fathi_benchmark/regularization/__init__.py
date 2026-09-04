"""Fathi-style material regularization utilities.

This package is intentionally additive.  It does not mutate the certified
unregularized CURRENT production lineage.
"""

from .tv_q1 import TVQ1Result, assemble_smoothed_tv_q1
from .tv_weight import Eq24Weight, fathi_eq24_weight

__all__ = [
    "TVQ1Result",
    "assemble_smoothed_tv_q1",
    "Eq24Weight",
    "fathi_eq24_weight",
]
