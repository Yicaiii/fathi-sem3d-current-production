# Gate 4C bootstrap-parent provenance integration

Regularized `iter000` is a certified bootstrap parent, not a predecessor
Armijo promotion.

The exact-reverse wrapper now distinguishes
`BOOTSTRAP_CERTIFIED_PRIMAL_PARENT` from `PROMOTED_ARMIJO_PARENT`.
Bootstrap trace identity is certified through the canonical primal-forward
summary, the certified-parent receiver SHA256, the immutable TRUE receiver
SHA256 in the certified reference, and the bootstrap accepted data objective.

Exact-reverse objective quadrature uses `certified_reference.contract.dt`.
The numerical driver timestep remains separately validated within the existing
certified tolerance.

The corrected-gradient bridge mirrors the same bootstrap/promoted distinction.
No numerical forward, reverse, SEM3D, optimizer, or mathematical algorithm is
changed by this compatibility patch.
