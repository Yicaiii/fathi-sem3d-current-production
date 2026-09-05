# Gate 4C exact-reverse summary dt contract

The regularized exact-reverse wrapper replays the parent data objective using
the immutable certified quadrature timestep from
`certified_reference.contract.dt`.

The certified reverse delegate already consumes the wrapper-provided
`objective_weights`, so the numerical reverse is correct. Before this patch,
however, the final reverse summary serialized `driver.dt` into
`objective.dt`, even when the objective was defined using `objective_dt`.

This patch changes summary provenance only:

- `objective.dt` records the actual quadrature timestep used for `J_data`;
- `objective.driver_dt` records the numerical driver timestep separately;
- `objective.dt_source` records whether the value came from the certified
  contract or from the legacy driver fallback.

No forward, reverse, SEM3D, optimizer, TV, or adjoint mathematics is changed.
