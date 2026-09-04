# Gate 4C parent-forward dt root cause

The first regularized iter000 parent forward completed all 1040 samples and
produced a receiver array that is bitwise identical to the certified baseline,
but the live wrapper blocked on `objective_bitwise_equal_to_accepted`.

The root cause is now reproduced exactly:

- `ExternalForwardDriver.dt = 0.0004999999999999999`
- certified `reference.contract.dt = 0.0005`
- the two binary64 values differ by exactly 1 ULP
- the existing wrapper accepted this through its `abs_tol=1e-18` dt gate
- the wrapper then formed the objective quadrature with `driver.dt`
- this produced an objective exactly 1 ULP below the certified scalar
- recomputing the same receiver with `reference.contract.dt` reproduces the
  certified objective bitwise (ULP distance 0)

This was therefore a contract inconsistency in the certification wrapper, not a
forward-operator, receiver, material, TRUE-data, TV, or optimization failure.

## Frozen correction

The external forward operator keeps its runtime `driver.dt`.  The data-objective
quadrature uses the immutable certified objective timestep:

`dt_J = reference.contract.dt`.

The parent-forward and regularized candidate-objective paths both use this
canonical convention.  The driver/reference dt compatibility gate remains
strict (`rel_tol=0`, `abs_tol=1e-18`) and both binary64 values are persisted for
provenance.

The already-completed iter000 parent forward is recertified from its durable
receiver/checkpoint artifacts; no numerical forward rerun is authorized or
required.
