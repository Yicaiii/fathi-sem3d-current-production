# Fathi TV Gate 4B — regularized lineage bootstrap

This gate prepares a **new** regularized run namespace without executing any
forward, reverse, SEM3D, or optimizer step.

## Why a separate bootstrap is required

The reusable CURRENT driver deliberately starts at accepted `iter_001`.
The historical `iter_000` workspace has no canonical `accepted_summary.json`,
and the certified external-reference manifest is run-identity bound.

Therefore the regularized lineage must not be started by pointing the old
driver at `K=0` or by reusing the old optimizer history.

## Frozen bootstrap policy

- baseline: `fathi_s43_repro_p20_t052`
- regularized run: `fathi_s43_repro_tv_p20_t052`
- `iter_000` material values are byte-identical to the certified baseline
- only the three material HDF5 files are copied into the new accepted parent
- the certified data objective remains in `objective.accepted` for compatibility
  with the already-certified parent-forward / exact-reverse route
- the regularized total objective is **not** stored as a single permanent
  accepted objective because Eq.24 weights change between outer iterations
- a parent-local frozen-weight total-objective manifest will be created after
  the parent gradient determines `beta_lambda` and `beta_mu`
- baseline optimizer history is never reused
- the external-reference manifest is derived with a new `run` identity while
  immutable operator / TRUE assets retain their original certified provenance

## J0 consistency rule

The derived runtime uses the already-frozen optimizer
`fixed_reproduction_scaling.J_ref` as the certified data `J0`. This avoids
propagating any stale `production_objective.J0` field from an older runtime
configuration.

## Gate 4B itself is non-numerical

Expected run counters:

- SEM3D: 0
- external forward: 0
- exact reverse: 0
- optimizer: 0
