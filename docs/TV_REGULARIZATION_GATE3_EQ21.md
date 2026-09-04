# Fathi TV regularization — Gate 3 Eq. (21) control-space assembly

This gate is additive and audit-only. It does not modify the certified data-only lineage and does not run SEM3D, an external forward, the exact reverse, L-BFGS, or Armijo.

## Certified splice point

The CURRENT gradient bridge first maps the exact reverse material VJP to the full Q1 material-control grid and then restricts it to the canonical active Mtilde controls. The arrays

- `full_control_covector_lambda.npy`
- `full_control_covector_mu.npy`

are the full physical-lambda/mu pre-Mtilde data covectors. The canonical active data vectors are their restriction by `active_h5_indices.npy` and are stored as

- `rhs/full_grid_trace_RHS_total_lambda.npy`
- `rhs/full_grid_trace_RHS_total_mu.npy`.

Only after this stage is `Mtilde g = rhs` solved to create the registered physical Pa-space Riesz gradients.

Therefore Fathi Eq. (24) is evaluated on the pre-Mtilde active vectors, not on `mtilde_solve/g_lambda.npy` or `g_mu.npy`.

## TV coordinate conversion

The TV quadrature kernel uses MPa as its material coordinate. If

`q_MPa = dQ/dm_MPa`, then

`q_Pa = 1e-6 q_MPa`

because `dm_MPa = 1e-6 dm_Pa`.

The Gate-3 vector is flattened in the CURRENT H5 C-order and restricted with the same `active_h5_indices.npy` used by the certified data bridge.

## Eq. (24)

For each material parameter,

`beta = varrho * ||g_mis||_2 / ||g_reg||_2`

with both vectors in the same active pre-Mtilde physical-Pa coordinate. The gate verifies

`||beta g_reg||_2 / ||g_mis||_2 = varrho`.

For the current p20 regularization profile, `varrho = 0.5` is a paper-informed stage policy. Section 4.3 does not explicitly state its exact value.

## Mtilde regression

Before solving the total right-hand side, the gate solves the existing data-only right-hand side with the existing Mtilde and requires reproduction of the already-registered data-only gradient. This certifies the splice point without altering the baseline.

Then it forms

`rhs_total = rhs_data + beta * rhs_tv`

and solves

`Mtilde g_total = rhs_total`.

The total solve is still audit-only. It is not registered as a production optimizer gradient.

## Objective normalization deliberately deferred

Fathi Eq. (9) writes a factor `R/2` in the scalar TV functional, while Eq. (17) and Appendix C use compressed notation for the regularization gradient. Gate 3 does not silently choose a scalar normalization. Production Armijo remains blocked until a total-objective directional derivative gate fixes one scalar/gradient convention and verifies it numerically.
