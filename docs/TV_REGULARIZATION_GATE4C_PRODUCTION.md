# Gate 4C — first production regularized transition

Source-code baseline:

- required certified source ancestor: `917c721`
- repository root is resolved dynamically from `FATHI_BENCHMARK_ROOT`
- run identity is read from the regularized engine/runtime configs
- the current branch and remote are recorded as provenance, not hard-coded as execution requirements

The old heavy repository is used only as the runtime/artifact root through
`FATHI_RUNTIME_ROOT`. It is not a source-code authority.

## Scope

Gate 4C initially permits **only**:

`regularized iter000 -> iter001`

This intentionally avoids generalizing L-BFGS history before the first
regularized closure is certified.

## Preserved components

- certified external forward
- exact discrete reverse
- corrected data-gradient bridge
- Mtilde machinery
- physical L-BFGS implementation
- Eq.25 lambda bias
- raw candidate generation

## Added regularized components

1. Gate-3B Eq.21/Eq.24 total RHS assembly.
2. Registered total physical gradient in the new run namespace.
3. Parent-local frozen-weight objective:
   `J_total = J_data + beta_lambda Q_lambda + beta_mu Q_mu`.
4. Candidate TV evaluation after the certified candidate data forward.
5. Armijo decision on `J_total`, with parent `beta_lambda`, `beta_mu` frozen
   across all trial alphas.
6. Promotion that keeps `objective.accepted = J_data` for compatibility with
   the certified forward/reverse route and records the accepted `J_total`
   separately.
7. Dedicated regularized final-closure audit.

## Important

The data-only CURRENT driver is not modified.

No unregularized `K=8` continuation is authorized.
