# Gate 4C regularized iter000 bootstrap-parent identity

The regularized `iter000` accepted summary is a certified bootstrap artifact.
It predates any regularized transition and therefore intentionally does not
contain `parent_iteration`, `child_iteration`, or `transition`.

`require_identity()` is the correct contract for promoted accepted parents,
but applying it directly to this bootstrap summary incorrectly manufactures a
requirement for predecessor-transition provenance that does not exist.

The regularized artifact layer now distinguishes two strict cases:

- promoted parents: full transition identity remains mandatory;
- regularized iter000 bootstrap: run id, `iter=0`, exact bootstrap lineage
  classification, `optimizer_history_reused=false`, and material SHA
  provenance are mandatory, while transition fields must all be absent.

This does not modify the bootstrap accepted summary, the gradient values,
Eq.21/24, TV, Mtilde, L-BFGS, forward, or exact reverse mathematics.
