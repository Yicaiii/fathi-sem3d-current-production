# Fathi TV regularization — Phase 1 mathematical gate

## Scope

This change is additive and audit-only. It does **not** alter the certified
`fathi_s43_repro_p20_t052` data-misfit baseline and does **not** run a forward,
reverse, L-BFGS, Armijo, or promotion stage.

The purpose is to freeze and test the missing regularization mathematics before
it is connected to the CURRENT iteration engine.

## Frozen paper equations

The new regularized lineage targets

\[
J(\lambda,\mu)=J_{data}(\lambda,\mu)+R_{TV}(\lambda,\mu).
\]

For one material component `m`, the unweighted TV component is represented as

\[
Q(m)=\int_{\Omega^{RD}}\sqrt{|\nabla m|^2+\epsilon}\,d\Omega.
\]

Its discrete Q1 covector is assembled pointwise at each quadrature point, in
accordance with Fathi Eq. (17) and Appendix C. The implementation uses the
literal `+ epsilon`; it does not use `epsilon**2`.

The paper's Eq. (24) factor is

\[
R=\wp\frac{\|g_{mis}\|_2}{\|g_{reg}\|_2}.
\]

Eq. (24) must only be applied after `g_mis` and `g_reg` have been mapped into
the same pre-Mtilde active control coordinate and units.

## Coordinate contract

- array order: `field[iz, iy, ix]`
- flatten order when later restricted to CURRENT active controls: `C`
- spatial coordinates: metres
- TV material coordinates: MPa
- epsilon: `0.01 (MPa/m)^2`
- Q1 integration: `2x2x2` Gauss-Legendre
- denominator: evaluated independently at every quadrature point

The Pa↔MPa chain rule is intentionally **not** connected to the CURRENT
pre-Mtilde RHS in Phase 1. That conversion is part of Gate 3 and must be audited
against the existing gradient bridge before production integration.

## Gate 1

1. constant material -> zero regularization covector;
2. layered piecewise-constant material -> finite non-zero interface response;
3. layered material -> quadrature denominators vary pointwise;
4. `max(|grad m|^2) / epsilon >= 100` for the certified S43 layered target;
5. no SEM3D/external-forward/exact-reverse execution.

## Gate 2

The directional derivative must satisfy

\[
\frac{Q(m+h\delta m)-Q(m-h\delta m)}{2h}
\approx g_{reg}^T\delta m.
\]

This is the decisive test against an incorrect global denominator.

## Later gates — not implemented by this commit

### Gate 3

- restrict full-grid TV covector to CURRENT active controls;
- audit MPa→Pa chain rule;
- compute parameter-specific Eq. (24) weights;
- assemble `g_mis + R g_reg` before the existing Mtilde solve;
- verify the total objective directional derivative;
- freeze `R_lambda`, `R_mu` for the entire Armijo line search.

### Gate 4

Create the separate `fathi_s43_repro_tv_p20_t052` lineage, reset L-BFGS
history, run one regularized `iter000→iter001`, and require final closure audit
before any further iteration is authorized.

The existing unregularized `iter000→iter008` results remain immutable baseline
artifacts.


## Factor convention audit

The printed Eq. (9) contains a factor `R/2`, while Eq. (17) and Appendix C
write the regularization control vector without a corresponding `1/2`.  Phase 1
therefore freezes the **Appendix-C base pair**

\[
Q(m)=\int\sqrt{|\nabla m|^2+\epsilon}\,d\Omega,
\qquad g_{reg}=Q'(m),
\]

and later applies the Eq. (24) algorithmic weight to this pair.  This convention
keeps the objective and covector exactly derivative-consistent.  Because Eq. (24)
normalizes by `||g_reg||`, an alternative half-scaled base pair gives the same
weighted regularization term and weighted gradient.  Gate 2 enforces this
derivative consistency explicitly.
