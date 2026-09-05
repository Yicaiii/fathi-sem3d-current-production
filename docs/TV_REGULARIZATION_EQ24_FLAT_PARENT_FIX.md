# Gate 4C — Eq.24 flat-parent dual-gate singularity handling

## Mathematical issue

For smoothed TV

`Q(m) = integral sqrt(|grad m|^2 + epsilon) dOmega`,

an exactly homogeneous parent satisfies `grad m = 0` pointwise and therefore
`dQ/dm = 0` exactly.  Fathi Eq. (24),

`beta = varrho ||g_mis|| / ||g_reg||`,

is singular at that state.

The first production regularized Armijo run exposed this boundary condition.
Thirteen completed candidate forwards all decreased the data objective while
`Delta Q` scaled quadratically with alpha, confirming a zero first derivative
of Q at the homogeneous parent.  The old Eq.24 path had normalized a tiny Q1
cancellation residual to 50% of the data covector norm.

## V2 dual-gate flat-parent classification

The singular branch is now state-triggered, not iteration-triggered.  Eq.24 is
deferred only if BOTH conditions hold.

1. Gradient-roundoff gate

The observed `max |grad m_h|^2` must be below a Q1 constant-field floating
roundoff bound derived from:

- IEEE-754 unit roundoff;
- the eight-term Q1 derivative dot product (`gamma_8`);
- the actual material magnitude in MPa;
- the actual minimum x/y/z mesh spacing.

No fixed `4096 * eps` threshold is used.

2. Scalar epsilon-floor gate

The assembled Q must agree with the discrete theoretical epsilon floor

`Q_floor = sqrt(epsilon) * sum(element volumes)`

within a `gamma_n` accumulation bound where `n` is the actual number of Q1
Gauss evaluations.

Only when both gates pass is the effective beta set to zero for that outer
iteration.  At any non-flat parent, the literal Fathi Eq.24 expression remains
unchanged.

## Audit margins

The Eq21 summary records for lambda and mu:

- observed max-gradient-squared;
- scale/mesh-aware roundoff bound;
- bound/observed margin and log10 margin;
- Q and Q_floor;
- Q-floor relative error and gamma_n tolerance;
- both gate booleans;
- Eq24 status and effective beta.

For the certified current S43 homogeneous parent reconstructed on the
37x33x33, h=1.25 m grid with 80 MPa material values, the pre-application
forensic calculation gives approximately:

- observed `max |grad m|^2 = 1.54617e-30`;
- roundoff bound squared `= 3.87741e-26`;
- margin `= 2.5078e4` (about 4.40 orders of magnitude);
- Q = Q_floor = 7200 exactly in float64 for this grid;
- quadrature count = 294912;
- gamma_n Q-floor relative tolerance about `3.274e-11`.

The apply script recomputes these numbers from the actual accepted H5 files
before changing the source.

## Second-line status/beta contract

The registration layer independently rechecks:

- DEFERRED_FLAT_TV_PARENT -> both gates pass, beta = 0, weighted ratio = 0;
- ACTIVE_EQ24 -> not both gates pass, beta > 0, weighted ratio = varrho.

This prevents a malformed Eq21 summary from being registered as a physical
gradient.

## First secant-pair curvature policy

The regularized code now carries an explicit reusable policy:

`use (s,y) in L-BFGS history iff s^T y > 0; otherwise record and skip it`.

The current Gate 4C driver still authorizes only iter000->iter001 and starts
with empty history, so no s0/y0 pair exists yet.  The helper and tests are added
now so that the first pair spanning beta=0 -> active Eq24 cannot be silently
ingested when iter001's gradient is later computed.

## Unchanged mathematics

This patch does not change:

- Q1 smoothed-TV formula or local denominator;
- epsilon = 0.01;
- MPa -> Pa chain rule;
- non-flat Fathi Eq.24;
- Eq21/Mtilde definition;
- frozen-beta scalar convention;
- Eq9 factor-of-two mapping;
- Eq25;
- L-BFGS direction mathematics;
- Armijo parameters;
- any SEM3D forward or exact reverse operator.
