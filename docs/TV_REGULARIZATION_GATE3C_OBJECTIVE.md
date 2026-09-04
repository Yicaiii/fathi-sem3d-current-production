# Fathi TV regularization — Gate 3C objective convention

Gate 3C resolves the factor-of-two notation mismatch between Fathi Eq. (9) and the TV derivative written in Eq. (17)/Appendix C without changing the already-certified Gate-3B control assembly.

Let

\[
Q(m)=\int_{\Omega^{RD}}\sqrt{|\nabla m|^2+\epsilon}\,d\Omega,
\qquad q(m)=Q'(m).
\]

Gate 3B uses the Eq. (21)/(24) coefficient

\[
\beta=\wp\frac{\|g_{mis}\|_2}{\|q\|_2}
\]

and assembles

\[
b_{total}=g_{mis}+\beta q.
\]

For scalar/gradient consistency during line search, the frozen regularization objective is therefore

\[
J_{reg}=\beta Q.
\]

If Eq. (9) is written literally as \((R/2)Q\), the equivalent Eq. (9) symbol is

\[
R_{Eq9}=2\beta,
\]

so that

\[
\frac{R_{Eq9}}{2}Q=\beta Q,
\qquad
\frac{d}{dm}\left(\frac{R_{Eq9}}{2}Q\right)=\beta q.
\]

The code must not use \((\beta/2)Q\) while simultaneously using \(\beta q\) in the gradient. That combination is inconsistent by exactly a factor of two.

The Eq. (24) coefficient is computed once at the parent model and frozen for all candidate evaluations in that line search.

Gate 3C is audit-only. It executes no SEM3D run, no external forward, no reverse, and no optimizer. It performs a central directional finite-difference check of the frozen TV scalar objective against the Gate-3B regularization covector in the same active physical-Pa control coordinate.
