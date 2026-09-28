# Unknown-mean normal variance ESS: unresolved source conventions

This is an open part of BayesESS entry 154, separate from the completed
regression calculator. Sources are the pinned BayesESS `essNormal` function
in `R/internal.R`, its `R/ess.R` wrappers and support appendix 3.2.3.1;
their hashes are recorded in
[conjugate-ess-sources.json](../docs/conjugate-ess-sources.json).
Morita, Thall and Müller's original manuscript is recorded in
[regression-ess-sources.json](../docs/regression-ess-sources.json).
Algorithm 2 and the degrees-of-freedom adjustment are on printed page 10
(zero-based PDF page 9).

## Model and unadjusted curvature

Write v for variance, with v ~ InvChiSq(nu, s²) and
mu conditional on v ~ Normal(mu0, v/phi). The prior mean
vbar = nu*s²/(nu-2) requires nu>2. An equivalent inverse-gamma prior
has shape nu/2 and scale nu*s²/2.

The support appendix specifies an epsilon prior with conditional mean
variance c*v/phi, inverse-chi-square degrees of freedom 4+1/c and
scale variance vbar/2. At (vbar, mu0), differentiating those stated
densities gives prior curvatures

- mean: phi/vbar;
- variance: (nu-7)/(2*vbar²).

The prior-predictive likelihood information per observation is 1/vbar for
the mean and (1/2+1/phi)/vbar² for the variance. The unadjusted epsilon
prior variance curvature is -3/(2*vbar²). Consequently the unadjusted
continuous matches would be phi*(1-1/c) and (nu-4)/(1+2/phi).
These are **not** complete BayesESS software results: the source explicitly
applies a further degrees-of-freedom adjustment.

## Adjustment and source discrepancies

The support appendix says an adjustment is added but does not specify its
value. Morita's Algorithm 2 gives a difference of curvature quantities
evaluated at the minimum degrees of freedom and at zero, and states its
purpose as ensuring ESS exceeds that minimum. The native R variance code
subtracts 2/vbar² through its four-minus-zero comparison. Combining that
specific adjustment with the corrected density derivatives above gives
the candidate variance root nu/(1+2/phi), while leaving the mean root
phi*(1-1/c) unchanged.

The candidate can be below four even when nu>4 (for example nu=5, phi=1),
so the stated minimum-ESS rationale and the meaning of the paper's
adjustment notation still require reconciliation. It is not appropriate
to present this candidate as an unqualified reproduction of the complete
method.

Additional observed discrepancies are independent of that issue:

- The rendered support appendix's variance-curvature line has signs that
  conflict with the negative Hessian of its preceding density.
- `essNormal` computes `Dq2Adg` from a variance-component term and a
  mean-component term, while a separately calculated mean term goes unused.
- The inverse-gamma wrapper passes its scale-variance conversion to an
  internal argument that is squared again; the documented inverse-chi-square
  and internal scale conventions must be distinguished.
- The wrapper strips minus signs from parsed prior values, including mu0.
- Native computation samples each positive integer patient count, interpolates
  with R's default 50-point grid, chooses its closest value and rounds to two
  decimals. It does not return an exact continuous root or evaluate m=0
  as specified by Algorithm 2.

The implementation remains open until its corrected statistical convention
and any explicitly requested native compatibility mode are specified with
independent reference evidence. Existing known-mean conjugate ESS remains
unchanged. The missing method must not be filled by silently choosing one
of the conflicting formulas.
