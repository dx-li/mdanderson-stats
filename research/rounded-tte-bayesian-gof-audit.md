# Rounded TTE fitting: source-to-method boundary

## Source-backed input and diagnostic

The cached BCSTTE user guide, `research/raw/BCSTTE/guide.txt`, §2.2's
discrete/rounded survival-time option (guide lines 54–78), describes integer
rounded follow-up values and the interval interpretation `(t - 1/2, t + 1/2)`.
The source does not specify native prior defaults or enough fitter details to
claim native posterior-sampling parity. The exact likelihood for an observed
rounding interval is the model probability mass between its endpoints; this
follows directly from interval observation and introduces no midpoint-density
approximation.

Johnson's randomized CDF construction is implemented separately in
`bayesian_chi_square_discrete_cdf` and audited in
`bayesian-chi-square-discrete-audit.md`. This fitter supplies CDF endpoints
from the same paired posterior parameter draw used for its interval
likelihood, preserving the observation-level dependence required by that
diagnostic.

## Implementation crosswalk

| Concern | Community implementation | Boundary |
| --- | --- | --- |
| Rounded observation | Caller supplies finite interval bounds; the guide's integer convention maps to `[max(0,t-.5),t+.5]`. | No censoring or rounding-rule inference. |
| Likelihood | Family-specific stable log interval masses in `_rounded_tte_likelihood.py`. | No density evaluated at interval midpoint. |
| Prior and sampling | Proper caller-supplied Gaussian prior in transformed coordinates; serial elliptical slice sampling reuses `tte_family_bayesian_gof._sample_elliptical_slice_chains`. | This is a Python prior convention, not the old conjugate priors or recovered native defaults. |
| Diagnostic | Paired posterior CDF interval matrices feed the randomized discrete CDF diagnostic. | A CDF interval that collapses in float arithmetic raises; diagnostic can be explicitly disabled. |
| Numerical/resource limits | Bounds, prior dimensions/covariance, chain work, draw-observation cells, and retained parameter cells are checked before sampling. | Current caps are API limits, not source program limits. |

The likelihood helper selects the better separated log-CDF or log-survival
difference. It rejects intervals when both available tail gaps are at or below
`32 * machine_epsilon * max(1, endpoint_log_magnitudes)`; that is an explicit
precision floor, not a statistical censoring rule. It does not replace such
intervals with a midpoint density approximation. Extreme masses that underflow
both tails are reported as zero representable probability.

The seven model parameterizations are exponential (log scale, reciprocal
rate), Weibull (log shape/log scale), lognormal (`mu`, the mean of log-time,
and log sigma), Gamma, inverse-Gamma, log-logistic, and generalized
log-odds-rate. Existing
fitters remain unchanged: exact/right-censored exponential and fixed-shape
Weibull use conjugate Gamma-rate priors, lognormal uses a Normal-Inverse-Gamma
prior, and the other joint samplers use Gaussian log-parameter priors. Rounded
interval likelihoods remove the first three conjugacies. The new interface
therefore states its own transformed-Gaussian prior for every family instead
of silently changing an existing fitter's prior contract.

This API does not implement native templates, prior defaults, random stream,
or report layout. Its diagnostic inherits the asymptotic reference and
regularity limits of the Johnson construction; it is not an exact finite-sample
calibration guarantee.

## Focused validation

Three independent Gauss-Hermite posterior-mean checks cover exponential
(1-D), correlated-prior lognormal (2-D), and correlated-prior log-odds-rate
(3-D). Refining the tensor rule from 24 to 40 nodes per coordinate changed
the absolute transformed posterior means by at most `2.17e-5`; the largest
absolute Monte Carlo discrepancy normalized by the combined batch-means
MCSE and quadrature-refinement change was `1.47`. Maximum classical split
R-hat across these checks was `1.00198`. Additional focused checks exercise
all seven family adapters, time-unit invariance, interval-at-zero handling,
pre-RNG input/resource rejection, and CDF diagnostic construction. These are
small deterministic checks, not a convergence guarantee for arbitrary data.
