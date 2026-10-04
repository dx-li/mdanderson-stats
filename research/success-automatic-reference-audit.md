# Automatic success-cutoff calibration reference audit

Reference batch prepared 2026-10-03. This is an independent mathematical
reference for the automatic cutoff search, not a reconstruction of the Windows
application's optimizer or reports.

## Source contract

The cached primary text is `research/raw/BayesianCalibration/paper.txt`
(SHA-256 `4e3ffbc0c878d7217261574261741829e373d9cb51d11fd9008168e42dbd7ec0`)
and the cached application help is
`research/raw/BayesianCalibration/calibration-help.txt` (SHA-256
`7d2c730bcbcd6b9957308aac0f169e86208b6ab4059d5045273e51998954c8b5`). The
paper defines success using the strict posterior comparison `Pr(effective | D)
> c`, and PID as false-positive probability divided by success probability.
For the single-arm binary endpoint, the supplement enumerates all response
counts `x=0,...,n`, gives the beta-binomial predictive mass and beta posterior
effective probability, and states that the only numerical task in deriving
operating characteristics is determining the critical response count for
cutoff `c` (paper supplement pp. 45–46). The help says users may request a
calibrated cutoff for a target PID over a candidate range; it gives `[0.6,
0.999]` as the default range but does not describe the search algorithm.

For a normal endpoint, supplement §S2.1 derives the normal-normal posterior
decision boundary and joint bivariate-normal law of truth and observed effect.
The survival adapter uses the stated fixed-event approximation
`SE(log HR)=1/sqrt(D r (1-r))` and the favorable lower log-HR tail. For the
one- and two-arm cases used here, the design truth and posterior decision
statistic have positive covariance. Therefore conditioning on increasingly
favorable posterior evidence makes the probability of ineffective truth
nonincreasing; this justifies bounded bisection for these normal models. No
monotonicity claim is made for arbitrary user-supplied evaluators.

## Independent calculations

`tools/reference_success_automatic.R` is standalone base R. For binary outcomes
it calculates the beta-binomial mass as a beta-function ratio, evaluates beta
tails with `pbeta`, enumerates all distinct posterior breakpoints in the
requested interval, and evaluates each induced strict-decision state. It does
not use a binary search or assume the PID curve is monotone. Cases include both
success directions, unequal design and analysis priors, an exact one-patient
identity, a discrete jump, and an infeasible range. The smallest feasible
breakpoint is the Python convention; the help does not identify the
application's tie/search rule.

For continuous outcomes, base R independently integrates the bivariate normal
probability by conditioning on the standardized truth and integrating the
conditional success probability. It brackets and bisects the target PID, with
an explicit endpoint-feasible case represented by the survival scale-invariance
pair. The two-arm reference uses unequal arm priors and has a sign-reflected
less-direction case. The survival cases multiply event information by four and
divide all log-HR means, standard deviations, and margin by two; they should
retain the same cutoff and operating characteristics.

Fixtures record binary candidate states and selected cutoffs, normal curve
values and bisection brackets, and survival rescaling results under
`tests/fixtures/success-automatic-*.csv`. These references check the underlying
operating characteristics and the documented search conventions. They do not
establish native cutoff optimizer, GUI, report, or plotting parity.

## Validation

The reference generator completed with warnings treated as errors in 0.47 s
(89.2 MB maximum resident set, no swaps). The focused comparison against the
integrated Python source passed all three tests in 1.53 s (141.8 MB maximum
resident set, no swaps). Ruff check and format checks passed for the Python test
file. The run covered exact and asymmetric binary priors, strict tie handling,
the discrete jump and infeasible-range behavior, unequal-arm normal priors and
reflection, normal bisection brackets, and survival event-information
rescaling.
