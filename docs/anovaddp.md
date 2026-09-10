# ANOVA DDP: nonlinear observation and conditional-update kernels

MD Anderson entry 67 implements the ANOVA dependent Dirichlet-process model of
De Iorio, Müller, Rosner and MacEachern, “An ANOVA Model for Dependent Random
Measures,” JASA 99(465):205–215 (2004). Its response model describes repeated
measurements using subject-specific nonlinear curves; the distribution of those
curve parameters depends on subject covariates through a DDP mixture.

**Coverage is partial.** The observation curve, Gaussian likelihood and
conditional Gaussian amplitude update are available. The dependent-DP cluster
updates, remaining parameter updates, covariance/base-measure updates, full MCMC,
new-subject prediction, study contrasts, nadir summaries, source data readers and
plotting workflow remain to be ported. These kernels do not fit a DDP model by
themselves and are not a substitute for the full sampler.

## Supplied nonlinear curve

`anovaddp_curve(parameters, time)` reproduces `regressione.cpp`. Parameters have
final axis six: `(z1, z2, z3, tau1, tau2, b1)`. The source comment incorrectly
lists seven names; its actual implementation reads six and fixes the logistic
intercept at -2. A nonempty time vector is evaluated against every parameter
vector, yielding shape `parameters.shape[:-1] + (len(time),)`.

- Before tau1, the response is z1.
- From tau1 to tau2, it is `r*z1 + (1-r)*(z2+z3*expit(-2))`, with
  `r=(tau2-time)/max(tau2-tau1, 1e-6)`.
- At and after tau2, it is `z2+z3*expit(-2+b1*(time-tau2))`.

Branch order and the 1e-6 denominator floor follow the native code. That floor
can introduce a discontinuity when knots are closer than 1e-6. With reversed
knots, the default retains the original branch behavior rather than silently
sorting them. `repair_order=True` implements `fittare`'s separate prediction
convention: replace tau2 by tau1+1 only when tau2<tau1. A requested replacement
that cannot be represented as greater than tau1 raises an arithmetic error.

```python
from mdanderson_stats import anovaddp_curve, anovaddp_loglikelihood

theta = [2, -1, 4, 1, 3, 0.7]
times = [0, 1, 2, 3, 4, 6]
print(anovaddp_curve(theta, times))
print(anovaddp_loglikelihood(theta, times, [2, 1.9, 0.7, -0.4, 0.3, 2], 0.4))
```

`anovaddp_loglikelihood` sums independent Gaussian observation terms over time.
By default it omits normalizing constants, matching the native `loglik` used
when comparing theta at fixed residual variance. `normalized=True` includes
`-n/2 * log(2*pi*variance)`, which is necessary for comparisons across variances.
A scalar positive variance and finite aligned observations are required.

The sigmoid is evaluated with a stable special function, so very large positive
or negative slopes do not overflow an exponential. Nonfinite final curves,
likelihoods and posterior moments raise clear errors; the function does not
replace them with arbitrary finite values. Returned arrays are immutable.

## Conditional amplitude update

`anovaddp_amplitude_posterior` implements the first Gaussian block of `simtheta`.
It conditions a six-dimensional normal prior on the fixed values of tau1, tau2
and b1, then combines the resulting normal prior for z1,z2,z3 with the Gaussian
observation likelihood. Off-diagonal prior correlations are retained.

```python
import numpy as np
from mdanderson_stats import anovaddp_amplitude_posterior

posterior = anovaddp_amplitude_posterior(
    [2, -1, 4, 1, 3, 0.7],
    [0, 1, 2, 3, 4, 6],
    [2, 1.9, 0.7, -0.4, 0.3, 2],
    prior_mean=[1, 0, 3, 0.5, 2, 0.5],
    prior_covariance=np.eye(6),
    variance=0.4,
)
print(posterior.mean, posterior.covariance)
```

The result contains the three-dimensional posterior mean, covariance and the
observation design matrix. Linear solves and positive-definiteness checks
replace the native explicit matrix inverses. The design is evaluated from the
same curve kernel, ensuring that its matrix product with the amplitudes equals
the likelihood mean. This deliberately resolves a source inconsistency:
`simtheta` omits `regressione`'s 1e-6 denominator floor. Ordinary separated knots
match both native formulas; very close knots follow the observation curve.
These are conditional moments, not samples or unconditional DDP posterior moments.

## Source findings to resolve in the full sampler

The archived `vardati` signature accepts beta0 but never uses it; it draws
residual variance with inverse-gamma shape `(alpha0+N)/2` and scale `SSE/2`.
The R manual instead specifies inverse-gamma prior shape alpha0/2 and scale
beta0/2, which would give posterior scale `(beta0+SSE)/2`. This is an identified
source discrepancy, not an implemented variance update. The complete sampler
needs an explicit, documented resolution before native MCMC comparisons can
be interpreted correctly.

Other native routines and the DDP model structure remain under audit. The source
also offers replacement of `regressione.cpp` for user-defined response curves;
that extension point will require a consistent likelihood/conditional-update
contract rather than a callback attached only to the prediction function.

## Validation and provenance

The unmodified `regressione.cpp` was compiled using a minimal adapter providing
only the NEWMAT RowVector indexing it requires. Three parameter vectors were
evaluated at eight times, covering every branch, exact boundaries, reversed
knots and the narrow transition floor. Python matched the native double outputs
exactly on this platform. The conditional update was independently evaluated
with R linear algebra for a correlated six-dimensional prior: maximum mean
and covariance differences were below 5e-15 and 4e-16 respectively.

Two focused tests also cover likelihood normalization, stable sigmoid limits,
explicit knot repair, immutable outputs, positive-definite covariance, and
agreement between the conditional design matrix and the nonlinear curve.
No CI jobs or dependencies were added.

[Source archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/AnovaDDP/anovaddp_1.0.tar.gz).
The case-sensitive download folder is AnovaDDP. Every regular archive member was
checked against its extracted bytes and inventoried in `anovaddp-sources.json`.
That inventory establishes provenance, not completion of every source routine.
The MDACC portion's original Artistic-license declaration is preserved in
`notices/mdanderson-anovaddp-COPYING.txt`. Original source, NEWMAT/RANDLIB
implementations, sample data and the article are not redistributed.
