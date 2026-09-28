# Lognormal Bayesian goodness of fit

`lognormal_complete_data_bayesian_gof` fits both log-location and log-variance
under an explicit proper Normal-Inverse-Gamma prior. Independent joint
posterior draws feed the existing [Johnson chi-square diagnostic](bayesian-chi-square.md)
for complete positive event times. The lognormal family follows §4.5 of the
[BCSTTE guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/BCSTTE/BCSTTE_UsersGuide.pdf).
The conjugate prior is an explicit Python choice; native fitting defaults
are not established by that guide.

```python
import numpy as np
from mdanderson_stats import lognormal_complete_data_bayesian_gof

observed = np.array([.5, 1, 2, 4, 8])
fit = lognormal_complete_data_bayesian_gof(
    observed,
    prior_location=.2, prior_location_precision=1.5,
    prior_variance_shape=2, prior_variance_scale=.8,
    samples=2000, bins=3, rng=6609,
)
assert fit.posterior_location_precision == 6.5
assert fit.posterior_variance_shape == 4.5
np.testing.assert_allclose(
    fit.posterior_location, (1.5*.2+np.log(observed).sum())/6.5,
)
assert fit.diagnostic.statistic.shape == (2000,)
log_locations = fit.centered_location_samples + fit.location_offset
print(log_locations.mean(), fit.diagnostic.area_against_reference)
```

## Model and interpretation

Let `Y=log(T)`. The lognormal model uses `Y | mu,V ~ Normal(mu,V)`, where
`V=sigma**2` and `sigma` is the guide's log-time standard deviation. The prior
is `V ~ InverseGamma(a0,b0)` in the shape/scale convention, and
`mu | V ~ Normal(m0,V/kappa0)`. Supply all four hyperparameters; `kappa0`,
`a0` and `b0` must be strictly positive.

For `n` observations with mean log time `ybar`, the posterior parameters are:

```text
kappa_n = kappa0 + n
m_n = (kappa0*m0 + n*ybar)/kappa_n
a_n = a0 + n/2
b_n = b0 + sum((y-ybar)**2)/2 + kappa0*n*(ybar-m0)**2/(2*kappa_n)
```

Each draw first samples `Z ~ Gamma(a_n, rate=1)`, sets `V=b_n/Z`, and samples
`mu | V ~ Normal(m_n,V/kappa_n)`. Every observation's CDF uses the same paired
`mu,V` draw. The diagnostic evaluates observed data under posterior parameters;
it does not replace observations with predictive draws.

Changing time units by factor `c` requires shifting `prior_location` by
`log(c)`. The other prior parameters remain unchanged. This transformation
preserves the diagnostic. The posterior `log_variance_samples` describe
variance of log time, not variance of the original event durations.

Location draws are stored as `centered_location_samples + location_offset`.
CDFs use centered coordinates to retain small relative differences under
large common time shifts. `posterior_log_variance_scale` retains `log(b_n)`;
calculations avoid forming an unrepresentably large inverse-gamma scale.
Positive posterior weights are computed separately to preserve weak-prior
contributions even when the prior mean is very large.

The function checks combined CDF/count/sample workspaces before allocating or
consuming randomness, with a 20-million-cell limit and at most 100,000 draws.
Unrepresentable posterior draws raise an error. Returned arrays are read-only.

The returned `diagnostic` has equal-probability bins, upper-inclusive boundaries
and `bins-1` reference degrees of freedom. Its posterior-average reference
probabilities and exceedance fraction are diagnostics; they are not calibrated
p-values for the posterior mean statistic. Censoring, rounded observations,
native prior defaults and native HTML reports remain outside this API.

## Independent evidence

The [base-R reference](../tools/reference_lognormal_bayesian_gof.R) verifies
conjugate posterior parameters and moments and the Student-t predictive CDF.
It also integrates the joint posterior diagnostic: conditional Normal location
intervals give exact bin-pattern weights, followed by scalar Gamma quadrature.
Thus the diagnostic reference preserves parameter dependence across all
observations rather than combining marginal CDF expectations.

Six cases cover broad data, a concentrated prior, equal times, and time units
scaled by `1e-200`/`1e200`. Eighty-eight posterior, paired-draw, CDF and
diagnostic summaries agree within 2.851 estimated Monte Carlo standard errors
using 16,000 draws per case. Focused checks also validate the conjugate update,
direct CDF calculation, unit invariance and a weak-prior/huge-location update.
