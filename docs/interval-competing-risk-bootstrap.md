# Bootstrap uncertainty for interval competing-risk coefficients

`bootstrap_interval_competing_risk_coefficients` estimates covariance and
standard errors for both causes' regression coefficients. It implements the
ordinary row-bootstrap workflow in `intccr::ciregic` with `nboot > 0`.
It is available alongside the point fitter's residualized-score covariance.

```python
import numpy as np
from mdanderson_stats import bootstrap_interval_competing_risk_coefficients

index = np.arange(1, 121)
x = np.column_stack((np.cos(index * 0.71) + index / 200, np.sin(index * 1.31) - index / 250))
u1 = ((index * 37) % 127 + 0.5) / 127
u2 = ((index * 53) % 131 + 0.5) / 131
t1 = -np.log(u1) / (0.3 * np.exp(0.4 * x[:, 0] + 0.2 * x[:, 1]))
t2 = -np.log(u2) / (0.25 * np.exp(-0.2 * x[:, 0] + 0.1 * x[:, 1]))
latent = np.minimum(t1, t2)
event = np.where(t1 < t2, 1, 2)
spacing = 0.3 + 0.013 * (index % 11)
lower = np.floor(latent / spacing) * spacing
upper = lower + spacing
censor_time = 2 + 0.023 * (index % 13)
right = upper > censor_time
lower[right] = censor_time[right]
upper[right] = np.inf
event[right] = 0

result = bootstrap_interval_competing_risk_coefficients(
    lower, upper, event, x, alpha=(0, 1), k=0.5, replicates=8, rng=166,
)
assert result.coefficient_samples.shape == (8, 4)
assert result.successful_replicates >= 2
print(result.fit.coefficients, result.standard_error)
```

Eight draws demonstrate the interface; they do not establish precise Monte
Carlo uncertainty. Inspect successful and failed counts and choose replication
appropriate to the precision needed, within the resource limits.

## What each replicate does

Each replicate samples the original number of rows uniformly with replacement.
Repeated rows remain in the dataset: they affect the empirical quantiles used
to choose the spline knots. Each refit rebuilds its knots and time boundaries
using its own sampled data, with the same link parameters, knot-rate setting
and explicit starting-boundary approximation as the original fit.

Coefficient rows contain all cause-1 slopes followed by all cause-2 slopes.
Covariance is the sample covariance of successful rows, using denominator
`successful_replicates - 1`. Standard errors are the square roots of its
diagonal. With fewer than two successful fits, these summaries are NaN.
No baseline or cumulative-incidence confidence bands are implied.

`result.fit` contains the original point fit for prediction and contours. Its
separate residualized-score covariance is unavailable (NaN): the bootstrap
workflow does not calculate that alternative estimator. Use
`result.covariance` and `result.standard_error` for bootstrap uncertainty.
The ordinary `fit_interval_competing_risk` API continues to provide its
residualized-score covariance.

## Reproducibility and failures

Use `rng` with a seed or NumPy generator. Alternatively, provide
`resample_indices`, an integer matrix with shape `(replicates, observations)`
containing zero-based original row indices. A tape and `rng` are mutually
exclusive; fixed tapes reproduce sample composition without claiming the
same random stream as R.

Both causes must occur in the original data. By default, an invalid resample,
including one missing a cause, raises an error with its replicate index.
Explicitly choosing `on_invalid_resample="record"` retains these failures in
the result; this is a Python extension to the source's error behavior.
Nonconvergent fits are recorded without replacement draws. Every requested
replicate retains its position in the coefficient/status/error ledger.
Covariance is conditional on successful fitting; heavy failure can distort
the intended bootstrap distribution.

Replicates run serially, with combined sampling, fitting and storage limits
checked before randomness is used. The same constraints and convergence
checks as the point fitter apply. The [source audit](../research/interval-competing-risk-bootstrap-audit.md)
distinguishes native source compatibility from numerical validation: known
defects in the native constraint derivatives prevent treating every native
optimizer result as a certified coefficient target.
