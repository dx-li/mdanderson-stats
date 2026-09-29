# Bootstrap uncertainty for interval-PH coefficients

`bootstrap_interval_survival_coefficients` fits the ordinary interval-censored
PH model, resamples observations and estimates coefficient covariance and
standard errors. It follows the coefficient bootstrap in `icenReg::ic_sp`.
The returned fit can also be used for survival prediction and contours.

```python
import numpy as np
from mdanderson_stats import bootstrap_interval_survival_coefficients

index = np.arange(1, 97)
x = np.column_stack((np.cos(index * 0.71) + index / 200, np.sin(index * 1.31) - index / 250))
probability = ((index * 37) % 97 + 0.5) / 97
latent = 3 * (-np.log(probability) / np.exp(0.45 * x[:, 0] - 0.3 * x[:, 1])) ** (1 / 1.4)
lower = np.floor(latent)
upper = lower + 1
exact = (index % 7 == 0) & (latent < 6)
lower[exact] = upper[exact] = latent[exact]
lower[latent >= 6] = 6
upper[latent >= 6] = np.inf

result = bootstrap_interval_survival_coefficients(lower, upper, x, replicates=20, rng=17)
assert result.successful_replicates >= 2
assert result.coefficient_samples.shape == (20, 2)
assert result.successful_replicates + result.failed_replicates == 20
print(result.fit.coefficients, result.standard_error)
```

Twenty draws keep this example small; they demonstrate the workflow and do not
establish a precise Monte Carlo estimate of uncertainty. Inspect the number of
successful draws and increase replication as appropriate within the work limits.

## Weights and reproducibility

With no weights, each replicate draws the original number of observations with
replacement. With positive case weights, it draws `ceil(sum(weights))` row
indices with selection probabilities proportional to those weights. Repeated
rows are collapsed and their **sampled counts** become the replicate's weights.
They are not multiplied by the original weights. Rescaling all original weights
can therefore change the bootstrap sample size even when the original fitted
coefficients stay the same.

Supply an integer seed or a NumPy generator through `rng`. For exact replay,
instead supply `resample_indices`, an integer matrix with one row per replicate
and `ceil(sum(weights))` zero-based input indices per row. A tape and `rng`
cannot be supplied together. This reproduces the chosen resamples, without
claiming the same random stream as R.

## Failed fits and interpretation

`coefficient_samples` retains every requested replicate in order. A failed fit
has a NaN row and a corresponding entry in `replicate_status` and
`replicate_errors`. Failures are not replaced by fresh draws. Covariance uses
successful rows with denominator `successful_replicates - 1`; standard errors
are the square roots of its diagonal. Both are NaN if fewer than two fits
succeed. The result explicitly labels these summaries as conditional on fit
success. Substantial failure means that the surviving draws may inadequately
represent the intended bootstrap distribution.

This API provides coefficient covariance and standard errors for ordinary PH
regression with at least one covariate. It does not attach sampling confidence
bands to the nonparametric baseline or reinterpret the existing survival
identification bounds. Stratified and clustered bootstrap policies are separate
workflows. Fits run serially; combined storage and conservative total fitting
work, including the iteration allowance, are bounded before randomness is used.
These limits may require a smaller request or iteration allowance.
The hard ceilings include 5,000 replicates, 100,000 sampled rows per replicate,
50 million total sampled rows and two billion estimated fitting work units.
Covariance entries outside the representable floating-point range raise an
error; rescale covariates to suitable units before fitting in that case.

The [source and numerical audit](../research/interval-survival-bootstrap-audit.md)
records the native weight, failure and covariance conventions and the fixed
resample references.
