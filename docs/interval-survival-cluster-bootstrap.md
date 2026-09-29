# Cluster bootstrap for interval-PH coefficients

`bootstrap_interval_survival_cluster_coefficients` estimates regression
uncertainty when several observations belong to the same patient or group.
Each replicate samples whole clusters with replacement and refits the
interval-censored proportional-hazards model. All observations from a selected
cluster stay together, including when that cluster is selected repeatedly.

```python
import numpy as np
from mdanderson_stats import bootstrap_interval_survival_cluster_coefficients

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

# Synthetic unequal groups illustrate the resampling interface.
cluster_ids = np.repeat(np.arange(16), [2, 4, 6, 8, 10] * 3 + [6])
result = bootstrap_interval_survival_cluster_coefficients(
    lower, upper, cluster_ids, x,
    replicates=8, rng=29, on_fit_failure="record",
)
assert result.cluster_draw_indices.shape == (8, 16)
assert result.coefficient_samples.shape == (8, 2)
assert result.successful_replicates >= 2
print(result.standard_error, result.resampled_row_counts)
```

Eight replicates demonstrate the interface, not precise Monte Carlo
uncertainty. Use meaningful independent sampling clusters in a real study,
inspect failures and choose replication appropriate to the precision needed.
Resampling cannot compensate for an insufficient number of independent groups.

## Sampling and replay

With G original clusters, every replicate makes exactly G uniform cluster
draws. Unequal group sizes mean that the expanded observation count can vary
between replicates. `resampled_row_counts` reports that count. The fitter
represents repeated observations with integer frequencies, preserving the
likelihood and Turnbull support without allocating the expanded row matrix.

Cluster IDs must be either integers or nonempty strings, with one type used
throughout and one ID per input row. At least two distinct clusters and one
covariate are required. `cluster_labels` records the deterministic sorted
label map; integer IDs are preserved without converting them to floating point.

Supply a seed or NumPy generator through `rng`, or provide
`cluster_draw_indices` with shape `(replicates,G)`. Each integer indexes
`cluster_labels` from zero. These choices are mutually exclusive, and the
returned draw tape supports exact replay. NumPy's stream and Python's string
ordering are not claimed to reproduce R's RNG or locale-dependent ordering.

## Failures and inference

The default `on_fit_failure="raise"` follows the native cluster wrapper's
abort-on-error behavior. Explicit `"record"` is a Python extension: each
failed replicate remains in the coefficient/status/error ledger, with NaN
coefficients and no replacement draw. Covariance then uses successful rows
with denominator `successful_replicates-1`. Covariance and standard errors are
NaN with fewer than two successful fits. Substantial failure can distort the
intended bootstrap distribution.

`fit` is the original working-independence PH fit, usable for predictions and
contours. Its log likelihood remains an optimization diagnostic for that row
model; it does not become a joint likelihood for dependent observations.
The cluster bootstrap covariance is returned separately. This API does not
produce baseline confidence bands or change survival identification bounds.

The workflow implements the semiparametric PH portion of `icenReg::ir_clustBoot`.
It does not add case-weighted cluster sampling, stratification, or parametric
interval-censoring models. Fits run sequentially, with combined storage and
sampling/fitting-work limits checked before random generation. See the
[source audit](../research/interval-survival-cluster-bootstrap-audit.md) for
native references and expanded-row equivalence checks.
