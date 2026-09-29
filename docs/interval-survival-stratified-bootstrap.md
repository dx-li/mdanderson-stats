# Shared-coefficient stratified interval-PH bootstrap

`bootstrap_stratified_interval_survival_coefficients` refits the shared
regression coefficients and stratum-specific baselines from the existing
stratified interval-PH likelihood. The available sources do not specify a
native bootstrap for this stratified model, so choose a resampling policy
explicitly:

```python
import numpy as np
from mdanderson_stats import bootstrap_stratified_interval_survival_coefficients

lower = [0, 1, 0, 1, 0, 1, 0, 1]
upper = [1, np.inf, 1, np.inf, 1, np.inf, 1, np.inf]
x = np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, None]
strata = ["A"] * 4 + ["B"] * 4
weights = [2, 8, 5, 5, 4, 6, 7, 3]

result = bootstrap_stratified_interval_survival_coefficients(
    lower,
    upper,
    x,
    strata=strata,
    weights=weights,
    resampling="within_stratum",
    replicates=20,
    rng=17,
)
assert result.successful_replicates + result.failed_replicates == 20
print(result.fit.coefficients, result.standard_error)
```

With `resampling="within_stratum"`, each first-seen stratum is resampled
independently. Its draw count is `ceil(sum(weights_g))`, and its row
probabilities are proportional to its positive case weights. This keeps the
stratum sample sizes fixed under the declared weighted-row convention. With
`resampling="pooled"`, the ordinary interval-PH weighted-row design is used:
draw `ceil(sum(weights))` rows globally with probabilities proportional to
all case weights. A pooled replicate omitting any original stratum is retained
as a failed replicate because it cannot refit the same stratified design.

Repeated selected rows are compressed into sampled frequency weights. The
original case weights determine selection probabilities and draw counts; they
are not multiplied into the replicate weights. `within_stratum_sample_sizes`
records each fixed per-stratum draw count. `resampled_stratum_counts` records
the actual number of sampled rows from each group in every replicate.

For exact replay, pass a zero-based `resample_indices` integer matrix instead
of `rng`. Its shape is `(replicates, sum_g ceil(sum(weights_g)))` for
within-stratum resampling and `(replicates, ceil(sum(weights)))` for pooled
resampling. The tape is retained read-only in the result. Within-stratum tape
columns are concatenated by the original first-seen group order, and every
index in each block must belong to that group. A seed or NumPy generator is
also supported, but its stream does not match R.

Failed fits stay aligned as NaN coefficient rows and are not redrawn. The
reported covariance uses denominator `successful_replicates - 1` and its
standard errors are conditional on successful refits; both are undefined
with fewer than two successes. These are coefficient summaries only. They do
not quantify baseline-mass uncertainty or turn interval-survival
identification bounds into confidence bands.

The conservative preflight limits requests to 5,000 replicates, 100,000
draws per replicate, 50 million total sampling operations, two million
combined result/scratch cells, and two billion worst-case optimizer work
units. The underlying fit retains its 20,000-row, 2,000 combined-support,
20-million likelihood-work and 5,000-iteration limits. Requests over a bound
are rejected before RNG use. Twenty replicates make a runnable example, not a
precise uncertainty estimate; use substantially more when the bounded work
permits and inspect the successful-fit count.

The ordinary `icenReg::ic_sp` bootstrap is documented as a global weighted
row resample, and its separate cluster helper samples subjects. Neither
defines stratified-bootstrap semantics. Both policies above are explicit
Python choices around the implemented shared-coefficient interval-PH target,
not native stratified-app parity. See the
[source and numerical audit](../research/interval-survival-stratified-bootstrap-audit.md).
