# Proportional Density full-data goodness-of-fit bootstrap

`proportional_density_full_bootstrap` implements the full-data disease-curve
procedure in section 3.1 of the [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721282/).
It resamples both observed diagnoses and censoring records, refits the model
and nonparametric disease curves, and calibrates their integrated squared
control-arm difference. This is separate from the cheaper
[failure-only bootstrap](proportional-density.md#failure-only-goodness-of-fit-bootstrap).

```python
import numpy as np
from mdanderson_stats import proportional_density_full_bootstrap

result = proportional_density_full_bootstrap(
    time=list(range(1, 21)),
    event=[1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    treatment=[0, 1] * 10,
    replicates=19,
    seed=7809,
)
assert result.tau == 19
assert result.bootstrap_statistics.shape == (19,)
assert np.isnan(result.bootstrap_statistics).sum() == result.failed_replicates
print(result.statistic, result.pvalue)
print(result.pvalue_lower, result.pvalue_upper, result.failure_reasons)
```

Nineteen replicates demonstrate the interface, not accurate tail calibration.
Choose a replicate count suitable for the desired Monte Carlo precision. The
returned simulation error accounts for a finite number of replicates, not
uncertainty in the fitted-model bootstrap approximation.

## Resampling and statistic

Within each arm, the original numbers of observed events and censored records
remain fixed. Event times are sampled with replacement from the fitted pooled
failure support, using that arm's fitted **observed-case** masses. Censored
record times are sampled independently with replacement from the original
censored records in that arm. These samples retain their event/censor labels;
no latent failure/censor pairing or cure indicator is generated. An arm with
no censoring records contributes an empty censor sample.

This fixed-status construction makes the paper's separate resampling steps
explicit. It is a stated Python convention, not verified output from a native
bootstrap driver: the downloaded archive contains no such driver. Disease-
conditional masses are used for fitted disease curves, not as the event
resampling distribution.

Each replicate re-estimates the censoring distribution and both disease
curves. `equal_censoring=True` applies the existing pooled-censoring assumption
to the original fit and every replicate; the default estimates censoring
separately by arm. The test compares fitted and nonparametric conditional
control survival curves using

```text
Delta_n = integral from 0 to tau of (S_fitted(t) - S_nonparametric(t))**2 dt.
```

Unit weight is used, as in the archive. Integration is the exact integral of
the right-continuous step curves, including the constant tail through `tau`.
The original-data endpoint is fixed across replicates. Its default is the
smaller arm-specific maximum follow-up; an explicit `tau` may be smaller.
This differs from the archive's right-endpoint rectangle statistic returned
by `proportional_density(...).goodness_of_fit`.

## Failed fits and replay

Discrete resampling can produce separated or unidentified failures, or a
censoring distribution with no support at a pooled failure time. Such replicates
remain `NaN`, with counts and reasons. They are never dropped or redrawn.
Original-data fitting errors still raise immediately.

With `B` replicates and `E` statistics at least as large as the observed value,
the upper-tail estimate is `(1+E)/(B+1)`. If `K` fits fail, `pvalue` and its
Monte Carlo standard error are `None`; bounds are `(1+E)/(B+1)` and
`(1+E+K)/(B+1)`. These describe unresolved replicate contributions, not
confidence intervals. A single p-value is reported only when all fits resolve.

A local integer seed reproduces sampling. An explicit
`ProportionalDensityFullBootstrapTape` can instead supply four integer matrices:
`control_event`, `treatment_event`, `control_censor`, `treatment_censor`, each
with one row per replicate. Event indices address the original fitted pooled
failure support; censor indices address raw within-arm censor records in input
order. The tape replaces sampling and cannot be combined with a seed. Seeded
sampling visits those four components in the listed order for each replicate.

Replicate counts and total resampled records have explicit limits. Supplied
index tapes have a separate retained-storage bound and are checked before
conversion. Full fitted histories are not accumulated across replicates.

This procedure calibrates goodness of fit. Unequal-censoring treatment-effect
null calibration, parameter confidence intervals and disease-curve confidence
bands require separate statistical constructions and remain outside this API.

Independent R `survival::survfit` and `stats::glm` references match nine
original/replicate statistics and 104 curve rows, including separate/pooled
censoring, changed censor records and unresolved replicates. Maximum absolute
area and curve errors are `4.30e-14` and `7.46e-14`. The
[reference generator](../tools/reference_proportional_density_full_bootstrap.R)
and [audit](../research/proportional-density-full-bootstrap-audit.md) document
the comparison.
