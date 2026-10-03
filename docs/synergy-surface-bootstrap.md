# SYNERGY wild-bootstrap resampling

`bootstrap_synergy_surface` implements the documented two-stage wild-bootstrap
data generation and REML refits for the semiparametric SYNERGY surface. It
returns bootstrap departures and descriptive summaries, not confidence limits:
the primary source's exact SD centering and denominator remain unresolved.

```python
import numpy as np
from mdanderson_stats import bootstrap_synergy_surface

dose1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1], dtype=float)
dose2 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1], dtype=float)
response = 2 + 0.5 * dose1 - 0.25 * dose2 + 0.2 * dose1 * dose2
response[-1] += 0.07  # repeated combination-dose observation
result = bootstrap_synergy_surface(
    dose1,
    dose2,
    response,
    replicates=20,
    rng=20261003,
    store_draws=False,
)
print(result.original_departure)
print(result.departure_mean)
print(result.departure_standard_deviation)
print(result.replicate_smoothing_at_boundary)
```

Responses are supplied on the caller's transformed scale `Y=g(E)`. The original
surface fit selects REML lambda unless `smoothing_parameter` is supplied. The
procedure then fits anchor surfaces at lambda/2 and 2*lambda. For every row,
including single-drug marginal rows, it constructs

```text
residual_i = Y_i - Fp_hat(d_i) - f_hat_lambda/2(d_i)
Y_i_star = Fp_hat(d_i) + f_hat_2lambda(d_i) + residual_i * w_i
```

where `w_i` independently takes `(1-sqrt(5))/2` and `(1+sqrt(5))/2` with
probabilities `(sqrt(5)+1)/(2*sqrt(5))` and `(sqrt(5)-1)/(2*sqrt(5))`. Each
pseudo-dataset refits its marginal baseline and selects a fresh REML smoothing
parameter. Duplicate dose rows receive independent row-wise multipliers.

`departure_draws` retains the fitted spline departures at the original
observations by default. Set `store_draws=False` to retain only the row-wise
mean and ordinary sample SD across bootstrap estimates (draw-mean centered,
denominator `B-1`). That SD describes bootstrap spread; it is not the Monte
Carlo standard error of the mean and is not used to form a normal interval.
`replicate_residual_variances_scaled` must be interpreted with its paired
`replicate_response_scales`. Both raw-dose and log-dose baselines are
supported; log-dose examples need positive marginal doses for both drugs and
matching-direction nonzero fitted marginal slopes.

The default is 100 replicates; 2–2,000 are accepted. Supply either `rng` or a
replayable `multiplier_tape` of shape `(replicates, observations)` containing
the two Mammen values. Execution is serial. A failed or nonconverged replicate
raises an error naming its index. The total estimated fitting work is capped at
two billion units. Retained draw matrices and multiplier tapes are each capped
at one million cells, checked before random-number consumption. For the log-
dose baseline, any both-zero observation is rejected because the source
baseline and residual are undefined there. No native random-stream equivalence
or interval convention is claimed.

The [numerical audit](../research/synergy-wild-bootstrap-audit.md) records the
independent fixed-tape R comparison and remaining source limits.
