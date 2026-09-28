# Time-to-event BOP2-DC

`bop2_dc_survival_design` monitors an exponential survival endpoint using
separate reference and clinically meaningful median survival times. Larger
median survival is better, so `lrv < cmv`.

```python
from mdanderson_stats import bop2_dc_survival_design

design = bop2_dc_survival_design(40, lrv=3, cmv=5, looks=[10, 20, 30, 40])
state = design.monitor(sample_size=40, events=20, total_time=140)
print(state.decision)  # final_consider
```

The prior is inverse-gamma on the **exponential mean**, with explicit
`prior_shape` and `prior_scale`. This follows the paper's parameterization;
the scale is in the same time units as follow-up. It differs from the
median-scale prior accepted by `bop2_survival_design`. Multiply a mean-scale
prior by `log(2)` to obtain the corresponding median-scale prior.

For `d` observed events and total observed follow-up `T`, the posterior
shape is `prior_shape + d` and the posterior mean-scale parameter is
`prior_scale + T`. The median-scale parameter is the latter multiplied by
`log(2)`. Censored patients contribute their observed follow-up to `T` but
do not increase `d`.

Interim no-go requires both posterior criteria to fail their current
cutoffs. At the final sample size, both must pass for go or both must fail
for no-go; other cases yield consider. Comparisons are strict, so equality
at a final cutoff yields consider. Only configured interim looks can stop
early. Monitoring does not retain prior stopping decisions.

The paper's illustrative `1e-6` prior shape and scale are the defaults;
they do not establish calibrated error control. With zero observed events,
this prior can imply very high survival probabilities. Choose the prior
explicitly and examine its implications for the intended time units.

This implementation assumes an exponential event-time model. Calendar replay
and operating-characteristic simulation are available below. Parameter
calibration remains open. See [source notes](bop2-dc-survival-source.md) and
[independent reference calculations](bop2-dc-reference.md).

Batches are limited to 100,000 scenarios. Posterior ratios use normalized
contributions to avoid overflowing intermediate sums. If the raw mean-scale
sum exceeds floating-point range, `posterior_scale` is infinite; posterior
probabilities can remain finite and usable. Unrepresentable probability
arguments raise an error instead of silently changing the decision.

## Trial replay and operating characteristics

`run_bop2_dc_survival_trial` accepts a full potential enrollment schedule and
event durations from enrollment. Positive infinity represents no event during
observation. At each configured interim, patients are administratively censored
at the last enrolled patient's arrival. At the final sample size, observation
continues for the supplied `final_followup` beyond the last arrival. Events
exactly at the analysis time are observed. Input order resolves tied arrivals.
The replay returns the first stopping decision and its complete look history.

```python
import numpy as np
from mdanderson_stats import (
    bop2_dc_survival_design,
    run_bop2_dc_survival_trial,
    simulate_bop2_dc_survival,
)

design = bop2_dc_survival_design(
    6, lrv=3, cmv=5, looks=[2, 4, 6],
    prior_shape=1, prior_scale=2, lambda_lrv=.8, lambda_cmv=.5,
)
trial = run_bop2_dc_survival_trial(
    design, [.5, 1, 2, 4, 5, 6], [np.inf]*6, final_followup=12,
)
assert trial.decision == "final_go"
assert trial.events == 0
oc = simulate_bop2_dc_survival(
    design, true_median=6, accrual_rate=1, final_followup=12,
    n_trials=1000, arrival="poisson", rng=156,
)
assert oc.decision_count.sum() == 1000
print(dict(zip(oc.decision_labels, oc.decision_probability)))
```

Simulations use exponential event times with mean `true_median/log(2)`.
`arrival="fixed"` places the first patient at `1/accrual_rate`, with equally
spaced subsequent arrivals. `arrival="poisson"` uses independent exponential
arrival gaps of that mean, including the first gap from origin zero. These
are explicit Python schedules: the primary paper specifies an accrual rate
but does not identify a gap distribution. Dropout is not simulated.

The four disjoint outcome labels are `stop_no_go`, `final_go`,
`final_consider`, and `final_no_go`. Sum the early and final no-go probabilities
for total no-go risk. Counts, probabilities and binomial Monte Carlo errors
are returned, along with mean enrollment, events, observed follow-up and
duration and their sample-based Monte Carlo errors. A zero estimated decision
error when no events occur does not establish zero underlying risk. Sample
errors for continuous summaries are undefined (`nan`) with only one trial.

`duration`, `analysis_time`, and the replay's `calendar_times` all use calendar
origin zero. They include the initial wait for enrollment; for a replay with
a shifted origin, subtract that origin if elapsed duration is desired. Risk
time is computed from relative enrollment differences so calendar rounding
does not silently erase final follow-up. Unrepresentable final clocks fail
explicitly.

The returned `rng_seed` replays an entire aggregate call with the same
arguments and numerical environment. A supplied Generator contributes one
seed; it is not consumed trial by trial. Per-trial summaries allow follow-up
analyses without retaining every patient's simulated history.

Work limits are checked before randomness is consumed: at most 100,000 trials,
one million trial-by-patient cells, 30 million repeated-look patient visits,
and two million retained summary cells. Generation is chunked and serial.
These bounds limit resource use; they do not certify Monte Carlo precision.

Eight independent R calendar cases check 18 looks, including all terminal
decisions, boundary events, no events and extreme time-unit changes. A
one-patient scenario has analytic decision probabilities and expected observed
follow-up; 16,000 simulated trials agree within 1.447 estimated Monte Carlo
errors. See the [audit](../research/bop2-dc-survival-oc-audit.md).
