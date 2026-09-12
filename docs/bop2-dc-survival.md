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

This implementation assumes an exponential event-time model. It provides
posterior monitoring, not yet calendar-time trial simulation or parameter
calibration. See [source notes](bop2-dc-survival-source.md) and
[independent reference calculations](bop2-dc-reference.md).

Batches are limited to 100,000 scenarios. Posterior ratios use normalized
contributions to avoid overflowing intermediate sums. If the raw mean-scale
sum exceeds floating-point range, `posterior_scale` is infinite; posterior
probabilities can remain finite and usable. Unrepresentable probability
arguments raise an error instead of silently changing the decision.
