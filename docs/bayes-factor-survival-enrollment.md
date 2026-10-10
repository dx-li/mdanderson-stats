# Recovered BayesFactorTTE enrollment workflow

`bayes_factor_survival_enrollment_trial` and
`simulate_bayes_factor_survival_enrollment` implement the original v1.1
integer-day enrollment schedule. This is a separate workflow alongside the
[explicit-calendar API](bayes-factor-survival-calendar.md).

The original DLL establishes these timing rules: months equal 30.4375 days;
first arrival is day zero; event durations and exponential interarrival gaps
are individually truncated to integers. Zero-day events and tied arrivals are
retained. Checks follow enrollment of patients 2 through the maximum, in input
order. Only previously enrolled patients contribute to that look. Events
strictly before the check are observed; an event tied with it remains censored.
At maximum enrollment the trial ends without additional follow-up.

```python
from mdanderson_stats import (
    bayes_factor_survival_enrollment_trial,
    simulate_bayes_factor_survival_enrollment,
)

trial = bayes_factor_survival_enrollment_trial(
    [0, 0, 2, 3],
    [1, 0, 3, 2],
    null_median_months=4,
    alternative_median_months=5.5,
    inferiority_cutoff=0,
    superiority_cutoff=1,
)
assert trial.decision == "inconclusive"
assert trial.history[-1].events == 2
trial.write_json("enrollment-trial.json")

study = simulate_bayes_factor_survival_enrollment(
    null_median_months=4,
    alternative_median_months=5.5,
    true_median_months=4,
    accrual_rate_per_month=2,
    max_patients=5,
    repetitions=3,
    seed=89,
)
print(study.stopping_probability)  # inferiority, superiority, inconclusive
print(study.mean_patients, study.patient_quantile(0.9))
study.write_json("enrollment-study.json")
```

Positive-time evidence uses the existing independently validated iMOM kernel.
Integer truncation can produce observed events with zero total exposure; this
workflow evaluates the model's limiting likelihood there. The ordinary
continuous-data API still requires positive exposure when events are observed.

Each simulated trial records a PCG64 child seed, all planned day tapes and
actual looks. Native RNG identity is not claimed. Patient-count quantiles use
the original sorted-count index `floor(p*repetitions)` for `0 <= p < 1`, with no
interpolation. Unrounded stopping probabilities/means and .05/.10/.50/.90/.95
count quantiles are saved in JSON. Run multiple named scenarios by calling this
workflow with recorded explicit seeds; the existing HTML scenario reporter
continues to describe its own caller-defined calendar.

`bayes_factor_survival_day_boundaries(max_patients=..., ...)` returns thresholds
for event counts 0 through maximum minus one. Inferiority means integer total
exposure **less than** the first noninferiority day; superiority means exposure
**at least** the first superiority day. Disabled sides are -inf/+inf. Continuous
roots are converted with ceiling and floor-plus-one respectively. Python omits
the original arbitrary finite search horizon and its possibly incomplete lists.
This is a numerical quadrature/root calculation, not a claim of CLR integrator
identity. Boundaries outside int32 days raise explicitly.

Trial size is 2..500: the original one-patient simulation skips its entire check
loop, so it does not define a useful analyzed trial. Nonnegative integer day
tapes, calendar dates and exposure must fit int32; overflow raises rather than
wrapping. Simulation work, worst-case evaluations and conservative storage are
checked before RNG creation. Original binary/input/report bytes and native
random streams remain compatibility differences. The
[original-control audit](../research/bayes-factor-tte-native-control-audit.md)
records the independently executed methods and substitutions.

The native boundary exporter also treats cutoffs within `1e-9` of 0/1 as
disabled. Python uses the exact requested cutoffs, consistent with monitoring;
only exact 0/1 disable a side. CLR TimeSpan rounding for arbitrary fractional
months and numerical integration identity remain compatibility differences.
