# STPLAN survival power

Five forward calculations cover the censored-survival options in STPLAN's
current main menu. They assume uniform accrual, administrative censoring at the
end of follow-up, and no dropout or competing risks. These are the source's
planning models and approximations, not a simulation of a fitted survival model.
See [STPLAN coverage](stplan.md) for other outcome families and remaining work.

```python
from mdanderson_stats import (
    stplan_censored_exponential_one_sample_power,
    stplan_george_desu_survival_power,
    stplan_information_survival_power,
)

# Mean survival 10 versus 15 months; 5 patients/month accrued for 12 months,
# with another 6 months of follow-up after accrual ends.
one_sample = stplan_censored_exponential_one_sample_power(10, 15, 5, 12, 6)
george_desu = stplan_george_desu_survival_power(10, 15, 5, 12, 6)
information = stplan_information_survival_power(10, 15, 5, 12, 6)
# Approximately 0.7548493, 0.3256637, and 0.3376567, respectively.
```

## Exponential survival

The first three APIs take `mean1, mean2, accrual_rate, accrual_duration,
followup_duration`, with the first two named `null_mean, alternative_mean` for
the one-sample method. Means and accrual parameters are positive; follow-up is
nonnegative. Time units must agree. Randomized comparisons allocate half the
expected participants to each arm.

For mean survival `m`, accrual duration `A`, and follow-up `F`, event probability
is the average of `1 - exp(-t/m)` over patient ages `t` in `[F, F+A]`.
Expected enrollment is accrual rate times `A`. Stable exponential differences
and small-argument series preserve very small event probabilities.

The one-sample method uses the **alternative** event probability to obtain a
Poisson expected event count. It averages the uncensored one-sample exponential
pivot power over positive event counts. Zero events imply no rejection. Under
the null, power is therefore `alpha / sides * (1 - exp(-expected_events))`,
not necessarily the nominal significance level. This Poisson-event mixture is
the source's censored-trial approximation. Its infinite sum is evaluated without
renormalization, with omitted probability at most `tail_tolerance=1e-10` by
default. This bounds truncation error; floating-point evaluation adds rounding
error. The native routine instead stops when individual weighted contributions
fall below a tolerance.

The George–Desu method estimates arm-specific event counts `d1, d2`, forms
`s = (mean1*d1 + mean2*d2)/(mean1 + mean2)`, and evaluates the exponential F-ratio
pivot with `2*s` degrees of freedom in both parts. Fractional expected counts
are intentional. The information-time method uses total expected events `D`
and normal noncentrality `abs(log(mean1/mean2)) * sqrt(D/4)`.

All five functions accept `alpha=0.05, sides=1`, with `0 < alpha < 0.5`.
As in other STPLAN procedures, `sides=2` halves alpha and evaluates a single
rejection direction; it does not sum both rejection tails.

## Historical controls

```python
from mdanderson_stats import stplan_historical_survival_power

power = stplan_historical_survival_power(
    0.05, 0.1, 5, 12, 6, 40, 20,
    control_allocation=0.2, continued_followup=True,
)
# Approximately 0.8898854.
```

Positional inputs are `experimental_hazard, control_hazard, accrual_rate,
accrual_duration, followup_duration, historical_deaths, historical_alive`.
Hazards and historical deaths are positive; historical survivors are
nonnegative. Planning counts may be fractional. `control_allocation` is a
fraction in `[0, 1)`, not a percentage. The default is zero.

With continued follow-up, future control events include both surviving
historical participants followed for the entire new study and newly allocated
controls subject to uniform accrual. With `continued_followup=False`, the source
sets all future control events to zero, including events from new controls.
The allocation fraction still reduces experimental enrollment. Use allocation
zero when modeling a fixed historical cohort without any new control arm.

Writing future experimental and control events as `E` and `C`, and past control
deaths as `H`, the null boundary SD is `sqrt(1/(H+C) + 1/E)`. Alternative
variability is `sqrt(C/(H+C)^2 + 1/E)`, because the past control deaths are fixed.
The signed effect is `log(control_hazard/experimental_hazard)`: the rejection
direction favors longer experimental survival. Reversing the effect lowers
power. The native historical-control interface is one-sided. Python's `sides=2`
option is an extension that halves alpha while retaining this fixed direction.

## Piecewise-exponential survival

```python
from mdanderson_stats import stplan_piecewise_survival_power

power = stplan_piecewise_survival_power(
    0.2, 0.05, 3, 1.5, 5, 10, 5, model_arm="lower_hazard",
)
# Approximately 0.3222513.
```

Inputs are `hazard_before, hazard_after, change_time, hazard_ratio, accrual_rate,
accrual_duration, followup_duration`. The change time is a participant's age
since enrollment, not calendar study time. Hazards and change time may be zero.
The hazard ratio is at least one. `model_arm="lower_hazard"` means the supplied
hazards belong to the better-survival arm; the other arm multiplies them by the
ratio. With `model_arm="higher_hazard"`, the other arm divides by the ratio.
These explicit labels avoid contradictory numeric arm labels in the native
interface and calculation module.

For each arm, survival at age `t` is
`exp(-hazard_before*min(t,change_time) - hazard_after*max(t-change_time,0))`.
Python averages event probability over the uniform follow-up-age interval,
splitting it at the change time. Breakpoints before or after all observed ages
are handled directly. The resulting total expected events enter the information
formula with effect `log(hazard_ratio)`. Zero expected events give the normal
approximation's limiting value `alpha / sides`; this is not an exact
zero-event rejection probability.

The native piecewise routine fails to normalize its within-segment integrals
correctly and can evaluate unbounded entry intervals. This can produce negative
expected deaths or a nonfinite power. Python corrects the integral. Two of the
three retained native reference cases produce NaN, whereas independent
integration gives powers `0.3222513` and `0.3520292`. The third native result is
`0.0666544`, compared with the corrected `0.0676619`. Native piecewise outputs
are defect evidence, not correctness targets.

## Numerical checks and limits

Inputs share the 200,000-value and broadcast-case limits of the other STPLAN
APIs. The one-sample mixture additionally limits aggregate support terms to
200,000 and evaluates them in bounded chunks. Oversized individual cases raise
an error; oversized batches can be split. Invalid or unrepresentable
calculations raise errors rather than returning a nonfinite power.

`tools/reference_stplan_survival.R` independently integrates event probabilities
and computes the five planning formulas for 18 cases. It uses base R only.
`tools/reference_stplan_survival.f90` probes the original numerical routines with
15 of those inputs. Twelve non-piecewise native outputs agree within their
numerical tolerances: roughly `9e-8` for the truncated one-sample mixture,
`2e-8` for George–Desu, and rounding error for the two normal approximations.
Six piecewise cases use independent quadrature as their reference, including
changes within the follow-up-age interval and zero hazards in either segment.
See [provenance and reproduction inputs](stplan-sources.json). The original
source and manuals are not redistributed.
