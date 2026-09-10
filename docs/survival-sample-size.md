# Time-to-event sample-size planning

Catalog entry **138**, MD Anderson's
[Nsurvival calculator](https://biostatistics.mdanderson.org/shinyapps/Nsurvival/),
is implemented for historical-control equality and two-arm equality, superiority,
noninferiority and equivalence. The
[technical document](https://biostatistics.mdanderson.org/shinyapps/Nsurvival/Log-rank-survival-help.pdf)
and application snapshot are pinned in [provenance](survival-sample-size-source.json).

```python
from mdanderson_stats import survival_sample_size

one = survival_sample_size(3, 7, accrual=3, followup=6)
assert one.group_sizes.tolist() == [17]
assert one.rounded_events == 9

two = survival_sample_size(10, 20, accrual=12, followup=6, allocation_ratio=1)
assert two.group_sizes.tolist() == [59, 59]
assert two.rounded_events == 52
print(two.required_events, two.expected_events, two.power)
```

**`followup` is additional follow-up after accrual ends.** The site's maximum
follow-up L equals `accrual + followup`; its A=12, L=18 example therefore uses
`followup=6`. Durations must be nonnegative with a finite sum. Zero accrual means
simultaneous enrollment. With no observation time, event probability is zero
and sample-size planning raises an error.

By default, `control` and `treatment` are median survival times. With
`parameter="hazard"`, they are positive exponential hazard rates in reciprocal
time units. Medians and hazards are related by `hazard=log(2)/median`. All rate,
duration, allocation and margin arrays broadcast over scenarios. Group axes are
last; two-arm order is control, treatment. One-arm results count only treatment
participants, not historical controls.

## Hypotheses and information

The effect is `b=log(treatment_median/control_median)=log(control_hazard/treatment_hazard)`;
positive b indicates better survival with treatment. Margins use this **log scale**,
not a median difference or an untransformed hazard ratio.

| Objective | Alternative |
| --- | --- |
| `equality` | Nonzero b; the sign selects direction in one-sided planning |
| `superiority` | b > margin |
| `noninferiority` | b > -margin |
| `equivalence` | -margin < b < margin |

Margin objectives require two arms. Equality requires zero margin; other
objectives require a positive margin and an expected effect inside the alternative.
`sides` is 1 by default and affects only equality; two-sided equality retains
the dominant normal rejection tail at alpha/2. These are **asymptotic event-based
planning powers**, not finite-sample log-rank guarantees.

One-arm information is the event count D. Two-arm information is `D*pc*pt`, where
pc and pt are the **planned randomization fractions**. `allocation_ratio` is
treatment/control and may range from `1e-6` to `1e6`. Continuous enrollment is
split using these fractions, then each group is rounded up separately. Power uses
the resulting expected number of events with the planned fractions; rounding
does not redefine the randomization ratio.

`required_events` is fractional. `rounded_events` is its ceiling for reporting.
Enrollment uses the fractional requirement, matching the source examples.
Consequently, `expected_events` can be below `rounded_events` while still exceeding
`required_events`. To plan a trial stopping at a fixed integer event target, use
that integer with `survival_event_power` and assess accrual separately.

## Event observation models

For one arm, both modes integrate exponential event probability under uniform
accrual exactly. For two arms, the default `event_method="source"` uses the manual's
Simpson approximation to mean control survival at F, F+A/2 and F+A, then raises
that mean survival to the treatment/control hazard ratio. This treatment
transformation is an approximation and depends on which group is control.

`event_method="uniform"` instead integrates each arm's exponential event
probability under uniform accrual independently:

```python
from mdanderson_stats import exponential_event_probability

print(exponential_event_probability([0.03, 0.06, 0.12], accrual=12, followup=6))
uniform = survival_sample_size(
    10, 20, accrual=12, followup=6, allocation_ratio=1, event_method="uniform"
)
print(uniform.group_sizes, uniform.event_probabilities)
```

Both modes assume no dropout and no competing risks. Exact event integration
does not make the subsequent asymptotic power formula exact. Stable exponential
differences, small-argument series and effective-exposure calculations preserve
small event probabilities and avoid overflowing hazard ratios.

## Equivalence conventions

```python
conservative = survival_sample_size(
    9,
    10,
    accrual=12,
    followup=6,
    allocation_ratio=1,
    objective="equivalence",
    margin=0.4,
)
assert conservative.group_sizes.tolist() == [347, 347]
assert conservative.rounded_events == 395

joint = survival_sample_size(
    9,
    10,
    accrual=12,
    followup=6,
    allocation_ratio=1,
    objective="equivalence",
    margin=0.4,
    equivalence="joint",
)
assert joint.required_events < conservative.required_events
```

The source example uses a conservative failure allowance of beta/2 for each
one-sided rejection condition. The default `equivalence="conservative"` matches
it, using the closer equivalence boundary to bound both failures. `"joint"`
solves the actual intersection probability under the working normal distribution.
This can reduce enrollment when the expected effect is off-center. It remains
normal-approximation survival planning. Both modes support event-power curves:

```python
from mdanderson_stats import survival_event_power

print(
    survival_event_power(
        [100, 200, 400],
        0.1,
        allocation_ratio=1,
        objective="equivalence",
        margin=0.4,
        equivalence="joint",
    )
)
```

Defaults are alpha .05 and target power .8; require `0 < alpha < .5` and
`alpha < target_power < 1`. Results own read-only arrays. Unrepresentable event
requirements or enrollment at or above `2**53` raise errors rather than returning
unusable counts.

## Source discrepancies and validation

The manual omits squares from log-effect denominators in its equality event
formulas and reverses the hazard/median direction in some hypotheses. Its printed
equivalence formula uses beta instead of the beta/2 needed to reproduce the
worked example. We use the proper squared log effect and the conventions above.
The help prose also mixes parameters in the equivalence example: the live input
defaults (medians 9 and 10, margin .4) reproduce its 395 events and 347 per arm.
These are source-document discrepancies, not claims of equivalent live-server bugs.

Ten focused tests reproduce all five source examples. Numerical integration and
simulation of exponential event times independently validate uniform-accrual
event probabilities. Gaussian rejection-region simulation validates joint
equivalence power; further checks cover tiny and overflowing hazard-duration
products, median/hazard input equivalence, time-unit scaling, unequal allocation
and invalid follow-up. Python APIs cover the statistical calculator; the original
Shiny interface and report layouts are not reproduced.
