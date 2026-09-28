# STPLAN survival inputs

The survival planners use exponential means or hazards. These helpers convert
the alternative inputs offered by STPLAN: medians, survival at a specified time,
historical event counts and person-time, or points on a two-segment survival curve.
Use one consistent time unit throughout the conversion and subsequent planning.

## Exponential and historical inputs

```python
from mdanderson_stats import (
    stplan_exponential_hazard,
    stplan_historical_control_hazard,
)

experimental = stplan_exponential_hazard(36, parameter="median")
control = stplan_exponential_hazard(0.8, parameter="survival", time=7.75)
historical = stplan_historical_control_hazard(25, 433)
```

For a median `m`, hazard is `log(2)/m`; for a mean `m`, it is `1/m`.
For survival `S` at time `t`, hazard is `-log(S)/t`. Historical hazard is deaths
divided by total observed person-time. The latter is an exponential rate estimate;
person-time is the sum over participants, not the calendar length of the cohort.

`parameter` defaults to `"median"` and also accepts `"mean"` or `"survival"`.
Supply `time` only for survival probabilities. Means, medians, observation times,
and person-time are positive. Survival lies in `(0, 1]`, and historical deaths
are nonnegative. A survival probability of one or zero observed deaths gives
hazard zero; downstream planners retain their own positivity requirements.
Inputs broadcast using the existing STPLAN array rules.
For a forward API parameterized by mean survival, use `1 / hazard` when the
converted hazard is positive.

## Two-segment survival curves

```python
import numpy as np
from mdanderson_stats import stplan_piecewise_from_survival

# Survival curve with hazards 0.2 before time 4, and 0.05 afterward.
model = stplan_piecewise_from_survival(
    [2, 8],
    np.exp([-0.4, -1.0]),
    change_time=4,
)
# model.hazard_before = 0.2, model.hazard_after = 0.05

# Infer the change time from three points instead.
inferred = stplan_piecewise_from_survival(
    [2, 8, 10],
    np.exp([-0.4, -1.0, -1.1]),
)
# inferred.change_time = 4
```

With a supplied change time, pass two points. The second time must exceed the
change time; the first may lie before, at, or after it. Without a supplied change
time, pass three points. The first is treated as lying before the change and
the other two after it, allowing equality at the inferred boundary.

Times must be positive and strictly increasing, and survival must be
nonincreasing in `(0, 1]`. The solver works in cumulative-hazard space
`H(t) = -log(S(t))`. It rejects negative hazards, an inferred change outside
the specified segments, and curves that cannot be reconstructed consistently.
These are exact parameter conversions, not a statistical fit to noisy curve
estimates.

The result is an immutable `STPLANPiecewiseModel` with `hazard_before`,
`hazard_after`, `change_time`, and boolean `change_time_identified` arrays.
An exponential curve has no unique change time: the flag is false. The result
preserves a supplied change time or uses the midpoint of the first two times
when it must choose a representation. The represented survival curve remains
the same. Nearly equal hazards require particular care because change-time
recovery is ill-conditioned. Hazards within 64 machine epsilons of each other,
relative to their scale, use the constant-hazard representation; reconstruction
of the supplied cumulative hazards is still checked.

The point dimension is the final axis. Leading dimensions broadcast across
studies, with the same bounded array sizes used elsewhere in STPLAN. Pass the
resulting hazards and change time to `stplan_piecewise_survival_power` or use
them as fixed inputs to `stplan_solve`.

## Evidence

The source workflow is described in STPLAN's user manual, chapter 26, and its
`a2exp.f` interface. Exponential and historical conversions also appear in
`aschc.f`. `tools/reference_stplan_survival_inputs.R` independently solves linear
systems in cumulative-hazard space for 15 cases, including both known-change
branches, an inferred change, zero hazards, constant hazards, very small survival
probabilities, and rescaled times. Original source and manuals are not distributed.
See [source provenance](stplan-sources.json).
