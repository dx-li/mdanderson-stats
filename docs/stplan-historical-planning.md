# STPLAN historical-control allocation planning

`stplan_historical_allocation_plan` jointly finds accrual duration and the fraction
of new patients assigned to controls, targeting a specified power under STPLAN's
historical-control survival model. It implements the scientific calculation behind
the original program's optimal-allocation option. The Python API uses explicit
search bounds.

```python
from mdanderson_stats import (
    stplan_exponential_hazard,
    stplan_historical_allocation_plan,
)

plan = stplan_historical_allocation_plan(
    experimental_hazard=stplan_exponential_hazard(36),
    control_hazard=stplan_exponential_hazard(0.8, parameter="survival", time=7.75),
    accrual_rate=3,
    followup_duration=0,
    historical_deaths=10,
    historical_alive=40,
    target_power=0.8,
    accrual_bounds=(0.001, 1000),
)
# plan.control_allocation: approximately 0.263054
# plan.accrual_duration: approximately 71.50138
# plan.achieved_power: approximately 0.8
```

This is the example in STPLAN's manual, chapter 25.5. With all new participants
assigned to the experimental arm, the required accrual duration is approximately
93.82301 in the same time units. Adding about 26.3% new controls reduces it to
71.50138. Other designs have a boundary solution at zero new controls; the planner
compares the allocation endpoints explicitly.

## Model and bounds

The calculation uses `stplan_historical_survival_power`, including its fixed past
control observations, uniform accrual, administrative censoring, and favorable
direction of a lower experimental hazard. See [the survival model](stplan-survival.md)
for its event-count and variance formulas. It is a planning approximation, not a
fit to individual historical observations.

Inputs describe one study and must be finite scalars. The experimental hazard
must be below the control hazard, and target power must exceed `alpha/sides` and
be below one. `alpha=0.05` and `sides=1` follow the native one-sided convention;
`sides=2` uses the package's existing half-alpha extension.

`accrual_bounds` is a positive increasing pair in the same time units as the
hazards and follow-up. `allocation_bounds` defaults to `(0, 0.99)`; both endpoints
lie in `[0, 1)`. Constraints on allocation are respected, including solutions at
either boundary. A target already attained at the lower accrual bound returns
that bound with `at_accrual_lower_bound=True`; its achieved power may exceed the
target. A target not attained by the searched allocations at the upper bound
raises an error.

With `continued_followup=False`, the original model suppresses all future control
events, including events from newly enrolled controls. Allocating more new
patients to controls then only reduces experimental information, so the planner
uses the lower permitted allocation. This retains the original model's convention.

## Numerical search and result

At each accrual duration, the planner searches allocation using a bounded grid
and local refinement, keeping the endpoints as candidates. It then solves for
target power in log accrual time and re-evaluates the returned design. This is a
numerical search within the supplied bounds; it does not certify a global optimum
over arbitrary parameter ranges. As with the original inverse calculation, the
intended accrual bracket should contain the relevant power crossing.

The result reports accrual duration, control allocation, expected enrollment,
target and achieved power, bounds, evaluation count, and an immutable `inputs`
mapping that can be passed directly to `stplan_historical_survival_power`.
The final power residual is checked against `power_tolerance=1e-8`, except when
the lower accrual bound already attains the target.

The default allocation grid has 17 points, with a maximum of 129. A shared
`max_evaluations` limit bounds all forward-power calls, including inner allocation
searches; its maximum is 20,000. These are serial, bounded-memory calculations.

## Validation

Seven independent R cases cover an interior optimum, a constrained boundary,
zero new controls, a small effect, and disabled continued follow-up. Six of those
cases also match original Fortran `qschc` option 7, including both manual examples.
The R reference inverts duration separately for each allocation and then minimizes
duration, providing a different organization of the numerical calculation from
Python's power-envelope search. See [source provenance](stplan-sources.json) and
the two reference harnesses under `tools/reference_stplan_historical_allocation.*`.
Original source and manuals are not distributed.
