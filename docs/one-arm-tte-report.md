# OneArmTTE multi-scenario report

`simulate_one_arm_tte_scenarios` runs the existing validated OneArmTTE
simulator once per scenario, in the order supplied. Each scenario has a name,
true mean/median TTE, Poisson accrual rate, and its own required positive seed.
The seed is reset independently for every scenario, so reordering the list does
not change another scenario's simulated stream. TTE and accrual rate must use
the same caller-selected time unit.

```python
from mdanderson_stats import (
    OneArmTTEScenario,
    one_arm_tte_design,
    simulate_one_arm_tte_scenarios,
)

design = one_arm_tte_design(
    standard_prior=(4, 12),
    experimental_prior=(1, 3),
    parameterization="mean",
    maximize=True,
    cutoff_inferiority=0.05,
    delta_inferiority=0,
    cutoff_superiority=0.95,
    delta_superiority=0,
    max_patients=40,
    minimum_patients=5,
    periodic_interval=1,
    monitor_at_accrual=True,
    followup_period=3,
)
report = simulate_one_arm_tte_scenarios(
    design,
    (
        OneArmTTEScenario("inferior", true_tte=2.5, accrual_rate=2, seed=981),
        OneArmTTEScenario("promising", true_tte=6, accrual_rate=2, seed=982),
    ),
    repetitions=100,
    credible_level=0.95,
    time_unit="months",
)
report.write_html("one-arm-tte-results.html")
```

The report captures a revalidated copy of the design, scenario order and inputs,
repetition count, quantile level, and compact results. It includes the design
settings, a row per scenario, and a detail section with early/final inferiority
and superiority frequencies and Monte Carlo standard errors, mean patients,
events, exposure, accrual-stop and final times, and sample-size/final-time
quantiles. Rule frequencies remain separate and can overlap. Means are derived
from the simulator's per-replication summaries; raw replication arrays are
discarded after each scenario is summarized.

HTML is escaped and standalone. `write_html` atomically writes UTF-8 output.
Opening the saved file displays the report; it is not an editable design file
and does not restore a simulation session. The report records the Python inputs
and results rather than matching the Windows application's report formatting,
version banner, or random streams.

The workflow accepts 1–20 scenarios and preflights all scenario data before
starting. Total potential patient draws across scenarios cannot exceed 100,000.
Actual accrual-phase monitoring checks are accumulated across scenarios and
limited to 100,000 by default; `max_total_monitoring_checks` can set a smaller
nonnegative ceiling. If stochastic calendar paths exceed that ceiling, the run
raises instead of returning a partial report. A single simulated trial remains
limited to 10,000 accrual-phase checks. Final assessment is not counted as an
accrual-phase check.

The report summarizes the existing statistical model. It does not introduce a
new operating-characteristic estimator or alter strict stopping comparisons.
See the [source and validation notes](one-arm-tte-source.md) for model,
calendar, and cross-language conventions.
