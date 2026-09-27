# TITE-BOIN and Rolling Six comparison

The [TITE-BOIN guide](https://biostatistics.mdanderson.org/shinyapps/TITE-BOIN/Guide.pdf)
and [BOIN desktop specification](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/99)
offer Rolling Six as a comparator. `compare_tite_boin_rolling_six` runs both
existing Python simulators under the same toxicity, event-time and arrival
scenario, then returns their original outputs plus numerical summaries.

```python
import numpy as np
from mdanderson_stats import (
    BOINDesign,
    compare_tite_boin_rolling_six,
    tite_boin_rolling_six_report,
)

comparison = compare_tite_boin_rolling_six(
    BOINDesign(target=0.3),
    [0.05, 0.15, 0.3, 0.45, 0.6],
    window=3,
    accrual_rate=2,
    cohorts=6,
    trials=24,  # Small reproducible example; increase for precise estimates.
    rng=99,
)
np.testing.assert_allclose(comparison.tite_boin.selection_probability.sum(), 1)
np.testing.assert_allclose(comparison.rolling_six.selection_probability.sum(), 1)
assert comparison.tite_boin_max_patients == 18
assert comparison.rolling_six_max_patients == 30
assert np.all(comparison.rolling_six.patients <= 6)
report = tite_boin_rolling_six_report(comparison)
assert "No recommendation" in report
print(comparison.tite_boin_summary.mean_duration)
print(comparison.rolling_six_summary.mean_duration)
```

## Interpretation

The two simulations use independent draws from one generator advanced serially.
They share scenario parameters, not patient outcomes. TITE-BOIN runs first.
An integer seed makes the entire comparison reproducible; an existing generator
is advanced by both simulations. Both branches' parameters and aggregate
resource bounds are checked before simulation starts.

Each design retains its own enrollment and stopping rules. TITE-BOIN's maximum
is `cohorts*cohort_size`; Rolling Six defaults to `min(6*doses,200)` and accepts
an explicit `rolling_six_max_patients`. The comparison does not force equal
realized enrollment. Each call supports at most 100,000 trials, 200 planned
patients per design, ten million combined output cells and one hundred million
potential patient-by-dose scenario cells across the two designs. Jobs run
serially and retain trial summaries rather than all potential-outcome arrays.

Selection probabilities include a no-recommendation bin followed by one bin
per dose. Their Monte Carlo standard errors are `sqrt(p_hat*(1-p_hat)/trials)`.
A selected dose is a recommendation from that design, not proof that it is a
true MTD. In particular, Rolling Six distinguishes `mtd` from
`highest_planned_dose`, `no_safe_dose` and `inconclusive`. These statuses are
retained in its original result and their frequencies appear in the report.

The `tite_boin_summary` and `rolling_six_summary` records expose mean enrollment
and DLT counts by dose, total enrollment/DLTs, duration, suspension time and
status probabilities. Original per-trial arrays remain available for other
analyses. The Markdown report includes those estimates, the scenario and design
settings; it does not fabricate correct-selection rates or native app output.

Event timing can be uniform, piecewise uniform through trimester masses,
Weibull or log-logistic. Arrival gaps can be fixed or exponential. True event
timing and TITE-BOIN's analysis prior are separate inputs. See
[TITE-BOIN](tite-boin.md) and [Rolling Six](rolling-six.md) for their conduct,
time-unit and scheduling conventions.

This implements the documented comparison capability with explicit Python
simulation/report conventions. It does not claim the desktop's unpublished
scheduler, random sequence, sample matching or native Word/HTML layout.
