# BOP2 protocol reports

The BOP2 online application presents six endpoint families and report/OC views.
This Python workflow builds a self-contained HTML snapshot from actual design
inputs and computes the reported operating characteristics with the existing
Python kernels. It is a specified-design report; it does not claim to reproduce
the application's optimizer, Word output, Chinese-language report, or animation.

```python
from mdanderson_stats.bop2_protocol_report import (
    BOP2ProtocolScenario,
    bop2_binary_efficacy_report,
)

report = bop2_binary_efficacy_report(
    20,
    null_rate=0.20,
    scenarios=(
        BOP2ProtocolScenario("null", 0.20),
        BOP2ProtocolScenario("alternative", 0.40),
    ),
    cutoff_scale=0.80,
    gamma=0.50,
    looks=[10, 15, 20],
)
report.write_html("bop2-report.html")
```

`BOP2ProtocolScenario` stores one named truth. For binary efficacy/toxicity,
`probabilities` is a scalar rate. For ordinal endpoints it is
`[CR, CR+PR]`; for multiple efficacy it is `[endpoint1, endpoint2]` and requires
`joint_probability`. Scenario labels are descriptive labels supplied by the
caller; the report does not infer null/alternative status from a label.

The six builders are `bop2_binary_efficacy_report`,
`bop2_binary_toxicity_report`, `bop2_ordinal_report`,
`bop2_multiple_report`, `bop2_efftox_report`, and `bop2_survival_report`.
They accept the corresponding design settings and construct the design
internally, so the report cannot attach arbitrary settings to an unrelated
precomputed result. Binary adverse-event rates remain adverse-event rates;
binary efficacy stops for posterior futility at scheduled looks and concludes
at the final analysis. Binary toxicity stops for a sufficiently low posterior
safety probability; safe success is the complete-negative final conclusion.
Both binary rules continue at equality with the cutoff. Ordinal cells are
ordered CR/PR/other. Multiple and EffTox cells are ordered 11/10/01/00, with
EffTox efficacy first. EffTox reports H00/H01/H10/H11 using the supplied joint
rates or the documented product defaults. The survival builder reports null
and alternative exponential medians, uses one explicit seed sequentially across
the two simulations, and includes Monte Carlo standard errors for success.

Ordinal and multiple efficacy stop for futility only when **both** marginal
criteria fail; either endpoint passing avoids futility, and either marginal
success is sufficient for a positive final conclusion. EffTox has a different
rule: it stops when either assessed efficacy or toxicity criterion fails, and
both efficacy and safety must pass at the final analysis. Survival futility
compares total observation time with the displayed analytic boundary. All
survival time inputs (medians, accrual rate, follow-up, and exposure) must use
the same units; equality behavior is captured in the report settings and rule.
The stop-by-look vector counts non-success termination only. Early positive
efficacy conclusions are included in success probability, not counted again as
failures; final negative conclusions are included in the final stop entry. Thus
stop probability plus success probability partitions the trial.

Exact binary and categorical OC quantities come from the existing exact
recursions. The TTE report includes event-count/total-time boundaries and
Monte Carlo enrollment, event, calendar-time, and success summaries. The report
retains compact tuples rather than simulated patient histories. Work and
scenario limits are checked before simulations/recursions. Results are
descriptive for the specified truths and settings; Monte Carlo standard errors
do not guarantee error control. Use the underlying optimizer APIs when an
explicit finite-grid calibration is desired and report that optimization
result separately. Each scenario summary distinguishes scalar `truth_values`
from the derived categorical `category_probabilities`.
