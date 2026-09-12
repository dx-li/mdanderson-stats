# BOIN12 utility-based dose finding

BOIN12 jointly considers binary toxicity and efficacy to choose an optimal
biological dose (OBD). This port covers the single-stage method behind
[MD Anderson's BOIN12 app](https://biostatistics.mdanderson.org/shinyapps/BOIN12/),
catalog entry 148. It provides posterior calculations, rank-based desirability
tables, dose decisions, final selection and complete-outcome simulation.

The [reference audit](boin12-reference.md) records source versions and numerical
fixtures. The executable comparator is the independent CRAN `escalation`
implementation; it is not the MD Anderson application's private backend.

```python
from mdanderson_stats import BOIN12Design, boin12_rank_desirability

design = BOIN12Design(toxicity_limit=0.35, efficacy_limit=0.25)
patients = [3, 6, 3, 0, 0]
toxicities = [0, 1, 2, 0, 0]
efficacies = [0, 3, 1, 0, 0]
decision = design.next_dose(patients, toxicities, efficacies, current_dose=2)
assert decision.next_dose == 2

table = boin12_rank_desirability([0, 3, 6, 9], toxicity_limit=0.35, efficacy_limit=0.25)
selection = design.select_obd(patients, toxicities, efficacies)
assert selection.obd == 2
```

## Outcomes and utilities

Counts are one-dimensional vectors indexed by dose; returned dose labels are
one-based. The four utility entries have this order:

| Outcome | Default utility |
| --- | ---: |
| Efficacy without toxicity | 100 |
| Neither efficacy nor toxicity | 40 |
| Both efficacy and toxicity | 60 |
| Toxicity without efficacy | 0 |

The default utilities are additive, so marginal toxicity and efficacy counts
suffice to calculate utility. For non-additive scores, also supply
`efficacy_without_toxicity`: this identifies the joint outcome counts without
assuming that toxicity and efficacy are independent.

Posterior utility calculations use fractional utility-weighted event counts
in a quasi-beta-binomial model. Separate Beta(1,1) marginal posteriors supply
toxicity and futility probabilities. `toxicity_limit` also supplies the BOIN
boundary target; it is not a second parameter distinct from that target.
All returned probabilities use the interval `[0,1]`, including
`utility_probability`; the R comparator displays the latter as a percentage.
`utility_mean` retains the 0–100 utility scale.

Desirability tables rank all admissible outcomes across the requested sample
sizes together. Exact ties receive average ranks, so ranks may be fractional.
Inadmissible rows have NaN ranks. Enumeration is limited to 100,000 rows and
currently supports additive utilities; posterior calculations and simulation
also support nonadditive utilities with joint counts.

## Decisions and final selection

The design compares posterior desirability among nearby admissible doses.
`stay_patients` controls when a toxicity rate inside the stay interval removes
the higher neighbor from consideration. `exploration_patients` controls
exploration of an untreated higher dose, subject to admissibility. Toxicity
above the de-escalation boundary directs the trial toward the lower dose.
An unavailable recommendation is represented by `None`.
`early_stop_patients=None` disables the enrollment stop; use 12 to match the
current app default. The exploration and stay thresholds default to 9 and 6.
Pass retained exclusions through `eliminated` to later decisions and final
selection. `decision.admissible` describes local candidate eligibility, not a
global elimination mask. A dose excluded for toxicity may be left for a safe
lower neighbor; exclusions prohibit assigning patients to it again.

Final selection applies unweighted isotonic regression to observed toxicity
rates at treated doses. Exact distance ties for the toxicity-limit MTD favor
the higher dose. The OBD maximizes posterior desirability among eligible doses
at or below that MTD. Returned fitted values retain full precision; untreated
positions are NaN.

## Joint-outcome simulation

```python
from mdanderson_stats import simulate_boin12

# Each row gives probabilities in the four-outcome order above.
simulation = simulate_boin12(
    BOIN12Design(0.35, 0.25, early_stop_patients=12),
    [[0.30, 0.60, 0.05, 0.05], [0.45, 0.35, 0.10, 0.10], [0.40, 0.20, 0.20, 0.20]],
    cohorts=12,
    cohort_size=3,
    trials=1000,
    rng=148,
)
print(simulation.obd_probability)
print(simulation.obd_mcse)
```

The simulator draws a multinomial outcome vector for each complete cohort,
preserving the specified dependence between efficacy and toxicity. Results
include per-trial marginal and joint counts, exclusions, final OBD/MTD labels,
stopping reasons, selection frequencies and Monte Carlo standard errors.
Index zero in selection summaries means no selected dose.

The focused reference checks cover 15 posterior cases (maximum absolute
probability difference `4.45e-16`), all 166 desirability-table rows for sample
sizes 0, 3, 6 and 9, and complete-data dose/OBD examples. A local 1,000-trial
run of the example took about 1.91 seconds; a 6,331-row desirability table
through 36 patients took about 0.019 seconds. These timings are observations
on the development machine, not performance guarantees. Source differences,
including the Python restriction to admissible final choices, are documented
in the reference audit.

## Remaining coverage

Nonadditive joint-outcome desirability tables, the current application's
two-stage option, 3+3 run-in, tradeoff utility mode, multilevel endpoints and
generated reports/protocols remain outstanding.
Late-onset outcomes belong to the separately cataloged TITE-BOIN12 design.
