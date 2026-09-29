# TITE-BOIN12 published patient observation ledger

`tests/fixtures/tite-boin12-paper-calendar.csv` transcribes the TITE arm of
Table 2 in [Zhou et al., TITE-BOIN12](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/),
DOI 10.1002/sim.9337. Times are days, the toxicity assessment window is 45
days, and the efficacy window is 60 days. An infinite delay denotes no event
within that endpoint's window. Dose labels and enrollment dates are the
published realized assignments, not counterfactual outcomes at other doses.

An endpoint becomes known at enrollment plus its event delay, or at the
assessment deadline for a non-event. Outcomes known exactly at an analysis
time count in that analysis. Counting all previously enrolled patients at the
current dose gives this independently derived observation ledger:

| Analysis day | Dose | Patients | Pending toxicity | Pending efficacy | Observed toxicity | Observed efficacy |
| --- | --- | --- | --- | --- | --- | --- |
| 71 | 1 | 3 | 0 | 1 | 0 | 0 |
| 142 | 2 | 3 | 0 | 1 | 1 | 1 |
| 213 | 3 | 3 | 0 | 1 | 1 | 1 |
| 284 | 4 | 3 | 0 | 1 | 2 | 0 |
| 315 | 3 | 6 | 2 | 2 | 2 | 2 |
| 396 | 3 | 9 | 0 | 0 | 3 | 5 |

At day 315 the denominator includes both cohorts already treated at dose 3.
For the final cohort, patient 16 has toxicity ascertained on day 361 and an
efficacy event on day 366; patient 17 has toxicity ascertained on day 371 and
an efficacy event on day 376; patient 18 has a toxicity event on day 366 and
efficacy ascertained on day 396. Thus all endpoints are known by day 396.

This ledger verifies observation chronology independently of the Python
calendar engine. It does not alone establish identical adaptive assignments,
native supplemental posterior conventions, or the general arrival/monitoring
policy. The illustrated one-day gap between a decision and subsequent
enrollment is a feature of the published example, not an inferred universal
software default.

## Adaptive-decision comparison

The published example is explicitly the AL version with toxicity/efficacy
limits 0.35/0.25 and utilities `(100, 40, 60, 0)`. The calendar driver reproduces
its enrollment dates using gaps `[1, 10, ..., 10]`, three-patient cohorts and
a one-day decision lag. Replaying potential-outcome tapes whose assigned-dose
entries match Table 2 reproduces dose assignments through day 305. At day 315,
however, the implemented AL rule recommends dose 2 while the illustration
assigns the next cohort to dose 3.

At that look, dose 3 has two observed toxicity events, four ascertained toxicity
outcomes and two pending outcomes with follow-up 20 and 10 days. The effective
sample size is `4 + 20/45 + 10/45 = 14/3`, and the declared AL toxicity estimate
is `2/(14/3) = 3/7`. This exceeds the standard BOIN de-escalation boundary for
target 0.35. All four doses remain admissible; dose-2 and dose-3 utility tail
probabilities are approximately 0.1558 and 0.1172 respectively. Both the toxicity
comparison and these utility rankings explain the Python decision. No different
settings or BDA variant were identified in the example text that resolve it.

This is an unresolved discrepancy with the published illustration. The
implementation is not adjusted to force that dose sequence. Table 2 remains
useful as an independent observation-ledger reference, but it is not asserted
as a successful full adaptive-trial reference. The finite outcomes at unrealized
doses are unknown; filling those tape cells with infinity is an explicit
synthetic extension, not recovered source data. After the dose sequence
diverges, those cells cannot establish published trial outcomes or final OBD.
