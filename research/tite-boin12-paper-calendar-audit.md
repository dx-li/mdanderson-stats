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
