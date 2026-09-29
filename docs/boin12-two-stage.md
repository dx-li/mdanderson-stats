# BOIN12 two-stage design

`boin12_two_stage_next_dose` and `simulate_boin12_two_stage` implement the
BOIN12 application's optional toxicity-only run-in followed by utility-based
dose optimization. Supply `stage1_threshold` explicitly; the application
requires an integer from 6 through 12 and recommends 6.

```python
from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.boin12_two_stage import simulate_boin12_two_stage

design = BOIN12Design(toxicity_limit=0.35, efficacy_limit=0.25)
truth = [
    [0.60, 0.25, 0.10, 0.05],  # noT/E, noT/noE, T/E, T/noE
    [0.45, 0.25, 0.20, 0.10],
]
result = simulate_boin12_two_stage(
    design, truth, stage1_threshold=6, cohorts=8, cohort_size=3, trials=100, rng=21
)
```

The cohort that first brings any dose to `stage1_threshold` remains a Stage 1
cohort. The transition is evaluated after that cohort, so Stage 2 makes the
next assignment. `transition_cohort` is the one-based triggering cohort, or
zero if the threshold was not reached. `stage1_cohorts` and `stage2_cohorts`
count cohorts assigned under each stage. All joint outcome counts are retained
and used for final OBD selection, including trials that stop before entering
Stage 2.

Stage 1 uses the BOIN toxicity escalation/de-escalation boundaries. Safety is
the BOIN12 posterior tail `Pr(pi_T > toxicity_limit | data)` under the same
Beta(1,1) prior, with a dose admissible only when this tail is strictly below
`toxicity_cutoff`. A failed safety check sticks for that dose; the Python
policy does not add a minimum sample-size guard or automatically exclude
higher doses. If the current dose is excluded above dose 1, the decision moves
down one level only when that adjacent dose is admissible; otherwise it stops
with `stop_no_admissible_neighbor`. Exclusion of dose 1 stops for safety. For
an admissible current dose, rate `<= escalation_boundary` moves up one level,
rate `>= deescalation_boundary` moves down one level, and an interior rate
stays; a blocked escalation stays, while a blocked de-escalation stops. The
global no-admissible-dose safety stop and safety exclusion at dose 1 precede
`early_stop_patients`. The precision stop then precedes ordinary movement and
the nonterminal fallback from an excluded middle dose.

Stage 2 uses `BOIN12Design.next_dose`, including its safety, efficacy,
utility, and early-stop rules. Final selection always calls
`BOIN12Design.select_obd` on all observed counts. Cohorts are fully observed
and assigned sequentially in this simulator; delayed outcomes are outside its
scope.

The source help specifies the S threshold and which endpoint criteria belong
to each stage, but does not fully describe stage-boundary ordering, persistence
of safety exclusions, or stopping precedence. Those operational details above
are explicit Python policies, not a claim of full native-app parity.
