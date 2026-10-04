# BOIN12 simulation report

`boin12_report` captures a validated `BOIN12Design`, bounded joint-outcome
scenarios, and one seed, then runs the existing single-stage or optional
two-stage simulator serially. It returns an immutable HTML-ready summary and
can save a self-contained HTML file. The report summarizes the simulations; it
does not retain per-trial count arrays.

The four scenario columns are, in order: no toxicity/efficacy, no
toxicity/no efficacy, toxicity/efficacy, and toxicity/no efficacy. Each dose
row must be a probability distribution. Keeping the full joint row preserves
toxicity–efficacy association, including for nonadditive utilities.

```python
from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.boin12_report import BOIN12ReportScenario, boin12_report

design = BOIN12Design(
    toxicity_limit=0.35,
    efficacy_limit=0.25,
    utilities=(100, 35, 65, 0),
)
scenario = BOIN12ReportScenario(
    "illustrative",
    [
        [0.72, 0.18, 0.06, 0.04],
        [0.55, 0.20, 0.15, 0.10],
        [0.35, 0.20, 0.25, 0.20],
    ],
)
report = boin12_report(
    design,
    (scenario,),
    cohorts=4,
    cohort_size=3,
    trials=20,
    start_dose=1,
    stage1_threshold=6,
    seed=148,
)
report.write_html("boin12-report.html")
```

Set `stage1_threshold=None` (the default) for single-stage BOIN12. A threshold
from 6 through 12 selects the existing two-stage toxicity-only-to-utility
workflow. Its triggering cohort remains Stage 1; the following assignment is
governed by Stage 2. The report includes the transition cohort distribution,
with 0 meaning no transition, and mean Stage 1/Stage 2 cohort counts.

OBD and MTD selection vectors include index 0 for no selection, followed by
one-based dose labels. Probabilities and Bernoulli MCSEs are unconditional over
all simulated trials. Mean patients, toxicities, and efficacies are reported
per dose; stop-reason frequencies also use all trials. The report records the
complete four-cell scenario truth, utilities, design cutoffs, derived BOIN
movement boundaries, cohort settings, optional transition threshold, and
seed. Scenario runs consume one NumPy generator serially in the order passed.

The report limits a design to 200 planned patients per trial and checks total
scenario/trial/dose work before simulation. The run-in's interaction with
utility selection when a 1/3 DLT pattern occurs remains unspecified in the
recovered source; this report records the current explicit Python two-stage
policy and does not claim to resolve that native precedence. It also does not
claim native HTML/Word formatting or random-stream parity. Multilevel outcomes
remain outside the current binary-endpoint method.
