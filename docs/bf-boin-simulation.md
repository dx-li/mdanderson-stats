# BF-BOIN calendar simulation

`simulate_bf_boin` replays BF-BOIN with asynchronous enrollment and delayed
DLT assessment. The default inter-arrival gap is uniform on
`(0, 2/accrual_rate)`; exponential gaps are available with
`arrival_distribution="exponential"`. The first patient starts at calendar
time zero. A DLT endpoint is calibrated with a Weibull distribution satisfying
`F(window)=p` and `F(window/2)=p/2`; assessment is at the earlier of the DLT
time and the window. At `p=0` no DLT occurs and assessment is at the window;
at `p=1` the limiting DLT time is `window/2`.

The required `true_response` vector is sampled at enrollment and observed at
the DLT window; positive observed responses gate backfill activity. Each
escalation cohort is enrolled at its current dose. While its outcomes are
pending, subsequent arrivals may be assigned to the highest eligible lower dose
under the core BF-BOIN rules. Only observed assessments affect eligibility and
decisions. The `assigned` result counts every assignment, while `patients` and
`toxicities` are final evaluated counts after follow-up. Per-trial assignment
counts and per-patient dose, arrival, assessment, DLT, response, and backfill
records are retained in the corresponding history fields for reproducibility
checks. Escalation decisions occur only after the current escalation cohort's
DLT assessments are complete; pending backfill outcomes are carried into final
follow-up.
`escalation_end` records the last escalation decision; `trial_duration` includes
final follow-up.

The simulator supports the primary escalation-time backfill mode. Expansion
after escalation and titration are separate workflows and are intentionally not
accepted as silently ignored options.

DLT and response are sampled independently. Responses from either enrollment
component can establish activity. Decisions occur at completion of the current
cohort's DLT assessments; the third-party R reference instead processes decisions
at arrival times. Posterior elimination is retained at cohort decisions. Final
follow-up includes all enrolled patients and their response windows. These
choices differ from the R reference's truncation of pending backfill outcomes,
so full-trial numerical parity is not claimed.

```python
from mdanderson_stats import BFBOINDesign, simulate_bf_boin

result = simulate_bf_boin(
    BFBOINDesign(n_cap=12, n_stop=9),
    true_toxicity=[0.05, 0.15, 0.30],
    true_response=[0.20, 0.40, 0.50],
    cohorts=6,
    trials=20,
    accrual_rate=6,
    rng=162,
)
print(result.selection_probability)  # first entry represents no selected MTD
```

The maximum retained-record bound is 1,000 per trial and 100,000 across a
request, calculated before allocating output arrays. An arrival limit of
100,000 per trial raises an error instead of changing the renewal schedule.
Requests outside the supported floating-point timing range also fail explicitly.

The example's 20 trials ran in 0.066 seconds on the development machine and
produced 212 backfill assignments. A separate history audit checked observed
activity before each backfill, lower-dose allocation, caps and final count
conservation. Process peak resident memory was about 109 MiB, including imports.
This is a bounded execution check, not a precise operating-characteristic study.
