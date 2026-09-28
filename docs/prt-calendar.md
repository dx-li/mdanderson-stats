# PRT calendar replay

`run_prt_calendar` connects the PRT posterior, isotonic risks and conduct rules
to explicitly supplied arrivals and dose-specific toxicity delays. It supports
cohort enrollment, suspension, FIFO waiting or declining arrivals, resumed
enrollment and final selection after enrolled patients' outcomes resolve.
Catalog entry 69 remains partial; see the [model and source limitations](prt.md).

```python
import numpy as np
from mdanderson_stats import run_prt_calendar

trial = run_prt_calendar(
    arrival_times=[0, 0.25, 0.5, 0.75],
    potential_toxicity_delays=[[1, 1], [0.25, 0.25], [2, 2], [np.inf, np.inf]],
    interval_endpoints=[1, 2],
    starting_dose=0,
    cohort_size=2,
    max_patients=4,
    waiting_policy="queue",
    target=0.99,
    prior_mean=-1.3,
    prior_variance=0.02,
    draws=32,
    warmup=8,
    chains=2,
    max_likelihood_evaluations=3000,
    rng=np.random.default_rng(9101),
)
assert trial.patient_dose.tolist() == [0, 0, 1, 1]
assert trial.duration == 2.75
print(trial.status, trial.selected_dose)
```

This tiny example checks event bookkeeping. Its target, prior and short chains
are illustrative, not calibrated design or convergence recommendations. The
default prior mean and increment variance are -14 and 28 from the archived
settings; sampling effort must be supplied explicitly.

Each delay-table row is a candidate, and each column is a zero-based dose.
Only the assigned dose's outcome becomes observable. Follow-up starts at actual
enrollment, including for queued candidates. Infinity or a delay beyond the
assessment window denotes no event during that window. Arrival times must be
nonnegative and nondecreasing; ties are allowed. The caller supplies the timing
distribution rather than inheriting an undocumented native simulation law.

Intervals follow the paper's explicit half-open convention: `[0,t1)`,
`[t1,t2)`, and so on. An event on an internal boundary belongs to the next
interval; an event exactly at the final endpoint falls outside the modeled
window. Only observed events and fully completed event-free intervals enter
the likelihood. An incomplete interval contributes no count. At a tied time,
follow-up updates precede enrollment. All remaining-risk rows and total risks
use aligned, projected posterior draws.

The starting cohort enrolls at the supplied starting dose. Conduct is evaluated
after each cohort and reconsidered during suspension when arrivals or observed
follow-up change. `waiting_policy="queue"` retains suspended arrivals in FIFO
order; `"decline"` does not enroll them later. These are explicit Python replay
policies. A partial cohort is analyzed if the arrival tape ends. A safety stop
remains permanent, while already enrolled patients' outcomes are followed.
Otherwise, final selection uses published rule 7 once outcomes resolve and may
select an untried dose. No final posterior is fitted after a permanent stop.

The result includes immutable patient records, compact analysis summaries,
posterior and predictive risk summaries, likelihood/work counts and fit count.
It distinguishes enrollment ending from follow-up completion and a safety stop
from ordinary tape exhaustion. `duration`, `enrollment_end_time` and
`stopping_time` are elapsed times from the first supplied arrival; patient and
analysis times use the original input origin. Undefined Rhat is retained, and
diagnostics do not certify convergence. Identical inputs and a freshly seeded
generator replay the same path.

At most 500 candidates, 10 doses, 10 intervals and 200 analyses are supported.
Input and retained-array limits plus cumulative likelihood/work budgets bound
resource use. Unchanged likelihood counts reuse one posterior; the full
posterior history is not retained. Predictive probabilities normalize verified
floating-point count mass, while materially invalid isotonic probabilities
still raise an error. In particular, the published full-covariance projection
can fail for otherwise valid trial data; the native safeguard is unresolved.

The [source audit](../research/prt-calendar-audit.md) records the independent R
interval ledger and replay checks. Native patient-file conversion, the
unavailable Appendix-B event-time generator, aggregate native operating
characteristics and exact executable equivalence remain outstanding.
