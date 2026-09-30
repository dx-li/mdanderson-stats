# Importance fitting in the Phase I/II calendar

`simulate_phase12_calendar` can use either the existing elliptical-slice MCMC
fit (`posterior_backend="mcmc"`, the unchanged default) or the bounded
adaptive vector-importance fit (`posterior_backend="importance"`). The
calendar sends the importance summaries directly to the same interim and final
decision rules. This is an optional Python integration, not a claim of native
RNG or optimizer parity.

```python
import numpy as np

from mdanderson_stats import simulate_phase12_calendar

trial = simulate_phase12_calendar(
    [0.04, 0.06, 0.09, 0.12, 0.16, 0.20],
    [0.10, 0.15, 0.22, 0.30, 0.38, 0.45],
    max_patients=20,
    posterior_backend="importance",
    rng=np.random.default_rng(20260929),
)
print(trial.reason, trial.future_selected)
for analysis in trial.analyses:
    if analysis.posterior_backend == "importance":
        print(analysis.time, analysis.posterior_integrations, analysis.posterior_converged)
        print(analysis.ratio_mc_se)
```

Each trial draws independent data and posterior child seeds from the supplied
generator and retains both seeds in its result. The data stream is separate
from posterior fitting, so importance draws do not consume or advance the data
stream. Different posterior estimates can still change decisions, allocations,
and stopping; identical seeds therefore do not guarantee identical records
across backends. The latest fit is
available as `trial.last_fit`; each analysis also retains its own importance
integration count, raw-integral MCSE vector, normalized-ratio MCSE vector, and
log evidence. `max_split_rhat` is `None` for importance analyses because
R-hat is an MCMC diagnostic and is not manufactured for this backend. Importance
draws are not retained.

The default importance limits are 10,000 integrations per fit, a `.001` raw
integral relative-error target, and up to 1,000 mode-optimizer iterations. A
calendar-level preflight bounds aggregate importance work before it draws the
two child seeds. Work is counted as component evaluations (67 integrand
components per integration); mode-optimizer iterations have a separate
2,100,000-iteration ceiling. For example, an 80-patient cap permits at most 17
distinct analyses, so the default integration limit requires at most
`17 * 10_000 * 67 = 11,390,000` component evaluations, below the default
50,000,000 aggregate budget. `max_total_posterior_component_evaluations` may be
raised up to 100,000,000. The per-fit integration ceiling may be raised up to
1,000,000, subject to that aggregate budget. If fewer than 100 integrations
remain at runtime, the calendar raises instead of returning a partial trial.

The fit's `posterior_converged` flag means its source-style raw-integral
criterion passed; it does not ensure each posterior ratio has a small MCSE.
The source's one-percent evidence floor and zero-hit indicator behavior also
apply. Inspect `ratio_mc_se`, particularly when a decision is near a cutoff.
As in the existing calendar, outcomes are analyzed only when available at a
look, and the default final fit uses the last attempted arrival time;
`complete_followup=True` explicitly extends follow-up when the trial has not
already stopped.
