# Six-dose posterior probability summaries

The archived six-dose kernel reports ten rows of six componentwise means and
variances. The first rows are reference-superiority probabilities, efficacy
threshold probabilities, future-study threshold probabilities, and the
row-major 6-by-6 strict pairwise-superiority probabilities. The last row is the
posterior second moment of response probability at each dose, `E[p² | data]`;
it is not another event probability. Toxicity exceedance is computed
separately from the beta-binomial model.

`summarize_phase12_importance_fits(fits)` computes those across-analysis-call
summaries from an explicit iterable of `Phase12ImportanceFit` results. It
streams the inputs rather than retaining a scenario's fit list. Component
labels and their exact order are returned with the 60-element mean and sample
variance arrays; dose indices in labels are zero-based (0 through 5). A single
fit has zero sample variance, matching the native running accumulator; an
empty iterable is rejected. Nonconverged fits are
included and counted, since the native integrator also returns its estimate at
the cap. The count does not certify integration quality.

The native accumulator resets at each scenario and receives one vector after
each successful kernel integration. These calls arise from both stopping-rule
evaluation and winner selection; they are analysis calls, not one record per
simulated trial. To summarize manually, supply every successful analysis
result from that scenario, including both call sites. The calendar OC captures
its importance-backend invocations automatically, including calls that reuse
a cached fit, and reports actual posterior refits separately. This matches
the Python invocation population but does not reproduce the extra integration
noise from native fresh integrations on repeated tallies. The summary makes no
independence or Monte Carlo-error claim. Its Welford-updated means are
mathematically equivalent to the native arithmetic means, with possible
last-bit differences from the native `sum/n` accumulation order; component
variances use the native Welford sample-variance convention (`M2/(n-1)`).

```python
import numpy as np

from mdanderson_stats import (
    fit_phase12_importance,
    summarize_phase12_importance_fits,
)

empty = np.zeros((6, 4))
responses = empty.copy()
responses[:, 1] = np.arange(1, 7)
fits = [
    fit_phase12_importance(empty, max_integrations=200, rng=np.random.default_rng(21)),
    fit_phase12_importance(responses, max_integrations=200, rng=np.random.default_rng(22)),
]
summary = summarize_phase12_importance_fits(iter(fits))
print(summary.analysis_call_count, summary.nonconverged_analysis_call_count)
for label, mean, variance in zip(
    summary.component_labels,
    summary.component_means,
    summary.component_sample_variances,
    strict=True,
):
    print(label, mean, variance)
```

This tiny configuration demonstrates the workflow; it is not calibrated OC
accuracy. The calendar OC computes the same summary online for each importance-backend
trial and pools it across trials without retaining fit histories:

```python
from mdanderson_stats import simulate_phase12_calendar_oc

oc = simulate_phase12_calendar_oc(
    [0.03, 0.06, 0.10, 0.16, 0.24, 0.34],
    [0.12, 0.20, 0.31, 0.42, 0.48, 0.50],
    n_trials=2,
    seed=20261004,
    max_patients=24,
    max_attempts=100,
    posterior_backend="importance",
    importance_max_integrations=100,
    importance_max_mode_iterations=100,
)
print(oc.posterior_probability_summary.analysis_call_count)
print(oc.posterior_refit_count)
print(oc.posterior_parameter_fit_trial_count, oc.posterior_parameter_no_fit_trial_count)
print(oc.posterior_mode_mean, oc.posterior_mode_laplace_mixture_variance)
```

The explicit-fit helper is limited to 100,000 inputs. The calendar OC instead
uses its aggregate trial/work bounds. Only the five probability arrays used by
the native vector are validated; unrelated fit diagnostics do not affect the
aggregation. The calendar keeps no fit histories and does not reproduce the
native random stream or adaptive integration details. See
the [source and output crosswalk](../research/parallel-phase12-scenario-report-audit.md)
for the C++ call sites and component mapping.
