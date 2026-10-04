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
simulated trial. To reproduce the same call population, supply every successful
analysis result from that scenario, including both call sites. The existing
calendar exposes `last_fit`, not the complete analysis-call sequence, so a
caller-supplied set of terminal fits has a different scope. The summary makes
no independence or Monte Carlo-error claim. Its Welford-updated means are
mathematically equivalent to the native arithmetic means, with possible
last-bit differences from the native `sum/n` accumulation order; component
variances use the native Welford sample-variance convention (`M2/(n-1)`).

```python
import numpy as np

from mdanderson_stats import fit_phase12_importance
from mdanderson_stats import summarize_phase12_importance_fits

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

The iterable is limited to 100,000 fits. Only the five probability arrays used
by the native vector are validated; unrelated fit diagnostics do not affect
the aggregation. This does not alter calendar behavior, retain fit histories,
or reproduce the native random stream and adaptive integration details. See
the [source and output crosswalk](../research/parallel-phase12-scenario-report-audit.md)
for the C++ call sites and component mapping.
