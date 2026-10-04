# Original 2010 GAO cohort trials

`u2oet_gao2010_decision` and `simulate_u2oet_gao2010_trial` implement the
original GAO utility and cohort-allocation rules from
[Houédé et al. (2010)](https://doi.org/10.1111/j.1541-0420.2009.01302.x).
They use the [original 2010 posterior model](u2oet-gao2010-fit.md).

The decision maximizes posterior mean utility over the elementary joint
outcomes (equations 10–11). Equation 12 first stops the trial only when, at
**every dose pair**, the posterior probability that the highest toxicity
category exceeds `toxicity_limit` is strictly greater than
`stopping_probability`. Both comparisons are strict. This is a global stop;
it does not remove individual pairs from the utility ranking.

For an interim decision, previously treated pairs and coordinatewise lower
pairs remain eligible. Untried escalation is restricted to the paper's three
adjacent upper neighbors. The paper does not settle an untried mixed-direction
move, so this implementation conservatively excludes it. Ties use row-major
grid order. These two choices are explicit Python conventions. A final
recommendation considers the full grid without interim movement restrictions.
The standalone decision accepts the fitter's `fit.joint` array, an outcome
utility matrix, and, for interim decisions, `current_pair` and a `treated`
count grid.

The simulator starts at the caller's explicit pair without a prior safety
analysis, then fits after each cohort or final patient-budget remainder.
It accepts supplied categorical joint truth and an outcome-independent
efficacy-evaluability probability. Unevaluable efficacy contributes to the
toxicity-only likelihood. The following fixed-parameter example demonstrates
the interface and outcome tape; it is not a posterior convergence example.

```python
import numpy as np

from mdanderson_stats import simulate_u2oet_gao2010_trial

truth = np.full((2, 2, 2, 2), 0.25)  # dose1, dose2, efficacy, toxicity
tape = np.array([[0.0, 0.1], [0.3, 0.9], [0.6, 0.2], [0.9, 0.7]])
trial = simulate_u2oet_gao2010_trial(
    [0.0, 1.0],
    [0.0, 2.0],
    truth,
    [[0.0, 0.0], [1.0, 1.0]],
    prior_mean=np.zeros(12),
    prior_sd=np.zeros(12),
    fixed_association=0.0,
    starting=(0, 0),
    n_patients=4,
    cohort_size=2,
    efficacy_evaluability=0.5,
    toxicity_limit=0.99,
    draws=8,
    warmup=0,
    chains=2,
    outcome_uniforms=tape,
    rng=np.random.default_rng(2026),
)
assert trial.patients.complete.sum() == 2
assert trial.patients.toxicity_only.sum() == 2
assert trial.posterior_fits == 2
assert trial.selected_pair == (0, 0)
```

Each tape row supplies a flattened joint-outcome CDF uniform and an independent
efficacy-evaluability uniform. Results retain the full tape, child seeds,
patient counts, per-look decisions and compact diagnostics. Repeating the
same inputs with the same caller-generator state reproduces the trial.
Sampling at each look uses the existing explicit-prior fitter; its native
Gibbs sampler and elicited-prior calibration are not reproduced. Clinical
within-patient adaptation and a calendar/pending-outcome process remain open.

Numerical work is serial. The driver preflights minimum cumulative fitting
work and retained storage before consuming the caller's random state, counts
initial-domain validation against its work budget, and releases posterior
draws after each look. Actual work can exhaust the budget during sampling;
that raises an error rather than returning a partial recommendation.
Split-Rhat and utility MCSE are descriptive diagnostics, not proof of
convergence. Utility summaries are scaled to avoid overflow where possible;
unrepresentable summaries raise an arithmetic error.

The [independent base-R reference](../research/u2oet-gao2010-conduct-reference-audit.md)
checks supplied posterior draws, both strict thresholds, intermediate
eligibility, final selection and Monte Carlo summaries. It does not establish
native executable parity or reproduce published operating characteristics.
