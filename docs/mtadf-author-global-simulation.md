# MTADF author global-logistic simulation

`simulate_mtadf_author_global` runs bounded serial operating-characteristic
trials for the recovered author global logistic rule. Each cohort draws a
toxicity count and then an efficacy count at the assigned dose; unassigned
dose outcomes are not generated. The supplied integer seed is retained
exactly in the result, and `selection_probability` and `selection_mcse`
summarize the final dose selections. The result also keeps compact observed
cohort outcome tapes and assignment indices so a completed trial can be
replayed; it does not retain all-dose potential outcomes or per-fit traces.

```python
from mdanderson_stats.mtadf_author_global_simulation import simulate_mtadf_author_global

result = simulate_mtadf_author_global(
    true_toxicity=[0.08, 0.16, 0.28, 0.42],
    true_efficacy=[0.15, 0.40, 0.65, 0.72],
    cohorts=4,
    cohort_size=3,
    trials=20,
    rng=2718,
)
print(result.selection_probability, result.mean_patients)
```

For a saved observed trial, `replay_mtadf_author_global_trial` accepts one
toxicity and efficacy count per cohort, for the dose actually assigned at that
review. Its `assigned_dose` and outcome tapes form a compact auditable record;
the input is not a table of potential outcomes at every dose.

The replay starts at dose index zero. After every cohort, including when only
the first dose remains safety-admissible, it fits the global quadratic logistic
model over the full dose grid. The next assignment moves one step toward the
rightmost fitted efficacy maximum, capped using the safety prefix computed
before that cohort. It then refreshes the cap. Final selection uses the last
fit and the refreshed cap. This preserves the source simulation's lagged-cap
order; direct global decisions use a fresh cap.

The Python simulation generates assigned-dose outcomes sequentially and does
not reproduce R's random-number stream or `arm::bayesglm` execution. The
bounded author-global fit reports its IRLS iteration count and convergence
status. Simulation stops with an error if a fit fails to converge rather than
returning an incomplete operating-characteristic result. Fit, outcome-work,
and retained-summary-cell budgets are checked before RNG use. The source
crosswalk and limitations are in the [audit](../research/mtadf-author-global-simulation-audit.md).
