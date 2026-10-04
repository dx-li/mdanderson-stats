# GAO adaptive calendar precision audit

## Source contract and claim boundary

The cached U2OET guide's four-corner precision criterion, target interval,
retained-draw requirement and source gaps are recorded in
[`u2oet-adaptive-precision-audit.md`](u2oet-adaptive-precision-audit.md).
This integration connects that stated criterion to the existing 2017 GAO
calendar posterior. It does not infer native GAO adaptive defaults or claim
that the original executable supports this exact workflow. Priors remain the
explicit Python coordinate means and SDs required by the GAO fitter.

## Calendar behavior

Adaptive mode is opt-in through the shared `U2OETAdaptiveSettings`. At each
analysis where complete or toxicity-only counts differ from the prior
sufficient-statistic state, the GAO adaptive fitter starts from the caller's
initial parameter values and continues within that fit until all four corner
ratios pass or its cap is reached. Unchanged counts reuse posterior and
precision metadata, including at final follow-up. A cap miss raises before
the associated allocation/final decision. Runtime exhaustion from the
cumulative evaluation/work budgets also raises; the driver does not return a
trial containing a decision made from an under-precise fit.

Each interim decision and the final trial expose target status, retained draws
per chain and the four chain-by-corner MCSE/SD ratios. Existing model-parameter
split-Rhat and maximum split-Rhat remain separate diagnostics. Design JSON
stores the adaptive settings so replicate summaries still require exact
design equality. With adaptive mode omitted or explicitly `None`, the fixed
fit, metadata and random-number path remain unchanged.

## Resource policy

The standalone adaptive wrapper provides a shared draw/work/memory plan for
this driver. Before the input Generator advances, the trial computes the
maximum possible distinct-fit count as the patient cap and checks minimum
aggregate evaluation/work requirements, standalone adaptive retained/live
fit bounds, and combined adaptive-live, posterior, history, precision-ratio
and random-tape storage. The effective work cap is
`min(max_work, adaptive_precision.max_total_work)`; likelihood-evaluation and
work counters from every successful adaptive fit are accumulated, including
the wrapper's per-chain deterministic start-likelihood checks. The remaining
global budgets are passed to each changed-state fit. Rejection proposals can
exceed minimum work, so a later runtime failure is possible by design and is
reported rather than silently dropping rows or resetting the budget.

## Validation

Nine focused tests passed: the five new adaptive-trial cases and all four
existing GAO calendar tests. The adaptive cases cover delayed-data caching and
refitting after final counts change, an unmet target before decision,
pre-RNG whole-trial minimum-budget rejection, runtime cumulative exhaustion,
and exact equivalence between omitted and explicit-`None` fixed mode. The run
took 4.42 seconds, peaked at 145,162,240 bytes RSS and reported zero swaps.
Root integration passes 18 affected checks across the shared planner,
standalone adaptive fitter, adaptive/fixed GAO calendar and the separately
integrated OOB Brier evaluator. That serial run used 5.653 seconds, 148.03 MiB
process peak RSS and zero swaps. Targeted Ruff/format/mypy pass, and independent
read-only review found no material budget, cache, metadata or memory defect.
No native executable parity is claimed.
