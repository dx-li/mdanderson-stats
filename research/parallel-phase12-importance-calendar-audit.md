# Phase I/II importance-calendar integration audit

## Statistical and workflow contract

The six-dose posterior model and vector-importance proposal are implemented in
`parallel_phase12_importance.py`; the source equations and component order are
audited in `parallel-phase12-importance-audit.md`. This change only wires that
fit into the existing calendar. The calendar continues to create independent
data and posterior child seeds, analyze snapshots at the same interim/final
times, cache a fit only when the observed endpoint tally is unchanged, and send
the resulting object to the existing source interim and final selection
functions. The MCMC backend remains the default and its draw settings, fit
calls, and diagnostics are unchanged.

Importance analyses report their actual integration count, convergence flag,
mode-optimizer iteration count, raw-integral MCSEs, and paired ratio MCSEs.
`max_split_rhat` is `None`: the fit has no MCMC chains and no R-hat estimate is
fabricated. The latest actual fit is retained in `last_fit`; calendar history
stores only compact fixed-size diagnostics, not proposal draws. A cached tally
can produce another timed analysis record without rerunning the posterior fit;
that record reuses the fit diagnostics that actually generated its decision.

Source cap-return behavior is retained: an importance fit that reaches its
integration cap without satisfying the stopping criterion still supplies its
posterior estimates to the source decision, while `posterior_converged=False`
and MCSE fields expose the limitation. This is not a claim that the native
optimizer, Hessian proposal, random stream, or floating-point results match.

## Resource bounds

Before consuming either child seed, the calendar computes a conservative upper
bound on unique analyses as `max_patients // 5 + 1`, multiplied by the
per-analysis integration cap and 67 component evaluations. The requested
aggregate cap defaults to 50,000,000 component evaluations and has a hard
ceiling of 100,000,000. This unit counts vector integrand components, not
individual arithmetic operations. Mode iterations are bounded separately at
2,100,000 aggregate iterations, with a per-fit limit no greater than 10,000.
At runtime, the remaining component budget determines the maximum allowed
integration count for the next fit. The importance fitter retains summary
accumulators rather than draws. Analysis history stores only two compact
67-/66-element error vectors per fit.

For the public default of 80 patients and 10,000 integrations per fit, the
preflight allows no more than 17 unique analyses, or 11,390,000 component
evaluations. The deliberately separate mode-iteration budget avoids treating
optimizer iterations as if they were integration components.

## Validation

Focused command used the repository virtualenv, `PYTHONDONTWRITEBYTECODE=1`,
single-thread OpenBLAS/OpenMP/vecLib, and warnings-as-errors. Ten tests passed
in 2.77 seconds across `test_phase12_calendar_importance.py`,
`test_phase12_calendar.py`, and `test_parallel_phase12.py`; peak process RSS was
143,867,904 bytes and `ru_nswap` was zero. The new checks exercised seeded
replay, an actual importance fit used at phase-II interim and final analyses,
pending-toxicity masking, component-work accounting, and rejection of an
over-budget request before the caller RNG state advanced. After that run, the
cache test was narrowed to a 24-patient endpoint that should reuse the interim
fit for the final analysis; an explicit repeated-tally assertion was added.
That last test-only tightening was not rerun in this checkout because the
numerical slot transferred to another lane. Targeted Ruff check/format and mypy on
`parallel_phase12_calendar.py` passed. `git diff --check` is run before commit.

The central independent checker is `tools/reference_phase12_model.R`; it
validates posterior moments for an importance mixture, not native proposal or
stopping parity. The existing elliptical-slice reference tests remain separate
and continue to exercise the default MCMC backend.
