# MTADF logistic complete-cohort simulation audit

## Source and simulation contract

The design methods follow Zang, Lee and Yuan, “Adaptive Designs for
Identifying Optimal Biological Dose for Molecularly Targeted Agents,”
*Clinical Trials* 11(3), 2014, publisher-hosted manuscript at
<https://pmc.ncbi.nlm.nih.gov/articles/PMC4239216/>. The global and local
posterior models, Cauchy priors, beta-binomial toxicity rule, local first-`l`
cohort ramp and local final isotonic rule are implemented in
`mtadf_logistic.py`. This simulation module executes those Python posterior
and decision APIs; it does not reproduce a native random stream or hidden
application conduct settings.

`MTADFLogisticSimulationConfig` requires separate dose-specific binary
toxicity and response probabilities and names its outcome model
`independent_marginals`. Cohort outcomes are drawn independently from those
marginals, matching the existing MTADF simulator convention. Thus reported
operating characteristics are conditional on patient-level independence;
they do not represent scenarios with toxicity/response association. The
global design starts at its lowest admissible dose. The local design starts
at dose zero and assigns one complete cohort to each of the first `l` dose
levels; a simulation with fewer than `l` planned cohorts is rejected. Both
designs stop when the safety rule leaves no admissible dose. Global final
selection uses the fitted posterior mean efficacy; local final selection uses
the paper's isotonic method. These are complete-outcome cohort paths without
calendar timing.

Outcome and posterior-sampler seeds are spawned independently for every
trial and retained as uint64 pairs. `simulate_mtadf_logistic_trial` replays one
pair exactly. Aggregate execution is serial, stores only per-trial dose counts,
selection, stop reason and compact diagnostics, and discards posterior draws
after each look and trial. The preflight bounds worst-case cohort fitting work
(including final global fits and distinct local bounce fits after the initial
ramp), aggregate arrays and their freeze-time copies, and peak live posterior
draw storage before seed generation. Identical local windows are counted as
one fit. Aggregate storage limits therefore cover output plus a conservative
posterior peak, not output arrays alone.

The global method has no calibration claim beyond the documented Python
conduct choices. The simulation assumes the supplied probabilities are known
truths and does not estimate them. MCMC convergence diagnostics are summarized
per trial: mean chain acceptance, maximum split R-hat, and for local fits the
mean positive-slope batch-means MCSE and maximum positive-slope R-hat. These
diagnostics are descriptive and are not convergence guarantees. Monte Carlo
standard errors for selection and early stopping use the trial-level Bernoulli
formula; no selection and zero-enrollment trials remain in the denominator.

## Validation

Focused tests exercise global and local seed-pair replay, cohort count
conservation, marginal binary bounds, immutable compact results, early safety
stop before enrollment, local ramp preflight, aggregate work rejection,
one-dose global execution, and rejection of unsupported outcome models. The
seven new simulation tests passed. Across the four affected MTADF test files,
25 tests passed in 5.21 seconds with warnings treated as errors. Peak RSS was
134,283,264 bytes (131,136 KiB), with zero process swaps. Ruff check/format
and targeted mypy passed.
