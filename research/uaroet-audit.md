# UAROET coverage audit

Baseline main d4f8643. The previous goal turn made progress: MTADF's core and
serial simulation were integrated with independent R comparisons and isolated
wheel examples. Catalog totals were 62 implemented, 58 partial and 18 pending.
The full catalog and GitHub publication goal remains active and incomplete.

This batch targets pending entry 92. Root uses feat/uaroet-core and one Luna
implementation agent uses feat/uaroet-luna in the existing mda-efftox-core
checkout. Only one numerical job runs at a time with BLAS/OpenMP threads set
to one. A starting memory-pressure reading reported 53% system memory free.

The official desktop page confirms UAROET 1.8 and Thall/Nguyen (2012). The
institutional published paper and author preprint were read for Sections 3–5.
The preprint resolves corrupt Greek characters in the published PDF extraction.
The PMC reader returned a browser challenge; the accessible institutional paper
was used instead. Source links are recorded in docs/uaroet-sources.json.

The saturated logistic continuation marginals and Gaussian copula are distinct
from the existing U2OET fitted FGM model. Reuse normal-rectangle quadrature and
truncated-normal sampling primitives where appropriate, not U2OET's likelihood
or allocation policy. Efficacy-by-toxicity axes follow repository conventions;
the paper labels toxicity first. Monotone endpoint logits are a baseline plus
nonnegative dose increments, each having an underlying truncated-normal prior.

Equation 6 uses a strict toxicity-tail event and a strict exclusion probability.
Equations 7–9 define near-optimality and probability-best globally, before
intersecting with safety. All exact per-draw utility ties satisfy the equality
event for probability-best. Equation 10 uses predictive probabilities of a
good-outcome set, not posterior mean utilities, as randomization weights.
There is no separate marginal efficacy-futility criterion in this paper.

Section 4.4's final global argmax notation is ambiguous if the highest-utility
dose is unsafe. Expose both the highest utility within the acceptable set and
the literal global maximum (gated by a nonempty acceptable set), with the
lowest dose resolving an exact tie; this distinction must remain visible.
Initial assignment is physician-specified, and the
no-skipping restriction applies to subsequent escalation, not final ranking.

The paper describes eliciting prior means by many large pseudo-trials and
choosing prior variances through ESS and operating-characteristic calibration.
This first posterior workflow takes explicit priors; it must not fabricate the
unpublished calibrated mean parameters from Table 2 marginal probabilities.

An independent base-R reference integrates copula cells over uniform efficacy
quantiles and checks the published Table 2 example at correlation .1. A reduced
one-dose binary model with fixed zero correlation factorizes into two
logistic-normal posteriors, enabling direct quadrature of posterior moments.

While implementation checks were running, entry 140 (ComPAS) was triaged
again. PubMed 30419609 and the current trialdesign.org catalog confirm the
Tang/Shen/Yuan 2019 article, DOI 10.1002/sim.8026. PubMed has no PMC link,
the trial-design application link resolves to an empty shell in the reader,
and the Wiley article is inaccessible through that reader. This verifies the
identity but does not provide enough equations to implement its adaptive
shrinkage model. It remains pending; do not substitute BPCC or guess the model.

## Integrated checkpoint

Luna commit `3eea72f` was integrated as `88558e8`. The three public modules
provide probability evaluation, explicit-prior posterior fitting and allocation.
Root added public exports, the coverage record, documentation and independent
R fixtures. Entry 92 moves from pending to partial: 138 entries now comprise
62 implemented, 59 partial and 17 pending. This is progress, not completion of
the full catalog goal.

The posterior uses Gaussian elliptical slices with nonnegative-increment
indicators and an independent-uniform Metropolis proposal for correlation.
No-data analyses draw from the prior directly. Counts in zero-observation cells
are excluded from the log-likelihood sum; independence uses analytic log
probabilities even when probability-scale values underflow. Correlated observed
cells that cannot be represented and integration failures raise explicit errors.
Separate likelihood and rectangle counts sum to the work budget. Retained cells
are bounded before allocation, and minimum workload checks precede RNG use.

Validation was deliberately small and serial with BLAS/OpenMP threads set to
one:

- The independent base-R generator ran without warnings. Its six scenarios
  cover the published probability example, binary and ordinal outcomes,
  positive/negative association, independence and both limiting copulas.
- All five focused tests passed in 1.91 seconds with warnings treated as errors;
  measured process elapsed time was 2.00 seconds, peak RSS 135.55 MiB and zero
  process swaps. Two tests compare independent copula/posterior integrations;
  three cover model structure and meaningful allocation behavior.
- The reduced posterior reference uses direct integration at fixed zero
  correlation, with 1,200 retained draws in each of two chains after 400 warmup
  iterations. Means agree within the recorded Monte Carlo uncertainty; split
  R-hat is below 1.05 in this check. This does not validate convergence for all
  possible user data or priors.
- Luna's small inferred-correlation check returned finite draws and acceptance
  fractions 0.375 and 0.5, using 59 likelihood calls and 472 rectangle calls.
- Ruff and formatting checks passed for six affected Python files. Mypy passed
  for the three new source modules using `--follow-imports=silent`.
- The source distribution and wheel built with the cached Hatchling runtime.
  Isolated-wheel imports confirmed all eight exports and exact module/catalog
  bytes. Both documented examples ran. Additional compact checks verified
  strict toxicity boundaries, all-tied-best events, final removal of no-skipping,
  extreme independent log probabilities, and resource rejection before RNG
  consumption. This process took 1.98 seconds, peaked at 114.42 MiB and had zero
  process swaps.

No full-suite run, large simulation or dependency installation was performed.
Automatic prior elicitation/ESS calibration, complete trial simulation, native
prior/settings files and executable/report equivalence remain open. The earlier
GitHub write restriction remains unresolved; this checkpoint is local and no
alternative publication transport was attempted.
