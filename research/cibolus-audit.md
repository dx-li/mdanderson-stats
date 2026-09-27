# CiBolus coverage audit

Baseline main `d3bc42a`. The previous goal turn made progress: Dose Schedule
Finder's core workflow, independent references and packaged examples were
integrated. Coverage is 62 implemented, 60 partial and 16 pending entries.
The full catalog/publication objective is not complete.

Root uses `feat/cibolus-core`; one Luna agent uses `feat/cibolus-luna` in the
existing independent checkout. Initial system memory reported 49% free.
Numerical jobs run serially with one BLAS/OpenMP thread. No new dependencies,
full-suite run or large simulation is planned for this port.

Entry 86 is CiBolus, with the catalog recording `CIBOLUS_V1.1.zip`. The live
detail page again returned a web-reader error, and PMC returned a browser
challenge. The institutional author-hosted article was accessible; its model,
likelihood, utility, decision, prior and computation sections were read. PDF
screenshots failed. No native archive was retrieved or executed.

Implementation follows the explicit failure indicator in the toxicity model.
The printed likelihood's final term uses time one as shorthand, but the
model and accompanying explanation distinguish failure from response exactly
at one. This resolution needs to be disclosed; it is not executable parity.
The safety quantity is conditional toxicity for response at one, which also
differs from both failure toxicity and marginal toxicity.

Engineering checks focus on cancellation in the response-hazard antiderivative,
underflowed transformed concentrations, nearly constant hazards, observed
intervals and log likelihoods. The published lognormal prior is broad; testing
only moderate parameter values would be insufficient. A numerical failure
must not be silently converted into a posterior rejection. Analytically
justified limiting probabilities need to be distinguished from such failures.

The independent base-R generator ran successfully in 0.31 seconds with warnings
treated as errors. It integrates continuous hazards and interval densities
directly, independently of the Python analytic antiderivative. Fixtures include
ordinary parameters, no bolus, full bolus, a nearly constant transformed bolus,
underflowed concentration powers with a finite limiting hazard, published prior
medians, six mixed observation records and a one-dimensional posterior with
only log baseline toxicity varying.

The first Luna core checkpoint `84d8030` was integrated as `903f2e4`. Its
initial two focused tests and two independent reference tests passed. A small
published-prior probe then found an uninitialized local variable in the
large-increment calculation. The corrected model also shares its direct
interval log-mass calculation between prediction and likelihood, preserving
small positive interval masses on the log scale.

Luna's subsequent 32-draw published-prior probe returned normalized joint
predictions. Root added a steep-response quadrature case, regenerated all R
fixtures in 0.32 seconds, and ran both independent core reference tests against
the corrected child checkout: two passed in 1.22 seconds with warnings treated
as errors. References now cover 28 point evaluations, five category-distribution
cases and six mixed observation records. The reduced posterior's directly
integrated baseline-toxicity mean is .202164169283463; its conditional-toxicity
mean at concentration .2 and bolus .1 is .222807878172775. Posterior sampler
comparison subsequently passed against the completed Luna fitter.

The stability follow-up `cac32eb` was integrated as `b9b01f8`. All three
independent reference tests passed against the child source in 4.50 seconds
(4.595 seconds measured process time, 133.39 MiB peak RSS, zero swaps).
The posterior comparison retained 1,200 draws from each of two chains after
400 warmup iterations, with only log baseline toxicity varying. Log baseline
toxicity, baseline toxicity, conditional toxicity and its exceedance probability
all agreed with direct integration within six batch-means Monte Carlo errors
plus .003; each checked quantity had split R-hat below 1.05. This verifies a
reduced posterior, not convergence under every dataset or the broad published
prior.

The fitter/decision checkpoint `0159e54` was integrated as `0f38cdf`.
It includes preflight allocation/work budgets, serial elliptical slice sampling,
independent prior draws without observations, diagnostics and immutable results.
Prediction reuses validated toxicity parameters. Indeterminate predictors raise
instead of being mapped to certain response; analytically saturated positive
exposures retain their limiting behavior.

Final validation used one numerical process and one BLAS/OpenMP thread:

- All seven focused tests passed with warnings treated as errors in 4.82
  seconds; process elapsed time was 4.931 seconds, peak RSS 134.91 MiB and
  process swaps zero. Decision checks distinguish strict event/cutoff equality,
  actual three-concentration no-skip restrictions and unrestricted final selection.
- Ruff and formatting checks passed on seven affected Python files. Mypy
  passed for the three new source modules with `--follow-imports=silent`.
- Wheel and source distribution builds succeeded using cached Hatchling.
  An isolated wheel import verified all 14 public exports and exact source/catalog
  bytes. Both documentation examples ran; the posterior example used 742
  likelihood evaluations and 9,112 patient/prediction work units.
- Compact packaged checks verified resource rejection before random-number
  consumption, an analytically constant-hazard interval whose ordinary CDF
  subtraction loses its mass, and rejection of an indeterminate extreme
  predictor. This process took 2.711 seconds, peaked at 114.33 MiB and reported
  zero swaps. No native executables, archives or article PDFs are in the wheel.

Entry 86 advances from pending to partial. The catalog now has 62 implemented,
61 partial and 15 pending entries, totaling 138. Prior elicitation/calibration,
complete trial simulation, native input/report workflows and executable parity
remain open. No full-suite run, large simulation, dependency installation or
CI change was performed. The temporary packaged verification script remains
under ignored `research/raw`; it is not a distribution artifact.

This goal turn made progress. Pinnacle is the next uncovered method; primary
method/manual and Rice wavelet source leads are recorded in its scouting note.
The earlier GitHub tool restriction remains unresolved: publication requires
approval, while the session forbids approval. This checkpoint is local; no
alternative publication transport was attempted. The full catalog/publication
goal remains active.
