# MTADF author-local simulation audit

The recovered source is the author `llogistic()` procedure in
`targetAgentDF.r`, lines 184–309. The existing source crosswalk
`research/mtadf-author-logistic-source-audit.md` records the local response
model: independent Cauchy priors with scales 10 and 2.5, logistic likelihood,
and globally standardized ordinal dose coordinates
`(x-mean(x))/(2*sd(x))`.

## Source-to-code map

- Lines 191–197 solve the fixed beta prior equation, pool posterior overdose
  probabilities with increasing PAVA, and enforce a one-dose admissibility
  floor.
- Lines 199–220 define the local likelihood, coefficient priors, and slope
  probabilities. The local regression windows are adjacent dose pairs on the
  global standardized ordinal scale.
- Lines 231–246 place the first cohort at the lowest dose, provisionally set
  dose index one next, and force the lowest dose without a posterior fit when
  the cap is one.
- Lines 248–287 apply the four author slope branches. A decision uses the
  cap carried from before the current cohort; the cap is recomputed after
  movement. This local simulator's cap-one preassignment override is preserved
  explicitly rather than generalized into ordinary one-step movement.
- Lines 290–296 perform final selection from the fresh cap and all-dose
  epsilon-adjusted efficacy rates, choosing the rightmost fitted maximum.

The author thresholds are `thetaf=ce1`, `thetab1=1-ce2`, and
`thetab2=1-ce1`. At dose one only the forward pair is fit; at the top only the
backward pair is fit. An interior dose with an untried next level uses only
the backward pair. If that next level has observations, both adjacent pairs
are fit and the source escalation/de-escalation conjunctions determine the
move. A cap of one bypasses all of these fits for that cohort.

The replay accepts bounded cohort-by-dose potential binomial count tables;
only assigned-dose counts are used. Simulation draws toxicity and efficacy
tables separately with NumPy and stores independent outcome/sampler seed
pairs. Trials execute serially, posterior arrays are discarded between
reviews, and aggregate output reuses the package's compact local-logistic
simulation result. A fit-free trial's MCMC diagnostic entries are NaN to mark
that they were not estimated. Monte Carlo selection errors use all requested
trials. The examples/tests exercise conduct, replay and a small actual MCMC
path; they do not establish native `metrop` random-stream or output parity,
or replace full operating-characteristic validation.

The source's native MCMC run length is not reproduced as a NumPy stream. The
Python API uses explicit draw, warmup, chain and seed settings, while reusing
the existing stable likelihood, sampler and chain summaries. The separate
paper-policy local implementation and the isotonic author simulator remain
unchanged.
