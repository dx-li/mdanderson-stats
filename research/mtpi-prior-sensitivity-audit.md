# mTPI prior-sensitivity audit

## Source contract

The primary article text is `research/raw/TPI/paper-bioc.xml` (source provenance
is recorded in `docs/mtpi-sources.json`). METHODS offsets 14473–15088 specify
that the six penalty constants are calibrated under independent Uniform
priors, for which the prior expected penalties of de-escalation, staying and
escalation are equal. The paper notes that equal-penalty constructions can be
obtained with arbitrary Beta priors but calls their performance ongoing
research.

At METHODS offset 22168, Table 3 explicitly fixes `ε1=ε2=0.05`, varies common
Beta(a,b) dose priors, and states the mTPI penalties remain those calibrated
under the Uniform(1,1) prior. It reports (1,1), (.05,.05), (1,3), and (.1,.3).
Accordingly, the source-supported sensitivity calculation uses the usual
posterior UPM rule with the changed posterior `Beta(a+x,b+n-x)` and unchanged
interval widths/loss calibration. The safety rules at offsets 10624–11264
remain posterior overdose probabilities against the same target and cutoff,
computed under the chosen prior. The posterior mean used for final isotonic
selection likewise changes with the posterior.

This does not infer new loss constants. The alternative-prior penalty
recalibration mentioned at offset 22817 is ongoing research with no reported
formula or results in the source; it is not implemented. Original TPI's
separately calibrated `K1`/`K2` is a different procedure.

## Python interface and scope

`MTPIDesign` accepts one common prior shape pair with each shape at least
`1e-6` and total at most `1e6`, defaulting to `(1,1)`. These numerical limits
match the established TPI beta-prior support and contain all Table-3 cases.
The pair applies independently at every dose and flows through posterior
UPMs, safety, final isotonic selection, complete-outcome simulation, and
isotonic posterior intervals. The appended dataclass fields preserve previous
positional arguments and default behavior. Dose-varying priors, elicitation,
and loss recalibration are excluded.

## Validation

Focused checks compare integer-shape nonuniform posterior interval masses and
safety probabilities with independent polynomial beta-CDF identities, and a
fractional Table-3 prior with independent beta-density quadrature. They verify
nonuniform isotonic posterior means and interval draws, and compare a seeded
simulation with a repeated run under identical prior settings. Existing
uniform screenshot/decision tests and isotonic interval tests check default
compatibility. The focused posterior/simulation/selection checks and both
existing mTPI and isotonic-interval test files passed (21 tests). Targeted Ruff
check/format and mypy passed. The test process took 3.76 seconds, reached
136.16 MiB peak RSS on macOS (`ru_maxrss` 142,770,176 bytes), and reported zero
swaps. No broad suite or package build was run.
