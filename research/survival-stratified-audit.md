# SurvivalContour stratified Cox audit

The preceding goal turn made progress: ordinary Cox fitting/prediction surfaces,
source comparisons, plotting and packaging were integrated at `33600b9`.
Catalog totals at this checkpoint are 62 implemented, 66 partial and 10 pending.
The overall catalog/publication goal remains active; entry 166 is partial.

Root uses `feat/survival-stratified-cox` and the existing sole Luna worker uses
`feat/survival-stratified-luna` in its separate checkout. Numerical jobs are
serial, with one BLAS/OpenMP thread. No installation or full-suite run is
planned. The GitHub publication approval blocker is unchanged; read-only source
retrieval and local implementation can proceed.

## Model contract

Stratified Cox regression shares one coefficient vector across strata, sums
partial likelihoods with risk sets restricted to each stratum, and estimates
a separate baseline hazard per stratum. It does not fit independent coefficient
vectors for each group. The global coefficient-identification and monotone-
likelihood checks must use the joint model. In particular, individual strata
can each have monotone likelihood in opposite directions while the shared
model has a finite maximum.

The workflow uses a common global continuous-covariate grid and mean/explicit
adjustment profile. Each returned contour has its own baseline, time grid and
pointwise intervals, while sharing the joint fit. Default times contain each
stratum's observed times and zero if absent. An event at zero gives its
post-event survival at zero. A stratum with no events contributes zero partial
likelihood and has fitted survival and confidence bounds one; other strata
must still identify the coefficients. Optional common prediction times use
right-continuous steps. Allocation preflight must cover all returned strata.

## Pinned source and reproduced helper defects

Read the author repository at
[`d4645f69f23fc1146c07432f576b4c40f85e1bba`](https://github.com/YushuShi/survivalContour/tree/d4645f69f23fc1146c07432f576b4c40f85e1bba).
The source-only files were saved under ignored `research/raw/survivalContour/R`
and their exact Git blob hashes verified:

| File | Git blob |
| --- | --- |
| coxStrataContour.R | `46cbd57ff949c18a934c3a94461727b191533ff7` |
| coxStrata3DContour.R | `e048504289e912b34735ed385e65edadf8d7ce03` |
| survivalContour.R | `a713dbde56c9c9b6dd351fd7f6fa90a1ed0652cc` |

With the conventional formula `Surv(time,status) ~ x1+x2+strata(group)`,
R `coxph` stores an x-level name `strata(group)`. The author helpers use that
name as a new-data column and leave the actual `group` column at its modal
value. Executing the unchanged helpers against distinct groups reproduced
identical first-group surfaces for every requested group. The 3D helper also
retains a repeated time vector: the five-profile reference has 46 time values
but only 10 surface columns. Both defects are recorded in the native fixture.
The unconditional time-zero/survival-one prefix is also inappropriate for an
event observed at zero. Python follows the correctly specified stratified model
and direct `survfit` predictions instead of duplicating these helper defects.

Only rendering sinks and a colorbar-styling pipe were replaced when capturing
the unchanged source. An initial probe's naive pipe implementation evaluated
the styling call without its argument and stopped; the final capture ignores
that purely visual pipe. Neither R model code nor helper source was changed.

## Independent reference fixtures

`tools/reference_survival_stratified.R` requires installed `survival` 3.6-4 and
checks the author-helper MD5 hashes. Its 28 records cover two informative strata
and a third without events, with tied events/censors and an event at time zero.
Two tie methods cross two adjustment profiles. For every group, both default
and custom common times are predicted at five grid points and five covariate
quantiles. One-profile direct `survfit` calls avoid ambiguity in the native
helper's flattened multiple-profile output.

The generator ran with warnings as errors in 1.36 seconds. Five CSV fixtures
preserve the input, shared fits/covariances/likelihoods, 780 surface rows,
780 quantile rows and the reproduced native defects. Expected Efron coefficients
are `[-0.105603215946071, 0.630305239381879]`; Breslow coefficients are
`[-0.119793462266714, 0.624593511301248]`. Prediction errors are converted from
`summary.survfit`'s SE(S) back to SE(log S) by division by survival, which is
positive for this fixture.

## Integrated implementation and verification

Luna commit `efa3889` was integrated as `bbfd4dc`. `survan_cox` now accepts
optional string/integer strata and retains their first-seen labels on the fit.
`survival_stratified_cox_contour` returns `SurvivalStratifiedCoxContour`, with a
shared fit and separate `SurvivalCoxContour` results accessible by label. Both
existing plotters work on individual groups. The ordinary contour builder was
factored into a private helper, without exposing internal fit overrides in the
public interface.

The likelihood aggregates within-stratum contributions and centers each part
to reduce cancellation from group-specific offsets. One sparse LP uses shared
coefficient columns and separate risk-set variables. Array blocks are retained
instead of expanding sparse entries into Python scalar lists. Information
matrices are summed sequentially. Up to 100 groups are allowed within the
existing row/design limits; the two-million-cell output budget sums actual
per-group timeline sizes and covers all groups before fitting.

The worker's nine focused ordinary/stratified Cox tests, targeted Ruff and mypy
checks passed. Root independently ran all direct-R comparisons with warnings
as errors: maximum errors were `7.1052e-11` for survival, `6.2879e-11` for
confidence limits, `3.4305e-10` for cumulative hazards and `3.7413e-10` for
SE(log survival). Fits and covariance, all 780 surface rows and all 780 quantile
rows agreed at their specified tolerances. The same check verified unit scaling
by `1e100`/`1e-100`, reversed row/stratum order, shared fit identity, no-event
groups, and post-event zero-time values. It took 0.178 seconds after imports,
peaked at 118.1875 MiB and reported zero swaps.

Additional root checks verified aggregate preflight before a patched fitter
could run, acceptance of 100 small strata, and preservation of distinct labels
`1` and `"1"`. A scalar string was incorrectly accepted as a character sequence;
root reproduced it, added a focused regression and rejected scalar strings and
bytes at the input boundary. The three affected stratified tests then passed
in 1.34 seconds. No broad suite was repeated.

The reference generator was tightened to assert equality of the third native
3D group as well, then rerun successfully; fixture values were unchanged.
Two group contours and the third group's flat 3D surface were rendered and
visually inspected. Export margins were adjusted to retain the 3D axis labels.
The final rendering peaked at 157.25 MiB with zero swaps. No dependency or CI
configuration was added, and no worker remains active.
