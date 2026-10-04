# Full-training OOB survival Brier reference

`tools/reference_random_survival_oob_brier.R` builds four tiny survival-forest
objects and calls the unchanged `get.brier.survival` helper from the pinned
randomForestSRC 3.2.2 source. It independently evaluates the same fixed-grid
IPCW equations and stops if the source's censor-survival projection,
row-by-time contributions, Brier curve, CRPS, or standardized CRPS differ.
The helper source is not copied into this repository; supply its local cached
path as the script's first argument. No R package installation is needed.

The independent calculation uses the source's censoring estimate
`G(t) = exp(-sum(d_c / Y_c : c <= t))`, where `Y_c` counts all observations
with observed time at least `c`, and `d_c` counts censors at `c`. It projects
that curve onto the forest's supplied event-time grid, which may omit observed
event times. For an observed event by grid time `t`, the contribution is
`S_i(t)^2 / G_grid(tau_i)`, where `G_grid(tau_i)` is the projected value at the
greatest grid time no later than `tau_i` (or one if there is no such grid time);
for a subject still at risk after `t`, it is
`(1-S_i(t))^2 / G(t)`; a censor by `t` contributes zero. Missing OOB curves
remain missing and are excluded only from that time point's average. CRPS uses
the ordinary trapezoid over the supplied grid as-is; no time-zero point is
inserted.

The four cases cover no censoring and an event at time zero; a censor before,
between, and tied with event times; an uneven event grid with missing OOB rows;
and a reduced grid that omits an observed event while retaining an intervening
censor. `tests/fixtures/random-survival-oob-brier-input.csv` stores every input
row/grid/prediction combination. The companion contribution and curve files
store the per-row terms and summaries. The generated fixtures are deterministic
and do not depend on an R random-number stream.

The final generator was run with base Rscript against the cached source. All
four event-grid, censor-survival, contribution-matrix, score, CRPS and
standardized-CRPS comparisons passed. The command completed in about 0.4
seconds; peak memory was not measured.

Source provenance is randomForestSRC 3.2.2 at Git revision
`b4d099e262423362a8872c13c468e6dbe2f9e9da`; `R/utilities.survival.R` has blob
`9ed62c12c118edd11d78fb85f2ec230448646406`. The implementation is intentionally
validated against that helper's actual tied-time indexing: `sIndex(x, y)`
counts `x <= y`, so a censor tied with an event can affect the event's projected
denominator. This records the source convention rather than substituting a
textbook `G(t-)` convention.
