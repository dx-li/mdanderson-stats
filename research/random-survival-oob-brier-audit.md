# Random survival forest OOB Brier/CRPS audit

## Primary-source contract

The source is the cached `R/utilities.survival.R` from randomForestSRC 3.2.2,
pinned at `b4d099e262423362a8872c13c468e6dbe2f9e9da`; its blob
`9ed62c12c118edd11d78fb85f2ec230448646406` is recorded in
[`random-survival-forest-audit.md`](random-survival-forest-audit.md). The
relevant functions are `get.brier.survival` (the `cens.model="km"` branch) and
`trapz`, approximately lines 221–404 of the cached helper. `get.brier.survival`
uses OOB survival curves when present, estimates censor survival from all
training outcomes, forms inverse-censor-weighted squared residuals per subject
and event-interest time, averages those values over nonmissing rows, then
integrates the score by trapezoids. The normalized summary divides by the
maximum time on the source grid.

For censor time `c`, the helper counts `Y_c = #{i: tau_i >= c}` and
`d_c = #{i: tau_i = c, delta_i = 0}`, then computes
`G(c) = exp(-sum_{u <= c} d_u / Y_u)`. The source calls this a KM censor model,
but its actual update is exponential Nelson–Aalen, not the product-limit KM
product. Its `sIndex(x, y)` is `sum(x <= y)`. Thus a censoring denominator is
mapped onto the event-interest grid, and an event at `tau_i` uses the projected
`G` at the greatest grid time `<= tau_i`; a censor tied at that grid point may
already be included. At evaluation time `t`, events by `t` contribute
`S_i(t)^2 / G_i`, observations with `tau_i > t` contribute
`(1-S_i(t))^2 / G(t)`, and censors by `t` contribute zero. The Brier score is
the column mean over rows with OOB survival predictions. CRPS integrates this
score on the supplied event-interest grid without inserting zero.

The Python contract narrows native behavior deliberately: only the complete
training cohort is accepted, because cached subset indexing is ambiguous; only
the `km` censoring path is implemented, not a separate censor forest. The fit's
training fingerprint and row order are checked before scoring. Rows with zero
OOB contributors stay NaN and are excluded only from score averages. The
singleton-grid policy is explicit: zero trapezoid area, with standardized
score NaN when its sole time is zero. Full native forest/RNG parity is not claimed.

## Implementation and validation

Implementation lives in
`src/mdanderson_stats/random_survival_forest_brier.py`; its result contains the
row contribution matrix, per-time score, projected censor survival, contributor
row count and CRPS summaries. Preflight bounds the result plus workspace cells
and arithmetic work. The function fails clearly for mismatched training data,
malformed OOB fit metadata or nonrepresentable contributions.

The independent fixtures cover a no-censor case with an event at time zero,
censoring tied with and between event times, an uneven grid with a missing OOB
row, and a reduced grid that omits an observed event. They retain source-grid
times, projected censor survival, every row/time contribution, mean scores,
CRPS and its maximum-time standardization. The focused Python test reconstructs
the input OOB matrices, compares all four result layers to those fixtures, and
checks full-training fingerprint enforcement, resource preflight and the
singleton/large-time integration policies. Four focused tests pass. Ruff
check/format, module-scoped mypy with silent imports, and `git diff --check`
pass. One serial focused run took 3.17 seconds, peaked at 141,328,384 bytes
RSS, and reported zero swaps; no full suite or installation was run.

Root integration repeated these comparisons alongside the affected GAO
workflow checks: 18 checks passed in 5.653 seconds, with 148.03 MiB process
peak RSS and zero swaps. Root targeted Ruff, format and mypy checks pass.
