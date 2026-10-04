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
training cohort is accepted, because cached subset indexing is ambiguous. The
default `km` path remains unchanged. An explicit `rfsrc` path fits the
source-configured censoring forest; it is a Python implementation of the
specified fit and projection contract, not a claim of native tree/RNG parity.
The fit's training fingerprint and row order are checked before scoring. Rows
with zero OOB contributors stay NaN. The singleton-grid policy is explicit:
zero trapezoid area, with standardized score NaN when its sole time is zero.

## Separate censoring-forest contract

For `cens.model="rfsrc"`, cached `get.brier.survival` fits `Surv(time,cens)~.`
to a data frame containing the original grow `time`, `cens = 1*(event==0)`,
and every original grow predictor. The call fixes `ntree=50`, `nsplit=1`,
`splitrule="random"`, `nodesize=set.nodesize(n,p)`, and `perf.type="none"`;
other arguments, including `na.action`, retain `rfsrc` defaults. `set.nodesize`
returns 2 when `n<=300,p>n`, 5 when `n<=300,p<=n`, 10 when `300<n<=2000`,
and `n/200` otherwise (truncated by the native integer interface). Predictions
are made for each requested row's original predictor vector. Source
`sIndex(x,y)=sum(x<=y)` projects each row's censor-forest curve to the outcome
grid, selecting the last censor-model time no greater than the outcome time
and supplying 1 before its first time. The resulting source matrix is
time-by-row; Python exposes the same values row-by-time for consistency with
its OOB prediction layout.

The original helper computes both IPCW terms literally for every row/time.
Consequently an inactive `0/0` term can turn a contribution into NaN, and its
`colMeans(..., na.rm=TRUE)` then omits that row only at that time. Python reports
both total OOB contributor count and per-time defined score count, preserves
these NaNs, and does not clip zero censor survival. Infinite positive
contributions fail explicitly; literal `0 * Inf` NaNs remain undefined and
are omitted as the source does. For no censor observations, the unchanged helper
skips model fitting but returns a vector that later receives matrix indexing;
Python uses the direct all-one censor-survival matrix, the natural `G(t)=1`
extension.

The source fit defaults to `na.action="na.omit"`, while the helper overlays
stored imputed predictors for prediction when available. Python requires
finite complete training data and does not attempt to reproduce that mixed
missingness behavior. Validation uses fixed-curve R references for the
contribution/projection/score layer and separately checks the 50-tree Python
fit and row prediction metadata; it does not compare random tree structures
or random streams across languages.

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
