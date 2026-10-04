# Random survival forest source and reference audit

SurvivalContour entry 166 names randomForestSRC 3.2.2 for its forest models.
The CRAN Git mirror's `3.2.2` tag resolves to
[`b4d099e262423362a8872c13c468e6dbe2f9e9da`](https://github.com/cran/randomForestSRC/tree/b4d099e262423362a8872c13c468e6dbe2f9e9da).
DESCRIPTION identifies version 3.2.2, dated 2023-05-23, by Hemant Ishwaran and
Udaya Kogalur, with GPL >=3 licensing. The tag and file blobs were retrieved
through the read-only GitHub API; the ignored source directory is not itself
a Git checkout. Provenance uses the verified tag and each file's Git blob hash,
not the containing project's current Git revision.

## Exact source bundle

| File | Git blob |
| --- | --- |
| DESCRIPTION | `d00a1128d6e2336006dd91fb201a5e231da14985` |
| man/rfsrc.Rd | `6f441a8d027d20764ce75d26a3aefcc54188ee82` |
| R/rfsrc.R | `5be0607d365a5422e4b1d58e03c3e0dfe92c8045` |
| R/utilities.survival.R | `9ed62c12c118edd11d78fb85f2ec230448646406` |
| R/utilities.R | `25b40cd53edc9489f53e7799383d5db74494f6ba` |
| R/utilities.factor.R | `1c6d8d398d5a0ee240134392d61dbe6dd19738d2` |
| src/splitCustom.c | `337a084d630f4871a61efd37caaa67ae1cfb2e6b` |
| src/randomForestSRC.c | `e9c6e896c4f93c6eb1b85bdf37992964d74376ea` |
| src/randomForestSRC.h | `6355ed999d06447747f78b01ee2a370e6bc43d1e` |

The original author contour helper is `R/rfsrcContour.R`, blob
`03b81a98d11b0d2f4daef1f70fdf5e3431f805bc`, at SurvivalContour revision
`d4645f69f23fc1146c07432f576b4c40f85e1bba`. It accepts an already fitted model,
uses 30 continuous-covariate points between the empirical 2.5th and 97.5th
percentiles, adjusts numeric covariates at means unless supplied, and prepends
time zero with survival one only when the forest event grid starts after zero.

## Native algorithm contract

- Default 500 trees; survival `mtry=ceil(sqrt(p))`, `nodesize=15`, `nsplit=10`.
- Default root samples are without replacement, with size `round(.632*n)`.
  The default with-replacement size is `n`. R uses nearest-even rounding.
- A parent must have at least `2*nodesize` sampled rows to split. The inspected
  log-rank path requires only nonempty children, so `nodesize` is not a strict
  minimum child size. All-censored nodes stop. All-event nodes with identical
  times stop; mixed tied event/censor nodes can still split.
- Continuous cutpoints are distinct observed values except the maximum,
  routed by `x <= cut`. They are not midpoints. `nsplit=0` considers all;
  otherwise up to `nsplit` are sampled without replacement and sorted.
- The split score is absolute log-rank numerator divided by its hypergeometric
  standard deviation. For event times j, numerator is
  `sum(d_left - Y_left*d/Y)` and variance is
  `sum((Y_left/Y)*(1-Y_left/Y)*(Y-d)/(Y-1)*d)` over `Y>=2`.
  A statistic improvement must exceed `1e-9`; ties retain the earlier candidate.
  A finite zero score is eligible, rather than necessarily stopping the tree.
- Leaf risk sets include observations censored at an event time. Leaf survival
  is the Kaplan–Meier product of `1-d/Y`; leaf cumulative hazard is the
  Nelson–Aalen sum of `d/Y`. Forest outputs average those two curves separately.
  In general `mean(KM)` differs from `exp(-mean(Nelson–Aalen))`.
- Curves are right-continuous. The native default common grid selects up to
  150 event times with rounded evenly spaced **one-based** indices; zero means
  all event times. The grid controls output, not the time resolution used to
  calculate within-leaf event/risk counts.

Relevant main-C functions include `logRankNCR`,
`getPreSplitResultGeneric`, `selectRandomCovariatesGeneric`,
`stackAndConstructSplitVectorGenericPhase1/Phase2`,
`updateMaximumSplitGeneric`, `getAtRiskAndEventCount`, `getLocalRatio`,
`getLocalSurvival`, `getLocalNelsonAalen`, `mapLocalToTimeInterest`, and
`updateEnsembleSurvival`. Final ensemble normalization divides the accumulated
survival and cumulative-hazard sums by the contributing tree count.

## Executed native references

`tools/reference_random_survival_forest.py` verifies source hashes and extracts
unchanged bodies for the custom log-rank example and four main-C leaf/step
kernels. A small standalone C wrapper supplies one-based arrays, allocation
helpers and event/risk counts. It compiles only those functions with the
existing compiler. The custom example and main log-rank implementation use the
same numerator and variance equations. Native R `get.grow.event.info` is sourced
unchanged to check common-grid selection with `ntime=0,1,5,150`.

The fixture contains 43 native split scores and 91 survival/cumulative-hazard
pairs over seven datasets: ties, all events, all censoring, one event, identical
times, zero-time events, and repeated bootstrap-style rows. Four additional
grid references use 401 event times. An independent one-feature, full-sample,
exhaustive-split tree driver uses those unchanged C scores and leaf kernels
to produce six deterministic whole-tree reference cases: tied or duplicate
rows with `nodesize=1,2,15`. It records splits and predictions at seven profiles
and 13 times. This driver does not call the Python package under test and is
not the full native tree engine. Hand calculations independently checked a
tied Kaplan–Meier step, a no-event leaf and a one-event log-rank score.

The initial C-only run took 0.57 seconds,
with a 41.1 MiB child RSS peak. The complete C/R run took 1.22 seconds, with
29.3 MiB parent and 82.6 MiB child peak RSS; zero swaps were reported. Jobs ran
sequentially with one BLAS/OpenMP thread and no dependency installation.
The expanded reference run, including all six tree cases, took 0.95 seconds,
with 28.4 MiB parent and 83.3 MiB child peak RSS and zero reported swaps.

These are executable split/leaf/time-grid references. They do not execute the
full native tree-growing, bootstrap, out-of-bag or forest prediction engine.
Original code and compiled objects remain ignored under `research/raw/` and
are not redistributed.

## Python implementation and integration

The public `fit_random_survival_forest` and `predict_random_survival_forest`
interfaces implement the numeric log-rank path with sequential seeded sampling,
packed binary trees, sparse leaf event steps, and explicit limits on sampled
rows, split work, stored events, nodes, prediction work and combined output
allocations. The output survival average is accumulated in the log domain;
Nelson–Aalen hazards are averaged separately. Integer node indices and all
returned arrays are read-only. Python's random stream differs from RF-SRC's.

Four focused tests compare all recorded native split/leaf/grid values and the
six independent deterministic tree references, and check seeded repeatability,
replacement sample fractions and zero-time failures. The comparisons use
relative and absolute tolerances of `2e-14`. The final forest plus existing
shared-contour suite passed seven tests in 1.37 seconds. Ruff checks, formatting
of the new modules/tests, and mypy on the forest, contour and plotting modules
also passed. The full repository suite and CI were not run or expanded.

The public guide's two examples ran with warnings treated as errors. Its
32-tree, 160-observation fit stored 450 nodes and 2,544 leaf event records,
with 216,680 split-work units. Predictions stayed unchanged within `1e-12`
under covariate-unit changes of `1e-100` and `1e100`, and a time-unit change of
`1e100`. Four contour cases checked mean/explicit profiles and default/custom
times, including percentile curves and direct-prediction agreement. A
zero-time failure preserved its actual survival jump. Both plot views were
rendered and inspected; requesting unavailable confidence limits raises a
clear error. The final integration run took 0.74 seconds, peaked at 153.9 MiB
RSS and reported zero process swaps, using one BLAS/OpenMP thread.

This adds the ordinary numeric forest family to SurvivalContour entry 166.
At that checkpoint the entry remained partial: interval-censored and neural
families, native simulation-based intervals and full application workflows
were still open. The initial forest interface did not implement missing-value imputation,
competing-risk forests or alternative split rules. Subsequent additions cover
categorical predictors, additional split rules, and single-pass missing-data
fitting; see `random-survival-missing-reference-audit.md` for the latter's
source contract and remaining limits. Categorical splitting was subsequently
added with explicit caller-declared columns and retained level maps; categorical
VIMP validates the raw training fingerprint and encodes profiles before tree
routing. Subsequent OOB and permutation-importance additions are documented
below.

## Source contract: OOB diagnostics and permutation importance

The pinned native source supplies a concrete next statistical workflow. In
`research/raw/randomForestSRC/src/randomForestSRC.c`, `updateEnsembleSurvival`
(around line 32285) includes a training row only for trees where it is out of
bag, keeps a per-row contributor count, and averages leaf Kaplan–Meier and
Nelson–Aalen curves separately. The R wrapper exposes both OOB matrices on
`time.interest` (`R/rfsrc.R`, around lines 1185–1201). These outputs require
retaining sampling membership. A row without contributors has no OOB estimate;
an in-bag fallback
would change the method. NaN with an explicit zero count is a suitable Python
representation.

`getMortality` (around line 32117) sums cumulative hazard over the native
interest-time grid. It is not a time integral or the last cumulative-hazard
value. `getConcordanceIndex` (around line 32495) returns one minus concordance,
excluding pairs with zero contributor counts. Earlier observed failures are
comparable to later observations; tied event/censor pairs put the failure
first. For two tied failures, tied mortality gets full concordance and unequal
mortality gets half. Ordinary comparable pairs with tied mortality get half.
The source uses `EPSILON` for time and mortality ties. A read-only retrieval
of [`src/randomForestSRC.h` at the pinned revision](https://github.com/cran/randomForestSRC/blob/b4d099e262423362a8872c13c468e6dbe2f9e9da/src/randomForestSRC.h)
confirms `EPSILON = 1.0e-9`; the header is now cached alongside the C source.
This absolute native tie tolerance must be documented because changing time
units can change which near-tied observations count as tied.

The manual (`man/rfsrc.Rd`, VIMP section around line 564) distinguishes
`importance="permute"` from the default `"anti"`. For permutation importance,
`getPermuteMembership` (around line 4290) permutes among each tree's OOB rows,
reroutes those cases, and averages the perturbed OOB mortality across each
tree block. `finalizeVimpPerformance` (around line 3805) averages the blockwise
increase in concordance error over the unperturbed OOB error. A block of all
trees compares the whole OOB forest; smaller blocks are not equivalent to a
single whole-forest perturbation. Do not label permutation VIMP as native
default importance.

## OOB implementation and independent native comparison

The optional `compute_oob=True` fit now retains bit-packed sampling membership
and returns OOB survival/hazard curves, contributor counts, mortality and
concordance error. Disabled OOB computation retains the previous fitting and
prediction path. Rows without OOB contributors remain undefined; there is no
in-bag fallback. Pair comparisons use row-sized vectors rather than an n-by-n
matrix. Separate combined-array and work bounds cover OOB computation.

`tools/reference_random_survival_oob.py` extracts the unchanged native
`getConcordanceIndex` into the existing small C harness, checks pinned source
blob hashes and produces 16 reference cases. Cases include censor/event and
event/event ties, exact versus just-above `1e-9` boundaries, absent OOB
contributors and no comparable pairs. All 16 Python results match exactly.
The fixture is `tests/fixtures/random-survival-oob-concordance.json`; native
source and compiled libraries remain ignored and are not distributed.

A separate independent root comparison reconstructs each sampled one-feature
tree with native C split scores and leaf KM/NA kernels, then averages only its
OOB rows. All 264 survival/hazard values agree exactly over 12-tree forests
on seven- and four-point grids. Native and Python concordance errors agree at
0.31818181818181823 and 0.2954545454545454, respectively, with 44 comparable
pairs. Replaying single-node tree sampling independently verifies membership
both with and without replacement. Enabling OOB computation preserves all
packed tree arrays exactly. The comparison takes 0.965 seconds after imports,
including the tiny native harness build, at 120.45 MiB peak RSS and zero swaps.

This change in error with the chosen grid is native behavior. `rfsrc.R`
passes the already thinned `get.grow.event.info(..., ntime=ntime)` grid to C
(around lines 301 and 534–535), and `getMortality` sums over that grid. It
does not secretly use every unthinned failure time. Grid choice and time-unit
dependence of absolute tie tolerances therefore remain explicit.

Existing tree checks plus focused new OOB checks pass: six tests in 1.57
seconds. Targeted lint, formatting and type checks pass. These are kernel and
independent-tree comparisons, not a claim of native full-forest RNG parity.
Categorical splitting, alternative split rules and missing-value handling
were separate scope at this checkpoint; later sections and the dedicated
missing-data reference audit document subsequent additions.

## Permutation importance and independent native comparison

`permutation_random_survival_forest_importance` now implements explicit OOB
permutation importance. Each feature is shuffled separately within each tree's
OOB rows; perturbed mortality is averaged over each complete tree block before
computing concordance error. The native `RF_perfBlockCount=floor(ntree/perfBlock)`
rule excludes incomplete tail trees. `finalizeVimpPerformance` excludes blocks
with undefined baseline or perturbed errors before averaging differences.
Negative differences are retained. Results expose each component and the
number of valid blocks rather than hiding undefined blocks.

The Python default `block_size=None` uses the entire forest. Native explicit
permutation importance defaults to `block.size=10`; this different Python
default is documented, and the native block size can be supplied explicitly.
Native `importance=TRUE` defaults to the separate anti-split importance
interface below. Random-stream equivalence is not claimed.

`tools/reference_random_survival_vimp.py` independently reconstructs sampled
one-feature trees using native C split/leaf kernels, replays OOB permutations,
and evaluates errors with the unchanged native concordance kernel. A constant
second feature must have zero importance. Seven trees and seed 1772 check:

| Block size | Complete blocks | Valid blocks per feature | Ignored tree indices | Importance |
| --- | --- | --- | --- | --- |
| 7 | 1 | 1 | none | `[0, 0]` |
| 3 | 2 | 2 | `[6]` | `[-0.09702380952380951, 0]` |
| 1 | 7 | 6 | none | `[-0.08333333333333333, 0]` |

All baseline errors, perturbed errors, block differences and mean importance
values agree exactly. The root comparison took 0.9344 seconds after imports,
including the tiny native harness build, at 113.28 MiB peak RSS with zero swaps.
The nine focused forest/OOB/importance tests passed in 1.72 seconds; targeted
lint, formatting and type checks passed. A training fingerprint checks exact
canonical training values and row order, without storing another feature matrix.
Prospective work bounds cover routing and pair comparisons, with row-sized
temporary vectors and bounded feature-by-block output arrays.

## Anti-split importance

The pinned randomForestSRC source describes `importance=TRUE` as anti VIMP
and the R help (`man/rfsrc.Rd`, VIMP section) says this sends each case to the
opposite daughter at a target-variable split. In `src/randomForestSRC.c`,
`getAntiMembership` routes OOB members through `antiMembershipGeneric`
(approximately lines 4058–4114). The generic routine compares each node's split
variable with the requested feature and flips its ordinary daughter when a
uniform draw is at most `RF_vimpThreshold`; non-target splits retain their
ordinary branch. The pinned `utilities.R` helper `is.hidden.vimp.threshold` (lines 1165–1180)
confirms the default is 1.0 and documents a probability in [0,1], with zero
disabling flips. That helper forwards an explicit option without checking its
range; the Python API validates the range. It exposes `vimp_threshold` with
the same default, but its
NumPy stream is not native tree-specific RNG parity. It reuses the existing
OOB block concordance estimator, complete-block tail exclusion, undefined-block
handling, training fingerprint, and bounded workspace contract.

The same C file's `randomMembershipGeneric` (approximately lines 4542–4595)
defines the separate `importance="random"` route. At target-feature nodes it
consumes one uniform alpha; if alpha <= q, it chooses left iff alpha <= L/(Nq),
otherwise it keeps the ordinary daughter. Here N and L are represented parent
and left-child sample counts including bootstrap duplicates. The same alpha
controls both comparisons; two independent random steps would change the law.
`randomMembershipJIT` repeats this routing rule. The pinned R helper sets q=1
by default and documents q in [0,1]. The Python per-feature API defaults to 1,
requires an explicit RNG or seed, and documents that its stream is not native
R tree-specific RNG parity. At q=0 it consumes the target-node draw but keeps
the ordinary branch, including the exact alpha=0 endpoint where native's
`ran1D` generator does not ordinarily land.

With `compute_oob=True`, each packed tree now retains immutable per-node
represented counts, while the existing bit-packed in-bag mask continues to
represent distinct in-bag rows. The count root equals the bootstrap sample
size and each internal node equals the sum of its two child counts. OOB fit
preflight charges the count vector against the node/cell budget before random
sampling; fits without OOB diagnostics do not retain it. The random-routing
importance reuses the existing blockwise OOB mortality/concordance estimator,
training fingerprint, categorical feature encoding, complete-block tail
exclusion, and undefined-block handling. Python `block_size=None` means the
whole forest; native VIMP requests default to blocks of 10. No feature-group
routing or native random-stream parity is claimed.

`tools/reference_random_survival_forest_random_vimp.py` extracts the unchanged
pinned C generic routing kernel after verifying its Git blob hash, and wraps
only a deterministic uniform and ordinary daughter predicate. Fifteen route
cases cover q=0, .4, and 1, including unequal represented daughter counts,
threshold equality, target/non-target nodes and terminal nodes. The independent
C run matched all 15 expected terminal/draw-count rows. Focused Python route,
OOB count-retention and block-importance checks passed: 9 tests in 1.46 seconds,
135,348,224 bytes peak RSS and zero swaps, with numerical threads capped at 1.
Root's integrated run passed all 31 cluster-bootstrap and affected forest
checks in 3.50 seconds, including the q=0/alpha=0 endpoint, categorical
relabeling, retained multiplicities, core OOB behavior and native routing.
Peak process RSS was 145.64 MiB with zero swaps and one numerical thread.
Count-retention and seeded block replay are covered. These checks cover the routing
kernel and small OOB fits, not full native forest or RNG parity.
The portable C reference generator reproduces the committed CSV byte-for-byte
in 0.480 seconds at 36.66 MiB peak child RSS with zero swaps. A generator-only
argument collision and output newline inconsistency were fixed during review;
neither changed the routing results.

The integrated categorical/anti checkpoint matches the unchanged native
branch/mask kernels and passes 20 focused checks. Detailed source scope,
resource measurements, category-relabeling invariance and preserved numeric
permutation references are recorded in
[the categorical audit](survival-forest-categorical-audit.md#integrated-numerical-validation).
