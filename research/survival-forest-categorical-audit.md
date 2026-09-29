# Random survival forest nominal-feature audit

Pinned randomForestSRC 3.2.2 documentation (the `Allowable data types and
factors` section of `man/rfsrc.Rd`) says unordered factors split by subsets
versus complementary subsets and use a shared factor map to align train and
test codes. The cached C implementation in
`randomForestSRC.c::stackAndConstructSplitVectorGenericPhase2` obtains the
candidate level count from levels present in the current node while storing
membership at the global factor-map width. Consequently a globally known
level absent at a node has an unset membership bit and routes right.
`makeFactor`, `bookFactor`, and `bookPair` enumerate unordered partitions
without retaining both complements. Their exact mode requires at most 32
levels and, for `nsplit > 0`, partition count no greater than both `nsplit`
and node row count; for `nsplit == 0`, partition count must be strictly less
than node row count. Otherwise source sampling chooses a subset-size group in
proportion to its binomial count (halved for equal-size complements), then
samples uniformly among subsets of that size.

The Python forest accepts explicit categorical feature indices over finite
numeric labels. Sorted global labels are retained as immutable maps; split
nodes store only their chosen left-level codes in one bounded packed array.
Candidate partitions are yielded one at a time, never materialized as a
powerset. The source's numeric and categorical split candidates share the
existing log-rank score and split-work limit. Numeric-only fits retain their
previous ordered-cut path and RNG calls. The default profile uses raw-space
numeric means and per-reference-population categorical modes; ties choose the
smallest observed numeric label. Contour axes must be numeric; nominal
adjustment columns remain fixed at the modal label and reference profiles are
validated against the fitted maps.

Python rejects unseen prediction levels rather than mapping them to a
synthetic factor level. The relevant native `check.factor` source does append
a synthetic level for novel labels, so this is a deliberate conservative
Python contract and not native novel-level parity. Category labels are numeric
in this API; callers may encode string categories consistently before fit and
prediction. The forest's randomized subset/RNG sequence is not claimed to
match R.

## Integrated numerical validation

`tools/reference_random_survival_extensions.py` verifies the pinned C/header
Git blob hashes recorded in the main forest audit, then extracts unchanged
`bookPair`, `splitOnFactor`, and `antiMembershipGeneric` bodies. A tiny wrapper
provides allocations, an explicit uniform tape and a simplified ordinary
numeric-polarity adapter. It does not execute the native forest engine or its
random-number generator. Extracted sources and compiled objects remain ignored
under `research/raw`; only the generator and numerical CSV are retained.

The Python comparison matches ten native categorical partition masks in order,
four global-map routing cases including a level absent from a node, and three
anti-routing cases including the number of uniform draws consumed. Inclusive
`nsplit>0` and strict `nsplit=0` exact-enumeration boundaries are checked against
the inspected source contract. The bounded native rebuild took 0.768 seconds,
with 26.63 MiB parent and 52.75 MiB child peak RSS and zero reported swaps.

All 20 focused forest/OOB/importance/shared-contour checks pass after integration,
including the final source-boundary, modal-profile and category-encoding fixes.
This run took 1.946 seconds, peaked at 148.39 MiB and reported zero swaps.
An additional mixed continuous/categorical bootstrap case exactly preserves
predictions, OOB curves and both blockwise importance estimators after an
order-preserving relabeling, including labels of magnitude `1e100` and stochastic
anti threshold 0.4. Four invalid-input/work rejections preserve the caller RNG.
A 40-category `nsplit=0` example stays within 18,439 split-work units. That
integration check took 0.119 seconds at 121.70 MiB, zero swaps.

The existing independent native-kernel permutation reference still agrees
exactly for block sizes 7, 3 and 1 after categorical integration (including
constant features, undefined blocks and omitted tails): 0.769 seconds,
125.30 MiB, zero swaps. Targeted Ruff/format and all three affected forest-module
mypy checks pass. No broad numerical suite or new CI workflow was introduced.
