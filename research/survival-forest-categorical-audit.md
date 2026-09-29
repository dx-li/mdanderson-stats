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
