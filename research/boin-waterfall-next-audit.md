# Next method gap: full BOIN waterfall simulation

Catalog entries 99 and 128 need a full waterfall simulator. Existing
`boin_waterfall.py` reproduces interactive `next.subtrial` planning, and
`boin_combination_simulation.py` handles ordinary single-MTD combinations.
Neither simulates the complete sequence of waterfall subtrials.

The reference is the already acquired BOIN 2.7.2 source
`research/raw/BOINComb/R/get.oc.comb.R`, especially the nested
`waterfall.subtrial.mtd`, `waterfall.subtrial` and `get.oc.comb.waterfall`
functions. `docs/boin-combination-source.md` records the archive provenance.
The source remains reference-only. Source inspection establishes the following
contract. Six deterministic original-R subtrial probes now validate the local
conduct rules; complete-wrapper and final-contour checks are still needed.

* The first search space follows column one down the dose grid and then the
  last row across. Subsequent spaces traverse columns 2 onward of an earlier
  row. A vector of cohort budgets supplies one budget per executed subtrial,
  with as many entries as rows. The default precision limit is 12 patients.
* Within a subtrial, complete cohort DLT counts update only its ordered space.
  Elimination uses the integer BOIN safety table and excludes higher positions.
  Movement uses the BOIN escalation/de-escalation table. The native precision
  check occurs **after movement**, at the destination dose's current enrollment.
* Subtrial selection uses weak-prior means, inverse-variance weighted PAVA,
  a tiny increasing tie adjustment and an optional strict de-escalation bound.
  It also records whether the selected candidate meets the escalation rule.
* If the first subtrial selects an earlier row in column one, the simulator
  excludes lower rows. When that candidate supports escalation, it first runs
  a special additional subtrial across columns 2 onward of that **same row**.
  This branch is not represented by simply chaining the interactive planner.
* Otherwise the next subtrial uses the preceding row, starting at one column
  above the selected column (clipped to the last column). Afterward the final
  contour is based on an unrounded bivariate isotonic fit, retained exclusions
  and row-to-row continuity. The ordinary standalone selector's rounded path
  is not an equivalent final-selection contract.

Static inspection also identifies behaviors that need explicit verification
and decisions, not silent copying: the first safety-stopped subtrial can break
the outer loop before its new counts are assigned back; custom indifference
probabilities/offsets are omitted from the shared native boundary-table call;
the interim extra-safe prior differs from final subtrial selection; and some
reported percentages divide by a fixed 1000 rather than `ntrial`. The final
contour also has untreated-row/continuity edge cases. Python must preserve
actual enrolled patients and DLTs, and calculate frequencies using the actual
trial count, even where source output is defective.

Prefer a shared bounded deterministic trial engine with a replay API and serial
simulation wrapper. Retain cohort/subtrial histories, per-trial counts and
exclusions, row-wise contour recommendations, selection probabilities and
Monte Carlo errors. Preflight dimensions, total enrollment and aggregate output
size before advancing a caller RNG. Reuse existing BOIN boundaries and isotonic
helpers where their exact contracts agree. Avoid replacing the source method
with the simpler interactive planner merely to obtain passing tests.

The temporary BOIN/Iso R installation used by earlier reference generation is
no longer present; the default R library contains neither package. Base-R
boundary and subtrial routines can still be sourced from the pinned files
without installation. Small deterministic original-R traces can validate
conduct and subtrial transitions independently; existing Iso reference fixtures
remain available for the shared bivariate fit.

`tools/reference_boin_waterfall.R` generated the six small cases in
`tests/fixtures/boin-waterfall-subtrials.csv` and the dose histories in
`tests/fixtures/boin-waterfall-traces.csv`. The unchanged original routines use
locally supplied uniform draws that encode specified cohort DLT counts. This
checks their actual requested doses, not a reimplementation of their conduct.
Generation took 0.29 seconds and required no external R libraries. The checks
include the native precision stop after a downward move to a previously
treated dose. These fixtures alone do not establish full waterfall coverage.
