# Dose-allocation overdose-risk source audit

The optional BOIN overdose summaries use the pinned `BOIN` source in the cached
`BOINComb` package. In `research/raw/BOINComb/R/get.oc.R`, lines 262–274 gate
the summaries on at least one dose whose true toxicity equals the target, then
count patients allocated to doses with `p.true > target`. The event thresholds
are strictly more than 60% and 80% of planned enrollment (`npts`, initialized
from planned cohort count and size at line 142), not realized enrollment.
Lines 275–293 omit both outputs if there is no exact-target dose. The summary
printer at `research/raw/BOINComb/R/summary.boin.R:214–256` presents the
percentages as optional results.

The cached Keyboard implementation independently uses the same availability
gate and strict planned-enrollment thresholds in
`research/raw/KeyboardComb/Keyboard/R/get.oc.kb.R:234–247`. The Python helper
therefore reports fractions, with Bernoulli Monte Carlo standard errors; users
can multiply probabilities by 100 for native-style percentages. It takes
planned counts explicitly and validates each against realized allocation.

The R BOIN code's exact-target check uses a scalar `if` on the result of
`which`, which can fail when several doses equal target. This implementation
uses the stated probability rule for all exact-target doses and works for
nonmonotone truth profiles as well. It does not infer target doses or planned
enrollment. If the exact-target gate is absent, all risk estimates are
explicitly unavailable.

The cached native calculation also uses `rowSums` after subsetting above-target
columns, which drops to a vector for one selected column and errors for none;
the helper handles both cases. The source R documentation says “or more” in
its prose, while the implementation and returned metric names use strict
“more than”; this port follows the executable comparisons (`> 0.6*npts` and
`> 0.8*npts`).
