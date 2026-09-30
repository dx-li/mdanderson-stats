# Random survival forest log-rank-score audit

The pinned randomForestSRC 3.2.2 manual (`research/raw/randomForestSRC/man/rfsrc.Rd`,
Survival analysis) advertises `splitrule="logrankscore"` and cites Hothorn and
Lausen (2003). The official randomForestSRC survival vignette gives the score
transform and standardized split statistic. The optional Python split
criterion follows that equation and the vignette's maximum-rank handling of
tied times. This is separate from the default log-rank criterion.

The independent pinned `coin` package source verifies the tie convention:
`R/Transformations.R::logrank_trafo` uses `rank(time,ties.method="max")`,
orders records by time and status, forms `n_risk=n-unique(or)+1`, aggregates
`n_event` at each maximum rank, and returns the cumulative weighted event score
minus the event indicator. This is the global sign reversal of the vignette's
individual score, which leaves the absolute centered split statistic unchanged.
Pinned source: cran/coin commit `200380dead826eaba431af5e07bf1397ac78044d`;
`Transformations.R` blob `06aa98fb1c63ad9c57b3ddc0e0c4ce60f29d3a98`,
`SurvivalTests.Rd` blob `c63308c71283f75306565951a4d3976e9abc00f4`.

The cached RF-SRC C source (`research/raw/randomForestSRC/src/randomForestSRC.c`,
`logRankNCR`, approximately lines 19848–20190) computes this alternative but
mixes covariate-sorted positions with original subject indices when constructing
scores and aggregating them for numeric and factor candidates. The Python code
computes one score per sampled node row in node order, then applies each existing
candidate mask in that same order. Duplicated bootstrap rows remain distinct
observations for ranking, candidate sizes and the variance denominator. The
Python API therefore implements the documented method, not native
`SURV_LRSCR`'s apparent indexing behavior. It also does not add the separately
advertised `bs.gradient` rule, competing risks or missing-value imputation.

A four-row source-index diagnostic makes the mapping discrepancy concrete.
Take `(time,event,x)` rows `(1,1,3)`, `(4,0,2)`, `(2,1,4)`, `(3,0,1)`. In
covariate order RF-SRC's `indxx` is `[4,2,1,3]`; maximum time ranks in that order
are `[3,4,1,2]`. The documented scores in original row order are
`[3/4,-7/12,5/12,-7/12]`, so the left daughter containing rows 4 and 2 has
standardized absolute score `7/sqrt(17) ~= 1.69775`. The C score-construction
loop around 19959 uses the first `Gamma` rows of `indxx` as though those rows
were ordered by survival time; its resulting position-indexed vector is
`[-1/4,-7/12,1,1]`. The numeric candidate loop around 20051 then adds
`survivalRank[indxx[k]]` even though the vector was built by sorted position.
For the same first-two-in-X candidate, that mapping gives approximately `0.201`
instead. This is an explanatory scalar reconstruction of the inspected loops,
not a compiled native RF-SRC forest comparison.

`tools/reference_random_survival_logrankscore.R` sources the unchanged pinned
`coin::logrank_trafo` body (no package installation) and emits four small cases:
unique times, tied times, bootstrap-style duplicated rows, and censoring before
an event. All 23 Hothorn–Lausen score values match the Python transform after
the documented sign reversal. Focused Python checks also compare tied-time and
duplicate-row scores to a scalar implementation, verify daughter statistics,
check the best exhaustive numeric split, and verify categorical splits retain
row alignment under a row permutation. The default remains `logrank`; all 12
tests in `tests/test_random_survival_forest.py` passed with warnings as errors.
Targeted Ruff check/format and mypy passed. The serial validation sequence took
3.78 seconds; its maximum child RSS was 169,328,640 bytes (about 161.5 MiB) and
reported zero swaps. A 32-tree guide example produced splits in all 32 trees.
These references do not execute RF-SRC's flawed `SURV_LRSCR` branch or claim
its native split decisions.
