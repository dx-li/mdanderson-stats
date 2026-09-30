# IPDfromKM report-diagnostics audit

## Pinned report contract

The report diagnostics are from CRAN IPDfromKM 0.1.10,
[source commit `16ea3e163b8ad409e51e035154c52803dcb1c28b`](https://github.com/cran/IPDfromKM/tree/16ea3e163b8ad409e51e035154c52803dcb1c28b).
In `R/getIPD.R`, lines 287–292 evaluate the reconstructed KM curve at each
preprocessed time, round only the fitted `estsurv` to three places, compute
`diff=round(estsurv-surv,3)`, then drop rows with missing values. The input
observed `surv` is not rounded: `R/preprocess.R` line 76 only divides it by
100 when percentage input is selected. Lines 294–306 calculate and name the
summary fields and call `suppressWarnings(ks.test(dat$estsurv,dat$surv))`.

The source precision formulas are:

- RMSE: `round(sqrt(sum(diff^2)/(n-1)),3)`;
- mean absolute error: `round(sum(abs(diff))/n,3)`;
- value labeled maximum absolute error: `round(max(diff),3)`.

The last formula is signed. The Python result preserves it under
`legacy_signed_max_error` and separately gives the correctly computed
`max_absolute_error`. This helper does not change metrics on existing
`ReconstructedIPD` results, which continue to expose their own unrounded curve
fit errors.

Rounding follows the R 4.4.1 `fround` distance comparison rather than NumPy's
round-to-even scaling: form the decimal lower and upper candidates, compare
their floating-point distances to the input, and on an exact distance tie
choose the upper candidate when the lower integer is odd. The implementation
was checked against 6,000 identical binary64 inputs transported to R 4.4.1
with `readBin`/`writeBin` (all rounded outputs matched). The inspected source
was `src/nmath/fround.c` at R's `R-4-4-branch` commit
[`702728a10fb4f4385cbc0818c55b106c423aab12`](https://github.com/wch/r-source/blob/702728a10fb4f4385cbc0818c55b106c423aab12/src/nmath/fround.c);
its header credits the R Core Team (2000–2020) and Ross Ihaka (1998). The
runtime reference was R 4.4.1 (2024-06-14).

Missing values are omitted by source rows, not independently in each sample.
Python therefore removes a pair if either vector has NaN and returns retained
indices and an omitted count. Infinities, finite non-probabilities, unequal
vector lengths and fewer than two complete pairs are rejected explicitly.
Inputs are bounded to 100,000 positions before downstream KS work.

## KS convention

The comparison is on rounded fitted values versus unrounded observed values.
The pinned run used R 4.4.1 `stats::ks.test.default`. That function chooses
`exact = (n_x*n_y < 10000)` when the caller omits `exact`; with ties it passes
the pooled observations to `psmirnov_exact`, so the exact reference is a
conditional permutation distribution respecting pooled tied-value blocks.
At or above that product threshold it uses `psmirnov_asymp`, whose two-sided
branch evaluates the limiting Kolmogorov distribution at
`sqrt(n_x*n_y/(n_x+n_y))*D`. This implementation follows those branch and
threshold rules, including tied observations. Other historical R versions can
have different defaults; Python records only `exact` or `asymptotic` and does
not claim parity with every R release.

The exact calculation is a bounded dynamic program over pooled value blocks.
It counts allocations of the first sample's labels that remain strictly below
the observed integer-scaled KS distance; subtracting this integer count from
`choose(n_x+n_y,n_x)` gives the tail probability without cancellation. The
exact branch only runs below the source product threshold. The asymptotic tail
uses the standard theta-transformed series for small arguments and the
alternating exponential series otherwise, avoiding large scratch matrices.
The reported D is computed from exact integer empirical-CDF counts before
conversion to floating point, avoiding cumulative-sum roundoff while retaining
the mathematical two-sample statistic.

The report's KS p-value is retained for source-compatible reporting only. The
two curves are paired at common times, and the fitted curve depends on the
observed one; `ks.test`'s independent-sample null does not describe this
dependence. It should not be interpreted as a valid inferential test of
reconstruction quality.

## Independent references

`tools/reference_ipdfromkm_diagnostics.R` uses only base R and the installed
R 4.4.1 stats implementation. It reproduces the report formulas and default
`ks.test` behavior for exact separated samples, exact tied and untied samples,
decimal rounding, the `n_x*n_y=10000` asymptotic branch, identical samples and
paired missing rows. Inputs and outputs are preserved in
`tests/fixtures/ipdfromkm-diagnostics-{input,summary}.csv`. The source package
is GPL-2; its source files remain ignored research inputs.


## Validation checkpoint

Ten independent R diagnostic scenarios cover nontrivial tied-label tails,
complete separation at 99 points, the 100-point asymptotic switch, precision
summaries and missing pairs. The exact 99-point separation tail agrees with
`2 / choose(198, 99)` (about `8.79e-59`) using a relative comparison against
the actual Python result. Two focused diagnostic checks and three existing
reconstruction checks pass with warnings treated as errors. The integrated
run took 1.795 seconds, peaked at 148.70 MiB RSS and reported zero swaps.
Targeted Ruff, formatting (including the guide) and worker mypy pass. The
rounding holdout described above compares all 6,000 values at identical binary
inputs; decimal text parsing was excluded from that comparison. Numerical work
ran serially with library threads limited to one, without installations or a
full local suite.
