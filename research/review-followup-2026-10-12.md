# Executed follow-up to independent static review

## ComPAS outer quadrature — confirmed narrower cases fixed

The initial six-arm50/100, variance1000, null.999 concern was not reproduced:
log evidence differed2.1e-14 from null.5, agreed with independent six-dimensional
Gaussian integration within1.4e-12, and passed on SciPy1.14.1 and1.18.1. The
reviewer withdrew that specific claim after accounting for QUADPACK's initial
endpoint-spike safeguard. That safeguard does not close every distant-split case.

Two subsequent bounded hypotheses were executed and **did reproduce** substantial
normalizing-mass loss despite tiny reported numerical errors:

| Six identical arms | Variance | Null | Before log evidence | Corrected log evidence |
| --- | --- | --- | --- | --- |
| 1/100 | 1000 | .5 | -30.530580136738973 | -30.530580136738973 |
| 1/100 | 1000 | .95 | -31.22371908456235 | -30.530580136738973 |
| 1/100 | 1000 | .999 | -31.223719084562838 | -30.530580136738973 |
| 50/100 | .000001 | .5 | -24.597977081979586 | -24.597977081979593 |
| 50/100 | .000001 | 5e-324 | -25.291124262539576 | -24.597977081979597 |

The outer integrator now always includes standardized local mode splits
-12,-4,-1,0,1,4,12 as well as the response-threshold transitions. This prevents
these extreme transitions from placing essentially all posterior mass in a thin
edge of an enormous finite interval. The integration domain remains the entire
real line; no posterior/prior tail is truncated by this change.

A separate fixed-node Gauss-Legendre/normal integral uses neither adaptive
quadrature, a mode finder, a response threshold nor production imports. Order
512/768/1024 and domain refinements agree within7e-13 in the sparse case and
1e-13 in the near-pooled case. Corrected production evidence agrees within1e-12.
See `tools/reference_compas_extreme_thresholds.py` and its portable fixture.
Five added regressions cover the confirmed cases, alongside the five-threshold
six-dimensional reference regression. The bounded review is closed by executed
evidence; numerical error estimates still are not certified bounds or clinical CIs.

## BLESS cancellation

The lung counterexample is confirmed. With ECOG/indicators0,
protein=2**100 and LDH=3100*2**100, the previous float-products/fsum calculation
returned-140737488355328 (-2**47), incorrectly giving survival1 at every time.
The published decimal expression cancels its variable terms and leaves1.93518.

The printed equations now use Decimal coefficients and exact conversion of the
normalized binary64 inputs, evaluated with400 significant digits before one
conversion of eta back to float. This preserves the centered decimal expression
without inventing clinical ranges or relying on fsum of already-rounded products.
The guard covers finite binary64 integer magnitudes and the printed decimal
coefficients/centers; existing finite-output checks remain. Fifteen regressions
exercise exact cancellation in lung, breast and GCMM at powers0,30,100,500,1000,
including saved-input replay and survival transformation. Existing499 BLESS
checks also pass. Scientific scope remains the rounded published seven-point grid.

## PRT source precision correction

See `prt-startup-precision-audit-2026-10-12.json` and the native recovery report.
Original CRT instructions in both binaries establish53-bit precision (0x23f),
correcting earlier probes' forced64-bit mode. Eight of20 fixture files change
numerically;5,044 installed-wheel PRT tests pass with corrected fixtures without
a production arithmetic change. An additional six initializer cases and111
calendar precision cases validate the correction. Historical37f results remain
in git history as alternate-precision diagnostics, not default desktop claims.

## Acceptance boundary

Earlier combined focused tests:5,635 passed before the narrower ComPAS cases.
The corrected quadrature is undergoing new focused and full-code acceptance. The earlier combined full run was interrupted at66,529 passing
tests when the BLESS review fix changed production code; it is not final
acceptance. The subsequent full run was also interrupted at64,814 passes when the ComPAS
defect was confirmed. Neither interrupted run is final acceptance. A fresh run
will include the corrected quadrature and all reference fixtures.
Do not promote PRT69, BLESS149 or ComPAS140 from partial on these results.

Final corrected acceptance:76,136 full tests pass (483.51s),5,640 fresh-wheel
focused tests,9,347 broader installed tests,28 plotting/TIFF tests and all three
examples. Corrected ComPAS passes82 tests on SciPy1.14.1 as well. Full evidence
and artifact hashes are in prt-conduct-and-review-validation-2026-10-12.json.
