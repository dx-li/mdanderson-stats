# Executed follow-up to independent static review

## ComPAS outer quadrature

The reported six-arm case was executed before any production change: each arm
has100 completed/50 successes, shared variance1000, tolerance1e-7. Log evidence
was-49.900093693853286 at null_response=.5 and-49.900093693853265 at.999, a
2.1e-14 difference. The proposed log(2) discrepancy is not reproduced.
Instrumenting the finite positive outer interval[0,3858.668285855197] returned
mass1.253314508371801; the negative interval returned1.2533142827139578. Thus
the positive half was not omitted in this executed case.

An independent six-dimensional integral of the marginal Gaussian logit prior,
without any response threshold, gives log evidence:

| Gaussian order per dimension | Log evidence |
| --- | --- |
| 8 | -49.90009369866118 |
| 10 | -49.900093693925214 |
| 12 | -49.900093693854615 |

Orders10/12 differ7.06e-11; production agrees within1.4e-12. Five thresholds
(.000001,.001,.5,.999,.999999) each agree with this reference within1e-9. This
is an explicit threshold-invariance regression, not a global proof against every
possible adaptive-quadrature failure. Existing error estimates remain estimates.
No ComPAS production algorithm change was made. Reproduce with
`tools/reference_compas_six_arm.py` and `tests/test_compas.py`.

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

Combined focused tests:5,635 passed in51.89s. Ruff, formatting and769-file mypy
checks pass. The earlier combined full run was interrupted at66,529 passing
tests when the BLESS review fix changed production code; it is not final
acceptance. A fresh full run uses all final code and corrected reference fixtures.
Do not promote PRT69, BLESS149 or ComPAS140 from partial on these results.
