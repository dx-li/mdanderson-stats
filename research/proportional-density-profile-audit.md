# Proportional-density profile inference

The primary [Shen–Qin–Costantino paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721282/),
section 3.2, gives a one-degree-of-freedom conditional likelihood-ratio test of
a specified beta under a common censoring distribution. The intercept is
profiled under the null. The implementation requires the caller to assert
that assumption; the unequal-censoring weighted-chi-square law is not replaced
by this calibration. Confidence limits invert the source test and are an
explicit Python reporting extension. They concern beta alone.

The cached primary HTML's MathML confirms the likelihood-ratio expression
(M42). The separate incidence expression (M49) has an apparent missing square
root and does not specify a unique pooled estimator; this addition leaves the
archive's incidence calculation untouched.

`tools/reference_proportional_density_profile.R` uses independent base-R
binomial GLMs for unrestricted and fixed-slope fits, and `uniroot` for interval
endpoints. Five cases cover ties, unequal arm/failure counts, positive/negative
slopes, nonzero null slopes and changed time units. Three confidence levels
(80%, 95%, 99%) produce 15 intervals.

`tools/check_proportional_density_profile.py` compared 90 estimates, null tests
and interval endpoints with those references. Maximum absolute error was
2.003e-13; the 30 endpoints reproduce their likelihood-ratio cutoffs within
8.882e-15. Zero-slope tests agree with the existing common-censoring estimator.
Arm exchange, time units 1e-150/1e150 and confidence nextafter(1,0) were checked.
The comparison took 0.0651 seconds after imports, peaked at 113.16 MiB RSS and
reported zero swaps.

Six focused existing/new tests passed with warnings as errors in 1.77 seconds,
peaking at 131.95 MiB with zero swaps. Targeted Ruff/format/mypy passed. Extremely
small confidence cutoffs below likelihood floating-point resolution raise
explicitly. Endpoint residuals are checked after root convergence. Work bounds
include unrestricted fitting, profile iterations, bracketing and residual
checks; no simulation or parallel worker is used.

Both downloaded R routines, common-censoring inference and the two
goodness-of-fit bootstraps are available. Unequal-censoring treatment-effect
null calibration and bootstrap parameter uncertainty remain open. Entry 78
therefore remains partial.
