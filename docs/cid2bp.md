# CID2BP: difference between two binomial proportions

`cid2bp_interval` implements five methods from catalog entry **38**,
[CID2BP](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/38).
The [version 1.2 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CID2BP/CID2BP_V1.2.zip)
contains complete Fortran 77 source, build instructions and reference output.
The source README is dated August 16, 2019.
[Provenance](cid2bp-sources.json) records the archive and inspected files;
original source and executables are not redistributed.

The software cites Lee, Serachitopol and Brown (1997), *Likelihood weighted
confidence intervals for the difference of two binomial proportions*,
Biometrical Journal 39:387–407. Source comments also identify Hauck and Anderson
(1986), Peskun (1993), and Cox and Snell's *Analysis of Binary Data*, second
edition, pp. 51–52, for the relevant methods.

## Supported calculations

All results estimate `p1-p2` for **independent** binomial samples. Inputs are
integer successes and trials; trials must be 1–1,000,000 per sample and successes
must lie between zero and trials. Confidence must be in `[1e-6,1-1e-12]`.

| Python method | Native menu | Calculation |
|---|---|---|
| `wald` | 1 | Ordinary unpooled normal approximation |
| `continuity_corrected` | 2 | Unbiased binomial variance estimates plus `1/(2*min(n1,n2))` in the half-width |
| `yates` | 3 | Wald half-width plus `1/(2*n1)+1/(2*n2)` |
| `peskun_native` | 4 | Native discrete-grid correction, including its upper-gap branch behavior |
| `cox_snell` | 5 | Profile binomial likelihood-ratio interval using a chi-squared(1) threshold |

`continuity_corrected` requires at least two observations in each sample.
The default is explicitly `cox_snell`; it is **not** the native menu's automatic
method 8, which switches between weighted-likelihood and Cox–Snell intervals.

The three normal methods include the native `adjust` routine's Peskun boundary
adjustments. These replace bounds when the observed difference is exactly ±1
or ±(1−1/max(n1,n2)); they are not generic Clopper–Pearson subtraction intervals.
The exact adjustments use log probabilities and `expm1` rather than large powers.
As in the native program, final bounds are restricted to [-1,1]. The normal
approximations can still be degenerate or have poor small-sample coverage;
these adjustments do not make every interval an exact-coverage procedure.

## Native Peskun calculation

`peskun_native` reproduces menu option 4, including two source conventions:
points within 1e-6 of the observed difference are excluded, and the loop's upper-
correction branch accepts a negative distance when it does not set a new lower
minimum. For example, at `3/5-0/4`, the upper correction uses half of .10, although
the nearest strictly positive distance is .15. This is explicitly native
compatibility, not a claim to repair or independently verify the paper's rule.
The upper-gap behavior can break sample-exchange symmetry.

NumPy cumulative minima reproduce the original loop order without Python loops
over grid points. At most two million `(n1+1)*(n2+1)` points are permitted. The
same special-case adjustments described above apply afterward. If an unadjusted
bound remains undefined or unordered, the function raises `ArithmeticError`
instead of returning NaN or silently fabricating a bound.

The original `cfpesk` subroutine was extracted unchanged and compiled with
`gfortran -std=legacy`. Its ordinary example and the asymmetric correction example
match Python to floating-point tolerance. Only this subroutine was compiled;
the full interactive executable has not been rebuilt.

## Profile-likelihood implementation

For a candidate difference `d`, write `p1=q+d`, `p2=q` and maximize over
`max(0,-d) <= q <= min(1,1-d)`. The binomial log-likelihood is concave in `q`.
The implementation checks boundary optima and otherwise solves its monotone
score equation. It then inverts the likelihood-ratio test on either side of
the observed difference. Zero-count log terms use their continuous limits.

The Fortran implementation uses a cubic solver, clips probabilities at zero or
one to 1e-5 or 1−1e-5, and uses a root tolerance of 1e-4. Python optimizes within
the actual probability domain, allowing exact boundary optima and more precise
roots. A stable logarithmic-remainder calculation evaluates likelihood loss
directly, avoiding cancellation between nearly equal log-likelihoods.
Differences near extreme data and in the last printed decimal are
intentional; the likelihood-ratio reference remains asymptotic, not an exact
finite-sample guarantee.

```python
from mdanderson_stats import cid2bp_interval

result = cid2bp_interval(7, 12, 1, 7, method="cox_snell")
assert abs(result.estimate - (7 / 12 - 1 / 7)) < 1e-14
assert abs(result.lower - (-0.001238004)) < 1e-8
assert abs(result.upper - 0.752218058) < 1e-8
native = cid2bp_interval(7, 12, 1, 7, method="peskun_native")
assert abs(native.upper - 0.7662527684207924) < 1e-12
```

The returned `BinomialDifferenceInterval` contains the estimate, limits,
confidence and selected method. This API handles one comparison at a time.

Focused tests check the supplied native reference output for the first four
methods, exact profile-likelihood limits for zero-event and separated samples,
sample-exchange symmetry through million-patient examples, and the source's
special normal-method adjustments. Native output comparisons allow its stated
1e-4 solver tolerance. The full original executable has not been rebuilt or run;
its supplied reference output and computational source were used, plus the compiled Peskun subroutine.

**Catalog status remains partial.** Weighted likelihood mid-P (6), conservative
weighted likelihood (7), automatic selection (8), exact intervals (9), and native session/report interfaces remain
pending.
