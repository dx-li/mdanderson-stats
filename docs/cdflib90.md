# CDFLIB90

CDFLIB90 is a library of cumulative distributions, complementary distributions,
quantiles and inversions with respect to distribution parameters. The catalog
archive contains Fortran 95 version 1.2 and additional C/Fortran DCDFLIB material.
The entry is **partial**: the beta distribution's four public interfaces are
implemented. The other 11 distribution modules, the remaining archived library
interfaces and the complete 106-file archive audit remain outstanding.

Source: [catalog entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/21)
and [archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/CDFLIB90/CDFLIB90%20%20_V90.tar.gz).
The original [legal terms](../notices/mdanderson-cdflib90-LEGALITIES.txt) are retained.
Original code and compiled reference executables remain outside the package.

## Beta distribution

```python
from mdanderson_stats import cdf_beta, cum_beta, ccum_beta, inv_beta

cdf = cum_beta([0.1, 0.5, 0.9], a=2, b=5)
survival = ccum_beta([0.1, 0.5, 0.9], a=2, b=5)
quantile = inv_beta([0.01, 0.5, 0.99], a=2, b=5)
shape = cdf_beta(3, x=0.2, cum=0.04, b=1).a  # a=2
extreme = cdf_beta(2, ccum=1e-100, a=1, b=1)
# extreme.x rounds to 1; extreme.cx still retains 1e-100.
```

`cdf_beta(which=1, *, cum=None, ccum=None, x=None, cx=None, a=None, b=None)`
returns an immutable `CDFBeta` containing all six broadcast arrays and `which`.
All result arrays own immutable backing storage; they cannot be made writable.
The computed group must be omitted rather than supplied as an initial guess:

| which | Computed group | Required input groups |
|---|---|---|
| 1 (default) | cum and ccum | x/cx, a, b |
| 2 | x and cx | cum/ccum, a, b |
| 3 | a | cum/ccum, x/cx, b |
| 4 | b | cum/ccum, x/cx, a |

Either or both members of each input pair may be supplied. Each lies in [0,1],
and a supplied pair must sum to one within eight double-precision machine
epsilons. The smaller member is retained and only the larger is reconstructed.
This preserves tiny complements supplied alongside a coordinate or probability
that has rounded to one. Omitted complements are calculated from the available
member; information already lost to rounding cannot be recovered.

The convenience functions return the corresponding array. Their positional
arguments are `(x, a, b)` for tails and `(cum, a, b)` for the inverse; use
`x=None, cx=...` or `cum=None, ccum=...` to supply only the complement.
`cdf_beta(which=2)` exposes both quantile coordinates, which is useful at extreme
probabilities. Shapes retain the original inclusive [1e-10,1e10] domain.
The implemented distribution is the standard Beta(a,b), with density proportional
to x**(a-1) (1-x)**(b-1). The manual's printed a/b exponents omit the minus ones;
the actual archived implementation and this port use the standard definition.

## Numerics and failure behavior

Forward evaluation reflects the beta parameters when the supplied complementary
coordinate is smaller. SciPy `betainc` and `betaincc` evaluate both tails directly.
The smaller computed tail is retained and the larger is reconstructed so the
result is a consistent complementary pair. At x=cx=0.5 with a=b, symmetry gives
exactly 0.5 for both tails. This also avoids platform-dependent kernel roundoff
at very large symmetric shapes.
Quantiles invert the smaller probability and compute both coordinates directly
using `betaincinv` and the reflected `betainccinv`; the smaller coordinate is not
obtained by subtracting a rounded larger coordinate from one.

Shape inversion uses monotonicity in a or b and a batched bisection in log shape
over [1e-10,1e10]. Each step evaluates the smaller requested probability using
the complementary-coordinate-aware forward function. Sixty-four steps reduce
the initial log interval to double-precision resolution. This preserves inputs
that a lower-CDF-only shape inverse can lose to subtraction or rounded x.
The result undergoes a forward check with relative probability tolerance 1e-7
plus 32 smallest-subnormal units. Failure raises `ArithmeticError`.

Shape requests require interior coordinate and probability pairs: both members
must be positive, although the larger may round to one. Endpoints cannot identify
a finite shape uniquely and raise `ValueError`. A requested shape outside the
search domain also raises `ValueError`; no boundary is returned as a solution.
Inputs are always validated. There is no unsafe skip-validation option or numeric
status code that callers might accidentally ignore.

These are double-precision calculations. Extremely small tails or quantile
coordinates can underflow to zero, and the larger member can round to one.
Providing the small complement preserves it when representable; it does not
provide arbitrary precision. Shape inversion cannot reconstruct a target that
has already underflowed to zero. Invalid numerical kernel results fail explicitly.

## Validation and source findings

`tools/reference_cdflib_beta.py` compiles eight unmodified original source files
with a separate driver. The fixture records archive/source hashes, compiler,
build command, driver, all inputs, six outputs and the original status for each
of **205 cases**. There are 64 forward cases over varied shapes/coordinates and
141 inverse cases generated from the original forward probabilities.

Forward probabilities agree directly with native output. For inversions, the
original generating coordinate or shape is known and provides a stronger check
than trusting the archived root-finder's output. Python recovers those values
with tight tolerances. Eleven original shape-b inversions return status -50
at the exact initial solution b=5, despite that solution being inside the bounds.
A uniform-beta inverse at lower probability about 0.01 reports status zero but
returns x=0.505 and cx=0.99. These outputs are preserved as source defect evidence,
not relabeled as valid statistical references. Successful native shape inversions
also have noticeably looser accuracy than the Python search.

Independent tests use exact rational binomial sums for integer beta shapes,
closed-form quantiles and shape inversions when one shape is one, endpoints,
logarithmic extreme-tail identities, broadcast batches, shape bounds, invalid
and unidentifiable requests, convenience interfaces and immutable ownership.
The known source status/quantile failures have explicit regression tests.

CDFLIB90's discrete distributions are embedded in continuous beta/gamma formulas;
its inversions may return fractional counts. Their future ports must preserve
that contract, rather than silently substituting integer-valued statistical
quantiles. This beta milestone does not claim those modules are implemented.


## Batching measurements

[Benchmark results](cdflib-beta-benchmark.json) compare one array call with
repeated scalar calls to this same Python API, for forward tails and shape-a
inversion. Both results are checked against one another and analytic beta
identities. The artifact records median-of-three timings, environment versions
and all input-generation parameters. It measures amortized Python overhead and
vectorized numerical work; it is not a comparison against native Fortran.
