# Legacy DCDFLIB Student's t

`cdft` and `cumt` implement the C/F77 Student's t interfaces in the pinned
CDFLIB90 archive, with separate evidence from the F95 distribution port.

```python
from mdanderson_stats import cdft, cumt

p, q = cumt(2, 10)
t = cdft(2, p=p, q=q, df=10).t
df = cdft(3, p=p, q=q, t=2).df
```

## Contract

| which | Computed group | Required input |
|---|---|---|
| 1 | p/q | t, df |
| 2 | t | p/q, df |
| 3 | df | p/q, t |

Omit the computed group. `DCDFLIBStudentT` returns which and owned immutable
broadcast arrays p, q, t, df. `cumt` returns paired lower/upper tails. Empty
batches are supported in every mode.

Input t may be any finite float, and input df any positive finite float.
Computed t retains the native search range [-1e100,1e100]; computed df retains
[1e-100,1e10]. The existing F95 `cdf_t` is unchanged: its input t is bounded by
±1e100 and df by [1e-3,1e10].

Inversion requires positive complementary p/q summing to one within three
machine epsilons. Either probability may be omitted in Python; the smaller
supplied tail is preserved. Median quantiles are exactly zero. A zero t or
median probability does not identify a unique df; the sign of t must match the
requested side of the median. Such requests, invalid inputs and unbracketed or
out-of-bounds answers raise `ValueError`. Numerical failures and failed forward
verification raise `ArithmeticError`, replacing native status/bound outputs.

## Numerical implementation

The smaller tail is half the regularized incomplete beta ratio
I_x(df/2,1/2), where x=df/(df+t²). Both beta coordinates are computed directly.
When t² overflows or underflows, logarithmic ratios retain the coordinates.
If x becomes subnormal or underflows, the implementation evaluates the leading beta term in
logarithms:

```text
small tail = exp((df/2)*log(x) - log((df/2)*B(df/2,1/2)) - log(2)).
```

The omitted relative correction is O(x), below floating-point precision in that
branch. Using a rounded subnormal coordinate instead can lose significant
relative precision (about 0.6% at t=1e161, df=1); transition cases are regression
tested. For shapes below one, the normalizer uses
logGamma(a+1)+logGamma(1/2)-logGamma(a+1/2), avoiding overflow in the log-beta
kernel for very small shapes. This preserves the half-probability limit for
subnormal df. If halving the minimum subnormal df rounds to zero, both tails
round to one half for every finite t. Compiled SciPy beta/log-beta/log-gamma
kernels are reused; their installed 1.18.1 interfaces were checked.

Quantiles invert the beta coordinates and reconstruct t with scaled square
roots, using logarithms when needed. For df>=1e20, quantiles use the limiting
normal kernel. For representable probabilities the normal quantile has magnitude
below 39, so the first relative t correction, (z²+1)/(4*df), is below 4e-18;
higher terms are smaller. These answers are still verified against the t CDF.
This avoids failed beta inverse evaluations at extreme finite df.

DF inversion reuses the vectorized legacy search with original-row indices:
endpoint checks, factor-five local bracketing, then 64 logarithmic bisections
retaining the best probability residual. The smaller tail decreases with df for
nonzero t, so this interface needs no user-supplied multiple-root bracket.
All inversions undergo smaller-tail forward verification at relative tolerance
1e-7 plus 32 minimum subnormals. Numerical identification becomes weak near
p=q=1/2; a valid input domain does not imply every inverse is identifiable or
resolvable to high parameter-relative precision.

## Native and independent validation

`tools/reference_dcdflib_t.py` compiles unchanged archived C sources/header and
all 64 unchanged F77 files into separate executables. Fixtures record archive
and source hashes, compiler versions, commands, drivers and empty adaptation
lists. The original sources and executables are not bundled in the package.

Each language contributes 100 ordinary cases: 35 tails, 35 quantiles and 30 df
inversions, all reporting native success. The grid uses t=-10,-2,-0.1,0,0.1,2,10
and df=0.2,1,2,10,40. Five invalid-input cases retain exact native statuses.
Five wider-domain cases per language record overflow, tiny df, large df and tiny t.
Both languages return zero upper tails at t=1e200 with df=1 or df=0.1, although
these tails are representable. Python repairs them rather than copying the
native overflow artifacts.

Independent tests use the df=1 Cauchy identity atan(1/abs(t))/pi, a stable df=2
closed form, standard-library log-gamma normalization for underflowed beta
coordinates, and erfc normal limits at extreme df. Coverage includes t through
1e308, subnormal df, df below the F95 input limit, df searches through 1e10,
quantile boundaries, weakly identified small df, batch alignment, ownership,
empty arrays, invalid probabilities, and overlap with the F95 API.

[Batch measurements](dcdflib-t-benchmark.json) compare one broadcast call with
repeated scalar calls to the same Python implementation, checking matching
answers. They do not measure speedup over native C or Fortran. On the recorded
machine, 64/256-element batches were about 34/81 times faster for tails, 43/93
times for quantiles and 41/81 times for df inversion (median of three runs).

These two legacy entry points are now covered. CDFLIB90 remains partial until
the other legacy distribution and public supporting contracts are resolved.
