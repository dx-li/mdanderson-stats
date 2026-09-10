# DRDIST distribution calculator

Catalog entry 31 is a seven-operation distribution calculator. `drdist` preserves
its menu numbers and tail conventions while reusing the package's shared
CDFLIB/CUMNOR numerical routines. Numerical inputs broadcast as NumPy arrays;
returned arrays are immutable. No new numerical dependency is introduced.

| Menu | Operation name | Value argument | Parameters | Result |
|---|---|---|---|---|
| 1 | `inverse_f` | Upper-tail probability | `numerator_df`, `denominator_df` | F quantile |
| 2 | `f` | F value | `numerator_df`, `denominator_df` | Upper-tail probability |
| 3 | `inverse_normal` | Lower-tail probability | `mean`, `sd` | Normal quantile |
| 4 | `normal` | Observation | `mean`, `sd` | Lower-tail probability |
| 5 | `inverse_t` | Lower-tail probability | `df` | Student's t quantile |
| 6 | `t` | t value | `df` | Lower-tail probability |
| 7 | `chi_square` | Chi-square value | `df` | Upper-tail probability |

```python
import numpy as np
from mdanderson_stats import drdist

critical = drdist(1, 0.05, numerator_df=5, denominator_df=10)
assert abs(float(critical) - 3.325834530413012) < 1e-10
assert abs(float(drdist("f", critical, numerator_df=5, denominator_df=10)) - 0.05) < 1e-12
normal_quantiles = drdist("inverse_normal", [0.025, 0.5, 0.975], mean=2, sd=3)
assert np.allclose(drdist("normal", normal_quantiles, mean=2, sd=3), [0.025, 0.5, 0.975])
```

Normal defaults are mean zero and standard deviation one. Standard deviations
must be positive. Degrees of freedom are required for the other distributions;
parameters that do not belong to an operation are rejected. Real degrees of
freedom in [1e-3,1e10] are supported, extending the original F/t integer inputs.
F and chi-square coordinates lie in [0,1e100], t coordinates in [-1e100,1e100],
following the shared CDFLIB kernels. Input values, means and standard deviations
must be finite; inverse probabilities must lie in [0,1].

Inverse endpoint probabilities return distribution limits: inverse normal/t
return negative/positive infinity at zero/one; inverse F returns positive
infinity at zero and zero at one. The original normal inverse's ±1e38 endpoint
sentinels are not retained. Interior quantiles must lie within the shared
kernels' supported finite ranges. Unrepresentable normal transformations raise
an error. Normal standardization and quantile transformation have scaled
fallback expressions to avoid intermediate overflow when the final value is
representable.

The driver calls its inverse-t input a p-value, but the underlying `studin`
function inverts the **lower** tail: 0.975 with 10 df gives +2.22814, while 0.025
gives −2.22814. This differs from inverse F, which inverts the upper tail.
For normal/t complementary-tail inversion without rounding `1 - tiny_probability`,
use the existing `cdf_normal` or `cdf_t` interfaces with `ccum=` directly.

## Source validation and intentional improvements

The original `drdist.f` was compiled with gfortran after removing only the
interactive main program. Tests compare all seven menu branches on ordinary
values at relative 2e-5/absolute 2e-6 tolerance, appropriate for the archive's
single-precision calculations. For the chi-square branch that comparison uses
x=10, df=2, where the source evaluates the exponential tail directly.

The archive's `overfl` placeholder always reports an overflow. Consequently,
other chi-square cases can be forced onto a cube-root approximation. At x=10,
df=5 the original upper tail is 0.07440710067749023, whereas Python and an
independent R `pchisq` calculation give 0.0752352461465122. The Python result
uses the accurate gamma-distribution identity and does not emulate that stub.

F and chi-square upper tails are evaluated directly, avoiding the original
`1 - CDF` cancellation. The normal calculation retains small tails beyond the
legacy hard cutoff at −13.27. Tests check tail direction, inverse round trips,
probabilities as small as 1e-300, broadcasting, endpoint limits, and normal
unit transformations near the float64 range limit.

## Bundled UTHLIB helper calculations

The source file also bundles supporting distribution functions beyond the seven
menu options. Most already have public equivalents in this package:

| Native helper | Existing Python calculation |
|---|---|
| `bin(n,p,m)` | `cum_binomial(m, n, p)` |
| `cdfvec(n,p)` | `cum_binomial(np.arange(n + 1), n, p)` |
| `pois(n,mu)` | `cum_poisson(n, mu)` |
| `betinc(x,a,b)` | `cum_beta(x, a, b)` |
| `gamin(x,a)` | `cum_gamma(x, a)` with unit rate |
| `zap(a)`, `zip(a)` | `gamma(a)`, `log_gamma(a)` |
| `zot(n)`, `fctrlg(n)` | `log_gamma(n + 1)` |
| `hytric(k,N,draws,successes)` | Lower-tail `fisher_exact` on the corresponding fixed-margin table, for k within its support |

These are mappings to the existing mathematical operations, not reproductions
of UTHLIB approximation branches or common-block storage. In particular, the
binomial vector has n+1 entries; the source's fixed 1,000-entry common block
cannot safely hold the n=1,000 case. The public APIs have their own documented
domains. DRDIST coverage refers to its complete seven-operation user interface.

Sources: [MD Anderson DRDIST](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/31)
and [original archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/DRDIST/DRDIST%20%20%20%20%20_V1.tar.gz).
Hashes are in `drdist-sources.json`. Original code remains in ignored research
storage and is not redistributed.
