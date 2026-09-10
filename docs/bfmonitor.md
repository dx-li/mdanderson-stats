# BFMonitor: explicit-shape implementation and coverage gaps

The online [BFMonitor application](https://biostatistics.mdanderson.org/shinyapps/BFMonitor/)
is a distinct catalog entry from the desktop [Bayes Factor Binary](bayes-factor-binary.md).
The live app inspected on September 10, 2026 identifies itself as PID 1027,
version 2.0.1.0, last updated `01/06/2026`. Its linked
[guide](https://biostatistics.mdanderson.org/shinyapps/BFMonitor/Guide.pdf) is version
2.0.0.0. [Source and live-output records](bfmonitor-sources.json) preserve the audit.

## Available numerical workflows

`IMOMBinaryPrior(null_rate, mode, shape)` represents a normalized one-sided iMOM
prior on `(null_rate,1)`, with `k=shape`, `nu=2*k`, and
`tau=(mode-null_rate)**2 * ((2*k+1)/(2*k))**(1/k)`. It provides density, CDF,
quantiles and `log_tau`/`tau` properties. Density is defined as zero outside the
open support; the CDF is zero below the null and one at/above one. These functions
accept arrays. Shapes from 0.1 through 100 are supported; unresolved finite
precision calculations raise an error. Likelihood quadrature is capped at 2,048
unit-width panels to bound memory use for extreme combinations of shape and scale. In particular, `tau` and interior quantiles
must be representable when requested.

`bayes_factor_binary_design` now accepts `imom_shape` and `strict_thresholds`.
The defaults remain one and `True`, preserving the desktop software's behavior.
Set `strict_thresholds=False` for inclusive comparisons, as displayed in the
current online app and used in the paper's simulation section. The guide's
method description instead writes strict inequalities; the choice is explicit.
`design.prior` returns the complete configured prior. Monitoring, exact operating
characteristics, simulation and HTML reports retain this configuration.

```python
import numpy as np
from mdanderson_stats import bayes_factor_binary_design

# Live app default: p0=.2, alternative mode=.4, ESS=5, superiority BF >= 9.
# The displayed prior is k=1.25, nu=2.5, tau=.0524. Supply k explicitly;
# this example does not perform an ESS-to-shape calibration.
design = bayes_factor_binary_design(
    50,
    null_rate=0.2,
    alternative_mode=0.4,
    imom_shape=1.25,
    inferiority_cutoff=0,
    superiority_cutoff=9 / (1 + 9),
    strict_thresholds=False,
)
np.testing.assert_array_equal(design.superiority_min, [6, 7, 9, 11, 12, 14, 15, 17, 18])
np.testing.assert_allclose(design.prior.tau, 0.0524, atol=0.00005, rtol=0)
probabilities = np.array([0.025, 0.5, 0.975])
np.testing.assert_allclose(design.prior.cdf(design.prior.quantile(probabilities)), probabilities)
oc = design.operating_characteristics([0.2, 0.4])
```

For a superiority Bayes-factor cutoff `B10`, use posterior cutoff
`B10/(1+B10)`; for a futility cutoff `B01`, use `1/(1+B01)` as the inferiority
cutoff. These conversions assume equal prior model odds. Zero inferiority and
one superiority cutoffs disable their respective rules. The Python design retains
three final conclusions; one-sided final-decision reporting in the native app
still needs a separate audit.

## Evidence and remaining work

All nine live default superiority boundaries agree with the explicit-shape
implementation. The exact transformed prior calculation uses
`t=k*log((theta-null_rate)/sqrt(tau))`; in this coordinate its density is
`2*exp(-2*t-exp(-2*t)+c)`, where `c=(tau/(1-null_rate)**2)**k`.
This reuses the stable log-space likelihood integration and backward recursion
of the desktop implementation. Ten focused tests cover both implementations,
including independent original-coordinate integration at shapes 0.5, 1.25, 2.04
and 5, prior normalization and quantile inversion, the live default boundaries,
and exactly represented equality cutoffs.

**Catalog status remains partial.** The app's ESS input is not interchangeable
with shape. Zhou, Lin and Lee's [2021 paper](https://doi.org/10.1002/pst.2139)
uses Morita's information-based ESS. Its published example assigns k=1.24 to
ESS=5 and k=2.04 to ESS=20 for null .2 and alternative mode .4, whereas the
current app displays k=1.25 for ESS=5. The guide does not specify the complete
calibration algorithm, rounding/search conventions or all reference-prior
constants. The available BioC text omits the mathematical expressions needed
to resolve those details. Neither the rounded display nor one matching boundary
table proves an arbitrary ESS conversion.

Remaining work is to verify and implement that calibration, audit the one-sided
final-decision convention, and cover the online protocol templates and CSV/Excel/
PDF/Word export workflows. The Python HTML report is available but is not a claim
of parity with those native templates. No native random-stream parity is claimed.
