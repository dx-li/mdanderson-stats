# Bayesian effective sample size: conjugate models

`conjugate_prior_ess` covers the seven conjugate-model choices in catalog entry
**154**, the [Bayesian Effective Sample Size Calculator](https://biostatistics.mdanderson.org/shinyapps/BayesESS/).
The app identifies Jaejoon Song, Satoshi Morita, J. Jack Lee and Ying-Wei Kuo,
PID 1077, version V0.0.5.0, updated March 25, 2026.

Its Sample R Code tab links to [github-js/BayesESS](https://github.com/github-js/BayesESS).
The reference inspected is package 0.1.19 at commit
`4bbf4df3789912b967774e8ff5c3a2d6d5646cdd`. The original `ess.R` and `internal.R`
functions were sourced in R and executed directly for these conjugate cases;
no native binaries or compiled package installation were needed.
The [support document](https://biostatistics.mdanderson.org/shinyapps/BayesESS/BayesESS_Support.pdf)
derives conjugate prior information and the more general epsilon-information
approach. [Provenance](conjugate-ess-sources.json) records the inspected sources.
Original source files, binaries and PDFs are not redistributed.

## Inputs and definitions

Parameters occupy the last axis; leading axes represent independent parameter
settings. All parameters are finite and positive, except the normal prior mean,
which can be any finite number. Results are read-only NumPy arrays; a single
parameter vector returns a zero-dimensional array.

| Model name | Parameters, in order | Information-based ESS |
|---|---|---|
| `beta_binomial` | alpha, beta | alpha + beta |
| `gamma_exponential` | shape, rate | shape |
| `gamma_poisson` | shape, rate | rate |
| `dirichlet_multinomial` | alpha_1, ..., alpha_K | sum of alpha values |
| `normal_normal` | prior mean, known sampling variance, prior mean variance | sampling variance / prior variance |
| `inverse_chi_squared_normal` | degrees of freedom, scale variance | degrees of freedom |
| `inverse_gamma_normal` | shape, scale | 2 × shape |

The last two models assume the normal sampling mean is **known** and the prior
is for its variance. The inverse-gamma density is proportional to
`v**(-shape-1) * exp(-scale/v)`; gamma priors use **rate**, not scale.
Gamma–Poisson assumes unit exposure per count observation. These are prior
information quantities, not MCMC effective sample sizes.

The normal model accepts prior variance directly, rather than requiring callers
to supply an already known ESS as in the native `n0` input. Positive fractional
Dirichlet concentrations and normal ESS are supported; the R wrapper restricts
those inputs to integers, although the conjugate formulas do not require that.

## Explicit gamma–exponential discrepancy

The R package and live app return the gamma **rate** parameter for this model.
This matches section 3.1.1.2 of the support document, but conflicts with its
information-based derivation in section 3.2.2.2 and summary table, which give the
**shape** parameter. Under exponential sampling, the gamma shape increases by
the number of observations, while the rate increases by their total follow-up.
Thus rate represents prior exposure in time units, not an equivalent observation
count. Rescaling time changes this native output.

The default `convention="information"` returns shape. Set `convention="native"`
to reproduce the R/app rate output explicitly. The convention does not change
any other supported model. This distinction is intentional and is not a claim
that all ESS definitions are interchangeable.

```python
import numpy as np
from mdanderson_stats import conjugate_prior_ess

np.testing.assert_array_equal(conjugate_prior_ess("beta_binomial", [[1, 1], [2, 3]]), [2, 5])
assert conjugate_prior_ess("gamma_exponential", [2, 7]) == 2
assert conjugate_prior_ess("gamma_exponential", [2, 7], convention="native") == 7
assert conjugate_prior_ess("normal_normal", [-2, 12, 3]) == 4
```

Calculations avoid simulation and operate directly on NumPy arrays. At most two
million parameter values are accepted. An ESS outside positive float64 range
raises `ArithmeticError`, rather than returning an infinite or zero information
value. Tests compare all seven native-convention results against the unmodified
R functions, verify conjugate information updates and time-unit behavior, and
check broadcasting, fractional concentrations and extreme variance scales.

**Catalog status is partial.** Unknown-mean variance models, the app's regression
and epsilon-information simulation workflows, CRM, TITE-CRM, nonconjugate survival
models, plots and native reports remain pending. The package already has separate
[regression ESS functions](regression-ess.md); parity with this app's regression
simulation choices has not been established.
