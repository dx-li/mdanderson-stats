# ESS Regression

These functions implement the regression prior effective sample size (ESS)
calculation of Morita, Thall and Müller, *Biometrics* 64:595–602 (2008),
[doi:10.1111/j.1541-0420.2007.00888.x](https://doi.org/10.1111/j.1541-0420.2007.00888.x).
Sources are the [public author manuscript](https://web.ma.utexas.edu/users/pmueller/pap/MTM08.pdf),
[archived publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC3081791/), and
[MD Anderson method guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ESSRegression/ESS_Method_Readme.pdf).
[Provenance](regression-ess-sources.json) records retrieved documents.

## Supported models and parameter conventions

Both regression interfaces support independent normal and gamma coefficient
priors, with up to 11 coefficients (an intercept and at most 10 covariates).
They reuse `ParameterDistribution`: normal parameters are **mean and variance**;
gamma parameters are **shape and scale**. A gamma rate in the paper must therefore
be inverted before passing it as the second parameter. No intercept, centering,
standardization or covariate distribution is inserted automatically.

`logistic_regression_ess(coefficient_priors, covariate_support, probabilities=None)`
accepts a finite covariate distribution: rows are support points and columns
correspond to coefficients. Supply a column of ones for an intercept. Weights
are normalized; absent weights specify a uniform distribution on the rows.
They describe the distribution of one hypothetical observation, not the number
of observed patients. The expectation is exact for that finite distribution.
For continuous covariates, supplied Monte Carlo samples or quadrature support
introduce an approximation that should be assessed separately.

`normal_regression_ess(coefficient_priors, covariate_second_moments, precision_prior)`
uses a gamma prior for the residual precision. Supply E[X_j²] for every design
column, including one for an intercept. Precision is the final result component.
Only these second moments are needed; synthetic outcomes need not be generated.

## Calculation and numerical behavior

The method matches the prior's log-density curvature to expected posterior
curvature from an epsilon-information prior with the same means and inflated
variances. `variance_inflation=10000` follows the manuscript's examples; it can
be changed to assess sensitivity. For each supported independent prior, the
curvature difference at its mean is `(1 - 1/c) / variance`.

Expected per-observation diagonal likelihood information is:

- Logistic coefficient j: E[X_j² p(X)(1-p(X))], with p evaluated at the prior mean coefficients.
- Normal coefficient j: mean precision × E[X_j²].
- Normal precision: 1 / (2 × mean precision²).

ESS for any chosen subvector is the ratio of summed prior-information differences
to summed observation information. This solves the paper's interpolated curvature
matching directly, avoiding Monte Carlo outcome generation and an integer search.
The paper defines distance as the absolute curvature difference; its positive
zero is the minimizer for these supported models.

The result exposes `ess`, `component_ess`, and `subvector_ess(indices)`, with
zero-based indices. Logarithmic equivalents and both information arrays are
retained. Ratios use log sums, logistic tails avoid saturation, and gamma curvature
differences are calculated without subtracting nearly equal negative quantities.
Zero observation information produces infinite ESS. A finite log ESS outside the
ordinary float64 range raises an error when converted, preserving the log result.

The overall trace ratio depends on parameterization and covariate units; it is
not generally invariant to rescaling individual coefficients. Subvector ESS is
not the sum or average of its component ESS values. These functions measure prior
information relative to a likelihood, not MCMC effective sample size.

```python
import numpy as np
from mdanderson_stats import ParameterDistribution as Prior
from mdanderson_stats import logistic_regression_ess, normal_regression_ess

log_dose = np.log(np.arange(1, 7) * 100)
support = np.column_stack([np.ones(6), log_dose - log_dose.mean()])
result = logistic_regression_ess([Prior("normal", -0.1313, 4), Prior("normal", 2.398, 4)], support)
np.testing.assert_allclose(result.ess, 2.3184809478983954)
np.testing.assert_allclose(result.component_ess, [1.41915284, 6.32959271])

normal = normal_regression_ess(
    [Prior("normal", 0, 1000), Prior("normal", 0, 1000)],
    [1, 1],
    Prior("gamma", 0.001, 1000),  # shape .001, rate .001
)
np.testing.assert_allclose(normal.subvector_ess([0, 1]), 0.0009999)
np.testing.assert_allclose(normal.component_ess[-1], 0.0019998)
```

Four focused tests reproduce the logistic Table 3 and normal-regression examples,
compare curvatures with independent finite differences, check gamma shape below
one, and exercise extreme logits, huge weights, and uninformative design columns.
The normal example's variance-inflation correction explains the slight difference
from the paper's rounded .001 and .002 results.

## Remaining coverage

**Catalog entry 80 is partial.** Native R input parsing, covariate-generation defaults,
example input fixtures and console workflows remain unverified because the source
archive has not been retrieved. The guide's example reporting ESS 2 does not include
its inputs, so it is not claimed as a reproduced fixture. These interfaces replace
the documented simulation search with explicit covariate-distribution calculations;
they do not claim native RNG or input-format parity. Original files are not
redistributed with the package.
