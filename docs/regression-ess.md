# ESS Regression

These functions implement the regression prior effective sample size (ESS)
calculation of Morita, Thall and Müller, *Biometrics* 64:595–602 (2008),
[doi:10.1111/j.1541-0420.2007.00888.x](https://doi.org/10.1111/j.1541-0420.2007.00888.x).
Sources are the [public author manuscript](https://web.ma.utexas.edu/users/pmueller/pap/MTM08.pdf),
[archived publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC3081791/), and
[MD Anderson method guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/ESSRegression/ESS_Method_Readme.pdf).
[Provenance](regression-ess-sources.json) records retrieved documents.

## Supported models and parameter conventions

The direct regression interfaces support independent normal and gamma coefficient
priors, with up to 11 coefficients (an intercept and at most 10 covariates).
They reuse `ParameterDistribution`: normal parameters are **mean and variance**;
gamma parameters are **shape and scale**. A gamma rate in the paper must therefore
be inverted before passing it as the second parameter. No intercept, centering,
standardization or covariate distribution is inserted automatically.
The simulation interface below supplies the original program's intercept and
independent uniform covariates explicitly.

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

## Original covariate simulation workflow

`simulate_regression_ess(model, coefficient_priors, ...)` implements the
original 2009 calculator recovered in the author-maintained BayesESS source.
It inserts an intercept and generates independent Uniform(-1,1) covariates,
accumulates information by patient, averages the paths over replicates and
interpolates the first crossing of the prior-information target. Model is
`"normal"` or `"logistic"`; a gamma `precision_prior` is required for normal
regression. Coefficient priors include the intercept; precision is a separate
argument and the final result component.

The following Python input replaces the original editable R example script.
All three population-expectation ESS values are 1.9998, rounded to two in the
guide. Finite simulated covariate paths vary around that expectation.

```python
from mdanderson_stats import simulate_regression_ess

priors = [Prior("normal", 0, 1)] * 4  # intercept plus three covariates
simulation = simulate_regression_ess(
    "normal", priors, precision_prior=Prior("gamma", 1, 1),
    max_patients=10, replicates=256, rng=np.random.default_rng(80),
)
print(simulation.whole_model.estimate)
print(simulation.crossing([0, 1, 2, 3]).estimate)  # all coefficients
np.testing.assert_allclose(simulation.crossing([4]).estimate, 1.9998)

population = normal_regression_ess(priors, [1, 1/3, 1/3, 1/3], Prior("gamma", 1, 1))
np.testing.assert_allclose(population.ess, 1.9998)
np.testing.assert_allclose(population.subvector_ess([0, 1, 2, 3]), 1.9998)
```

`crossing(indices=None)` selects the whole parameter vector or any nonempty
set of distinct zero-based indices. It returns status `"exact"`,
`"interpolated"` or `"not_reached"`, an estimate (or `None`), bracketing
patient counts and their log information. The first exact point on a plateau
is used. If the target is beyond `max_patients`, increase that bound; the
function does not extrapolate or return the native source's tie-related crash.

For reproducible cross-language comparisons, pass `covariate_draws` with shape
`(replicates, max_patients, number_of_nonintercept_covariates)`. Values must be
finite and within [-1,1]; explicit dimensions must match. Supplied covariates
replace random generation and are retained read-only. The native program draws
unused covariates as well, so equal R and Python seeds do not imply identical
paths. The original algorithm does not need simulated outcomes.

Results retain log mean-cumulative information for patient counts zero through
the cap, log prior-information gains and signed log prior/epsilon curvatures.
Crossing calculations use these quantities directly. Ordinary
`mean_cumulative_information`, `prior_information` and
`epsilon_prior_information` are read-only convenience properties: overflow
raises on access, while sufficiently small values can underflow to zero.
The retained log values preserve such information. This avoids subtracting
nearly equal negative gamma curvatures and avoids saturated logistic tails.

Patient and replicate counts are each limited to 10,000. Preflight limits
also cap covariate draws at two million cells, a cumulative path at 200,000
cells and replicate × patient × parameter work at twenty million. Replicates
are processed sequentially without retaining per-replicate information paths.

The [source and numerical audit](../research/regression-ess-simulation-audit.md)
records agreement with 396 original-R information rows and 12 ESS outcomes,
including three not-reached results. **Catalog entry 80 is implemented:** the
normal/logistic calculations and all six documented input choices are exposed
through the Python interfaces. Native console formatting and arbitrary R
script execution are replaced by Python arguments and results; original
source files are not redistributed with the package.
