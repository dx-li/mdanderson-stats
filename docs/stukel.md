# STUKEL generalized logistic models

Catalog entry 58 is partial. Implemented: the two-shape inverse link, transformed
log odds, prediction from supplied coefficients, and grouped-binomial likelihood,
gradient and observed Hessian, bounded regression estimation, covariance and
dispersion estimates for all six parameter families. Model scanning and
plotting/reporting remain pending.

The model is described in Thérèse A. Stukel, “Generalized Logistic Models,”
JASA 83(402), 426–431 (1988),
https://doi.org/10.1080/01621459.1988.10478613 .

```python
from mdanderson_stats import stukel_probability, predict_stukel

probabilities = stukel_probability([-2, 0, 2], alpha1=-0.5, alpha2=0.25)
predicted = predict_stukel([[1, 2], [3, 4]], [0.5, 1, -1], -0.5, 0.25)
```

For m=abs(eta), select a=alpha1 when eta>=0 and alpha2 otherwise. The magnitude
of the transformed log odds is expm1(a*m)/a for a>0, log1p(-a*m)/(-a) for a<0,
and m for a=0. Restore eta's sign, then apply the ordinary logistic inverse.
`stukel_log_odds` exposes the transformation; `stukel_probability` returns
probabilities. Inputs and shape parameters broadcast and must be finite. Zero
shapes recover ordinary logistic regression; eta=0 always gives probability 0.5.

`predict_stukel` accepts observations by covariates in the last two axes, with
optional leading batches. beta is one coefficient vector; by default its first
coefficient is an intercept. With intercept=False it contains only covariate
coefficients. Prediction shape parameters are scalars. No fit is inferred from
these coefficients and no uncertainty is calculated.

The archived S predict.glr script repeats `alpha1>0` in an else-if branch, skipping
the alpha1<0 transform. The default here follows the original Fortran FGH fitting
formula. `legacy_prediction=True` explicitly reproduces the skipped branch.

Small products use expm1/log1p ratios to retain the logistic limit. Large products
are evaluated in log space, including cases where exp(a*m) overflows but the
transformed log odds remain representable. Unrepresentably large log odds return
signed infinity and probabilities saturate to zero or one. This extends the
original optimizer's overflow cutoff; it is not a convergence policy for fitting.

`tools/reference_stukel.py` compiles original all.f privately and records 15 FGH
cases (165 transformed predictors), with source/archive hashes and compiler
provenance, in tests/fixtures/stukel.json. Tests compare those values and verify
90-digit Decimal calculations, broadcasting, symmetry, monotonicity, prediction,
endpoint saturation, invalid inputs, and the explicitly requested legacy bug.
Original Fortran uses truncated Taylor series near zero; comparison tolerances
account for that approximation. No original Fortran is bundled in the package.

## Likelihood and derivatives

```python
from mdanderson_stats import stukel_objective

result = stukel_objective([[1, -1], [1, 1]], [2, 8], [10, 10], [0, 1, 0.2], family=3)
print(result.negative_log_likelihood, result.gradient, result.hessian)
```

The design matrix must include its intercept column explicitly. Successes and
trials are integer count vectors matching design rows, with positive trials.
Coefficients begin with one beta per design column and then the free shapes:

| family | Shapes | Extra coefficients |
| --- | --- | --- |
| 0 | Both fixed by fixed_alpha | None |
| 1 | alpha1 free; alpha2 fixed | alpha1 |
| 2 | alpha1 fixed; alpha2 free | alpha2 |
| 3 | alpha1 = alpha2 | Shared shape |
| 4 | alpha2 = -alpha1 | alpha1 |
| 5 | Both free | alpha1, alpha2 |

`fixed_alpha` defaults to (0,0); only fixed entries are used. The objective omits
the binomial combinatorial constants, matching original FGH. The result includes
log odds and probabilities. Derivatives are with respect to the supplied
coefficient vector. The Hessian is the observed Hessian, including residual-times-
link-curvature terms; it need not be positive definite for generalized families.

The original family-4 FGH sets alpha2=-alpha1 but omits the corresponding signs
in the negative-tail shape score and mixed derivatives. This implementation applies
the chain rule. Its likelihood matches FGH; its derivatives match the appropriate
linear projection of the unrestricted family-5 derivatives and independent finite
differences. The incorrect family-4 derivatives are not used as an optimizer input.

At alpha=0 the source chooses the nonnegative-shape branch. The link is continuously
first differentiable in shape, but its two shape-side second derivatives differ.
At eta=0 the nonnegative-predictor branch is chosen; predictor curvature can also
be discontinuous there. The returned Hessian follows these branch conventions;
it is not a claim that a two-sided Hessian exists at those boundaries. Numerical
Hessian checks must respect the branches rather than differentiate across them.

Small shape products use differentiated series with 16 terms, improving the
original six-term approximation. Likelihood and scores use stable log probabilities
and complementary tails to avoid cancellation for saturated correct predictions.
Nonfinite objective/derivatives raise an explicit numerical-range error. No fit,
convergence, or parameter uncertainty is implied by this evaluation function.

`tools/reference_stukel_objective.py` records 24 native FGH cases spanning all six
families, negative/zero/positive/near-zero shapes, and the corrected family-4
projection. Tests compare objective, gradient and Hessian; independently difference
the objective and scores away from branch boundaries; recover ordinary logistic
information; and check saturated tails and invalid inputs.

## Regression fitting

```python
from mdanderson_stats import fit_stukel

fit = fit_stukel(x, successes, trials, family=5)
print(fit.coefficients, fit.alpha, fit.deviance, fit.dispersion)
print(fit.standard_errors, fit.inference_message)
predicted = fit.predict(new_x)
```

`fit_stukel` accepts an observations-by-covariates matrix or a one-dimensional
single covariate, and adds an intercept unless intercept=False. It requires a
full-rank design and positive residual degrees of freedom (rows minus number
of estimated coefficients). Missing/nonfinite data and invalid counts raise;
rows are not silently deleted as in the original S wrapper.

All six families use the parameter order described above. Free shapes are bounded
by +/-shape_bound (default 10); beta coefficients by +/-1e20, matching minimize.S.
The default start is beta=0 with start_alpha=(0,0). `initial` can provide the whole
coefficient vector. Fixed shape parameters are supplied separately in fixed_alpha.

L-BFGS-B with analytic gradients replaces David Gay DRMNHB. Columns are scaled
by their maximum absolute values (at least one), with bounds transformed exactly;
the optimizer uses negative log likelihood per trial. Defaults are relative
function tolerance 1e-12, projected-gradient tolerance 1e-8 in scaled coordinates,
and 2,000 iterations. Either convergence criterion may stop optimization. Solver
paths and local optima can differ, especially after correcting family-4 derivatives.
Convergence does not establish a global maximum. A numerical trial exceeding the
objective's range fails explicitly rather than supplying a fabricated derivative.

A bounded linear program checks complete/quasi-complete separation before fitting:
pure response groups admit nonnegative signed margins while mixed groups require
zero margins. A positive feasible margin means no finite interior maximum. This
check has numerical tolerances; it is not proof of global identification for all
shape families. Rank deficiency, separation, iteration failure, and numerical
failure are explicit errors.

The result includes coefficients, the two effective shapes, the full objective
and derivatives, deviance, residual_df, dispersion, covariance, standard_errors,
active_bounds, iterations, and inference_message. `null_deviance` and
`null_dispersion` describe the intercept-only empirical-frequency model from the
original raw-data summary; its degrees of freedom are rows minus one.

Default scale="pearson" estimates dispersion as Pearson chi-square/residual_df,
without flooring at one. scale="fixed" uses dispersion one. Deviance is computed
from stable log probabilities. Covariance is dispersion times the inverse observed
Hessian, obtained by a linear solve after checking positive definiteness. At active
bounds or detected nonregular zero branches, or when curvature is not positive
definite, covariance and standard_errors are None and inference_message explains
why. Such a fit is not silently presented as an ordinary interior Wald estimate.
These local approximations do not replace model checking or account for selecting
a family from the data.

The original minimize.S uses untransformed eta for family-0 fitted probabilities,
even when fixed shapes are nonzero. This port consistently uses the fitted
transformed link. It also avoids the source's substitution of a unit denominator
for a rounded zero binomial variance in Pearson calculations.

`tools/reference_stukel_fit.py` runs the archived FGH/DRMNHB optimizer on both
beetles and Warsaw data for all families. Only machine constants are replaced for
the current IEEE binary64 platform. Ten successful native fits agree in likelihood
within 1e-6. The two native family-4 fits report false convergence; their coefficients
are not treated as successful-fit oracles. The corrected fits improve their
likelihoods and pass stationarity and curvature checks. Tests additionally cover
closed-form intercept-only covariance/dispersion, fixed shapes, prediction,
separation, active bounds, rank deficiency, and failed convergence.
