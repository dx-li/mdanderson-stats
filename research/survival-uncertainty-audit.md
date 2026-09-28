# Simulated survival confidence limits

## Verified source contract

The locally archived flexsurv 2.3.2 files
`R/summary.flexsurvreg.R` and `R/distributions.R` define the reference
procedure. `cisumm.flexsurvreg` calls `normboot.flexsurvreg`, which draws the
optimized parameters jointly from a multivariate normal distribution using
the fitted full covariance. Fixed parameters stay fixed. Covariate effects
are added on the transformed parameter scale before inverse transformation.

Pointwise lower and upper limits are ordinary R type-7 empirical quantiles
at `(1-confidence)/2` and `1-(1-confidence)/2`. The native code drops NA/NaN
predictions when calculating these quantiles, but does not redraw parameter
samples. Its separately calculated sample SD does not drop missing values.
Finite/infinite predictions are not interchangeable with missing values.
The implementation must state any stricter handling of unrepresentable draws.

These are confidence limits for the survival curve from asymptotic parameter
uncertainty. They are neither simultaneous bands nor prediction intervals for
an individual's event time. Reusing the same joint parameter draws across all
profiles and times preserves dependence between requested predictions.

## Parameter coordinates

The existing Python fit objects retain numerically normalized coordinates and
their full covariance. Conversion to original data units within each family
is affine in the transformed parameters, so normal draws can be generated in
those normalized coordinates without changing the asymptotic distribution.

| Model | Joint Python coordinates | flexsurv correspondence |
| --- | --- | --- |
| Weibull | AFT coefficients, log(sigma) | log(shape) = -log(sigma); log(scale) is the linear predictor |
| Lognormal | AFT coefficients, log(sigma) | meanlog is the linear predictor; sdlog = sigma |
| Log-logistic | AFT coefficients, log(sigma) | log(shape) = -log(sigma); log(scale) is the linear predictor |
| Prentice generalized gamma | mu coefficients, log(sigma), Q | mu, log(sigma), Q with coefficient reordering |
| Stacy generalized gamma | log(scale) coefficients, log(shape), log(k) | the same transformed parameters with reordering |

Log-logistic `sigma` is the logistic scale, not its standard deviation; no
pi/sqrt(3) conversion belongs in this mapping. Covariance transformations must
preserve coefficient/ancillary cross terms, including the log-scale sign change
when expressing Weibull or log-logistic in the native shape coordinate.

Stacy-to-Prentice conversion is nonlinear: `Q=1/sqrt(k)`,
`sigma=1/(shape*sqrt(k))`, and `mu=log(scale)+log(k)/shape`.
Sampling a Gaussian in one family and then treating the converted parameter
distribution as a Gaussian in the other changes the confidence procedure.
Each fitted parameterization must retain its own sampling coordinates.

## Independent numerical reference

`tools/reference_survival_uncertainty.R` calculates pointwise survival limits
and sample SDs for supplied original-coordinate draws using base-R Weibull,
normal, logistic and gamma probability functions. Supplying the same draws
separates prediction/quantile agreement from random-number stream differences.
The Prentice gamma identity is evaluated away from its cancellation-prone
zero-Q limit; existing generalized-gamma native-kernel references separately
cover that limit. This reference does not refit models or require new packages.

Reference generation produced 180 pointwise rows for six cases: Weibull,
lognormal, log-logistic, positive-Q Prentice, negative-Q Prentice and Stacy.
Each case has 64 joint draws, three covariate profiles and ten time points,
including time zero and extreme upper tails. Means/covariances come from the
existing independent R metric fixtures (`*_covariates` cases). Input draws
use NumPy default generators with seeds 600 through 605, standard normal
scores and the covariance Cholesky factor; the 384 supplied draws are saved
in `tests/fixtures/survival-uncertainty-draws.csv`. This construction retains
nonzero parameter cross-covariances. The two Prentice cases' smallest absolute
Q values are .01889 and .05064, above the reference gamma identity's .005 guard.

Reproduce the base-R predictions and type-7 limits without refitting:

```sh
Rscript tools/reference_survival_uncertainty.R \
  tests/fixtures/survival-uncertainty-draws.csv \
  tests/fixtures/survival-uncertainty-summary.csv
```

Reference generation succeeded. Agreement with the new Python uncertainty
interface remains to be checked after its implementation is integrated.
