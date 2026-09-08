# STUKEL generalized logistic models

Catalog entry 58 is partial. Implemented: the two-shape inverse link, transformed
log odds, and prediction from supplied coefficients. Regression estimation,
analytic gradients/Hessians, covariance and dispersion estimates, family-specific
constraints, model scanning, and plotting/reporting remain pending.

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
