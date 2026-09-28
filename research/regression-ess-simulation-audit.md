# Regression ESS simulation source and references

BayesESS 0.1.19 at `4bbf4df3789912b967774e8ff5c3a2d6d5646cdd` includes
`ESS_RegressionCalc` and its compiled-helper variant in `R/internal.R`.
The file identifies Satoshi Morita's regression ESS program, version 1.0 of
11 August 2009, downloaded from the institutional software site in 2017.
This supplies a primary implementation for the regression workflow exposed by
BayesESS, in addition to the previously implemented explicit-distribution
calculations. File hashes are in
[conjugate-ess-sources.json](../docs/conjugate-ess-sources.json).

## Statistical contract

The native program supplies an intercept and up to ten independent
Uniform(-1,1) covariates. Coefficient priors are independent normals with
mean/variance parameters or gammas with shape/rate parameters. The normal
regression additionally has a gamma prior for residual precision. The Python
`ParameterDistribution` uses gamma shape/scale, so the native rate must be
inverted at the interface.

Prior and epsilon-prior log-density curvatures are evaluated at the original
prior means. Epsilon priors preserve those means and inflate variances by
10,000. Their curvature difference is `(1-1/10000)/prior_variance`; calculating
that difference directly avoids subtracting nearly equal negative gamma
curvatures when shape is below one.

Each replicate accumulates the observation-information contributions over
patient positions. The program averages these cumulative paths over replicates,
then interpolates the crossing of expected likelihood information with the
prior-information difference. Logistic contributions are `x_j^2*p*(1-p)`
at the prior-mean predictor. Normal coefficient contributions are
`E[tau]*x_j^2`; the precision contribution is `1/(2*E[tau]^2)`.
No generated response enters these formulas.

The native code computes whole-vector and two selected-subvector ESS values,
although its returned list contains only the two subvectors. If the crossing
is beyond the supplied maximum patient count, it returns NA. Equal closest
indices can cause its scalar branch to fail. The Python API should return an
explicit not-reached result when no crossing exists and define plateau/tie
handling, rather than extrapolate or reproduce that crash.

## Executed original-R references

`tools/reference_regression_ess_simulation.R` sources the original function
without compiled helpers or package installation. A replacement `runif`
supplies an archived uniform covariate sequence in the original call order,
including the unused covariate positions. An exit trace records the function's
computed curves and whole-vector result; its statistical arithmetic is unchanged.

The fixture contains 16 replicates of 32 patients with two covariates and
four prior/model cases: normal, logistic, zero-mean logistic and a logistic
prior whose information target cannot be reached within 32 patients. It
records 12 whole/subvector ESS results and 396 posterior-information path rows.
The three not-reached results are NA. Generation completed in .23 seconds.
The Python implementation and comparison are pending.

An additional source limitation was observed while preparing the reference:
an intercept mean of 40 makes the original logistic calculation round `p` to
one and `p-p^2` to zero. The resulting tied closest indices crash its `if`
condition. The unreached reference uses an intercept of 10 so the native
calculation can return NA normally. Python must calculate logistic information
from log tails and keep this numerical failure separate from a valid
not-reached conclusion. Native RNG equivalence is not a requirement of these
supplied-input comparisons.
