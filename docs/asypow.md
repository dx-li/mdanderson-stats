# ASYPOW asymptotic power

MD Anderson catalog entry 33, ASYPOW, calculates asymptotic power for nonlinear
models. This port currently provides the shared information-matrix calculation
and independent-group binomial, Poisson and exponential-survival information,
including regression, ordinal, multinomial and general design-matrix models.
Remaining S-plus SMO models, mixed constraints and complete native workflow coverage remain
pending; the catalog status is **partial**.

The original S-plus 2.1 archive has a broader scope than the later R archive.
In particular, it supplies separate SMO and multinomial routines documented in
the 1997 paper. Coverage is assessed against that broader original scope, not
just the R package's index. LR information-matrix calculations and binomial,
Poisson, multinomial and ordinal SMO expected-log-likelihood methods are available. SMO supports both conventions
for subtracting degrees of freedom, as described below.

```python
import numpy as np
from mdanderson_stats import asypow_group_information, asypow_information

p = [0.2, 0.4]
information = asypow_group_information(p, model="binomial", group_size=[1, 1])
design = asypow_information(p, information, [1, -1])  # H0: p1 = p2
assert abs(float(design.power(100)) - 0.608779484645457) < 1e-12
required = design.sample_size(power=0.8, significance=0.05)
assert abs(required - 156.977210186524) < 1e-9
assert int(np.ceil(required)) == 157
```

## Information and null constraints

`asypow_information(parameters, information, contrasts, null_values=0,
information_observations=1)` represents the linear null hypothesis
`C @ theta = b`. A single row can be a vector. `information` is the positive
definite Fisher information at the alternative parameter vector, for the
specified number of observations. Up to 500 parameters are supported. Contrast
rows must be nonzero and linearly independent; ambiguous numerical rank is
rejected. Fixed parameter values and equality constraints are special cases.

Writing `V = C I⁻¹ Cᵀ`, the noncentrality per observation is
`w = (C theta - b)ᵀ V⁻¹ (C theta - b) / information_observations`.
The degrees of freedom equal the number of independent contrasts. The returned
`null_parameters` are the information-metric projection onto the null:
`theta - I⁻¹ Cᵀ V⁻¹ (C theta - b)`. This is a local quadratic approximation,
not an exact constrained MLE for an arbitrary nonlinear likelihood.

The implementation uses Cholesky solves and singular values of the transformed
contrasts instead of forming matrix inverses. Contrast rows are rescaled before
solving. Noncentrality is normalized before squaring to preserve small
representable results. Ill-conditioned information can still make a local
approximation unreliable; the rank checks do not certify the model itself.

`asypow_group_information(parameters, model=..., group_size=..., duration=...)`
returns information for **one subject distributed across the groups**. Relative
positive allocations are normalized. Parameters and allocations are evaluated
in log space where useful, so large common allocation scales do not overflow.

| Model | Parameter | Diagonal information for group weight g |
| --- | --- | --- |
| `binomial` | Event probability p in (0,1) | g/[p(1-p)] |
| `poisson` | Mean lambda > 0 | g/lambda |
| `exponential` | Event rate w > 0 | g * Pr(event observed)/w² |

Exponential survival requires positive `duration`. Entry is uniform over a study
of that duration, with administrative censoring at its end and no other dropout.
Its observed-event probability is `1 - (1-exp(-w*duration))/(w*duration)`.
The port reuses the package's stable exponential event-probability calculation
and a log-domain small-rate limit when the probability itself underflows.
Unrepresentable information raises an error rather than returning zero/infinity.

## Regression designs

`asypow_regression_information(parameters, covariates, family="logistic",
observations=1, group_size=1, duration=None)` computes per-observation information
for the native linear and quadratic regression families.

```python
from mdanderson_stats import asypow_regression_information, asypow_information

theta = [-0.7, 0.3, -0.1]  # intercept, slope, quadratic coefficient
information = asypow_regression_information(theta, [-2, -1, 0, 1, 2])
design = asypow_information(theta, information, [0, 1, 0])  # zero slope
n = design.sample_size()
assert abs(float(design.power(n)) - 0.8) < 1e-12
```

One parameter vector defines one group; a matrix defines independent groups
by row. Each row has 2 coefficients for a linear predictor or 3 for a quadratic
predictor. A covariate vector is shared across groups, while a matrix supplies
one row per group. `observations` broadcasts over the covariate matrix and may
be zero at unused points; each group must retain a positive allocation.
`group_size` is a positive scalar or vector with one value per group.

Point weights are proportional to `group_size[g] * observations[g,j]`, normalized
across **all** groups and points. Thus groups with different total observations
need not receive the same final allocation even when `group_size` is equal.
Information is block diagonal, ordered by group and then coefficient. Pass
parameters flattened in that order to `asypow_information` for cross-group
contrasts. At most 500 coefficients and 1,000,000 covariate entries are supported.

For design vector v=(1,x) or (1,x,x²), each point contributes a scalar information
weight times `v vᵀ`:

| Family | Predictor eta | Information weight |
| --- | --- | --- |
| `logistic` | log odds | p(1-p), p=logistic(eta) |
| `cloglog` | log negative-log survival | z²/[exp(z)-1], z=exp(eta) |
| `poisson` | log mean | exp(eta) |
| `exponential` | log event rate | observed-event probability |

Exponential survival uses the same uniform-entry model as the group calculation;
`duration` is required and may be scalar or one value per group. Other families
reject it. For exponential log-rate parameters, the rate-squared transformation
cancels the rate-squared denominator of raw-rate information.

Predictor information and allocation normalization are calculated in log space.
Weighted design vectors are formed before their matrix product, retaining small
information weights that can be offset by large covariates. Zero-allocation
points are skipped before polynomial evaluation. Stable small-rate and saturated
limits replace native cancellation/overflow. A zero or singular information
matrix can legitimately result from an uninformative design or saturated
response; the shared power constructor rejects non-positive-definite information.
An overflowing polynomial or information matrix raises a rescaling error.

## Ordinal outcomes

`asypow_ordinal_information(cumulative, group_size=1)` accepts K-1 cumulative
category probabilities, excluding the final one. Each row is a separate group.
Probabilities must increase strictly inside (0,1), so all K category masses are
positive. The information is tridiagonal within each group: diagonal entries
are `1/p_category[i] + 1/p_category[i+1]`, with off-diagonal entries
`-1/p_category[i+1]`. Group blocks are weighted by normalized allocations.
The matrix is with respect to cumulative probabilities, not raw category masses.
Up to 500 total cumulative-probability parameters are supported.

`asypow_ordinal_regression_information(parameters, covariates, quadratic=False,
link="logistic", observations=1, group_size=1)` adds cumulative-link ordinal
regression. Each parameter row contains the ordered K-1 intercepts, followed by
the shared coefficient of x and, if `quadratic=True`, x². For category boundary i,
the cumulative probability is `G(intercept[i] + slope*x + quadratic*x²)`, where
G is logistic or complementary-log-log (`link="cloglog"`).

```python
from mdanderson_stats import asypow_ordinal_regression_information, asypow_information

theta = [-1.0, 0.5, 0.2]  # two ordered intercepts, one shared slope: 3 categories
information = asypow_ordinal_regression_information(theta, [-1, 0, 1, 2])
design = asypow_information(theta, information, [0, 0, 1])
assert abs(float(design.power(design.sample_size())) - 0.8) < 1e-12
```

Covariate and allocation rules match the other regression designs, with limits
of 500 total coefficients and 10,000 covariate entries. The output is block
diagonal across independent groups, with coefficients ordered group by group.
The computation sums category score outer products weighted by category
probabilities. Second-derivative link terms cancel in the sum; avoiding them
reduces cancellation compared with the native per-category Hessian routine.
Category gradients and their matrix products are vectorized within each design
point. Stable log category probabilities preserve distinctions between nearby
logistic probabilities even when both round to one. Zero-allocation points are
skipped. Unordered/collapsed thresholds or unrepresentable cloglog category log
probabilities raise errors rather than returning an invalid matrix.

The raw-information formula is independently checked against multinomial
probability gradients. Both regression links match unmodified ASYPOW R results
for two-group quadratic designs; reference outputs are in
`tests/fixtures/asypow-ordinal.json`. Additional checks verify reduction to the
existing binary models, cross-group slope power, and extreme-predictor cases.

## Multinomial category probabilities

`asypow_multinomial_information(probabilities, group_size=1)` supplies information
for the original S-plus multinomial model. Each row contains K-1 positive
**individual category probabilities** whose sum is strictly below one. The last
category has probability `1 - sum(row)`. A vector is one group. Up to 500 free
probabilities are supported; allocations are positive and normalized by group.
The residual probability uses compensated summation to retain precision near
the boundary where the supplied probabilities sum to one.

```python
from mdanderson_stats import asypow_multinomial_information, asypow_information

p = [0.2, 0.3, 0.1]  # the fourth category has probability 0.4
information = asypow_multinomial_information(p)
assert abs(information[0, 1] - 2.5) < 1e-14
design = asypow_information(p, information, [1, 0, 0], null_values=0.1)
assert abs(float(design.power(design.sample_size())) - 0.8) < 1e-12
```

For each group, information is `diag(1/p) + ones/p_last`, weighted by that group's
allocation. The original `info.multinomial.kgp.s` returns only `diag(1/p)` when
K>2, omitting the implicit final category. Python includes its contribution:
observing that category has score `(-1/p_last, ..., -1/p_last)`, whose expected
outer product is `ones/p_last`. With two categories this reduces to the ordinary
binomial formula, matching the source's special binary branch.

For p=(0.2,0.3,0.1), the original function returns diagonal (5, 10/3, 10).
The corrected matrix adds 2.5 to **every** element. This deliberate correction
can change LR noncentrality and power relative to the native software.
Validation independently calculates expected category-score products and checks
equivalence to cumulative-probability information after reparameterization.

## General design matrices and transformed parameters

`asypow_design_information(coefficients, design, model="logistic", observations=1)`
implements native `info.mvlogistic` and `info.mvloglin`. Supply one design row
per covariate pattern and one column per coefficient; include any intercept
column explicitly. Nonnegative observation weights are normalized over all
positive-weight rows. A design vector represents one row. Limits are 500
coefficients and 1,000,000 matrix entries.

For `model="logistic"`, event probability is `logistic(X @ coefficients)` and
the information is the weighted sum of `p(1-p) * x xᵀ`.

For `model="loglinear"`, coefficients must be positive. The native model is
**binomial with multiplicative probabilities**:
`p = exp(X @ log(coefficients))`. Every used row must imply `p < 1`; coefficients
themselves may exceed one when the full row still implies a valid probability.
Information is with respect to the supplied positive coefficients, with each
row contributing `p/(1-p) * (x/coefficients)(x/coefficients)ᵀ`.
This is separate from the Poisson log-link regression interface.

```python
from mdanderson_stats import asypow_design_information, asypow_information

theta = [0.2, 0.8]
info = asypow_design_information(theta, [[1, 0], [1, 1], [1, 2]], model="loglinear")
design = asypow_information(theta, info, [0, 1], null_values=1)
assert abs(float(design.power(design.sample_size())) - 0.8) < 1e-12
```

Log-space weights and scaled design vectors avoid overflowing intermediate
derivative products. Positive probabilities smaller than a representable float
can still contribute representable information. Rows with zero allocation are
skipped before model evaluation, so they need not imply a valid probability.
This differs from the native loop, which can reject unused log-linear rows.

`asypow_reparameterize(information, jacobian)` implements delta-method information
for a new parameter vector. Supply `J = d(new parameters)/d(old parameters)` at
the alternative. The result is the inverse of `J I⁻¹ Jᵀ`. For a log transformation
of positive parameters, for example, `J = diag(1/parameters)`. The Jacobian may
have fewer rows than columns for a reduced parameter vector, but must have full
row rank. The old information must be positive definite. Cholesky solves and
singular values replace explicit inverses, and row scaling makes the rank check
less sensitive to differing parameter units. Nonlinear transformations retain
the usual local delta-method interpretation.

Both general models match native R outputs on weighted, three-coefficient
designs. Reparameterization matches native R evaluation of a log transformation and an
independent variance calculation for a reduced contrast. Tests also cover very
small multiplicative coefficients, probabilities near one, zero-allocation rows,
and parameter units scaled by `1e-100`/`1e100`.

## Power and inversions

`design.power(sample_size, significance=0.05)` uses a noncentral chi-square
survival probability with df equal to the contrast rank and noncentrality `n*w`.
Sample size and significance broadcast as NumPy arrays; results are read-only.
Sizes may be continuous or zero. Survival functions avoid subtracting a CDF
from one. Significance and requested powers must lie strictly between zero and
one; total noncentrality is limited to `1e8`.

`design.sample_size(power=0.8, significance=0.05)` returns the **continuous**
solution. Round up for an integer total, then check allocation rounding for the
actual design. It returns zero if requested power is no greater than significance;
positive power gain is unattainable when `w=0`. Scalar targets are supported.
This is an asymptotic calculation, not finite-sample exact power.

`design.significance(sample_size, power=0.8)` inverts power for significance,
using the design's degrees of freedom. Inputs broadcast.

## SMO binomial designs

`asypow_smo_binomial(probabilities, null_probabilities=None, group_size=1,
subtract_df=True)` implements two common original S-plus SMO hypotheses:

- Omit `null_probabilities` to test equality across G>=2 binomial groups. The
  null probability is the allocation-weighted mean, which maximizes expected
  log likelihood under equality. Degrees of freedom are G-1.
- Supply a scalar or vector to fix all G null probabilities. Degrees of freedom
  are G, including when some alternative components happen to match the null.

Probabilities must be strictly in (0,1); up to 500 groups are supported, with
positive relative allocations. SMO uses `w = 2 * sum(weight * KL(Bern(p),Bern(q)))`,
the expected log-likelihood difference, instead of the LR quadratic approximation.
For near-equal probabilities, stable log remainders avoid cancellation; log-space
weighting retains small representable divergences.

```python
from mdanderson_stats import asypow_smo_binomial

design = asypow_smo_binomial([0.4, 0.3], group_size=[10, 9])
assert abs(design.sample_size() - 806.341114462205) < 1e-8
assert abs(float(design.power(100)) - 0.061231924153204) < 1e-12
```

The returned `SMOPower` exposes `power`, `sample_size`, and `significance` methods.
The default correction gives noncentrality `nu=n*w-df`; `subtract_df=False` uses
`nu=n*w`. Following the original software, power and significance calculations
reject nonpositive nu. It is not clamped to zero. Sample-size inversion returns
a continuous value, requires requested power above significance, and rejects
an exactly null alternative. Power/significance inputs broadcast; sample-size
targets are scalar. The shared noncentrality limit of `1e8` also applies.

The original example p=(0.4,0.3), allocation=(10,9), has null probability
0.35263157894736841 and w=0.01097409068025511. At significance 0.05 and power 0.8,
sample sizes are approximately 715.2173914 without correction and 806.3411145
with correction. Direct R calculations and evaluation of the original S-plus
binomial/noncentrality and power routines agree. For that reference execution,
the S-plus multi-value return was converted to an R list and the later R
chi-square wrapper supplied its distribution calculations; statistical formulas
were unchanged. Inversion was independently checked with a tighter R root.

The significance inverse uses actual df, correcting the original `self.sig.s`
hardcoded df=1, just as the LR inverse does. Mixed fixed/equality constraints,
exponential-survival models, regression SMO and generic expected-likelihood
optimization remain pending. These tests are asymptotic approximations, not
finite-sample exact binomial tests.

## SMO Poisson designs

`asypow_smo_poisson(means, null_means=None, group_size=1, subtract_df=True)`
provides the same fixed-null and all-groups-equal hypotheses for positive Poisson
means. The equality null uses allocation-weighted means and df=G-1; a specified
null fixes all G means and uses df=G. Inputs support up to 500 groups. The result
is `SMOPower`, with the same correction, inversion and noncentrality limits
as the binomial method above. Mixed constraints remain pending.

The per-observation divergence is twice the weighted sum of
`p*log(p/q) - p + q`. Near-equal means use a stable log remainder; large or small
means use log-space divergences and allocation weights before exponentiation.

```python
from mdanderson_stats import asypow_smo_poisson

design = asypow_smo_poisson([2, 4], group_size=[1, 2])
assert abs(design.sample_size() - 30.37914057915722) < 1e-9
assert abs(float(design.power(30)) - 0.7944166025642022) < 1e-12
```

The original S-plus `noncent.poisson.kgp.s` and `self.power.s`, evaluated through
R, give null means 10/3, w=0.29128080454643612, and the values above.
Compatibility changes were limited to converting the multi-value return to an
R list and aliasing `is.inf` to `is.infinite`; the later R chi-square wrapper
provided distribution calculations. The statistical formulas were unchanged.
Sample size was independently inverted with a tight R root tolerance.

## SMO multinomial and ordinal designs

`asypow_smo_multinomial(probabilities, null_probabilities=None, group_size=1,
subtract_df=True)` takes K-1 category probabilities per group, with the final
mass equal to one minus their sum. `asypow_smo_ordinal(cumulative,
null_cumulative=None, group_size=1, subtract_df=True)` takes K-1 increasing
cumulative probabilities, excluding the terminal 1. A vector represents one
group; a matrix has one group per row. At most 500 free parameters are supported.
All category masses must be positive, including the implicit final category.

Omitting the null compares the complete distributions across G>=2 groups.
The null is the allocation-weighted pooled distribution, with df=(G-1)*(K-1).
Supplying null parameters fixes every free parameter, with df=G*(K-1); a
single vector broadcasts to all groups. Mixed constraints, partial constraints,
and equality between selected categories remain pending. Null parameters in
`SMOPower` are always a G by (K-1) matrix in the input parameterization.

The two parameterizations produce the same divergence and power for the same
category masses and hypotheses. The noncentrality is twice allocation-weighted
categorical KL divergence. Nonnegative generalized-KL summands avoid subtraction
of nearly equal expected log likelihoods. Compensated summation computes the
multinomial final mass; calculations retain small representable divergences.
If rounding makes a pooled distribution invalid, it is rejected rather than
silently replacing a zero category with a positive floor.

```python
from mdanderson_stats import asypow_smo_multinomial, asypow_smo_ordinal

multi = asypow_smo_multinomial([[0.2, 0.3], [0.4, 0.1]], group_size=[1, 3])
ordinal = asypow_smo_ordinal([[0.2, 0.5], [0.4, 0.5]], group_size=[1, 3])
assert abs(multi.sample_size() - 172.853418531114) < 1e-9
assert abs(float(multi.power(100)) - 0.480630044419691) < 1e-12
assert abs(ordinal.sample_size() - multi.sample_size()) < 1e-9
```

The original S-plus multinomial and ordinal noncentrality routines give
w=0.06730956764893814 (within rounding) and df=2 for this equality example.
The pooled category masses are (0.35,0.15,0.5). The reference execution through
R changes only the S-plus multi-value return to an R list, with the later R
chi-square wrapper supplying distribution calculations. Power and sample-size
inversion agree with the original formulas. Binary-category reduction and
340-digit Decimal likelihood comparisons cover near-null and rare-category
behavior independently.

For a fully specified null with category masses (0.25,0.25,0.5) in both groups,
the multinomial source gives w=0.14959244615398992 and df=4. Its ordinal equivalent
is a valid null with cumulative probabilities (0.25,0.5). The original ordinal
routine incorrectly rejects it: repeated zero equality markers are interpreted
as duplicate equality constraints. This port accepts the valid null and gives
the same divergence as the multinomial parameterization.

## Native comparisons and intentional corrections

The archived R implementation gives the same noncentralities and ordinary
power values for the binomial example and Poisson means 2 and 4 with equal
allocations. Their per-observation noncentralities are 0.05 and 1/3 respectively.
The original sample-size inverse has a loose root tolerance: it returns
156.97763180694631 for the binomial example. An independent tighter R root gives
156.97721018652396, agreeing with Python.

Two source issues are corrected:

- `asypow.sig.R` uses central chi-square df=1 unconditionally. Python uses the
  actual contrast rank, so significance inversion also works for multiple df.
- The native nuisance-coordinate update can use the wrong sign. For theta=(1,2),
  I=[[2,1],[1,2]] and theta1=0, its null output is (0,1.5). The constrained
  quadratic optimum is (0,2.5), returned by Python; both calculations give
  noncentrality 1.5. The null projection is not used to replace the supplied
  information when calculating power.

Focused tests check native power examples, tighter R sample-size inversion,
forward/inverse consistency, multiple constraints, parameter/contrast rescaling,
null power, and exponential information at rates too small for a direct formula.
All four regression families also match the unmodified R source on two-group
quadratic designs with unequal covariate allocations. Fixtures in
`tests/fixtures/asypow-regression.json` were generated from those R calculations.
Additional checks cover cross-group slope tests, extremely small information
weights combined with large covariates, allocation scaling by `1e300`, and
survival time units scaled by `1e-200` and `1e200`.
The software archive is retained only in ignored research storage, with hashes
in `asypow-sources.json`; no native files are redistributed in the wheel.

Sources: [MD Anderson entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/33),
[Brown and Lovato's ASYPOW paper](https://doi.org/10.18637/jss.v002.i02), and
[archived R source](https://cran.r-project.org/src/contrib/Archive/asypow/asypow_2015.6.25.tar.gz).
The R archive identifies the original S authors and declares `ACM | file LICENSE`.
The site's registered download is not used as a native parity claim. Validation
uses R archive version 2015.6.25 and the original
[S-plus 2.1 supplement](https://www.jstatsoft.org/index.php/jss/article/downloadSuppFile/v002i02/asypow-2.1.tar.Z.tar).
Original archive and inspected routine hashes are recorded separately in the
provenance file. Neither archive is redistributed in the wheel.
