# SINGLE dose-response design precision

Catalog entry 55 is partial. Fixed one- and two-sample designs under point priors are
implemented for logistic and log-log models, with linear or centered predictors.
Independent uniform and correlated normal/log-normal prior criterion averaging
and design-derived prior correlations are implemented. Fixed-dose, one- and two-sample
point- and uncertain-prior allocation optimization is implemented, along with joint
dose/allocation optimization and automatic dose-point addition. Complete model/prior
workflow reporting remains pending.
Optimized numerical design reports are available.

```python
from mdanderson_stats import single_design_precision

result = single_design_precision(
    doses=[-2.399239, 2.399495],
    subjects=[90.741824, 9.258176],
    parameters=[0, 1],
    model="logistic",
    form="linear",
    quantile=0.05,
)
print(result.quantile_sd)  # approximately 0.444280, matching the supplied example
```

The logistic response is expit(u); the log-log response is exp(-exp(-u)). Linear
form uses parameters (intercept, slope) and u = intercept + slope*dose. Centered
form uses (slope, center) and u = slope*(dose-center). Doses must already be on
the desired model coordinate; no logarithm is applied implicitly.

Doses and allocations are matching vectors, with nonnegative allocations and at
least two distinct informative dose values. Fractional allocations are allowed,
as in the optimizer's approximate designs. Parameters end in two entries; their
leading axes broadcast with the requested response quantile. A nonzero slope is
required to identify the quantile dose. Quantiles lie strictly between zero and one.
Set `quantile=None` for slope-only precision: quantile dose and variance are then
`None`, and accessing `quantile_sd` raises an error. This permits identifiable
linear zero-slope models. Centered zero-slope models remain singular.

The result contains the expected Fisher information, response probabilities,
quantile dose, slope variance and quantile variance, with standard-deviation
properties for both criteria. These are local asymptotic precision calculations
at the supplied parameter values, corresponding to SINGLE's point-prior case.
They do not fit parameters or optimize the supplied design.

Information sums per-dose Bernoulli information weighted by allocations. The
quantile variance uses its parameter gradient and a Cholesky solve, avoiding an
explicit inverse. The logistic predictor information weight is expit(u)*expit(-u).
For log-log it is t²/(exp(t)-1), where t=exp(-u); separate small/large-t expressions
avoid overflow and cancellation. Information remains meaningful when a response
probability rounds to one. Invalid or numerically singular designs raise errors.

The original CPROB clamps probabilities to [1e-7,1-1e-7], and GEXP caps its exponent
at 85. The Python calculation uses stable, unclipped model probabilities and
information weights. It does not pretend these source cutoffs define the statistical
model. Native comparisons use values where neither cutoff applies.

`tools/reference_single.py` extracts unchanged CPROB, CDPDB, MIX and GEXP and uses
an independent driver to assemble information from the original probabilities and
derivatives. Twelve cases cover both models, both parameterizations and three
parameter pairs; one centered zero-slope case is retained as a singular reference
and explicitly rejected by the quantile-precision API. Source/extracted hashes and
compiler provenance accompany `tests/fixtures/single.json`.

Tests compare native probabilities/information, reproduce two printed criterion
values from the supplied test.out, verify a closed-form symmetric logistic design,
check equivalence of parameterizations and inverse allocation scaling, exercise
quantile broadcasting, and confirm stable information at probabilities rounded to
one. Source: SINGLE_V1.tar.gz, source/single, version 2.0 (August 1997).

## Two-sample comparisons

`single_two_sample_precision` implements the two comparisons offered by SINGLE's
CINIT menu: a location difference with a common slope, or a slope difference with
a common location parameter. Each group has separate dose and allocation vectors;
these may have different lengths. Parameters end in three entries, with any leading
axes evaluated together. The parameter order follows the original MIX routine:

| Form | Comparison | Parameters | Group 1 minus group 2 |
| --- | --- | --- | --- |
| linear | location | (a1, b, a2) | a1 − a2 |
| linear | slope | (a, b1, b2) | b1 − b2 |
| centered | location | (b, d01, d02) | d01 − d02 |
| centered | slope | (b1, d0, b2) | b1 − b2 |

For linear form, “location” means the intercept. In centered form it means the
center dose. In particular, the linear location criterion is the precision of an
intercept difference, not a transformed dose difference. Changing forms can also
change the shared-parameter constraint, so these comparisons are not automatically
equivalent under reparameterization.

```python
from mdanderson_stats import single_two_sample_precision

comparison = single_two_sample_precision(
    doses=([-1, 1], [-0.5, 0, 1.5]),
    subjects=([20, 30], [30, 20, 50]),
    parameters=[1.2, 0.3, 0.7],
    form="centered",
    comparison="location",
)
print(comparison.difference)  # -0.4
print(comparison.sd)
```

The result includes the joint 3×3 Fisher information, each group's response
probabilities, parameter difference, variance and standard deviation. It uses a
Cholesky solve for the contrast variance, including the covariance induced by the
shared parameter. It does not return a hypothesis-test p-value or test power.
Each group needs positive informative allocation, and the joint design must
identify all three parameters. One group can have a single informative dose if
the other group's doses provide the remaining information. Singular designs
raise errors; zero slopes are permitted when the chosen parameterization remains
identifiable. Numerically rank-deficient gradient matrices are also rejected.

`tools/reference_single_two_sample.py` compiles unchanged CPROB, CDPDB, MIX and
GEXP routines and assembles the joint information in an independent driver. The
16 native cases cover both models, both forms, both comparisons and two parameter
triples. The fixture records source, extracted-code and driver hashes and compiler
provenance. Tests compare probabilities and information, then check the contrast
formula used by CRIT against that native-derived information. CRIT itself is not
executed by this reference driver. Additional tests check closed-form symmetric
logistic variances, group swapping, batch evaluation, allocation scaling, and
identifiable versus singular designs. The unused power criterion commented out
in the source menu is not exposed.

## Uniform-prior criterion averaging

`single_uniform_criterion` integrates a selected local precision criterion over
independent uniform parameter intervals. The default six-point Gauss-Legendre
rule per varying parameter follows CEVAL/RECGS in SINGLE. Computation evaluates
all tensor nodes in a NumPy batch. Equal interval endpoints represent fixed
parameters and consume one node; this also permits a fully fixed point prior.

```python
from mdanderson_stats import single_uniform_criterion

averaged = single_uniform_criterion(
    doses=[-1, 0, 2],
    subjects=[20, 30, 50],
    lower=[0.2, 0.8],
    upper=[0.4, 1.2],
    criterion="quantile_sd",
    quantile=0.05,
)
print(averaged.value)
```

One-sample criteria are `slope_sd`, `slope_variance`, `quantile_sd` and
`quantile_variance`. For two samples, supply the two groups' dose/allocation
vectors, three parameter intervals, `comparison="location"` or `"slope"`, and
`criterion="sd"` or `"variance"`. The two-sample variance option extends the
original menu's standard-deviation criterion by averaging its square.

The returned value is the arithmetic average of the selected local criterion.
In particular, average SD is neither square root of average variance nor the
SD computed from average information. The result also exposes quadrature
parameters, normalized weights, local criteria and order for inspection.
Normal-prior integration in the source uses a reciprocal aggregation and is
not substituted for this uniform-prior calculation.

`order` may be increased from 6 up to 32 to assess convergence (cost grows as
order to the number of varying parameters). These are quadrature approximations,
without a certified integration error. A finite quadrature result alone does not
prove that an improper expectation exists: singularities between quadrature
nodes require care. One-sample quantile criteria reject slope intervals touching
or crossing zero. Other invalid or unidentifiable parameter values encountered
at nodes propagate errors from the fixed-design APIs. Slope criteria automatically
request slope-only precision, so linear zero-slope nodes are valid. No singular
nodes are dropped or replaced.

`tools/reference_single_uniform.py` compiles unchanged RECGS, RECGSX and QNXTIX
(with its entries), together with unchanged response/derivative routines. An
independent callback assembles information, inverts its small matrix, and evaluates
the criterion. It does not execute the original CRIT/CEVAL routines. Thirty-two
cases cover both response models, both parameterizations, one/two samples, and
SD/variance criteria. Reference tolerances allow the source's partly default-REAL
quadrature constants. Further tests use an analytic uniform-intercept integral,
point-prior reduction, allocation scaling, order convergence and Jensen's inequality
for SD versus variance averaging. NumPy's quadrature contract is documented in
the [leggauss reference](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html).

## Normal and log-normal priors

`single_normal_criterion` uses tensor Gauss-Hermite quadrature and SINGLE's
reciprocal aggregation: `1 / E[1 / local_criterion]`. This is a harmonic average
of the chosen SD or variance, whereas uniform priors use arithmetic averaging.
The choices of criterion, response model, form and two-sample comparison follow
`single_uniform_criterion`.

```python
from mdanderson_stats import single_normal_criterion

result = single_normal_criterion(
    doses=[-1, 0, 2],
    subjects=[20, 30, 50],
    mean=[0.2, 0.0],
    covariance=[[0.02, 0.01], [0.01, 0.03]],
    lognormal=[False, True],
    criterion="quantile_sd",
    order=8,
)
print(result.value)
```

Here the intercept is normal and log(slope) is normal. `mean` and `covariance`
always describe the **latent normal vector**, before exponentiating coordinates
selected by the boolean `lognormal` vector. For a logged coordinate with latent
mean m and variance v, the actual parameter mean is exp(m+v/2), and its variance
is exp(2m+v)·(exp(v)−1). Cross-covariances are likewise specified in latent space.
Logged parameters are positive. Negative signed log-normal priors are not defined
by the original PARRAW routine and are not added here.

SINGLE's ALNTON input conversion uses log(raw mean) and raw variance/raw mean²,
a first-order approximation rather than exact log-normal moment matching. Use
`single_prior_parameters` to convert raw marginal inputs explicitly. Its
`conversion="exact"` default matches log-normal marginal means and variances;
`conversion="legacy"` reproduces ALNTON. Correlations always describe latent normal
coordinates, including when the marginal means and variances are supplied on the
raw scale. `single_design_correlation` provides the reference-design correlation heuristic.

The default node scaling correctly reproduces the supplied latent covariance.
The original RECHRM divides physicists' Hermite nodes by √2, then TORAW applies
the covariance square root; this produces covariance/4. Set `legacy_scale=True`
to reproduce that scaling deliberately. The original constants for π and √2 are
rounded, so comparisons allow their small numerical discrepancy. This option
does not reintroduce CPROB's probability clipping, GEXP's exponent caps, or CRIT's
1e-10 local-criterion floor. NumPy's weight convention is documented in the
[hermgauss reference](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermgauss.html).

Covariance must be symmetric and positive semidefinite. Eigen-directions within
32 machine epsilons times the largest absolute covariance entry are treated as
zero to handle numerical rank deficiency; larger negative eigenvalues raise an
error. Integration operates only over positive eigen-directions. Thus zero
covariance reduces to a point prior and rank-deficient covariance avoids redundant
tensor dimensions. The result exposes transformed parameter nodes, weights, local
criteria, order and scaling mode. Allowed orders are 2–32 (the source tabulates
2–8); increasing order assesses convergence but does not certify an error bound.
Numerically invalid or singular nodes propagate errors. In particular, the
one-sample quantile criterion rejects a zero slope at a node; slope criteria
permit it in linear form. Exponentiation
overflow or underflow raises an error instead of clipping parameters.

`tools/reference_single_normal.py` compiles unchanged RECHRM and QINIX (including
entries) and the original response/derivative routines. Its independent callback
applies diagonal latent-prior transformations and computes the reciprocal local
criterion. The 64 reference cases cover both models/forms, one/two samples,
normal/log-normal priors and SD/variance criteria using original node scaling.
PRCOMP, ALNTON, PARRAW and CRIT are not executed by this driver. Independent tests
verify correlated normal moments, mixed normal/log-normal moments, the factor-of-four
scaling discrepancy, rank-one and point priors, allocation scaling and convergence.


```python
from mdanderson_stats import single_prior_parameters

latent_mean, latent_covariance = single_prior_parameters(
    mean=[0.2, 1.2],
    variance=[0.02, 0.03],
    correlation=[[1, 0.4], [0.4, 1]],
    lognormal=[False, True],
)
# Pass these arrays to single_normal_criterion with the same lognormal mask.
```

The input helper supports two or three parameters, nonnegative marginal variances,
fixed parameters, and positive semidefinite latent correlation matrices. It rejects
nonpositive log-normal means and inconsistent correlations. Exact log-normal
conversion works in log space to avoid overflowing the variance-to-squared-mean
ratio. Tests verify raw moments through quadrature, the original ALNTON algebra,
latent correlations, fixed parameters and extreme moment ratios. These helper
checks use mathematical identities; they do not execute native ALNTON.


## Reference-design prior correlations

`single_design_correlation(dose_bounds, parameters, ...)` constructs the DSTCOV
reference design and converts its inverse expected information to correlations.
Group 1 receives 50 subjects at each interior third of the dose interval. With
`comparison="location"` or `"slope"`, group 2 receives two 50-subject allocations
at the same midpoint. This follows the executable source despite its introductory
comment describing an evenly spaced design.

The result contains dose/subject vectors, covariance and correlation. The latter
can be passed to `single_prior_parameters`. Model/form/comparison conventions and
parameter batching follow the fixed-design APIs. Parameters are actual model
values, so callers using log-normal priors explicitly choose the evaluation point.
These correlations are a prior-elicitation heuristic, not an estimate from observed
data. Explicit evaluation parameters avoid the original CGTCOV/CCRIT dependency
on mutable prior-transformation state.

Covariance uses triangular solves. Singular designs raise errors: for example,
the second group's slope cannot be identified from a zero midpoint in linear form.
Tests verify closed-form one/two-sample logistic covariances, an independent 2×2
inverse-correlation identity for both models, batching and prior-input composition.
These tests do not execute native DSTCOV or CGTCOV; response information is
separately checked against original routines.


Slope-only regression tests cover zero-slope information in both response models,
unchanged precision at nonzero slopes, an analytic uniform-slope integral crossing
zero, normal priors with zero-valued quadrature nodes, zero-slope reference-design
correlations, and continued rejection of singular centered models.


## Fixed-dose allocation optimization

`single_optimize_allocations` minimizes one-sample local slope or quantile variance
at supplied dose points under a point prior. Minimizing SD gives the same allocation.
Dose locations stay fixed, and allocations are nonnegative continuous subject
counts summing to `total_subjects`; no integer rounding is performed.

```python
from mdanderson_stats import single_optimize_allocations

optimized = single_optimize_allocations(
    [-2.399239, 2.399495], [0, 1], total_subjects=100, criterion="quantile"
)
print(optimized.subjects)  # approximately [90.7414, 9.2586]
print(optimized.sd)  # approximately 0.444280
```

The result includes doses, counts, variance/SD, initial variance, iteration count
and a relative finite-dose optimality gap. Defaults initialize equally; optional
`initial_subjects` must match the dose vector and total. The solver uses SLSQP
with analytic allocation derivatives and a normalized variance objective. Per-dose
response information is computed once. Trials with singular information are
infeasible, and failure to converge raises an error rather than returning an
apparently optimized result.

For unit-total information M and criterion gradient c, the variance is cᵀM⁻¹c.
The derivative for dose i is −wᵢ(gᵢᵀM⁻¹c)². At the finite-dose convex optimum,
the maximum sensitivity divided by variance is at most one; active doses attain
one. `optimality_gap` is the positive part of this ratio minus one. The return
check requires a gap no larger than max(1e-5, 10√tolerance). This checks allocation
optimality on the supplied dose set, not optimal dose locations. Full-rank
information is required; a singular limiting optimum may cause failure.

Tests reproduce the supplied SINGLE allocation/SD example, compare two-point
solutions to analytic optimal splits, verify zero allocations at inferior doses,
and exercise scaling, permutation, initialization, failure reporting and zero
slopes. The original allocation optimizer itself is not compiled in these tests.
`single_search_design` adds the incremental support-point search.


## Allocation optimization under uncertain priors

`single_optimize_prior_allocations` takes a `(nodes, 2)` parameter array and
positive weights summing to one. Nodes and weights can be taken directly from a
`single_uniform_criterion` or `single_normal_criterion` result, preserving the
chosen quadrature order, transformation and covariance scaling. The nodes are
held fixed during optimization and per-node dose information is precomputed.

```python
from mdanderson_stats import single_normal_criterion, single_optimize_prior_allocations

prior = single_normal_criterion([-1, 2], [50, 50], [0.3, 0.9], [[0.01, 0.002], [0.002, 0.01]])
optimized = single_optimize_prior_allocations(
    [-1, 2],
    prior.parameters,
    prior.weights,
    criterion="quantile",
    measure="sd",
    aggregation="harmonic",
)
print(optimized.subjects, optimized.value)
```

Use arithmetic aggregation for SINGLE's uniform-prior convention and harmonic
aggregation for its normal-prior convention. `measure` chooses `variance` or `sd`;
these are distinct objectives under uncertain priors. The return fields `value`
and `initial_value` always describe that selected aggregate. Subject counts are
continuous and the dose vector stays fixed. The same source node set can be
reused across candidate dose sets without repeating prior construction.

The analytic gradient includes the SD transformation and reciprocal averaging.
The reported `optimality_gap` is a relative finite-dose first-order stationarity
gap, using the objective's allocation scaling exponent (one for variance, one-half
for SD). Successful termination and the gap threshold are both required. For
these general aggregated objectives this is not asserted to prove a global
minimum. Different starts and higher prior quadrature orders can assess solution
and integration sensitivity. Singular information at any positive-weight node
makes that trial allocation infeasible; no prior nodes are silently discarded.

Tests compare optimized values with separate prior evaluators for both response
models, both criteria, both measures and uniform/normal averaging, and compare
against a dense two-dose allocation search. Further checks cover point-prior
reduction, sample-size scaling, centered models, discarded interior doses and
explicit failure reporting. These tests do not compile the original optimizer.


## Two-sample allocation optimization

`single_optimize_two_sample_allocations` distributes `total_subjects` across both
sample groups and their candidate doses. This total is for the whole experiment,
not a separate total per group. Group dose vectors may differ in length, and each
returned subject vector corresponds to its input dose vector.

```python
from mdanderson_stats import single_optimize_two_sample_allocations

result = single_optimize_two_sample_allocations(
    doses=([-1, 1], [0]),
    parameters=[0, 1, 0],
    comparison="location",
    total_subjects=100,
)
print(result.subjects, result.value)
```

A three-entry parameter vector defines a point prior. For an uncertain prior,
supply `(nodes,3)` parameters and positive `prior_weights` summing to one, such as
nodes and weights from the two-sample prior evaluator. Shared-parameter conventions
follow `single_two_sample_precision`. `measure` defaults to SD; variance is also
available. Choose arithmetic aggregation for uniform priors or harmonic aggregation
for SINGLE's normal-prior convention. The result's `value` is that chosen aggregate.
Optional initial subject vectors must match the group dose vectors and sum to the
shared total. Default initialization distributes subjects equally across all dose
entries, rather than forcing equal group totals.

The common prior-allocation solver uses analytic derivatives and checks
identifiability at every positive-weight prior node. Its returned first-order
stationarity gap applies to allocation over both groups. It does not certify a
global optimum for arbitrary aggregated objectives. Allocation counts remain
continuous, dose locations stay fixed, and convergence failure raises an error.

Tests verify analytic between-group allocation splits, unequal-slope formulas,
symmetric prior optima, both response models, both parameterizations, group swaps,
sample-size scaling, independent precision recomputation and explicit failures.
The original two-sample allocation optimizer is not compiled by these tests.


## Joint dose-location and allocation optimization

`single_optimize_design` moves dose locations within a shared interval while
optimizing continuous allocations. It supports one/two samples, point/weighted
priors, both response models and parameterizations, and arithmetic/harmonic
averaging of SD or variance. Two-sample comparisons use the same three-parameter
conventions as the fixed-design APIs. Total subjects is shared across groups.

```python
from mdanderson_stats import single_optimize_design

result = single_optimize_design(
    initial_doses=[-1.111111, 3.333333],
    parameters=[0, 1],
    dose_bounds=[-10, 10],
    total_subjects=100,
    criterion="quantile",
    quantile=0.05,
)
print(result.doses, result.subjects, result.value)
# Approximately [-2.39936, 2.39937], [90.744, 9.256], 0.444280.
```

For two samples, provide two initial dose vectors and `comparison="location"` or
`"slope"`. A parameter vector represents a point prior; a node matrix requires
positive `prior_weights` summing to one. These can be reused from the prior
quadrature evaluators. Default initial allocations are equal across all dose
entries. Custom initial counts must match each dose vector and the shared total.
Dose/count pairs retain their group and entry correspondence; they are not sorted
or rounded after optimization.

The number of dose entries stays fixed, although allocations may become zero.
This is a local SLSQP search, not an automatic support-point search or guarantee
of a global optimum. Different initial designs can lead to different solutions.
Full-rank information is required at each positive-weight prior node; singular
trial designs are infeasible. Convergence failure raises an error. Results contain
final dose/count vectors, final and initial criterion values, iterations and a
`stationarity` diagnostic. This combines normalized projected dose gradients and
the relative allocation sensitivity gap; it is reported separately from SLSQP's
termination decision and is not a certified optimization error bound.

Dose derivatives include both the change in response-information weight and the
change in the predictor's parameter gradient. For logistic response, dw/du is
w·(1−2p); for log-log it is w·(t/(1−exp(−t))−2), t=exp(−u). The implementation
uses stable limiting expressions and analytic derivatives for both dose and
allocation variables, including SD and reciprocal prior averaging. Dose variables
are normalized to the interval and the objective to its initial value.

Validation reproduces the original printed two-dose slope and quantile designs,
checks bounded optima and a symmetric two-sample optimum, recomputes weighted-prior
objectives independently, and checks sample-size scaling and failure behavior.
Forty-eight finite-difference checks inspect the objective/gradient contract passed
to the external solver across both models/forms, one/two samples and all aggregation
choices. The original joint optimizer is not executed by these tests.


## Numerical design reports

An optimized result provides `report(digits=8)` and
`write_report("design.tsv", digits=8)`. TSV sections contain one row per dose,
with one-based group/entry numbers, dose and continuous subject allocation; group
and overall subject totals; initial/final criterion values; the stationarity
diagnostic; and iteration count. Input entry ordering and zero allocations are
retained. The criterion value represents the SD/variance and arithmetic/harmonic
objective selected for optimization; it is not relabeled as a different metric.

`digits` controls significant digits from 1 through 17. Seventeen digits preserve
binary64 values when parsed back. File export writes UTF-8, replaces the explicitly
supplied destination, and propagates filesystem errors. The numerical report does
not serialize model/prior configuration or constitute a complete reproducible
study file; callers must retain their optimization inputs separately. Tests parse
one/two-sample optimization reports back into values and check totals, ordering,
zero counts, explicit replacement and I/O errors.


## Automatic dose-point search

`single_search_design(parameters, dose_bounds, ...)` begins with two dose entries
per group, scans candidate starting doses, jointly optimizes them, then adds one
entry per group until the improvement rule stops the search or `max_doses` is
reached. It supports the same models, priors, comparisons and objectives as the
joint optimizer. Returned `best` is a `SingleOptimizedDesign`, including its report
methods. `steps` retains every optimized stage, its relative improvement and
whether it was accepted.

```python
from mdanderson_stats import single_search_design

search = single_search_design(
    [[-3, 1], [0, 1], [3, 1]],
    [-8, 8],
    prior_weights=[1 / 3] * 3,
    criterion="slope",
    max_doses=4,
)
print(search.stop_reason, search.best.doses, search.best.value)
```

The default grid has 10 points including endpoints (`scan_points` supports 2–20).
Initial seeds are all distinct grid pairs with equal allocations. As in DQSCAN,
two-sample seeds initially use the same pair in both groups. Extension seeds append
a common grid point in each group; joint optimization subsequently permits the
groups' doses and allocations to differ. Seeds assign the minimum positive group
allocation divided by 1, 2, 4, 8, 16, 32 or 64 to each new entry and proportionally
reduce previous group counts, preserving the experiment's total.

These seeds reflect DIQSCN's halving scheme, with explicit differences: all seven
fractions are scored, existing zero counts are excluded from the minimum, group
totals are preserved under the shared-total API, and extensions start from the
previous jointly optimized design. The original also retains an equal-allocation
intermediate design and may stop scanning fractions early. Feasible seeds are
ranked by initial criterion and passed to the local optimizer until one converges.
This is a deterministic heuristic, not an exhaustive search of all local optima.

An extension is accepted when `(previous − candidate) / previous` is at least
`relative_improvement` (default 0.01, matching SINGLE). Otherwise the previous
design remains best and the rejected larger design stays in `steps`. The reasons
are `relative_improvement` or `max_doses`; the latter limit is 2–10 entries **per
group**, including zero allocations and coincident entries. `scan_evaluations`,
`infeasible_seeds` and `failed_local_starts` expose attempted work. A stage with no
feasible seeds or no successful local optimization raises an error; it is not
reported as convergence. A finite grid and local stopping rule do not prove that
no larger or differently initialized design would improve the criterion.

Tests reproduce the printed two-dose quantile optimum and reject a negligible
third-dose improvement, accept substantial improvements for a broad three-node
prior, retain a better but sub-threshold rejected design in history, verify
one/two-sample totals and limits, independently recompute criteria, and check
failed-search behavior. The original search executable is not run by these tests.
