# SINGLE dose-response design precision

Catalog entry 55 is partial. Fixed one- and two-sample designs under point priors are
implemented for logistic and log-log models, with linear or centered predictors.
Independent uniform-prior criterion averaging is also implemented. Correlated
normal/log-normal priors, design optimization,
dose-point addition and original reporting workflows remain pending.

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
at nodes propagate errors from the fixed-design APIs. The one-sample API's
nonzero-slope restriction continues to apply at each node, including when
requesting slope precision. No singular nodes are dropped or replaced.

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
