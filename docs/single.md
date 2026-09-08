# SINGLE dose-response design precision

Catalog entry 55 is partial. Fixed one-sample designs under point priors are
implemented for logistic and log-log models, with linear or centered predictors.
Uncertain normal/log-normal/uniform priors, two-sample criteria, design optimization,
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
