# CiBolus

CiBolus studies a drug given as an initial bolus followed by continuous infusion.
This is an independent Python implementation of the model in
[Thall et al. (2011)](https://odin.mdacc.tmc.edu/~pfthall/main/Biometrics_IAtPA_2011.pdf),
associated with [catalog entry 86](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/86).

## Observation conventions

Time is standardized by the maximum infusion duration, so one means the end
of the planned infusion. Concentrations are positive and bolus fractions lie
between zero and one. Response can occur immediately after the bolus, at an
exact later time, or within an observed interval. If there is no response by
one, the observation records that failure separately. Toxicity is a completed
binary outcome.

Intervals are open on the left and closed on the right. An interval starting
at zero excludes the separately observed bolus response. Its toxicity
calculation uses the upper endpoint, when response is detected and infusion
stops. Irregular observation intervals are allowed. These conventions preserve
the treatment actually delivered instead of treating response time as an
independent efficacy measurement.

## Model and priors

The response model combines immediate-response probability with a continuous
hazard depending on the accumulated effective dose. A separate conditional
toxicity model depends on concentration, bolus fraction, time of stopping
infusion, and whether response failed to occur. Eleven positive parameters
are represented by independent normal priors on their logarithms. Their
ordering is `alpha0` through `alpha5`, then `beta0` through `beta4`.

Use an explicit prior for fitting. The published prior is available for
reproducibility, but its study-specific elicitation does not calibrate a new
study. Its standard deviations are nine on the log scale, so it spans very
different response curves. A zero prior standard deviation fixes a parameter
as a Python extension useful for reduced models and reference checks.

There is an ambiguity in the printed article: the likelihood uses time one
as shorthand for the no-response category, while the explicit toxicity model
adds a separate failure effect. This implementation retains that effect.
A response exactly at one has no failure effect. The
[source record](cibolus-sources.json) documents this resolution; it is not a
claim of equivalence to the original executable.

## Prediction and allocation

For an observation schedule ending at one, response categories are immediate
bolus response, each detection interval, and no response. The joint prediction
crosses these categories with toxicity absent/present. Utilities must use the
same category order, with columns for toxicity absent and present. Expected
utility averages those supplied utilities over the joint distribution.

The decision rule screens conditional toxicity for a response at time one and
the probability of response by one. Conditional toxicity at one, toxicity
following failure, and marginal toxicity are distinct quantities. Using the
marginal value in the published safety rule would change the design.

A pair is excluded if its posterior probability of exceeding the toxicity
limit is greater than the toxicity cutoff, or its posterior probability of
falling below the efficacy limit is greater than the efficacy cutoff. Both
tail events and exclusions are strict; equality at a cutoff is acceptable.
The next assignment maximizes posterior mean utility among eligible pairs.
Escalation cannot skip an untried concentration, while bolus fraction is not
subject to that restriction. Initial assignment uses the configured starting
pair. Final selection removes the concentration restriction. Exact utility
ties use lowest concentration, then lowest bolus fraction as an explicit
Python convention.

## Python example

This synthetic example uses four detection intervals. It is an interface
demonstration, not the paper's calibrated study configuration.

```python
import numpy as np
from mdanderson_stats import cibolus_predict

log_parameters = np.log(
    [
        0.5,
        0.7,
        0.8,
        0.08,
        1.4,
        1.6,  # response parameters
        0.135,
        0.9,
        0.12,
        0.25,
        0.2,  # toxicity parameters
    ]
)
concentrations = [0.2, 0.4]
bolus_fractions = [0.1, 0.2]
endpoints = [0.25, 0.5, 0.75, 1.0]
utility = [[100, 10], [90, 8], [70, 6], [50, 4], [30, 2], [0, 0]]
prediction = cibolus_predict(
    log_parameters,
    concentrations,
    bolus_fractions,
    endpoints,
    utility=utility,
)
print(prediction.expected_utility)
print(prediction.joint.sum(axis=(-2, -1)))
```

The joint prediction axes are concentration, bolus fraction, response category
and toxicity. Keep the same concentration/bolus ordering when supplying counts
of previously treated patients.

```python
from mdanderson_stats import (
    CiBolusObservation,
    CiBolusPrior,
    fit_cibolus,
    cibolus_decision,
)

observations = [
    CiBolusObservation(0.2, 0.1, "bolus", False),
    CiBolusObservation(0.4, 0.1, "exact", False, time=0.4),
    CiBolusObservation(0.2, 0.2, "interval", True, lower=0.25, upper=0.5),
    CiBolusObservation(0.4, 0.2, "failure", True),
]
prior = CiBolusPrior(log_parameters, np.full(11, 0.4))
fit = fit_cibolus(
    observations,
    prior,
    concentrations,
    bolus_fractions,
    endpoints,
    utility=utility,
    draws=128,
    warmup=64,
    chains=2,
    rng=np.random.default_rng(86),
)
decision = cibolus_decision(
    fit,
    treated=[[1, 1], [1, 1]],
    toxicity_limit=0.4,
    toxicity_cutoff=0.9,
    efficacy_limit=0.3,
    efficacy_cutoff=0.9,
)
print(decision)
```

These short chains demonstrate the interface. Check posterior risk precision
and sampling diagnostics before interpreting an analysis. A completed sampler
run alone does not establish convergence or adequate tail-probability precision.

`fit.log_parameters` has axes chain, retained draw and parameter. Grid results
have axes chain, retained draw, concentration and bolus fraction; `fit.joint`
also has response-category and toxicity axes. Parameter and utility summaries
include intervals, classical split R-hat and batch-means Monte Carlo errors.
R-hat is undefined for fixed coordinates. Empty observations produce independent
prior draws and do not use warmup. With observations, fitting uses serial
elliptical slice sampling, an explicit Python alternative to the paper's
coordinate-wise sampling scheme.

The fitter accepts at most 200 observations, 20 concentration levels, 20 bolus
levels and 20 detection endpoints. The combined retained parameter, likelihood,
joint-probability and grid-summary arrays cannot exceed two million cells.
Likelihood evaluation and patient/prediction work budgets are also enforced;
`max_evaluations` and `max_work` control these limits. Impossible minimum budgets
are rejected before sampling. Chains run sequentially. These bounds limit
allocations but are not a total process-memory guarantee.

Response calculations retain small interval probabilities on the log scale.
Known saturated exposure limits return zero survival; the instantaneous hazard
can be infinite at the origin without a bolus or at extreme parameter values.
Parameters outside floating-point range and other unrepresentable calculations
raise errors rather than being silently treated as posterior rejections.

## Numerical evidence and coverage

Independent R quadrature evaluates response hazards and interval densities.
The reference data include zero/full bolus limits, nearly constant hazards,
underflowed concentration powers, mixed observation types and a reduced
one-dimensional posterior. See the [audit](../research/cibolus-audit.md) for
the checks actually completed.

[Complete-outcome cohort simulation](cibolus-trials.md) now generates the
joint response/toxicity cells, fits after each cohort and applies the allocation
and final-selection rules. Full prior elicitation/calibration, calendar and
pending-outcome conduct, aggregate operating characteristics, native input/report
workflows and executable parity remain open. The article and original
executable are not bundled.
