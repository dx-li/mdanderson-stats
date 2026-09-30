# Proportional-density profile inference for beta

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2721282/),
section 3.2, tests a specified time-tilt coefficient `beta = beta0` while
profiling the intercept `alpha`. Under a common censoring distribution, the
conditional likelihood-ratio statistic has asymptotic chi-square calibration
with one degree of freedom. The function below exposes that test for any finite
`beta0`. The caller must explicitly set `equal_censoring=True`; it raises for
`False` and does not infer equality from the observations.

```python
import numpy as np
from mdanderson_stats import proportional_density_profile

time = np.arange(1.0, 15.0)
event = np.ones(time.size)
treatment = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
# Add equal arm-specific censoring records; their values do not enter the
# conditional failure likelihood, but the caller must justify the assumption.
censor_time = np.array([3.5, 8.5, 15.0])
time = np.r_[time, censor_time, censor_time]
event = np.r_[event, np.zeros(6)]
treatment = np.r_[treatment, np.zeros(3), np.ones(3)]

result = proportional_density_profile(
    time,
    event,
    treatment,
    beta_null=0.2,
    equal_censoring=True,
    confidence=0.95,
)
print(result.likelihood_ratio, result.pvalue, result.beta_interval)
```

The fit maximizes the same conditional logistic likelihood used by
`proportional_density`. Under the null it solves the intercept score equation,
equivalently choosing the intercept so the fitted treatment-arm failure
probabilities sum to the observed number of treatment-arm failures. The
likelihood-ratio test is source-defined; the confidence interval is an explicit
Python extension obtained by inverting that test at the requested confidence
level. It covers `beta`, not `alpha_star` or the incidence contrast.

The test does not address unequal-censoring calibration. The paper mentions
bootstrap critical values for that setting but does not specify the null-data
generation and restricted-refit procedure. The incidence contrast is a
different hypothesis and is not tested by this function. Existing fit defaults
and their archive-derived incidence Wald result remain unchanged.

The intercept/profile and interval roots use centered, scaled failure times.
The conservative event-vector work bound is checked before fitting and can be
lowered with `max_work`. Nonrepresentable predictors, exhausted brackets, or
nonconvergent roots raise explicit errors; interval endpoints are never
substituted when root finding fails. A test without interval inversion can use
`interval=False`.
