# Diagnostic populations and binormal ROC analysis

Catalog entries **104 (DIAG)** and **105 (DTROC)** are implemented as reusable,
broadcast Python APIs. The specifications are the official
[DIAG help](https://biostatistics.mdanderson.org/shinyapps/DIAG/DIAG.pdf) and
[DTROC help](https://biostatistics.mdanderson.org/shinyapps/DTROC/DTROC.pdf), by
Yanhong Zhou and J. Jack Lee. [Source hashes](diagnostic-and-roc-sources.json)
pin the inspected documents. These are independent mathematical implementations;
no unavailable Shiny server code or pixel-identical browser interface is claimed.

DIAG accepts either a classification table or sensitivity, specificity and
prevalence, projects a population of 10,000 by default, and varies predictive
values with prevalence. DTROC assumes Gaussian marker distributions in disease
and control groups; a higher marker value tests positive. It selects thresholds
by control percentiles and reports diagnostic quantities and AUC. All numerical
inputs to the source density, ROC, prevalence and population displays are exposed
as arrays. Browser sliders, button states and square-grid artwork are not
additional statistical methods and are not replicated.

## DIAG

```python
import numpy as np
from mdanderson_stats import diagnostic_population, diagnostic_population_from_counts

# Rows: test positive, test negative. Columns: disease positive, disease negative.
table = [[5, 4], [3, 2]]
result = diagnostic_population_from_counts(table)
print(result.sensitivity, result.specificity)  # 0.625, 0.33333333
print(result.positive_predictive_value)  # 0.55555556
print(result.negative_predictive_value)  # 0.4
print(result.expected_counts)  # table * (10000 / 14)

# Reuse estimated test performance at prevalences from another population.
prevalences = np.linspace(0, 1, 101)
curve = diagnostic_population_from_counts(table, prevalence=prevalences)
known_performance = diagnostic_population(0.9, 0.8, 0.5)
```

Count-based sensitivity and specificity reuse the completed CTA implementation.
A table must have final shape (2,2), nonnegative integer counts and total below
2**53. Both disease groups must be represented; otherwise their conditional test
performance cannot be estimated. `prevalence=None` estimates prevalence from the
table, while an override applies the estimated test performance to a new target
population. This does not imply that a case-control sample's disease proportion
is a population prevalence estimate.

`DiagnosticPopulation` includes sensitivity, specificity, false-positive and
false-negative rates, prevalence, PPV, NPV, FDR, joint probabilities and expected
counts. Its final two table axes follow the input convention. Counts are real
expectations, not rounded individuals. Population size is a nonnegative integer
below 2**53. A zero population yields zero counts but leaves conditional
probabilities defined by the model.

For prevalence p, sensitivity s and specificity c, joint probabilities are

```text
                Disease +        Disease -
Test +              p*s          (1-p)*(1-c)
Test -          p*(1-s)               (1-p)*c
```

Predictive values normalize the appropriate test row. Products and normalization
are evaluated in logarithms, with FDR computed directly rather than subtracted
from PPV. If a conditioning test outcome is impossible, its predictive value is
NaN. Inputs broadcast; a prevalence array therefore provides the source's PPV/NPV
curves without a separate simulation or fitting operation. Stored result arrays
are owned and read-only.

## DTROC

```python
from mdanderson_stats import BinormalROC

model = BinormalROC(control_mean=-1, control_sd=1, disease_mean=1, disease_sd=1)
point = model.at_percentile(0.8, prevalence=0.5)
print(point.threshold)  # -0.15837877
print(point.auc)  # 0.92135040
print(point.diagnostic.sensitivity)  # approximately 0.876646
print(point.diagnostic.expected_counts)

fpr = np.linspace(0, 1, 1001)
sensitivity = model.curve(fpr)
markers = np.linspace(-5, 5, 501)
density_values = model.densities(markers)  # final axis: control, disease
```

Percentiles and prevalences are fractions in [0,1], unlike the source sliders'
percent units. `at_threshold` also accepts a threshold directly, including
infinite all-positive/all-negative limits. For threshold t,
sensitivity is Phi((disease_mean-t)/disease_sd), and specificity is
Phi((t-control_mean)/control_sd). AUC is the probability that an independent
disease marker exceeds a control marker:

```text
Phi((disease_mean-control_mean) / sqrt(disease_sd**2 + control_sd**2))
```

Standard deviations must be positive and finite. Model parameters broadcast;
for several models evaluated on one grid, give parameter arrays a trailing
singleton dimension. `curve` accepts false-positive rates directly, avoiding the
loss from subtracting tiny FPRs from one. It computes standardized differences
without materializing thresholds, preserving curves even where a large marker
offset prevents representing individual quantile cut-points. `at_percentile`
raises `ArithmeticError` when its requested finite threshold cannot be resolved.

Direct normal log tails feed predictive-value calculations. Thus, for two
identical marker distributions at threshold 40 standard deviations, PPV still
equals prevalence although both positive-test joint probabilities underflow in
float64. Infinite thresholds represent truly impossible outcomes and produce
NaN for the corresponding conditional probability. This is a distinction between
floating-point underflow and an impossible conditioning event.

## Validation

Nine focused tests cover the manual examples, exact 2x2 ratios, prevalence
substitution, projected count conservation, impossible outcomes, log-tail
conditioning, diagonal and reflected ROC curves, endpoints, large scales,
quantile representability, batching and density normalization. AUC is checked
independently by integrating the ROC curve and by comparing 100,000 simulated
independent marker pairs. Related CTA diagnostic tests exercise the reused count
estimator. No full-package test run is required for each small API addition;
checks concentrate on these statistical contracts and their shared dependency.
