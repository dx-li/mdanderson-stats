# BARD categorical response model

This module calibrates a categorical logistic response model to user-supplied
population response rates. Supply the joint factor-profile distribution
explicitly: marginal factor frequencies alone do not determine dependence
between factors.

```python
import numpy as np

from mdanderson_stats.bard_response import bard_response_model

profiles = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
joint_weights = np.array([0.40, 0.10, 0.20, 0.30])
# Rows are factors; columns are their level-specific odds ratios versus level 1.
odds_ratios = np.array([[1.0, 2.5], [1.0, 0.6]])
fit = bard_response_model([0.25, 0.42, 0.58], profiles, joint_weights, odds_ratios)

print(fit.intercepts)
print(fit.conditional_probabilities)
print(fit.marginal_residuals)
print(fit.probability(2, [2, 1]))
```

For dose `d` and profile `x`, the model is

```text
logit Pr(response | dose=d, X=x)
    = intercept[d] + sum_f log(odds_ratio[f, x[f]])
```

Each intercept is solved so the joint-weighted conditional probabilities
reproduce that dose's supplied population response. Odds ratios compare
factor levels **conditionally**, holding all other factors fixed. Marginal
odds ratios after averaging over other factors generally differ. This code
does not infer or impose a dependence structure.

`factor_profiles` is a bounded integer matrix `(profiles, factors)` with
one-based category labels. `profile_probabilities` supplies one nonnegative
mass per row and must sum to one within `1e-12`; only that roundoff-sized
deviation is normalized. `response_odds_ratios` is a positive finite matrix
`(factors, levels)` with an exact 1 in each level-1 column. All rows share the
same rectangular category width. A category absent from the profile table is
allowed and may have zero population mass, but its odds ratio remains part of
the caller's explicit model. Zero-mass profiles do not affect calibration
and are retained in returned conditional probabilities.

Population rates of exactly 0 or 1 yield intercepts `-inf` or `+inf` and exact
all-zero or all-one response probabilities. Interior rates use a bracketed
log-tail inversion; an unresolved solve or numerically flat inverse raises
`ArithmeticError`. Inversion is rejected when the weighted response derivative,
divided by the smaller target marginal tail, is below `1e-8`. This sensitivity
guard is a Python numerical convention. Returned arrays are owned and
read-only. The evaluator
`bard_response_probabilities(intercepts, factor_profiles,
response_odds_ratios)` uses the same profile and odds-ratio contract and can
evaluate published scenario intercepts directly. Its intercepts may be
finite or infinite, but not NaN.

Bounds are 100 doses, five factors, 20 levels per factor, 100,000 profiles,
1,000,000 output cells, and 100,000,000 calibration profile-iterations.
These are Python resource limits, not BARD application limits. This is a
source-based response-model component; it does not implement BARD's full
two-stage operating-characteristic simulation or claim native GUI parity.
