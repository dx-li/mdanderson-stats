# Pooled-error CI for an observed combination

The observed-combination method in Lee and Kong (2009), Section 2, has a
specific fallback when an observed combination effect has no replicate-based
variance estimate. `interaction_index_pooled_error` uses it for an explicitly
supplied mean response `y` and combination dose vector. Fit each single-agent
median-effect curve from its own observations first.

For drug `i`, let `s_i²` be the OLS residual mean square on the transformed
response `log(y / (1-y))`, and let `df_i = n_i - 2`. This implementation uses
the residual-degree-of-freedom weighted pool

```text
s_pool² = sum(df_i * s_i²) / sum(df_i)
Var(y) ≈ y² (1-y)² s_pool²
```

The second equation follows from the paper's delta approximation for
`Var(logit(y))`. The variance contribution is evaluated directly on the
logit-response scale to avoid underflow from converting to raw-effect
variance and back. The fitted coefficient covariance matrices remain those of
the individual regressions; this procedure pools only the missing observed
combination response-error contribution. The interval uses the Section 2
Student-t degrees of freedom `sum(n_i - 2)`.

This is an approximation under the paper's constant transformed-error
variance assumption. It is not a pooled fit of the dose-response coefficients.
If replicate combination responses are available, use `interaction_index`
with the variance of their mean as `effect_variance`. For a fixed ray, use
`interaction_index_ray`: the paper's Section 3 delta method propagates the
separate covariance matrices for each single-agent regression and the
combination regression. The Section 2 pooled-error fallback does not replace
that ray calculation. Neither method creates a simultaneous confidence band.

```python
from mdanderson_stats import fit_median_effect, interaction_index_pooled_error

fits = [
    fit_median_effect([1, 2, 4, 8], [0.85, 0.70, 0.45, 0.22]),
    fit_median_effect([1, 2, 3, 4, 6, 8], [0.92, 0.78, 0.64, 0.51, 0.34, 0.21]),
]
result = interaction_index_pooled_error(fits, [0.4, 0.6], 0.5)
assert result.degrees_of_freedom == 6
assert result.log_interval.shape == (2,)
```

The primary contract is Section 2 equations (8) and (9) in [Lee and Kong,
“Confidence Intervals of Interaction Index for Assessing Multiple Drug
Interaction”](https://pmc.ncbi.nlm.nih.gov/articles/PMC2796809/). The cached
software readme lists its corresponding native routine as `CI.known.effect`.
This API implements the equation, not the original S-Plus/R interface or
random-number behavior.
