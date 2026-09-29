# Finite-grid binary rBOP2 calibration

`calibrate_rbop2_binary` searches only the finite lower/upper cutoff curves the
caller supplies. It follows the help-file objective: among candidates whose
overall positive probability under the declared null is no greater than
`alpha`, select the one with greatest overall positive probability under the
declared alternative. It does not infer a native cutoff grid or claim to
reproduce the application's calibration search.

The source describes the efficacy null at `experimental = control + margin`
and the toxicity null at `experimental = control - margin`. This API requires
the caller to provide the exact null and alternative rate pairs, so it can
also evaluate other explicitly declared scenarios. Feasibility uses a direct
floating-point comparison `type_i_error <= alpha`; the existing posterior engine's
strict-futility/inclusive-superiority cutoff rules are retained. A posterior
comparison too close to a cutoff is resolved using the same exact rational
case or tighter quadrature as binary monitoring, otherwise it raises.

The constraint applies only to the supplied null rate pair. It is not a
worst-case guarantee over a composite null or over unknown control-arm rates.

```python
from mdanderson_stats.rbop2_calibration import calibrate_rbop2_binary

fit = calibrate_rbop2_binary(
    [[1, 1], [2, 2]],
    calibration_prior=[[1, 1], [1, 1]],
    analysis_prior=[[2, 1], [1, 2]],
    endpoint="efficacy",
    margin=0,
    null_rates=(0.3, 0.3),
    alternative_rates=(0.6, 0.3),
    cutoff_candidates=[
        [[0.25, 0.8], [0.8, 0.8]],
        [[0.25, 0.9], [0.9, 0.9]],
        [[0.05, 0.95], [0.8, 0.8]],
        [[0.6, 0.9], [0.8, 0.8]],
        [[0.25, 0.9], [0.9, 0.9]],
    ],
    alpha=0.2,
)
if fit.selected_design is None:
    raise RuntimeError("no supplied cutoff curve met the null constraint")
print(fit.selected_index)  # 3
print(fit.calibration_power[fit.selected_index])  # 0.3696
print(fit.analysis_power[fit.selected_index])  # 0.6132
```

The calibration prior controls candidate ranking and the null constraint.
When `analysis_prior` is supplied, the selected cutoff curve is separately
evaluated under that prior and the result retains both sets of operating
characteristics. Its analysis-prior type-I error is reported, not promised to
remain at `alpha`. Without `analysis_prior`, the calibration prior is used for
both. Ties choose higher power first, then lower expected total enrollment at
the calibration null, then earlier input order. If no candidate is feasible,
the selected index and design are `None`; every candidate's feasibility,
power, type-I error, expected null enrollment, and largest estimated posterior
absolute error remain available for inspection. This statewise posterior
integration error is not an operating-characteristic or type-I-error bound.

The implementation shares each prior's statewise posterior probability
surface across all candidate curves and reuses the exact operating
characteristic recursion. Its deterministic work estimate is checked before
posterior evaluation. It does not use simulation or random-number generation.
