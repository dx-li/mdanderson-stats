# Exhaustive single-arm binary success-cutoff search

`calibrate_binary_success_cutoff` searches a closed cutoff interval for the
single-arm beta-binomial operating-characteristic model. It uses the existing
`SuccessCalibration` result returned by the explicit-grid
`calibrate_success_cutoff` helper.

```python
from mdanderson_stats.success_calibration_binary_search import (
    calibrate_binary_success_cutoff,
)

result = calibrate_binary_success_cutoff(
    40,
    0.05,
    margin=0.3,
    design_prior=(2, 4),
    analysis_prior=(1, 1),
)
print(result.cutoff)
print(result.operating_characteristics.incorrect_decision_probability)
```

The source help advertises calibrating posterior cutoff values to a target
probability of incorrect decision and exposes a candidate interval, defaulting
to `[0.6, 0.999]`. The source rule declares success when the posterior
probability of the favorable effect is strictly greater than the cutoff.
For a single-arm binary trial of size `n`, there are only `n + 1` possible
response counts and thus finitely many posterior-probability breakpoints. The
implementation evaluates the lower interval endpoint and the unique
breakpoints inside the interval. At each breakpoint it applies the strict
rule exactly, so these values represent every distinct decision set in the
closed interval.

The returned cutoff is the smallest evaluated threshold with positive success
probability and PID no greater than `target`; that smallest-feasible tie policy
is an explicit Python convention. `candidates_evaluated` counts the distinct
breakpoint states directly checked by binary search. If none meets the target
with positive success probability, the function raises `ValueError`. Each
checked result is computed with `binary_success_oc`, preserving that API's
result and floating-point conventions. Beta tails and masses use float64
arithmetic; extremely small positive probabilities may underflow, and this
function does not claim native application search or numerical parity. A
structurally nonempty success set with unrepresentable total success mass, or
tail calculations that violate monotone response ordering, raise
`ArithmeticError` rather than being treated as an infeasible design.

This search is limited to single-arm binary outcomes. The current paper and
application also describe two-arm, normal and log-hazard-ratio settings; this
function does not infer a global optimizer for those models. For arbitrary
caller-supplied candidate grids or those other models, use
[`calibrate_success_cutoff`](success-calibration.md#calibration-evidence-and-remaining-coverage)
with the matching operating-characteristic function. The separate two-arm,
normal and survival wrappers are documented in the
[`normal and survival section`](success-calibration.md#normal-outcomes-and-survival-approximation)
and are not used by this exact single-arm search.

The design prior generates the operating-characteristic truth and response
counts, while the analysis prior determines each posterior success
probability. `margin`, `direction`, and `null_rate` follow
`binary_success_oc`; when omitted, `null_rate` equals `margin`.

Evidence: cached help in `research/raw/BayesianCalibration/calibration-help.txt`
specifies target-PID cutoff calibration and the optional candidate interval;
the paper's Equation (2.1) supplies the strict posterior cutoff rule. The
search algorithm and smallest-feasible tie policy are Python implementation
choices, not verified native calibration behavior.
