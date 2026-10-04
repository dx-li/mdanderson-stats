# Automatic two-arm binary success-cutoff calibration

`calibrate_binary_two_arm_success_cutoff` searches the finite decision states of
the existing two-arm beta-binomial model. It minimizes the cutoff among
numerically distinguishable cutoffs in the supplied closed interval whose
computed probability of incorrect decision (PID) is at most `target` and whose
Bayesian success probability is positive.

```python
from mdanderson_stats import calibrate_binary_two_arm_success_cutoff

calibration = calibrate_binary_two_arm_success_cutoff(
    1,
    1,
    target=0.3,
    cutoff_range=(0.49, 0.99),
    margin=0.0,
)
print(calibration.cutoff)
print(calibration.operating_characteristics.incorrect_decision_probability)
```

The treatment and control outcomes are independent. Each arm has its own beta
design prior, which generates the joint predictive mass and truth classification,
and beta analysis prior, which determines the posterior probability that the
treatment-minus-control risk difference exceeds `margin` (`direction="greater"`)
or is below it (`direction="less"`). Success uses the strict rule
`posterior_probability > cutoff`; states tied at the cutoff are excluded.
`null_rate` is the control event rate used for frequentist type I error, and
`null_treatment_rate` defaults to that same value. Explicitly supply a
margin-boundary pair when that is the intended null.

The search prepares the two-arm response-count table once and examines all
distinguishable decision states. PID need not be monotone in the cutoff when the
design and analysis priors differ, so this search does not use binary search.
The smallest-feasible policy is a Python convention; the application help gives
the default cutoff interval `[0.6, 0.999]` but does not specify its optimizer or
tie policy. This function does not claim native search or report parity.

For positive cutoffs, the existing table rejects values within a state's
posterior quadrature-error interval. The search therefore considers only
representable float cutoffs outside conservatively outward-rounded versions of
those intervals. When an interval removes a decision boundary, the candidate
on its right is the first float above the merged conservative interval endpoint.
This is the smallest feasible cutoff among those conservative candidates, not
a claim that every guard-safe floating-point value was searched. If the
intervals leave no distinguishable cutoff or no feasible state can be assessed, the function raises
`ArithmeticError` and explains that mathematical feasibility is unresolved;
it does not label the case infeasible. At cutoff zero, the existing table's
special endpoint rule is preserved.

The result reports the full operating-characteristic record recomputed by the
existing table evaluator at the selected cutoff. The target comparison uses its
computed PID. The table retains analysis posterior integration errors but not
errors for design-prior truth probabilities, so this is not a rigorous upper
bound on the mathematical PID. Rare probabilities can underflow under the
existing float64 beta-binomial calculations.

Each arm supports at most 1,000 patients, with at most 40,000 joint response
count pairs. The search sorts the prepared states once and uses compensated
suffix sums; it does not build a candidate-by-state matrix. See the
[source and algorithm audit](../research/success-two-arm-automatic-audit.md).
