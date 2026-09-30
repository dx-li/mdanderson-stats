# U2OET 2010 GAO probabilities

The original 2010 GAO model is exposed separately from the 2017 GAO
comparison. It centers each agent's dose grid, allows endpoint-specific
interaction coefficients (including valid negative values), and combines the
ordinal margins with a Gaussian copula. This is a fixed-parameter probability
calculation, not posterior fitting or native-application parity.

```python
import numpy as np

from mdanderson_stats.u2oet_gao2010 import (
    U2OETGAO2010Marginal,
    u2oet_gao2010_probabilities,
)

dose_a = [1.0, 2.0, 3.0]
dose_b = [1.0, 2.0]
efficacy = U2OETGAO2010Marginal(
    intercepts=[[-1.0, -0.6], [0.1, 0.3]],
    slopes=[[0.35, -0.1], [0.1, 0.2]],
    lambda_=0.8,
    gamma=0.2,
)
toxicity = U2OETGAO2010Marginal(
    intercepts=[[-1.4, -1.0], [-0.1, 0.2]],
    slopes=[[0.3, 0.25], [0.15, 0.2]],
    lambda_=1.1,
    gamma=-0.03,
)
probability = u2oet_gao2010_probabilities(
    dose_a,
    dose_b,
    efficacy=efficacy,
    toxicity=toxicity,
    association=0.25,
)
utility = np.array([[20, 10, 0], [60, 40, 10], [100, 70, 20]], dtype=float)
expected_utility = probability.expected_utility(utility)

# Full grouped outcomes: dose_a x dose_b x efficacy category x toxicity category.
complete = np.zeros((3, 2, 3, 3), dtype=int)
complete[0, 0, 0, 0] = 2
complete[1, 1, 2, 1] = 1
# Optional toxicity-only outcomes for patients whose efficacy is unevaluable.
toxicity_only = np.zeros((3, 2, 3), dtype=int)
toxicity_only[2, 1, 2] = 1
log_likelihood = probability.loglikelihood(complete, toxicity_only=toxicity_only)
```

`intercepts` and `slopes` have one row per ordinal threshold and two columns in
agent order. Each dose grid has 2–5 strictly increasing nonnegative levels; a
zero dose is permitted. At each threshold, each linear predictor is
`intercept + slope * (dose - mean(supplied dose grid))`. The two endpoint
marginals have separate `lambda_` and `gamma`; `gamma` is shared across that
endpoint's thresholds. `association` is the Gaussian-copula latent correlation.
The returned probability object's last two axes are efficacy category then
toxicity category, both zero-based.

For threshold predictors `eta1` and `eta2`, the continuation probability is

```text
1 - (1 + lambda * (exp(eta1) + exp(eta2) + gamma * exp(eta1 + eta2)))**(-1/lambda)
```

`lambda_` must be positive. Negative `gamma` is allowed only when the
interaction bracket remains strictly positive at every threshold and dose
pair in the supplied grids. The implementation evaluates the negative-term
remainder in the log domain when terms are separated and uses bounded
standard-library Decimal arithmetic when cancellation is material. It rejects
invalid parameter/grid combinations. Use the actual candidate dose grids when
checking the constraint; validity on a smaller grid does not imply validity on
a larger one.

`loglikelihood` accepts grouped complete outcome counts with the same shape as
`log_joint` and optional grouped toxicity-only counts with shape
`dose_a x dose_b x toxicity category`. It omits combinatorial constants and any
separate probability model for evaluability. Thus it does not fit or model the
paper's `zeta` parameter or informative evaluability mechanism.

This module does not implement the paper's prior calibration, original-prior
posterior sampler, dose-selection rule or clinical trial workflow. The paper
states normal priors for the linear coefficients and interactions, lognormal
priors for endpoint link shapes, and a uniform prior for the copula correlation;
validity constraints couple negative interactions to coefficients and the
dose grid. The elicited prior-center calculation and the native application's
parameter-file coordinate ordering remain unresolved. Use the separate
[2017 explicit-prior GAO fitter](u2oet-gao-fit.md) only for its documented
2017 parameterization.
