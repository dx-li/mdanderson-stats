# U2OET generalized Aranda–Ordaz model

`U2OETGAOMarginal` and `u2oet_gao_probabilities` implement the GAO comparison
model in Appendix A of the [U2OET paper](https://odin.mdacc.tmc.edu/~pfthall/main/JRCCS_2017_ph12_2agent_utility.pdf).
They evaluate explicitly supplied parameters, joint likelihoods and expected
utilities. [Posterior fitting](u2oet-gao-fit.md) uses explicit caller-defined
priors. Native prior-file interpretation remains pending.

```python
import numpy as np
from mdanderson_stats import U2OETGAOMarginal, u2oet_gao_probabilities

probability = u2oet_gao_probabilities(
    [1, 3],
    [2, 5],
    efficacy=U2OETGAOMarginal([[-1, 0.3]], [[0.2, -0.1]], lambda_=1),
    toxicity=U2OETGAOMarginal([[-0.4, -1.2]], [[-0.05, 0.15]], lambda_=1),
    kappa=0.4,
    association=0.55,
)
assert probability.joint.shape == (2, 2, 2, 2)
np.testing.assert_allclose(probability.joint.sum(axis=(-2, -1)), 1)
counts = np.zeros((2, 2, 2, 2), dtype=int)
counts[0, 0, 1, 0] = 3
assert np.isfinite(probability.loglikelihood(counts))
utility = [[20, 0], [100, 50]]  # efficacy rows, toxicity columns
print(probability.expected_utility(utility))
```

These are illustrative coefficients, not fitted estimates or calibrated prior
centers. Each marginal has 2–4 categories. Its intercept and slope arrays have
shape `(categories - 1, 2)`, with thresholds in rows and the two agents in
columns. Intercepts and slopes are unrestricted finite real values; no
monotonic dose-response claim follows without further coefficient constraints.
Each dose grid contains 2–5 strictly increasing positive **raw** doses. Changing
dose units requires the corresponding slope transformation.

For endpoint `k`, threshold `y` and raw doses `d1,d2`, define

```text
eta1 = intercept[y, 0] + slope[y, 0] * d1
eta2 = intercept[y, 1] + slope[y, 1] * d2
S = exp(eta1) + exp(eta2) + kappa * exp(eta1 + eta2)
gamma[y] = 1 - (1 + lambda[k] * S)**(-1 / lambda[k])
```

`lambda_` is positive and endpoint-specific; `kappa` is a single positive
interaction shared across both endpoints. At `lambda_=1`, continuation is
`S/(1+S)`. A small positive shape approaches `1-exp(-S)`; zero itself is
rejected. Category masses follow the continuation-ratio recurrence: failure
at the first threshold, successive continuation followed by failure, and the
product of all continuations for the last category.

`association` is the latent Gaussian correlation in `[-1,1]`. The GAO joint
law uses a Gaussian copula. The separate PDS/CMI API uses an FGM parameter with
a different meaning. The shared `U2OETProbabilities` result supports grouped
complete counts, optional toxicity-only counts and expected utilities; its
likelihood omits parameter-independent multinomial constants.

## Numerical scope

The implementation uses log sums and stable log-softplus calculations for
marginal continuations and category products. Independence directly adds the
marginal log probabilities, preserving representable log likelihoods even if
ordinary probabilities underflow. For nonzero nonsingular correlation,
Gaussian quantiles use log tails and joint cells use deterministic conditional
normal quadrature. The returned grid must preserve both marginals and total
mass. Unresolved positive rectangles or unrepresentable thresholds raise an
error; this is not an extreme Gaussian log-tail integrator.

At correlation `-1` or `1`, ordinary probability-interval overlap gives the
limiting copula, including structural zero cells. Very small overlap intervals
can round away. The numerical guarantees of the separate FGM tail calculation
do not apply to this Gaussian computation.

Independent base-R equations and conditional-normal integration verify 440
joint cells and 15 grouped likelihoods across binary and ordinal outcomes,
different positive shapes, a small-shape limit, and correlations `-1`, `-0.65`,
`0`, `0.55` and `1`. See [reference generator](../tools/reference_u2oet_gao.R)
and [audit](../research/u2oet-gao-audit.md). These validate explicit equations;
they do not establish native executable, default-prior or fitting equivalence.
