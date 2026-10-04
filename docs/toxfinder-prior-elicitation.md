# ToxFinder physician-prior elicitation

The package can fit the paper's four Gamma-prior constraints for either
single-agent toxicity curve and combine the two fits with explicitly supplied
interaction moments. This adds a calculation that is separate from ToxFinder's
still-unresolved second-stage information criterion.

```python
from mdanderson_stats import elicit_toxfinder_prior

elicitation = elicit_toxfinder_prior(
    agent1_doses=[600, 1200, 1400, 2000],  # negligible, AD, high toxicity, above target
    agent2_doses=[350, 600, 700, 800],
    target=0.30,
    interaction_mean=[1.0, 0.05],
    interaction_variance=[3.0, 3.0],
)
prior = elicitation.prior
print(elicitation.agent1.probability_residuals)
```

Each dose vector contains `(d1, AD, d3, d4)`: the negligible-toxicity dose,
single-agent acceptable-dose reference, high-toxicity dose, and dose believed
almost certainly to exceed the target. Only `d1 < AD`, `d3 > AD`, and `d4 > AD`
are required; the paper does not require an ordering between `d3` and `d4`.
The fitted `alpha` and `beta` coordinates are independent Gamma distributions
parameterized by shape and scale. The result reports those parameters, their
means and variances, both fitted elicitation probabilities, probability
residuals, and the residual of the third equation.

For one agent, equations (7)–(10) in §3.3 of Thall et al. define the fit. Write
`zj = dj / AD` and `g(η) = η / (1 + η)`. The equations are

```text
Pr[g(alpha * z1**beta) < q_low] = confidence
E[alpha] = target / (1 - target)
E[alpha * z3**beta] = q_high / (1 - q_high)
Pr[g(alpha * z4**beta) > target] = confidence
```

The paper defines `G(a,b)` as Gamma(shape `a`, scale `b`), with mean `a*b`
and variance `a*b**2`. The target and the third equation fix `E[alpha]` and
analytically determine beta's scale for each beta shape. The two remaining
probability equations are solved numerically. These are expectations about
toxicity **odds** where written as `E[alpha * z**beta]`; they are not
expectations of toxicity probabilities.

The defaults `q_low=0.05`, `q_high=0.60`, and `confidence=0.99` reproduce the
paper's elicitation questions. The paper explicitly permits clinicians to
replace the 5% and 60% values. The numerical quadrature, bounded shape search,
multi-starts, stopping tolerance, and failure behavior are Python conventions;
they are not verified native executable settings. A fit outside the supported
solver range or failing its residual criterion raises an error instead of
silently returning a nearest prior.

The default fit uses 128 probability-space Gauss-Legendre nodes and validates
the fitted probability constraints again with adaptive integration over the
beta Gamma density. The search uses six starts, Gamma shapes from `1e-6` to
`1e4`, probability residual tolerance `2e-5`, and an adaptive quadrature error
limit `5e-7`. These bounds and tolerances are Python conventions.

The two-agent builder requires `interaction_mean` and `interaction_variance`
for `(alpha3, beta3)`, so it never guesses between the paper's general
recommendation `(means=(1, 0.05), variances=(3, 3))` and its case-study value
`beta3 mean=1, variance=0.9`. These are distinct published specifications.

The paper's Table 1 provides a reference at target `0.30`: for Gemcitabine,
the four doses are `(600, 1200, 1400, 2000)`, with alpha mean/variance
`(0.4286, 0.1054)` and beta mean/variance `(7.6494, 5.7145)`. For
Cyclophosphamide, the doses are `(350, 600, 700, 800)`, with corresponding
moments `(0.4286, 0.0791)` and `(7.8019, 3.9933)`. The paper's prose has an
index typo around the third elicited dose; the implementation follows its
four questions, equations, and Table 1, where `d2=AD` and `d3>AD`.
Those published rounded moments do not reproduce all four printed equations
at exactly `0.99` when evaluated under the stated independent Gamma model:
adaptive integration gives Gem probabilities about `0.9866` and `0.9922`, and
Cyclophosphamide probabilities about `0.9778` and `0.9788`. Therefore the
elicitor solves the printed constraints and does not force its answer to the
Table 1 moments; the difference is retained as a source inconsistency.

Source: [Thall et al., “Dose-Finding with Two Agents in Phase I Oncology
Trials” (2003), §3.1 and §3.3, DOI 10.1111/1541-0420.00058](https://odin.mdacc.tmc.edu/~jjlee/clinical_trials2012/Thall/References/11_Biometrics%20Two%20Agents%202003.pdf).
