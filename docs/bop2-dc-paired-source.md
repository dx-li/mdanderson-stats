# Paired BOP2-DC source notes

`bop2_dc_paired_design` implements the paired endpoint combinations described
in Section 2.1.4 and Section 2.2 of Zhao, Li, Liu, and Yuan, “Bayesian optimal
phase II designs with dual-criterion decision making” ([arXiv:2112.10880](https://arxiv.org/abs/2112.10880)).
The primary software reference is the [MD Anderson BOP2-DC app](https://biostatistics.mdanderson.org/shinyapps/BOP2-DC/).

Counts use the explicit Dirichlet category order `(E1E2, E1¬E2, ¬E1E2,
¬E1¬E2)`. Marginal endpoint posteriors are formed by summing the appropriate
Dirichlet cells; endpoint decisions are then combined, rather than replacing
the model with an independence approximation. Multiple efficacy uses OR at
the final go rule and AND at the final no-go rule. Efficacy/toxicity uses AND
for final go and OR for final no-go. Interim no-go uses AND for multiple
efficacy and OR for efficacy/toxicity.

For toxicity, the public `lrv` and `cmv` fields retain the user-supplied
toxicity cutoffs. The posterior is evaluated directly as the lower tail of the
toxicity-rate beta distribution. This is equivalent to transforming to
success=`no toxicity` with thresholds `1-LRV < 1-CMV`, while avoiding loss of
precision when a toxicity cutoff is extremely small.
Exact operating characteristics propagate probability over the two marginal
event counts using the supplied joint four-cell outcome distribution. This
state reduction is valid because the decision depends only on those marginals
and the fixed prior; the joint transition probabilities still retain endpoint
association. Early no-go states are absorbed before the next patient arrives.
This independently implemented recursion is checked against exhaustive path
enumeration, not against a recovered app backend. Calendar-time simulation and
finite-grid calibration remain separate work.
