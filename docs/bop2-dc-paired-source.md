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

For toxicity, the endpoint is transformed to success=`no toxicity`. Thus a
toxicity LRV/CMV pair with `LRV > CMV` becomes no-toxicity thresholds
`1-LRV < 1-CMV`, preserving the lower-is-better toxicity interpretation.
This module intentionally covers the paired monitoring core; simulation and
finite-grid calibration for these paired modes remain separate work.
