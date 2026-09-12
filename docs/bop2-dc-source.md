# BOP2-DC source notes

This module implements the binary efficacy BOP2-DC decision layer from Zhao,
Li, Liu, and Yuan, “Bayesian optimal phase II designs with dual-criterion
decision making” (Pharmaceutical Statistics, 2023, DOI 10.1002/pst.2296;
preprint [arXiv:2112.10880](https://arxiv.org/abs/2112.10880)). The primary
software source is the [MD Anderson BOP2-DC app](https://biostatistics.mdanderson.org/shinyapps/BOP2-DC/),
version V1.0.9.0 (updated 2026-03-23).

The efficacy parameter has two prespecified reference values: `lrv` for
statistical significance and `cmv` for clinical relevance, with `lrv < cmv`.
Posterior probabilities are computed under a Beta prior. At an interim look,
the trial stops for no-go only when both posterior probabilities fall below
their information-adaptive cutoffs. At the final look, both criteria must pass
strictly for `final_go`; both must fail strictly for `final_no_go`; all other
outcomes are `final_consider`. Equality therefore continues at interim and is
consider at the final analysis.

The implementation exposes explicit prior parameters and separate cutoff and
gamma values. It does not silently implement the app's other endpoint modes
(efficacy/toxicity, multiple efficacy, or time-to-event); those require their
own endpoint-specific posterior and decision-combination contracts.

Operating characteristics retain interim no-go mass separately in
`stop_no_go`; `no_go_probability` adds that mass to final no-go outcomes.
