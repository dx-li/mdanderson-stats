# BOIN12 implementation scope

This implementation follows Lin, Zhou, Yan, Li, and Yuan, “BOIN12: Bayesian
Optimal Interval Phase I/II Trial Design for Utility-Based Dose Finding in
Immunotherapy and Targeted Therapies,” *JCO Precision Oncology* 4 (2020),
1393–1402: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7713525/>.

The MD Anderson BOIN12 application and its current next-dose guide are the
primary software references:

- <https://biostatistics.mdanderson.org/shinyapps/BOIN12/>
- <https://biostatistics.mdanderson.org/shinyapps/BOIN12/BOIN12Nextdose.pdf>
- <https://biostatistics.mdanderson.org/shinyapps/BOIN12/BOIN12Stop.pdf>

The module covers complete, binary toxicity and efficacy outcomes. Toxicity
and efficacy use independent Beta(1,1) marginal posteriors. Admissibility is
`Pr(p_toxicity > toxicity_limit) < 0.95` and
`Pr(p_efficacy < efficacy_limit) < 0.90`, the defaults described in the paper.

Desirability uses the quasi-beta-binomial construction: utility-weighted
events divided by 100 are treated as the event count in a Beta posterior with
Beta(1,1) prior. The public API reports the posterior benchmark probability on
the natural `[0,1]` scale; an RDS display adapter can multiply it by 100 to
match the application table. The default utility order is
`(no toxicity/efficacy, no toxicity/no efficacy, toxicity/efficacy,
toxicity/no efficacy) = (100, 40, 60, 0)`. For utilities satisfying the
cross-sum condition `u01 + u10 = u00 + u11`, marginal toxicity and efficacy
counts determine the utility events; otherwise callers must provide the joint
`efficacy_without_toxicity` count.

Next-dose movement uses BOIN escalation/de-escalation boundaries generated from
the supplied toxicity limit, which is the BOIN12 boundary target in the
published single-stage method. It compares posterior utility
probabilities among admissible neighboring doses and implements the documented
exploration rule after more than eight patients at the current dose. Final OBD
selection isotonicizes empirical toxicity rates, identifies the dose nearest
the toxicity limit as the MTD, and maximizes posterior benchmark exceedance
probability among
admissible doses at or below that MTD. The `escalation` 0.2.3 implementation
was used as an executable third-party reference for numerical fixtures; it is
not treated as the native MD Anderson application backend, whose admissibility
and sticky-elimination behavior take precedence.

Late-onset outcomes, pending-data weighting, and multilevel endpoints are not
inferred here; they require their separate BOIN12/TITE-BOIN12 specifications.
