# TITE-BOIN12 final selection and BDA source audit

## Final OBD selection

The primary article (Zhou et al., *Statistics in Medicine* 41 (2022),
1918–1931, [PMC9199061](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/))
states that final OBD selection uses admissible doses no higher than the
estimated MTD, then chooses the dose with the greatest utility desirability.
The local `BOIN12Design.select_obd` already implements the corresponding
complete-data two-step rule: calculate admissibility, isotonic-regress the
observed dose toxicity rates, take the observed dose closest to the toxicity
limit as MTD, and maximize posterior utility desirability among admissible
doses at or below that MTD. `tite_boin12_select_obd` validates complete
patient-level TITE records, forms the exact joint binary counts, and delegates
to this existing implementation. Thus fully observed TITE histories reduce
exactly to ordinary BOIN12 selection. It does not select from pending data.

## Bayesian data augmentation findings

The article models the four joint binary outcomes in order
`(no toxicity/efficacy, no toxicity/no efficacy, toxicity/efficacy,
toxicity/no efficacy)` with a Dirichlet prior. Given complete or imputed cell
counts `n`, its P step is
`p | n ~ Dirichlet(a + n)`. It specifies prior total concentration one and
marginal prior means `Pr(toxicity)=0.5*phi_T` and
`Pr(efficacy)=phi_E`. Those three constraints do not determine all four joint
cell concentrations; an additional dependence convention is needed.

For each endpoint q, the source assumes event time uniform over its assessment
window A_q, so `w_q=t/A_q`. When toxicity is pending but efficacy is observed
as b, the main paper gives

`Pr(Y_T=1 | X_T>t, Y_E=b) = p_(1b)*(1-w_T) /
  [p_(0b) + p_(1b)*(1-w_T)]`.

The reciprocal formula for pending efficacy follows by exchanging endpoints.
The article points derivations of the three missing-data patterns to
Supplementary Section S.3. The both-pending conditional and the fully
specified four-cell prior were not recoverable from the local source cache or
the accessible primary-article text. No BDA sampler is introduced here, since
choosing a joint prior association or filling in the missing derivation would
be an unlabelled modeling assumption. Native chain length, burn-in, and other
sampler settings are likewise not specified in the recovered main article.
