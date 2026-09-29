# TITE-BOIN12 Bayesian data augmentation audit

The primary article is Zhou et al., “TITE-BOIN12: A Bayesian Phase I/II Trial
Design to Find the Optimal Biological Dose with Late-onset Toxicity and
Efficacy,” *Statistics in Medicine* 41 (2022), 1918–1931,
[doi:10.1002/sim.9337](https://doi.org/10.1002/sim.9337),
[PMC9199061](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/).
Section 2.2 specifies the BDA imputation and posterior steps. The main article
states conditional working independence of event times given their respective
binary endpoints and uniform event times over each endpoint's assessment
window. Thus for pending fractions `wT=tT/AT` and `wE=tE/AE`, the survival
weights in cell order `(01,00,11,10)` are

`S01=1-wE`, `S00=1`, `S11=(1-wT)(1-wE)`, `S10=1-wT`.

When both outcomes are pending, the cell probabilities are proportional to
`p_ab S_ab`. If only toxicity is pending while efficacy is observed as `b`,
the conditional toxicity-event odds are proportional to `p_1b(1-wT)` versus
`p_0b`. If only efficacy is pending while toxicity is observed as `a`, the
conditional efficacy-event odds are proportional to `p_a1(1-wE)` versus
`p_a0`. The observed endpoint's event-time factor cancels because it is
constant over the missing endpoint's alternatives. These are the three
missing-data patterns used by `_impute_pending`.

The P step draws each dose's joint cell probabilities from a Dirichlet
distribution with concentration equal to the prior vector plus completed
cell counts. The article specifies prior total concentration one, marginal
toxicity mean `0.5*phiT`, and marginal efficacy mean `phiE`; those constraints
do not identify all four positive concentrations (the association remains
free). The Python API therefore requires the four-cell Dirichlet vector
explicitly, either shared or dose-specific. It does not claim to reproduce an
unidentified native dependence default.

For decision summaries, each retained completed-data imputation is passed to
the existing BOIN12 posterior calculation, then its toxicity-tail,
efficacy-futility, utility-mean, quasi-Beta utility-probability, and
quasi-event-count summaries are averaged. Strict admissibility cutoffs apply
to the averaged toxicity and efficacy tail probabilities. These are kept
distinct from the P-step Dirichlet draws; the
implementation does not replace BOIN12's quasi-Beta utility posterior with a
Dirichlet utility distribution. No patient-level imputation trace is retained.

The paper does not provide a portable sampler contract (chain count, burn-in,
diagnostic thresholds, or random-number generator) in the main text. The
Python API exposes the NumPy generator, chain count, warmup, and retained draw
count as explicit computational settings. Its MCSE and split-Rhat are
diagnostics, not automatic convergence guarantees. Supplementary sections
were not needed to recover the three conditional formulas; any further
application defaults remain unverified.
