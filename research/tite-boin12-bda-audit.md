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

## BDA dose-conduct policy

Section 2.2.3 says BDA and approximate-likelihood conduct use the BOIN12 dose
algorithm with updated admissibility, marginal DLT probability, and dose
desirability. It also suspends accrual when **more than** half of patients at
the current dose have pending DLT or efficacy outcomes. The implementation
uses strict `>` thresholds, so exactly one half is allowed. It does not apply
the approximate-likelihood method's zero-effective-information guard: an
explicit Dirichlet prior still defines the BDA conditionals at zero pending
follow-up.

The article does not give an explicit BDA movement-rate formula in the main
text. The Python conduct function uses the posterior-averaged imputed
completed toxicity count divided by the fixed treated count. This preserves
the ordinary observed toxicity rate when outcomes are complete and is kept
separate from the P-step Dirichlet posterior mean. Utility desirability and
admissibility tail probabilities are the averages of ordinary BOIN12
quasi-Beta summaries over retained completed-data imputations. The optional
3+3 run-in and precision-stop ordering reuse the existing Python AL conduct
policy; they are not claimed as source-defined BDA behavior.

## Numerical validation

Twenty-five focused BDA, TITE conduct and reference tests passed with warnings
treated as errors in 1.77 seconds, at 134.97 MiB peak RSS and zero swaps.
Targeted Ruff checks, formatting and mypy passed. Complete-data summaries
reduce directly to the existing BOIN12 calculation; pending-state checks cover
reproducibility, repeated imputation, conservation and numerical bounds.

The independent `tools/reference_tite_boin12_bda.py` enumerates all 16 labeled
missing-state assignments for a four-patient example. Each term is integrated
analytically with the multivariate beta ratio of Dirichlet normalizers, then
the complete-data BOIN12 summaries are averaged using those exact state
probabilities. The synthetic prior `(1.2,0.8,0.3,0.7)` has concentration three;
it is a numerical reference, not the article's ESS-one prior.

A separate seeded sampler comparison uses four chains, 1,000 warmup steps
and 4,000 retained draws per chain. All 13 joint-probability, completed-count
and BOIN12 summaries agree within 1.372 estimated batch MCSEs. Maximum split
R-hat is 1.000205 for joint probabilities and 1.000271 for BOIN12 metrics.
The reference/comparison process took 3.321 seconds, peaked at 126.39 MiB RSS
and reported zero swaps. These checks validate the explicit model and Python
sampler; they do not establish native application equivalence.
