# 2010 U2OET GAO probability model audit

## Primary source and scope

The inspected primary source is Houede, Thall, Nguyen, Paoletti and Kramar,
“Utility-Based Optimization of Combination Therapy Using Ordinal Toxicity and
Efficacy in Phase I/II Trials,” *Biometrics* 66 (2010), 532–540,
DOI [10.1111/j.1541-0420.2009.01302.x](https://doi.org/10.1111/j.1541-0420.2009.01302.x).
The cached paper is `research/raw/U2OET/gao2010.pdf`, SHA-256
`9ffc425871b3cd3f837ef22ce0bbec45ae247b06269928b7ad265332ccc344b8`;
`research/raw/U2OET/gao2010.txt` is its local extracted text. Equations below
are from sections 3.1–3.2; prior facts are from section 4.1.

This component implements the 2010 fixed-parameter probability kernel only.
It remains separate from the 2017 GAO comparison in `u2oet_gao.py`, which uses
raw-dose predictors and a shared positive interaction. It does not claim native
application parity, posterior fitting, prior calibration, utility selection,
or trial conduct.

## Model contract

For each agent, center its supplied strictly increasing nonnegative dose grid
at the arithmetic mean of the possible values. Zero is permitted because the
paper's prior-elicitation table includes zero-dose monotherapy rows. At endpoint `k` and ordinal
threshold `y`, each agent has a separate linear predictor
`eta_j = alpha[k,y,j,0] + alpha[k,y,j,1] * centered_dose_j`. The two-agent
generalized Aranda–Ordaz continuation is

```text
xi* = 1 - {1 + lambda_k * [exp(eta_1) + exp(eta_2)
                           + gamma_k * exp(eta_1 + eta_2)]}^(-1/lambda_k).
```

The endpoint-specific `gamma_k` is shared across thresholds. The threshold
continuations yield ordinal marginal categories by the paper's continuation
ratio products. A Gaussian copula with latent correlation `rho` constructs the
joint probability table. The existing result object's grouped complete-count
and toxicity-only log likelihoods apply directly.

`lambda_k` is positive. For negative `gamma_k`, the paper requires
`gamma_k > -(exp(-eta_1) + exp(-eta_2))` at every dose pair and threshold.
This strict condition makes the continuation bracket positive. Equality gives
zero continuation, but it is not included as an implicit closure policy. The
implementation checks the condition on the caller's actual grids and uses a
log-domain subtraction for well-separated terms and a bounded standard-library
Decimal evaluation for cancellation-prone terms. It raises rather than
silently returning an invalid result. It does not impose an arbitrary
nonnegative-interaction restriction.

For complete outcomes and toxicity-only outcomes, the package likelihood sums
the observed log joint cells and log toxicity marginals, respectively, without
multinomial constants. It does not model `zeta` or informative evaluability.
When evaluability is conditioned upon or its fixed factor is independent of
the modeled parameters, those omitted factors do not alter this probability
kernel's parameter-dependent likelihood. The implementation does not claim
that assumption for settings with informative evaluability.

## Priors and remaining source limits

Section 4.1 states normal priors for the alpha coefficients and gamma
interactions, lognormal priors for `lambda_k`, and a uniform `rho` on `[-1,1]`.
The paper reports `Var(alpha)=10^2=100` and
`Var(log(lambda))=Var(gamma)=1.5^2=2.25`; these are variances, not standard
deviations. It obtains prior means by extending the Thall–Cook least-squares
calibration to match elicited marginal outcome probabilities. The native
parameter-file coordinate ordering and the exact prior-center solving
procedure are not established here. In particular, negative gamma validity
depends jointly on alpha and the chosen dose grid, so independent marginal
truncation or direct reuse of the 2017 prior sampler would change the prior.

Root's standalone base-R reference evaluates the published formula over binary
and ordinal grids, endpoint-specific positive, zero and negative interactions,
small link shapes, and Gaussian correlations `{-1, -0.65, 0, 0.55, 1}`. It also
covers complete and toxicity-only grouped likelihoods. The integrated checker
passes 680 joint cells, 60 complete/partial/combined likelihood values and 100
expected utilities, plus dose translations and unit changes through 1e±150.
The maximum absolute discrepancy is `4.84e-13`; the run took .087 seconds,
peaked at 120.02 MiB RSS and reported zero swaps. Reproduction uses
`tools/reference_u2oet_gao2010.R` and `tools/check_u2oet_gao2010.py`. The worker's focused checks
passed 10 tests in 1.85 seconds, including the existing 2017 GAO probability
tests, with warnings treated as errors, peak RSS 136,527,872 bytes and zero
swaps. Targeted Ruff lint/format and mypy checks passed. Negative-interaction
tests include the near-boundary probe `eta_1=eta_2=0`,
`gamma=-2+2^-40`, `lambda=1`, a nonzero-predictor Decimal reference, and the
large-predictor cancellation case `eta=(1000,0)`, `gamma=-1`. These validate the stated equations, not native executable parity.
