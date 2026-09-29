# MTADF logistic method coverage audit

## Source and implementation scope

This audit uses the publisher-hosted author manuscript: Zang, Lee and Yuan,
“Adaptive Designs for Identifying Optimal Biological Dose for Molecularly
Targeted Agents,” *Clinical Trials* 11(3), 2014, available through PubMed
Central at <https://pmc.ncbi.nlm.nih.gov/articles/PMC4239216/>. Sections 2.2
and 2.4 specify the global and L-logistic efficacy models and decision rules;
Section 2.1 supplies the independent beta-binomial toxicity model, isotonic
posterior-overdose safety rule, and prior elicitation. This implementation does
not access or claim parity with the separate MTADF application.

The global model uses
`logit(p_j)=alpha+beta*d_j+gamma*d_j**2`, with binomial efficacy likelihood
and independent `Cauchy(0,10)`, `Cauchy(0,2.5)`, `Cauchy(0,2.5)` priors. The
local model uses the last `l` adjacent doses through the current dose,
`logit(p_k)=alpha+beta*d_k`, with independent `Cauchy(0,10)` and
`Cauchy(0,2.5)` priors, and reports `Pr(beta>0 | data)`. The paper recommends
`l=2`; its examples calibrate local escalation/de-escalation cutoffs to 0.4 and
0.3. Both methods reuse `mtadf_decision` for the Section 2.1 toxicity rule.
The L-logistic final selection uses the existing double-sided isotonic
estimator, as stated in Section 2.4.

An independent tensor Gauss-Legendre reference in transformed Cauchy-prior
coordinates checked the global posterior for `n=[10,10,10]`,
`y=[2,7,4]`, `doses=[-1,0,1]`. Quadrature orders 128 and 192 agreed within
`3.32e-14`; their predictions were
`[0.22308657936014006, 0.6662037623234378, 0.40920684688553677]`. The Python
fit at seed `20260929` returned posterior means
`[0.2118507251, 0.6784212879, 0.4049271112]`, each within 1.82 reported batch
MCSEs. Split R-hat values were `[1.018, 1.030, 1.014]`; acceptance rates ranged
from 0.237 to 0.313. This is an independent statistical reference, not native
application or sampler-sequence parity.

Posterior inference uses a bounded, multi-chain random-walk Metropolis sampler
with warmup-only scalar-scale adaptation. The paper says to use MCMC but does
not specify a sampler, tuning, chain count, burn-in, or diagnostics. Draw count,
warmup, chains and RNG are therefore explicit Python settings. Dose coding is
used exactly as supplied: hidden centering/scaling would change the specified
Cauchy priors. Reports include draws, acceptance rates, split R-hat and batch
mean MCSE, including a batch-means MCSE for the local positive-slope
indicator; diagnostics do not certify convergence.

The paper does not define whether global “posterior estimate” means mean,
median or another summary; this implementation uses posterior mean. It also
does not fully specify dose-boundary movement or safety action when the current
dose is inadmissible. Python conduct conventions are documented in the public
function docstrings: lowest-index efficacy ties, one-level endpoint clamping,
safety-first drop to a safe dose, and stop when no dose is admissible. Local
movement similarly clamps at boundaries, rejects unsafe destinations, and
drops to a safe lower dose or stops if none exists. These choices provide a
usable and explicit Python policy; they are not asserted as author-program
behavior. Safety-only stop/drop paths do not require efficacy MCMC. Once
enrollment has started, an interim current dose must itself have observations;
the API does not substitute a different observed dose silently.

The local source starts with one cohort at each of the first `l` levels. This
API surfaces that as sequential initial-ramp actions, then uses the slope
thresholds and the paper's “next dose already tried” bounce guard. The initial
ramp stops if its next required dose is unsafe. The paper gives no separate
per-dose minimum sample size or early stopping threshold beyond toxicity
admissibility. Near the lowest dose boundary, a local window cannot contain
`l` lower-or-current levels; as an explicit Python convention, the first `l`
dose levels form the window there. Final isotonic selection requires no
current dose and runs without a logistic posterior fit. When an escalation
would invoke the source bounce guard using a previously treated next dose, the
next-window posterior can be supplied explicitly and its diagnostics remain in
the result. If boundary clamping makes the current and next windows identical,
the current posterior is reused for the bounce guard, avoiding an unnecessary
fit and Monte Carlo-noise disagreement. Combined distinct current/bounce draw
storage and transition work receive a decision-level preflight.

## Validation

Focused tests check seeded replay, posterior array dimensions and bounds,
local slope symmetry and an independent transformed-Cauchy quadrature
reference, fit/count matching, toxicity-first stop/drop actions,
treated-current validation, local bounce-guard diagnostics and final isotonic
selection, including same-window bounce reuse at the lowest-dose boundary. The
three affected MTADF test files passed: 18 tests in 2.20 seconds with warnings
treated as errors. Peak RSS was 141,475,840 bytes (138,160 KiB)
with zero process swaps. Ruff check/format and targeted mypy passed. These
checks support the independent Python model and policies; no native
random-stream or app-output parity is claimed.
