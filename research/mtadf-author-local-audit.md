# MTADF author-reference local logistic audit

## Source and posterior

This implementation follows the cached author reference
`research/raw/mtadf-author-reference/targetAgentDF.r`, with the local
per-trial rule in `df.llogistic` (lines 321–409) and the operating
characteristic loop in `llogistic` (lines 184–309). No source was fetched or
executed for this change. The related author isotonic safety helper and final
selection are implemented in `mtadf_author.py`.

The source defines `x = 1:J` and `xs=(x-mean(x))/(2*sd(x))`, where R's `sd`
uses sample standard deviation. Each local logistic posterior has two
coefficients, `logit(p)=beta0+beta1*xs`, with independent Cauchy scales 10 and
2.5 (`:338–350`). The Python fit uses the equivalent binomial count likelihood
and reuses the bounded logistic random-walk sampler; `mcmc::metrop` proposal,
seed stream, and implementation defaults are not claimed to match.

## Decision contract

The R rule maps `thetaf=ce1`, `thetab1=1-ce2`, and `thetab2=1-ce1`
(`:360–363`). Defaults are `ce1=.3`, `ce2=.4`. It fits these adjacent windows:

- At dose 1, current/next; move up only if `P(beta1>0)>ce1` (`:371–378`).
- At dose J, previous/current; move down only if `P(beta1<=0)>thetab2`
  (`:379–385`).
- At an interior dose with no observations at the next level, previous/current;
  move down if the backward nonpositive probability is `>thetab2`, up if it
  is `<=thetab1`, otherwise stay (`:386–393`).
- When the next level has observations, fit current/next first, then
  previous/current. Move up only when backward nonpositive probability is
  `<=thetab1` and forward positive probability is `>thetaf`; otherwise move
  down if backward nonpositive probability is `>thetab2`; otherwise stay
  (`:394–405`).

The two-sided order is relevant to seeded Python sampling. The code uses
zero-based dose indices while preserving source branch conditions. If the
admissible prefix has length one, `df.llogistic` sets the next dose to the
lowest without fitting (`:367–369`). Direct `final=True` delegates to the
author isotonic all-dose epsilon-rate rule, matching the final `llogistic`
selection (`:291–295`). The OC function hard-starts cohort 1 at dose 1 and,
when the initial cap exceeds one, assigns cohort 2 at dose 2; subsequent moves
use the cap computed before the cohort just completed, then refresh it
(`:226–287`). The public local decision uses a fresh cap, while a private
override helper supports that simulator lag without changing the normal
decision API.

## Independent posterior check

The seeded Python/reference comparison uses four chains, 8,000 retained draws
per chain and 2,000 warmup iterations. The observed diagnostic/error summary
was:

| Reference window | Slope-probability error | Slope MCSE | Slope R-hat | Max prediction error | Max prediction MCSE | Max prediction R-hat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lower pair; upper neighbor unobserved | 0.04040 | 0.02113 | 1.0449 | 0.04172 | 0.01770 | 1.0711 |
| Upper pair; lower neighbor unobserved | -0.02490 | 0.02160 | 1.0203 | 0.02999 | 0.01878 | 1.0278 |
| Interior pair; both observed | -0.01523 | 0.01048 | 1.0163 | 0.01867 | 0.01174 | 1.0827 |

The focused check requires each reported slope/prediction MCSE to be at most
0.03, R-hat maxima at most 1.1, and reference differences within six estimated
MCSEs plus the independent quadrature refinement error. These are bounded
sanity checks for this deterministic seed, not proof of MCMC convergence.

## Difference from the existing Python paper policy

`mtadf_local_logistic_posterior` uses the caller's dose values directly; the
author uses globally standardized ordinal indices. Because the intercept and
slope priors are independent, changing this coordinate system changes the
prior model and the posterior. The paper API also offers configurable
`window_length`, a single slope-probability gate, and a bounce guard, whereas
the author reference uses exact adjacent pairs and the branch-specific gates
above. Its ordinary local fitter requires every window dose to have
observations; the author fit can include an unobserved adjacent dose, which
simply contributes no likelihood. The author-specific posterior therefore
reuses low-level sampler internals, not the paper-policy public local fit.

This local API does not implement the global quadratic author `logistic()`
model or claim full MTADF family/live-app parity. The R file uses
`arm::bayesglm`, scaled linear/quadratic covariates, and other calibration
choices for that distinct model.
