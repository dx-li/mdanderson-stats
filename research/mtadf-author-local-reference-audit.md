# MTADF author local-logistic reference audit

The cached primary author file is
`research/raw/mtadf-author-reference/targetAgentDF.r` (SHA-256
`e27be5581fdcb7c71a7739a4f1b25026dc054896fcbf7134ff9b1a1c8960caac`). Its
local `df.logistic` function is at lines 313–409; its serial local simulation
is at lines 184–310. The likelihood and prior are declared at lines 199–205
and again at 345–351: a two-parameter linear logit with independent Cauchy
priors, intercept scale 10 and slope scale 2.5. The author uses `metrop`
starting from `(0,0)` for 2,000 iterations and summarizes rows 1,001–2,000.
This reference integrates the declared posterior directly; it does not
reproduce finite-chain Monte Carlo error or the app's random sequence.

For `J` doses, both author functions define `xs=(1:J-mean(1:J))/(2*sd(1:J))`
using R's sample standard deviation. Each local window selects adjacent
entries from this full-grid vector. The response counts are expanded by
`collapse(yeff,n)` and the corresponding `xs` values are repeated by the same
patient counts (simulation lines 251–280; real-trial lines 371–400). A
zero-count neighbor contributes no response rows, but the observed neighbor
keeps its full-grid coordinate. The likelihood is therefore not recentered
or rescaled to the observed subset. `df.logistic` also does not require every
window dose to have positive enrollment; it passes the possibly empty
neighbor rows through `rep(..., n)` to the posterior fitter.

The reference fixture contains two such one-observed-dose windows and one
two-observed-dose interior window. For one observed dose the likelihood
identifies only the linear predictor at that dose, creating a long intercept
/ slope ridge in direct product-Cauchy coordinates. The reference changes
variables from intercept to that identified linear predictor and performs
adaptive nested integration, separately integrating negative and positive
slope halves. It records the quadrature error estimates and repeats at tighter
tolerance. The two-observed-dose case uses independent Gauss-Legendre
quadrature in the two Cauchy CDF coordinates; the slope domain is split at
zero so the positive-slope indicator is not a discontinuity within a
quadrature interval. Its 192/256-order refinement is also recorded. These are
deterministic numerical references to the declared posterior, not native
Metropolis output.

With `ce1=.3` and `ce2=.4`, the exact local action inequalities are visible
in the source. At the lower boundary, move up only when
`P(beta>0)>ce1` (lines 251–257, 371–377). At the top boundary, move down only
when `P(beta<=0)>1-ce1` (lines 258–264, 378–384). When the next dose has no
patients, use the backward window: move down if its nonpositive-slope
probability is `>1-ce1`; otherwise move up if it is `<=1-ce2` (lines 265–272,
385–392). With both neighbors observed, move up only when the backward
nonpositive probability is `<=1-ce2` and the forward positive probability is
`>ce1`; otherwise move down when the backward nonpositive probability is
`>1-ce1`; otherwise stay (lines 273–285, 393–404). Thus equality at the
upper threshold can permit an increase, while equality at `ce1` cannot.

Both the simulation and `df.logistic` short-circuit when the admissible-dose
count is one: force dose 1 and do not fit a local efficacy posterior
(simulation lines 240–246; function lines 367–370). The accompanying gate
fixture records this safety-only path along with threshold-equality and
interior branch cases. The source caps proposed moves by the admissible count.

The independent reference generator is
`tools/reference_mtadf_author_local.py`; fixtures are
`tests/fixtures/mtadf-author-local-reference.csv` and
`tests/fixtures/mtadf-author-local-gates.csv`. This establishes reference
values for deterministic posterior integration and source-defined conduct
boundaries only. It does not claim MCMC, full-trial simulator, or historical
application execution parity.
