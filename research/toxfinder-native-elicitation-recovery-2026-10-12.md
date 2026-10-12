# Native ToxFinder prior elicitor: recovered contract, 2026-10-12

Read-only analysis of original ToxFinder.exe SHA256 3951dc95cdfaaad0da0526cfb25d28b7f970d4d82a6836203da4a469282d8070, extracted beneath research/raw/source-recovery-2026-10-12/toxfinder-native/SOURCEDIR/Program Files/ToxFinder/bin/. Addresses below are original VA. No original binary redistributed by these scratch references.

## Mathematical contract

405cb0 builds six independent Gamma priors in coordinate order alpha1,alpha2,alpha3,beta1,beta2,beta3. Calls405de0 for each single-agent prior. With target t, single-agent scaled doses zLow=d1/AD, z3=d3/AD, zHigh=d4/AD, and physician low/high toxicity probabilities qLow,qHigh:

- alpha mean ma=t/(1-t).
- beta mean mb=log(qHigh/((1-qHigh)*ma))/log(z3).
- Negative mb invokes404da0 error: Prior values are inconsistent. They require -ve beta.
- This fixes odds evaluated at mean beta, NOT the published E[alpha*z3**beta] Gamma-MGF equation. At t=.3,qHigh=.6,z3=1400/1200, mb=8.126872809767802. Existing paper-based elicitor must remain separately identified.
- Search variables are log(varAlpha),log(varBeta). 4052e0 exponentiates them; shape=mean^2/variance, rate=mean/variance.
- At EVERY objective evaluation, draw30,000 independent alpha,beta Gamma pairs and reuse each pair across both tails. Probabilities are strict P[p(zLow)<qLow] and P[p(zHigh)>t], p(z)=alpha*z**beta/(1+alpha*z**beta). 405290 criterion=(probLow-.99)^2+(probHigh-.99)^2.
- Gamma wrapper406330 narrows shape to float32 before gamma routine406d60, then divides by double rate.
- Config parser defaults qLow=.05,qHigh=.60. Interaction alpha3,beta3 use direct configured means/variances (defaults mean1,var.9 each), not this solver.

## Initialization, acceptance, failure and resources

405de0 traverses varianceAlpha (.1,.03,.3) outer and varianceBeta (3,1,10) inner. Initial log-space simplex steps(.1,.1). Each proposed start first goes through406200, which evaluates the probability routine at initial point and each one-coordinate +step vertex. Each tail probability must be strictly between1e-6 and.999999 at least once; the two flags can qualify on different vertices. Failing precheck skips that start.

Eligible starts run original4075a0/407660/407760 optimizer. Constructor407520 defaults3 attempts,20 iterations per attempt,relative tolerance1e-6. Elicitor sets objective cutoff2e-6 and refresh=true. All simplex vertices are reevaluated EACH outer iteration (stochastic objective); the usual initialization evaluations also occur. On attempt failure, steps multiply by-.5 and next attempt starts from previous returned point; steps therefore .1,-.05,.025. After3 attempts fail, move to next variance pair. First optimizer success exits immediately. On success, exponentiate output and store Gamma shapes/rates.

IMPORTANT acceptance is cutoff OR relative spread convergence. There is NO final requirement for objective<=2e-6. Original controlled constant-probability(.5,.5) fixture converges by zero spread and is accepted despite criterion=.4802. Consequently success flag cannot be represented as proof that elicited tail targets are satisfied.

If all9 starts fail,404da0 reports Prior has not converged for drug1/2!, then enters native termination routine4213a4 with exitcode1. No fallback/default calibrated prior is assigned. Host fixture intercepts error handler entry intentionally, avoiding file/process effects. Never interpret zero-initialized fixture shape arrays on error as native fallback values.

The elicitation logvariances have NO explicit box bounds visible in405de0/405290/4052e0. Do not borrow MAP bounds .005..30 from another routine. Native resource gates are30,000 pairs/evaluation,9 starts,3 attempts,20 iterations/attempt. Modern finite/positive/domain validation and overflow handling should be explicit Python guardrails; original zero beta, z3=1 and degenerate/overflow behavior are not yet independently characterized.

## Executed original references and precise scope

1. /tmp/toxfinder_elicitation_probability_reference.py and /tmp/toxfinder-elicitation-probability-fixtures.json execute original objective constructor4051f0, probability4052e0 and criterion405290. Host supplies Gamma draw tapes at406330 and runtime pow; original30,000-loop, strict comparisons and SSE execute. Each evaluation makes60,000 Gamma calls. Three tape fixtures yield probabilities(0,0),(1/3,1/3),(.5,.5), criteria1.9602,.8624222222222221,.4802. These are deterministic controlled-draw checks, not seed-equivalent original Gamma simulation.

2. /tmp/toxfinder_elicitor_control_reference.py and /tmp/toxfinder-elicitor-control-fixtures.json execute original top-level405de0,precheck406200,criterion405290,entire optimizer4075a0/407660/407760 and output transforms. Host replaces probability routine4052e0 with stated deterministic probability functions, supplies allocation/free/printing services, and intercepts native errors. Original initialization and retry/stop logic run unmodified. Five scenarios: cutoff success,all-start precheck failure,constant plateau success,all-start/attempt nonconvergence,interior objective success. JSON records all calls,starts,attempts,shapes/rates and errors. Nonconvergence executes9 starts/27 attempts and2675 probability calls. This verifies full optimizer/controller behavior but NOT a full30,000-draw calibration.

Launch both with PYTHONPATH=/tmp/efftox-research-env/lib/python3.12/site-packages python SCRIPT; scripts depend on /tmp/toxfinder_native_core.py, /tmp/toxfinder_objective_reference.py and /tmp/arand-python. They require packaging before becoming portable tools.

## Remaining evidence boundary

No whole native stochastic calibration under original Gamma RNG/seed has been executed. Exact original seed/stream contract and pathological input behavior remain unverified. A named Python native-contract implementation may faithfully implement recovered means,Monte Carlo criterion,starts,refresh,retries and acceptance, with explicit independent RNG and resource controls; it must not claim original seeded numeric parity or treat objective quality as assured. Before promoting full native calibration parity, obtain successful original unmodified stochastic fit plus a reproducible seed and verify output moments, error paths and stability independently.

## Portable implementation/reference integration

`src/mdanderson_stats/toxfinder_native_elicitation.py` now implements the named
native contract with an independent NumPy Gamma stream and additional quality
assessment. The published-equation elicitor remains unchanged. The scratch
reference functions above are integrated into the hash-pinned portable
`tools/reference_toxfinder_native.py`; no `/tmp` dependencies are needed by that
tracked runner. Fixtures preserve all 2,675 nonconvergence probability calls and
all successful/failing start/attempt controls. Original Gamma-loop references use
supplied draws, not original RNG identity. The public result exposes the separate
assessment and does not interpret native convergence as tail-target assurance.
