# Predicted risks of toxicity (PRT)

Catalog entry **69 remains partial**. Python now implements the discrete probit
hazard likelihood, conditional remaining toxicity probabilities, PRT predictive
criteria, cohort decisions and final selection. The state-space posterior fit and published covariance-weighted isotonic
transformation are now available below. Native projection safeguards, patient-file
conversion and full calendar trial simulation remain outstanding. The predictive
function accepts aligned isotonic posterior draws; fitting and projection are
separate steps.

Sources: [MD Anderson entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/69),
[conduct guide](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/PRT/readme_conduct.pdf),
and Bekele, Ji, Shen and Thall, [*Monitoring late-onset toxicities in phase I trials
using predicted risks*](https://doi.org/10.1093/biostatistics/kxm044), Biostatistics
9 (2008), 442–457. An [author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/Biostatistics_LateTox_2008.pdf)
is available. The April 2007 Windows archive predates the final paper; these
functions implement the published equations and rules, without claiming binary
program or historical simulation parity.

## Predictive risk

```python
import numpy as np
from mdanderson_stats import prt_predictive_risk

# Illustrative aligned draws, not an actual fitted clinical-trial posterior.
total = np.array([0.15, 0.2, 0.3, 0.4, 0.5])
pending = np.column_stack([total, total * 0.6])
risk = prt_predictive_risk(total, pending, target=0.3)
print(risk.predictive_negligible, risk.predictive_excessive)
print(risk.count_probability)  # Probabilities of 0, 1, or 2 future toxicities.
```

For one dose, `total_probability` contains posterior draws of its isotonic
probability of toxicity within the full assessment window. `pending_probability`
has one row for each matching draw and one column per pending patient. A column
contains that patient's probability of toxicity before the end of the window,
conditional on survival through their completed intervals. The rows must share
one underlying posterior draw across all columns. Permuting or independently
sampling the columns would change the predictive dependence structure.

The beta approximation in Section 4.2 matches the total-risk posterior mean and
variance. The empirical variance uses the second central moment (divisor equal
to the number of draws). Degenerate moments, boundary means or moments that do
not define a proper beta distribution raise an error; no pseudo-variance is
invented. Returned `beta_alpha` and `beta_beta` expose this approximation.

Conditional on a posterior draw, future toxicity indicators are independent with
patient-specific probabilities. A polynomial recursion computes the conditional
distribution of their sum, then averages it across draws. Only this sum is needed
for the beta update, so this is algebraically equivalent to equations (4.1)–(4.3)
without enumerating every binary outcome vector. Complexity is O(draws × pending²),
with memory bounded by `chunk_size × (pending + 1)` in addition to inputs.
The implementation supports up to 1,000 pending patients. It preserves posterior
mixing dependence and does not substitute a binomial distribution based on an
average risk.

For each possible future toxicity count s, the approximate updated distribution
is Beta(alpha+s, beta+pending-s). Its probability of exceeding `target` is returned
as `updated_exceedance`. `predictive_negligible` sums future-count probabilities
whose updated exceedance is **at most** `negligible_cutoff`; `predictive_excessive`
uses **at least** `excessive_cutoff`. Their complement is reported separately as
`predictive_acceptable`. Defaults are target .3 and cutoffs .3/.9.

`posterior_excessive` is the fraction of the supplied total-risk draws strictly
above target (the paper's xi). It is not the predictive excessive-risk probability,
and is not computed from the moment-matched beta distribution. The latter is
used only for predicting the effect of completed pending outcomes.

With no pending patients, the raw predictive criteria are still returned, while
`decision_negligible=1` and `decision_excessive=0` implement the paper's special
criteria for cohort decisions. With pending patients, these equal their raw
predictive counterparts. Arrays are immutable. The calculation is deterministic
conditional on the supplied draws; uncertainty from posterior fitting, finite
MCMC samples and beta moment approximation remains the caller's responsibility.

## Cohort and final decisions

```python
from mdanderson_stats import prt_decision, prt_final_selection

decision = prt_decision(
    exceedance=[0.1, 0.4, 0.8],
    predictive_negligible=[0.98, 0.2, 0.05],
    predictive_excessive=[0.01, 0.04, 0.2],
    pending=[0, 2, 1],
    current_dose=1,
)
print(decision.action, decision.dose, decision.rule)  # stay, 1, 5.1
winner = prt_final_selection([0.1, 0.4, 0.95], [0.1, 0.28, 0.31])
print(winner)  # 1
```

Dose indices are zero-based; `None` denotes suspension, trial stopping or no final
selection. Posterior exceedances must be nondecreasing across 2–10 doses, as
required by the isotonic model. Defaults use epsilon .05. Rules are applied in
published order: lowest-dose toxicity stops the trial, current-dose toxicity
triggers de-escalation, and acceptable/negligible current risk invokes predictive
stay, escalate or suspension rules. De-escalation chooses the highest lower dose
not exceeding the upper posterior cutoff. Escalation is by one dose and cannot
skip an untried intermediate dose. Equality at posterior cutoffs is acceptable;
the predictive checks use the inclusive inequalities specified in the paper.

Rule `6.1` returns either stay or suspend, according to its reference to rule 5.
Other returned labels are `3`, `4`, `5.1`, `5.2`, `6.2`, `6.3` and `6.4`.
The function overrides predictive criteria for every pending-zero dose. Apply
these rules after the last enrollment in the current cohort, then reevaluate
suspension when arrivals or follow-up updates justify it; cohort scheduling is
not performed by this function.

Final selection implements rule 7: among all doses whose posterior exceedance is
at most the upper cutoff, choose the posterior mean toxicity nearest the target.
Python resolves equal distances in favor of the lower dose. It does not add an
untried-dose exclusion absent from the published rule.

## Probit model components and validation

`prt_conditional_toxicity(beta)` accepts final axes `(interval, dose)`, preserving
leading draw axes. It returns risks after zero through all H completed assessment
intervals, ending in zero. H is explicit here; this avoids the paper's C versus
the archive guide's interval-count conventions. These are **raw** model risks,
before the covariance-weighted Bayesian isotonic transformation required by PRT.
Use isotonic draws, rather than arbitrary raw risks, in the predictive criteria.

`prt_interval_loglikelihood(beta, survived, events)` evaluates equation (2.1).
Counts are interval-by-dose: completing an interval without toxicity contributes
a survival count; toxicity in an interval contributes an event. An incomplete
pending interval contributes neither. Log-normal tails and `expm1` retain small
hazards and avoid subtractive cancellation in cumulative risks.

The [reference calculation](prt-reference.json) compares dynamic programming with
4,096 explicitly enumerated future-outcome vectors for 512 aligned draws and 12
pending patients. Run `uv run python tools/reference_prt.py` to reproduce it.
Focused tests additionally check exact beta moments and tails, predictive
posterior dependence, chunk invariance, all decision branches, final selection,
small hazards and extreme finite log likelihoods. This validates the implemented
criteria, not a fitted PRT posterior, native executable or published trial OCs.


## State-space posterior fit and isotonic transformation

```python
import numpy as np
from mdanderson_stats import fit_prt_model, prt_isotonic_projection

fit = fit_prt_model(
    survived=[[3, 2]],
    events=[[1, 3]],
    prior_mean=0.3,
    prior_variance=1.2,
    rng=np.random.default_rng(6910),
)
# Remove the terminal zero-risk row before estimating covariance.
raw = fit.conditional_toxicity[:, :, :-1, :]
projected = prt_isotonic_projection(raw.reshape(-1, *raw.shape[-2:]))
print(projected.probability.mean(axis=0))
```

`fit_prt_model` takes interval-by-dose survivor and event counts with 1–10
assessment intervals and 2–10 doses. Counts in the next interval cannot exceed
the preceding interval's survivors. Priors are independent between intervals.
Within each interval, dose coefficients follow a Gaussian random walk starting
from the fixed `prior_mean`, with every increment having `prior_variance`.
Defaults -14 and 28 come from the archived settings. These are variance inputs,
not standard deviations. Both can instead have one value per interval. The
prior covariance between doses k and l (one-based) is `min(k,l)*variance`.

The sampler targets the likelihood in equation (2.1) directly. Independent
interval/chain blocks update together using elliptical slice sampling, avoiding
latent-variable sign conventions and reproducing the stated posterior rather
than the original Gibbs random stream. Uninformed intervals receive independent
prior draws. Defaults retain 2,000 draws after 1,000 warmup iterations in each of
four chains; random state is explicit. A failed bracket or nonfinite likelihood
raises an error.

Results retain coefficients and raw conditional risks with axes
`(chain, draw, interval, dose)`. Risks have an additional terminal all-zero row.
Coefficient and nonterminal risk summaries report means, intervals, classical
split-Rhat and batch-means MCSE. These diagnostics estimate mixing and precision;
they do not certify convergence or a threshold decision. Prior settings and
likelihood-evaluation counts are retained with the immutable arrays.

`prt_isotonic_projection` implements the Section 3 min-max formula literally,
using the empirical full covariance matrix for each interval. It estimates
covariance over flattened retained draws, solves principal submatrix systems by
Cholesky factorization and processes the draw vectors with NumPy operations.
It does not replace full covariance by diagonal variance weights. The result
retains transformed draws, covariance matrices and the largest condition number.
Singular/non-positive-definite covariance or condition numbers exceeding 1e12
are rejected. No ridge or pseudoinverse is substituted.

**The published formula can produce values outside the probability interval.**
Inverse-covariance weights need not be nonnegative. In the [guide-history pilot](prt-fit-pilot.json),
the raw fit's maximum coefficient split-Rhat was 1.002 or less, but the projection
produced a minimum of about -0.0153. This is materially below zero, not roundoff.
The API raises an error rather than clipping or feeding invalid probabilities
to the predictive criteria. This finding concerns the literal published formula
with estimated covariance; the original executable's handling remains unverified.
Its resolution is required before claiming complete end-to-end PRT coverage.

Independent R tensor Gauss-Hermite integration verifies a two-dose posterior,
refining from 80 to 120 nodes per dimension. Both coefficient and risk means
agree with the sampler within six estimated Monte Carlo standard errors.
A separate R full-covariance min-max calculation verifies 180 transformed values;
prior-only sampling verifies the random-walk covariance. Regenerate both fixtures
with `Rscript tools/reference_prt_fit.R`. Run
`uv run python tools/pilot_prt_fit.py` to reproduce the guide-history diagnostic.
