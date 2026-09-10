# U2OET: two-agent ordinal efficacy/toxicity models

Entry 77 is **partial**. The Python implementation supplies PDS, CMI and hybrid
model probabilities, grouped likelihoods and conditional expected utilities.
Posterior summaries and new-cohort allocation from supplied posterior draws
and posterior fitting are also available, along with IID prior sampling,
beta-moment information and pseudo-trial prior calibration. Full trial conduct,
simulation and complete native input/report workflows remain pending. Gaussian
scenario construction and native scenario/dose/utility readers are available;
Patient snapshots, toxicity-only likelihoods and next-patient cohort decisions
are also available. GAO model fitting is still pending.

The sources are the [official U2OET 1.8 archive](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/U2OET/U2OET_V1.8.zip),
its user guide and the [author-hosted paper](https://odin.mdacc.tmc.edu/~pfthall/main/JRCCS_2017_ph12_2agent_utility.pdf)
by Thall, Nguyen and Zinner, JRSS Series C 66 (2017), 201–224,
doi:10.1111/rssc.12162. [Source hashes](u2oet-sources.json) record the archive,
its files and the paper. The archive contains a Windows executable and examples,
but no primary C++ source. Its comparison R programs implement other designs.
No original executable, paper, examples or comparator code is redistributed.

## Python API

```python
import numpy as np
from mdanderson_stats import U2OETMarginal, u2oet_probabilities

efficacy = U2OETMarginal(
    intercepts=[-0.8, 0.1, -0.3],
    slopes=[[0.7, 1.2], [0.4, 0.8], [0.3, 1.1]],
    powers=[0.4, 2],
    link=0.7,
)
toxicity = U2OETMarginal(
    intercepts=[0.8, -0.1, 0.3],
    slopes=[[1.2, 0.7], [0.8, 0.4], [1.1, 0.3]],
    powers=[2, 0.4],
    link=1.8,
)
probability = u2oet_probabilities(
    [4, 5, 6],
    [40, 60, 80],
    efficacy=efficacy,
    toxicity=toxicity,
    association=0.6,
)
# Axes: agent 1, agent 2, efficacy, toxicity. Indices start at zero.
assert probability.joint.shape == (3, 3, 4, 4)
counts = np.zeros((3, 3, 4, 4))
counts[0, 1, 2, 0] = 3
print(probability.loglikelihood(counts))
# Example utilities; rows are efficacy categories, columns toxicity categories.
utility = np.array([[20, 15, 5, 0], [50, 35, 20, 10], [75, 60, 40, 20], [100, 90, 65, 35]])
print(probability.expected_utility(utility))
```

These example parameters are illustrative, not calibrated prior centers or
fitted estimates. The result is conditional on these parameters, not a posterior
recommendation. Native prior-vector ordering has not been established and is
not inferred from the example files.

Each outcome has 2–4 ordered categories and each agent 2–5 strictly increasing,
positive doses. An intercept and two positive slopes are required for each
continuation threshold. Intercepts do not require ordering. The two positive
powers modify the interior standardized doses while anchoring the endpoints.
`link` is the positive Aranda–Ordaz shape; one gives logistic continuation
probabilities. A small positive value approximates the complementary log-log
limit; zero is rejected.

`model="pds"` requires zero interaction. `model="cmi"` requires both powers to
be one and log centering. `model="pds+cmi"` allows PDS powers and an interaction
coefficient shared across thresholds for each outcome. Interaction can destroy
monotonicity. `centering="linear"` uses standardized dose minus one; the default
uses its logarithm. Association lies in [-1,1] and is the FGM copula parameter,
not Pearson correlation.

## Numerical implementation

Dose standardization first divides by the maximum dose to avoid overflow of
raw means and differences. Changing the dose measurement units preserves the
model. Dose ratios that underflow are rejected.

Continuation probabilities and category products are calculated in log space.
The Aranda–Ordaz calculation uses log-softplus and `expm1`, with asymptotic
branches below machine precision. FGM rectangle probabilities are evaluated
as nonnegative sums rather than subtracting four nearly equal CDF values.
This also preserves tiny joint probabilities at association endpoints ±1.
`log_joint`, `log_efficacy` and `log_toxicity` are immutable arrays. Ordinary
`joint` probabilities can underflow; use the log arrays for likelihood work.

The grouped likelihood excludes parameter-independent multinomial constants.
Complete counts describe fully observed efficacy/toxicity pairs. An optional
`toxicity_only` array adds marginal contributions for pending efficacy. Empty counts give zero log likelihood. Impossible or
unrepresentably small observed probabilities can give negative infinity;
invalid finite inputs and overflowing linear predictors raise errors.

## Validation and remaining work

Two focused tests cover independent reference agreement and numerical extremes.
The independently written [R reference](../tests/fixtures/u2oet-reference.R)
evaluates the published equations directly, including four-CDF copula
rectangles, across five cases covering 2, 3 and 4 categories, the three model
variants, both centerings and association -1, 0 and 1. All 549 joint cells agree
with Python within 8e-16 absolute plus 2e-12 relative tolerance. This is
independent equation validation, **not native executable parity**.

Further checks cover marginal preservation, total probability, grouped
likelihoods, expected utilities, PDS monotonicity, dose rescaling by 1e±300,
link shapes 1e±300, and a rare/rare log probability near -3000 at association
-1. The latter has an analytic logistic/FGM reference. The R script generates
a fixed reference table; R is not a runtime or CI dependency.

Remaining coverage includes:

- GAO continuation probabilities and fitting (Gaussian scenario copulas are supplied).
- Validation of complete trial operating characteristics.
  Explicit-prior fitting, pseudo-trial centers and prior ESS are supplied below.
- Calendar simulation, final selection and adaptive MCMC precision targets.
  Open-cohort handling and toxicity-only likelihoods are supplied below.
- Calendar-based simulation, operating characteristics, scenario/native-file
  import and reports.

The original program is freely downloadable, but a general redistribution
license was not found in the inspected guide or README. This implementation
uses the published mathematics and does not assert a license for the vendor
software. The repository's release and licensing status is unchanged.

## Posterior criteria and new-cohort allocation

`u2oet_posterior(joint_draws, utility, criteria=...)` accepts equally weighted
posterior joint-probability draws with axes `(draw, agent1, agent2, efficacy,
toxicity)`. It returns posterior mean utilities, probabilities of unacceptable
efficacy/toxicity and a dose acceptability mask. It does not produce posterior
draws or assess MCMC convergence. Prior draws or scenario probabilities must not
be represented as a fitted posterior.

`U2OETCriteria` defaults to the paper's four-category example: efficacy at least
level 2 must have probability at least .40, and toxicity at least level 2 must
have probability at most .45. A pair is excluded when the posterior probability
of either violation is **strictly greater than .90**. The bad events themselves
also use strict inequalities. Binary outcomes require level 1; defaults are not
silently adjusted to another outcome scale. Utilities must be nonnegative for
utility-proportional randomization. Joint probability normalization is checked
per draw/dose; input is not silently rescaled. At most 20 million cells are
accepted per call.

For separately fitted draws, call
`summary = u2oet_posterior(posterior_draws, utility)`, then
`decision = u2oet_allocation(summary, treated_counts, surplus=3, top=2)`.
`decision.probabilities` contains the next-cohort assignment weights.

`u2oet_allocation` implements the new-cohort rule using counts of **all assigned
patients**, including those without completed outcomes. Each agent may advance
at most one level beyond its highest previously tried level. Both agents can
advance together, and an untried pair of previously tried individual levels is
eligible. Statistical acceptability and this escalation restriction jointly
define the candidate set for ranking and the surplus calculation.

The best candidate receives the cohort unless it has at least `surplus` more
patients than **every other candidate**. Once that condition holds, allocation
is proportional to utility among the best `top` candidates. The default is the
paper's surplus 3 and top 2. Use `top=3`, `top=4` or `top=None` for the guide's
alternatives, and `greedy=True` or `top=1` for pure greedy allocation. All-zero
utilities in the randomization set raise an error: proportional weights are
undefined. Scaling utilities before normalization prevents overflow of their
sum. A candidate with zero utility receives zero randomization probability.

With no assigned patients, an explicit zero-based `initial=(i,j)` is required.
The protocol starting pair is returned regardless of prior acceptability. With
patients already assigned and no acceptable eligible pairs, the result has
`best=None`, all-zero assignment probabilities and an explanatory reason.
This is a stop indication, not a probability distribution to sample.

Ties use ascending agent-1 then agent-2 indices. This is an explicit Python
convention; native tie handling has not been verified. Output arrays are
immutable. The function returns weights rather than consuming random numbers,
so a trial simulator can control its RNG and audit each assignment.

Two additional focused tests use hand-computable posterior draws and cohort
histories. They verify threshold equality, exactly-.90 posterior risk,
inefficacy stopping, utility orientation, first-cohort assignment, no skipping,
surplus comparison against a lower-ranked third pair, top-two/all-pair
randomization and greedy allocation. These validate the published decision
rules, not native executable parity.

The new-cohort function must not silently rerandomize patients within an
existing cohort. Use `u2oet_next_patient` below for open-cohort handling. Accrual
calendars, final selection and trial simulation remain pending.

## Posterior fitting

`fit_u2oet` now fits the PDS, CMI and hybrid models to fully observed ordinal
outcomes. Counts have the same four axes as `U2OETProbabilities.joint`.
All prior means and standard deviations must be supplied explicitly. Obtain
the Python coordinate names with `u2oet_parameter_names(LE, LT, model=...)`:

- Each outcome's intercepts, then threshold-major agent-1/agent-2 slopes.
- Log PDS powers (omitted for CMI), then log link shape.
- A shared interaction coefficient for CMI/hybrid (omitted for PDS).
- Association last, after both outcomes.

`prior_mean` and `prior_sd` cover all coordinates **except association**, which
has the published uniform [-1,1] prior. Intercepts and interaction coefficients
have independent normal priors. Slopes have independent normals truncated below
at zero; supplied means and SDs describe the underlying normals, not their
truncated moments. Log powers and log link shapes have normal priors, giving
lognormal physical parameters. This follows section 2.3 of the paper and the
CMI extension. The Python ordering is explicit and does not claim compatibility
with the unverified native prior-file packing.

```python
from mdanderson_stats import (
    fit_u2oet,
    u2oet_parameter_names,
    summarize_chains,
    U2OETCriteria,
    u2oet_posterior,
    u2oet_allocation,
)

# Illustrative priors and data; these are not the source trial's calibrated priors.
names = u2oet_parameter_names(2, 2)
mean = np.zeros(len(names) - 1)
sd = np.full(mean.size, 0.2)
for i, name in enumerate(names[:-1]):
    if ".slope." in name:
        mean[i] = 0.5
counts = np.zeros((3, 3, 2, 2))
counts[1, 1, 1, 0] = 3
fit = fit_u2oet(
    [1, 2, 3],
    [1, 2, 3],
    counts,
    prior_mean=mean,
    prior_sd=sd,
    draws=1000,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(77),
)
diagnostics = summarize_chains(fit.parameters)
posterior = u2oet_posterior(
    fit.joint.reshape((-1, 3, 3, 2, 2)),
    [[10, 0], [100, 40]],
    criteria=U2OETCriteria(efficacy_level=1, toxicity_level=1),
)
treated = np.zeros((3, 3))
treated[1, 1] = 3
allocation = u2oet_allocation(posterior, treated)
```

The sampler updates each outcome's normal coordinates as a block with elliptical
slice sampling, treating positive slope constraints as part of the likelihood
support. Association uses an independent uniform proposal and a Metropolis
likelihood ratio. This targets the stated posterior without tuning proposal
scales or adding transformed-slope Jacobians. It is a new Python sampling
algorithm, not a reproduction of the executable's Gibbs sampler. Unchanged
marginals are cached during each block update; only positive-count cells enter
the likelihood. Every retained sweep includes both outcome updates and the
association update. No thinning is applied.

Arrays retain `(chain, draw, ...)` axes. `fit.parameters` uses the named
coordinates, including log powers/link; `fit.joint` stores joint probabilities;
`fit.log_likelihood` omits parameter-independent multinomial constants. The
result also records association acceptance per chain and likelihood evaluations.
Use `summarize_chains` on parameters **and** quantities relevant to decisions.
Its classic split R-hat and batch-means Monte Carlo errors are diagnostics,
not convergence guarantees. The default starts use prior centers with positive
slopes, identically across chains. Supply dispersed `initial` rows for stronger
convergence assessment, especially with diffuse priors or weak identification.

Each run requires an explicit NumPy generator, at least two chains and eight
retained draws per chain. At most 20 million joint probability cells are stored.
A failed slice bracket or a power/link transform outside floating-point range
raises an error; proposals are not silently clipped to a different prior.
Proper priors make zero-observation fitting meaningful. Highly diffuse priors
and sparse outcome categories may still yield slow mixing.

Three focused numerical checks validate the sampler:

- A concentrated-nuisance CMI example reduces to a logistic-normal posterior
  for 3 successes in 10 trials. Independent R integration gives intercept mean
  -0.61233147641579189 and success-probability mean 0.36123314764157921.
  Four chains agree within five estimated Monte Carlo standard errors.
- With no observations, retained slope means recover the positive-truncated
  normal prior, other named coordinates recover their normal means, and
  association updates recover the uniform prior.
- A full PDS example with four complete observations compares all 13 parameter
  means against 200,000 independent R prior-importance samples. The importance
  effective sample size is approximately 165,118. All means agree within six
  combined Monte Carlo errors and fitted classic split R-hats are below 1.05.
  The independent [R script](../tests/fixtures/u2oet-posterior-importance.R)
  evaluates the published equations directly; it is not vendor source.

These checks validate the target and sampling implementation in the tested
settings. They do not establish convergence for other priors/data or reproduce
the published trial operating characteristics. GAO and full trial simulation
remain pending; toxicity-only outcomes are supported below. Prior calibration is now supplied below.

## Prior draws, information and pseudo-trial calibration

`sample_u2oet_prior` draws directly from the stated prior, independently across
samples. It accepts the same named means/SDs as the fitter and adds uniform
association draws. The result contains named coordinates and joint outcome
probabilities. It does not use MCMC. Positive-normal slopes use rejection from
a normal when its mean is nonnegative and an exponential-tail envelope when its
mean is negative. Tail draws are formed as positive excesses, avoiding
cancellation from subtracting a large truncation threshold.

`u2oet_prior_ess(prior.joint)` implements guide section 1.5: marginalize the joint
draws, moment-match a beta distribution for each dose/outcome category, and
report the mean and maximum information across all cells. With marginal mean
`m` and variance `v`, the beta concentration is `m*(1-m)/v - 1`. Efficacy and
toxicity cell arrays remain available for inspection. This is **prior
information**, not the effective number of MCMC samples.

The estimator uses empirical population variance (divisor number of draws),
which respects the variance bound for probabilities. Native C++ variance
normalization is unverified. A constant interior probability has infinite
concentration; an identically zero/one probability has undefined concentration
(NaN); a mixture entirely at the endpoints has limiting concentration zero.
Undefined cells are retained and make the overall summaries undefined. They
are not dropped or replaced with fabricated finite values. Marginal roundoff
within the admitted normalization tolerance is clipped to [0,1].

`calibrate_u2oet_prior` implements guide section 1.3. For each replicate it draws
a balanced pseudo dataset from an explicit joint scenario, fits the pseudo
posterior, and retains its coordinate means, Monte Carlo standard errors and
classic split R-hats. Averaging these pseudo-posterior means produces the
candidate prior centers. The supplied guide defaults are 100 pseudo patients
per dose pair and underlying normal means zero/SDs 100. Association remains
uniform and does not receive an estimated center. Log powers/link are averaged
on their named log coordinate scale. `pseudo_prior_sd` can be changed explicitly;
changing it defines a different calibration exercise.

The number of `repetitions` is required. The guide recommends at least 1000;
small values are useful for workflow checks, not adequate production calibration.
Returned `standard_error` describes variation of the average over independent
pseudo trials, including their MCMC noise. Inspect the per-trial MCMC diagnostics
as well. Trials with poor mixing are retained; numerical failures raise errors.
No failed or inconvenient pseudo dataset is silently discarded. Exact pseudo
counts are retained to permit replay. Scenario sums within 1e-12 of one are
rescaled to one before multinomial sampling; malformed scenarios are rejected.

```python
from mdanderson_stats import sample_u2oet_prior, u2oet_prior_ess

# Reuse the illustrative binary prior arrays and names from the fitting example.
prior = sample_u2oet_prior(
    [1, 2, 3],
    [1, 2, 3],
    efficacy_levels=2,
    toxicity_levels=2,
    prior_mean=mean,
    prior_sd=sd,
    draws=10000,
    rng=np.random.default_rng(78),
)
information = u2oet_prior_ess(prior.joint)
print(information.mean, information.maximum)
```

For calibration, pass an elicited `scenario_joint` with axes `(agent1, agent2,
efficacy, toxicity)` to `calibrate_u2oet_prior`, along with dose arrays,
`repetitions` and an explicit generator. The guide's Gaussian-copula scenario
construction is now supplied by `u2oet_scenario`, described below. An arbitrary
FGM scenario is not equivalent to that construction.

Validation covers independent normal-tail moment references from R at underlying
means -10, -2, 0 and 1, normalization of 12,000 IID prior draws, a hand-computed
beta-moment example (efficacy concentration 4, toxicity 14, overall mean 9),
constant/endpoint degeneracies, and pseudo-data scenarios with all low versus
all high efficacy. The latter use a specified SD .5 and show the expected
opposite movement of the inferred efficacy intercept while preserving balanced
allocation and all trial diagnostics. No 1000-replicate clinical calibration
or published operating-characteristic reproduction is claimed.

**Earlier block-only sampler limitation:** an additional PDS check used the guide's SD 100,
100 patients per pair, a 2×2 dose grid with binary uniform joint scenarios, two
pseudo trials, four chains, 2000 warmup and 2000 retained sweeps per chain
(seed 7708). Maximum classic split R-hats were **51.95 and 7.27**. Thus this
configuration did not mix adequately, and its finite prior-center estimates
must not be treated as calibrated. A shorter 64-draw check also failed strongly.
The SD-.5 workflow checks above do not validate the diffuse default. The
additional updates described below address this failure on the first dataset;
complete calibration and trial operating-characteristic reproduction remain
outstanding.

## Updates for diffuse priors

`fit_u2oet(..., coordinate_updates=True)` now combines each outcome-block
elliptical slice move with scalar elliptical slice moves and a joint
link/intercept/slope move. `calibrate_u2oet_prior` enables these extra moves by
default. The standalone fitter retains its previous, cheaper default;
`coordinate_updates=False` reproduces that update schedule. The fit result
records the setting. The prior distributions and likelihood are unchanged.

For the joint move, a symmetric normal increment changes log link shape. Each
threshold intercept is transformed to preserve its continuation probability
at **zero standardized covariates**, and its two slopes are scaled by the
inverse-link derivative. A shared interaction uses the first threshold's
scale. This map is invertible; its log Jacobian is three times the sum of the
log scales, plus the first log scale when an interaction is present. The
Metropolis acceptance probability includes both the normal-prior density ratio
and this Jacobian. Preserving the probability at zero covariates alone would
not justify accepting the proposal without that correction.

Numerical checks verify reversal using the opposite increment, the analytic
Jacobian against finite differences, and the preserved center probabilities
for PDS, CMI and hybrid models. The independent full-PDS R importance comparison
now exercises these extra updates. This tests the posterior target as well as
mixing. Added updates cost more per sweep, so compare Monte Carlo precision per
elapsed time rather than raw iteration counts.

On the first previously failing pseudo dataset (the exact counts are recorded in
[u2oet-diffuse-validation.json](u2oet-diffuse-validation.json)), the original
block-only sampler had maximum split R-hat 51.95. Adding scalar moves alone
reduced this to 1.193 in a 1000-warmup/1000-draw, four-chain run. Including the
joint link move gave maximum parameter split R-hat **1.003** with 1500 warmup
and 1500 retained draws per chain, seed 7711. That run took about 145 seconds
in this environment. The record also includes parameter means, Monte Carlo
errors and joint-probability diagnostics. This is a targeted difficult-case
check, not a universal convergence guarantee or a completed 1000-trial
calibration study.

## Gaussian scenarios and native scenario inputs

`u2oet_scenario(efficacy, toxicity, association=.1)` constructs a true-outcome
scenario from marginal arrays shaped `(agent1, agent2, category)`. It uses the
Gaussian copula specified for simulation in section 4.1 of the paper. This
association is a latent-normal correlation, not the observed ordinal Pearson
correlation. It is distinct from the FGM parameter in the fitted PDS/CMI/hybrid
model. The default .1 matches the paper's simulation scenario association.

```python
from mdanderson_stats import u2oet_scenario

# Illustrative constant marginals over a 2-by-2 dose grid.
efficacy = np.broadcast_to([0.1, 0.2, 0.3, 0.4], (2, 2, 4))
toxicity = np.broadcast_to([0.4, 0.3, 0.2, 0.1], (2, 2, 4))
scenario = u2oet_scenario(efficacy, toxicity, association=0.1)
assert scenario.joint.shape == (2, 2, 4, 4)
# scenario.joint can be passed to calibrate_u2oet_prior.
```

The implementation integrates conditional-normal probabilities over each
ordinal rectangle. It avoids subtraction of four bivariate CDF values and does
not use randomized multivariate integration. Upper-tail normal quantiles are
formed from independently accumulated survival probabilities. Integration
intervals split at the conditional transitions to resolve near-perfect
correlation. Independence and limiting correlations ±1 use direct formulas.
Zero marginal categories remain zero joint cells.

Every quadrature cell must meet an estimated absolute error tolerance of 2e-12,
and both reconstructed marginals are checked within 1e-11. Per-cell estimated
errors are returned in `quadrature_error`; these estimates are not rigorous
mathematical bounds. This API computes ordinary probabilities and does not
promise relative accuracy for extremely small cells or finite log tails.
Marginal normalization errors up to 1e-12 are rescaled; larger errors fail.

The following native file readers support the guide's formats:

- `read_u2oet_doses(path)` returns the two positive increasing raw-dose arrays.
- `read_u2oet_scenario(path, dose_counts=(M1,M2), efficacy_levels=LE,
  toxicity_levels=LT)` reads one-based dose indices and ordered marginal
  probabilities. An optional single-value association header must be strictly
  between -1 and 1. Without that header, association is zero. Row order is free;
  every dose pair must appear exactly once.
- `read_u2oet_utility(path, efficacy_levels=LE, toxicity_levels=LT)` reads
  zero-based efficacy/toxicity/value rows into an efficacy-by-toxicity matrix.
  Every cell is required; utilities must be nonnegative and weakly improve with
  efficacy and worsen with toxicity.

Readers accept whitespace, blank lines and UTF-8 BOMs. Invalid indices, duplicate
or missing cells, malformed numbers and inconsistent probabilities raise errors.
An invalid single-number scenario header is rejected; the guide describes the
native program defaulting invalid association to independence. The Python
reader deliberately does not conceal that invalid input. Model/prior/trial
parameter-file parsing remains pending; patient-data import is supplied below.

Two focused tests cover an independently written R integral over uniform
quantiles, analytic Gaussian quadrant probabilities, correlations within 1e-12
of ±1, exact limiting correlations, zero-probability categories and native-format
indexing/validation. The 48 R cell references agree within 2e-12 absolute error;
the [R reference script](../tests/fixtures/u2oet-gaussian-reference.R) is supplied.
The library does not depend on R.

An additional [archive audit](u2oet-scenario-audit.json) read all **96 scenarios**
from eight supplied configurations, including 2-, 3- and 4-category variants,
as well as every configuration's raw doses and utility matrix. All probability
and marginal checks passed. The maximum estimated quadrature error was
4.99e-14 or less. The nine scenario-1 expected utilities for PDS/4E_4T match the
paper's Table 4 to its printed one-decimal precision. This validates source
scenario construction, not native posterior/trial simulation parity. Original
scenario and utility files are not bundled.

## Patient snapshots and next-patient decisions

`read_u2oet_patients(path, dose_counts=(M1,M2), efficacy_levels=LE,
toxicity_levels=LT)` reads the five fields in guide section 3.1: patient ID,
one-based dose indices, and zero-based efficacy/toxicity levels. A pending
outcome is -1. `u2oet_patients(records, ...)` accepts the same format as an array.
IDs must start at 1 and strictly increase; gaps are allowed. An empty snapshot
is permitted for the first assignment. At most 2500 patient rows are accepted.
Invalid indices, duplicate/out-of-order IDs and malformed rows raise errors.

The result preserves immutable source-format records and aggregates:

- `treated`: every assigned patient, including those with pending outcomes.
- `complete`: joint counts when both outcomes are observed.
- `toxicity_only`: counts with observed toxicity and pending efficacy.
- `ignored_outcomes`: number of patients whose toxicity is still pending.

The last group remains in assignment counts. Following the guide, an efficacy
observation is not used for fitting until toxicity has been recorded. This
asymmetry is intentional. It does not mean efficacy-only data are generally
uninformative in other statistical models.

`fit_u2oet(..., toxicity_only=data.toxicity_only)` adds marginal toxicity log
likelihood contributions to the complete joint counts. The two count arrays
must represent disjoint patients. `U2OETProbabilities.loglikelihood` supports
the same keyword for direct likelihood evaluation. Missing outcomes are not
imputed or counted as responses; pending elapsed times are not used as
additional likelihood information. With no toxicity-only records, the existing
complete-data likelihood path is preserved.

```python
from mdanderson_stats import (
    u2oet_patients,
    fit_u2oet,
    u2oet_posterior,
    u2oet_next_patient,
    U2OETCriteria,
)

patients = u2oet_patients(
    [[1, 1, 1, 1, 0], [2, 1, 1, -1, 1], [3, 1, 1, 1, -1]],
    dose_counts=(3, 3),
    efficacy_levels=2,
    toxicity_levels=2,
)
# Reuse explicit illustrative binary prior arrays mean/sd from the fitting example.
fit = fit_u2oet(
    [1, 2, 3],
    [1, 2, 3],
    patients.complete,
    toxicity_only=patients.toxicity_only,
    prior_mean=mean,
    prior_sd=sd,
    draws=1000,
    warmup=500,
    chains=4,
    rng=np.random.default_rng(79),
)
posterior = u2oet_posterior(
    fit.joint.reshape((-1, 3, 3, 2, 2)),
    [[10, 0], [100, 40]],
    criteria=U2OETCriteria(efficacy_level=1, toxicity_level=1),
)
next_patient = u2oet_next_patient(posterior, patients, cohort_size=3, max_patients=60)
```

Before assigning each patient, recompute the posterior from the latest snapshot
and assess its Monte Carlo diagnostics. `u2oet_next_patient` follows the guide's
open-cohort rule: take the trailing count of identical dose pairs modulo the
cohort size. If nonzero and the last pair remains acceptable, continue there
with probability one, even if another pair has greater posterior utility. If
that pair has become unacceptable, close the cohort and apply the existing
new-cohort allocation rule. Every assigned patient contributes to the trailing
count, regardless of outcome availability.

The result reports assignment weights, whether the cohort continues, the next
position in the cohort and whether an unacceptable cohort was closed. A new
cohort starts at position 1; stopping gives position 0 and all-zero weights.
`surplus=None` uses the cohort size as the randomization surplus threshold;
explicit `surplus`, `top` and `greedy` options pass through to the new-cohort
rule. With no patients, supply an explicit zero-based `initial` pair. Reaching
`max_patients` stops enrollment and does **not** silently select a final dose.

This implements the guide's fixed cohort-size rule. The archive's separately
named first/new-dose/old-dose cohort settings are not mapped without verifying
their exact semantics. Calendar simulation, final selection, adaptive MCMC
precision targets and integrated trial reports remain outstanding.

Two focused tests cover mixed complete/partial/pending records, marginal
likelihood contributions, open-cohort precedence, early cohort closure,
multiple consecutive full cohorts and enrollment limits. A toxicity-only
3-in-10 posterior agrees with the independently integrated logistic-normal
reference; association proposals are all accepted because these observations
carry no information about association. All three supplied example patient
files were also read: each has 60 complete observations. The PDS/4E_4T and
hybrid examples end in a seven-patient run at zero-based pair (1,1); the binary
example ends in a three-patient run there. Original patient files are not bundled.
