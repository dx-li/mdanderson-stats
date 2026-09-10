# U2OET: two-agent ordinal efficacy/toxicity models

Entry 77 is **partial**. The Python implementation supplies PDS, CMI and hybrid
model probabilities, grouped likelihoods and conditional expected utilities.
Posterior summaries and new-cohort allocation from supplied posterior draws
are also available. Posterior sampling, prior calibration, full trial conduct,
simulation and native input/report workflows remain pending. GAO with its Gaussian copula is
also pending; it is not replaced with the FGM model.

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
Counts describe fully observed efficacy/toxicity pairs; missing or delayed
outcomes are not handled. Empty counts give zero log likelihood. Impossible or
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

- GAO continuation probabilities and its Gaussian copula.
- Priors, pseudosampling prior centers, effective sample size calibration and
  posterior fitting with convergence assessment.
- Open-cohort conduct, delayed/partial outcomes and final selection.
  Posterior acceptability and new-cohort allocation are now supplied below.
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

Still pending: generating the posterior draws, managing an open cohort (the
guide rechecks acceptability at every arrival and may close a cohort early),
handling delayed/partial outcomes, accrual calendars, final selection and trial
simulation. The cohort-boundary function must not be used to silently rerandomize
patients within an existing cohort.
