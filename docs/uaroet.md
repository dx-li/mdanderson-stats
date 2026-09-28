# UAROET: ordinal outcomes and adaptive randomization

UAROET chooses doses using ordinal efficacy and toxicity outcomes, elicited
utilities, and Bayesian adaptive randomization. The [official desktop entry](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/92)
identifies the [Thall and Nguyen (2012) method](https://odin.mdacc.tmc.edu/~pfthall/main/JBS_AR_Utility2012.pdf).
This is an independent implementation of its logistic continuation model and
Gaussian copula, with [complete-outcome trial simulation](uaroet-trials.md).
Catalog entry 92 remains partial while prior calibration and native input/output
workflows remain open.

## Probability model

Array axes are dose, efficacy category, toxicity category. Higher efficacy is
better; higher toxicity is worse. Doses and categories are zero-based. This
matches other Python APIs in this package; the paper labels toxicity first.
The model supports one to five ordered doses and two to four categories for
each outcome. It uses dose order rather than numerical dose spacing.

Each dose has one continuation logit per nonzero outcome category. With
`lambda[y] = expit(theta[y])`, the marginal probabilities are

```
P(Y = 0) = 1 - lambda[1]
P(Y = y) = product(lambda[1:y]) * (1 - lambda[y+1])
P(Y = highest) = product(all lambda)
```

`uaroet_probabilities` accepts the efficacy and toxicity logit matrices and a
common latent Gaussian correlation. It returns marginal and joint probabilities,
their logs and quadrature errors. Correlation zero gives independence; the
probability API also provides exact limiting copulas at correlations -1 and 1.
This correlation is not the Pearson correlation of the ordinal outcome scores.

```python
from mdanderson_stats import uaroet_probabilities

model = uaroet_probabilities(
    efficacy_logits=[[-1.0, 0.2], [-0.3, 0.7]],
    toxicity_logits=[[-1.5, -0.4], [-0.8, 0.1]],
    association=0.3,
)
utility = [[20, 10, 0], [60, 40, 5], [100, 70, 10]]
print(model.expected_utility(utility))
```

These are probabilities conditional on supplied parameters, not posterior
estimates. Continuation probabilities are formed in log space. Gaussian cells
use deterministic conditional-normal integration, with checks of normalization
and marginal preservation. The existing U2OET fitted FGM copula is a different
model and is not substituted here.

## Explicit priors and posterior fitting

`uaroet_parameter_names` defines the ordering used by `uaroet_logits` and
`fit_uaroet`. A monotone endpoint has a baseline logit at each threshold and
nonnegative increments between consecutive doses. This imposes stochastic
ordering of that endpoint with dose. A nonmonotone endpoint has independent
free logits at every dose and threshold. Both endpoints are monotone by default;
their flags can be changed separately.

Supply prior means and standard deviations for every named normal coordinate.
For increments, these describe the underlying normal before truncation at zero.
The common correlation has a uniform prior on (-1,1) unless explicitly fixed.
The fitting routine retains separate chain and draw axes so posterior means,
utilities and allocation probabilities can be checked for sampling precision.
`summarize_chains` provides classic split R-hat and batch-means Monte Carlo
standard errors; completing a sampler run does not establish convergence.

The paper elicits priors by fitting many large pseudo-trials and then calibrating
prior variances with effective sample size and trial operating characteristics.
Taking logits of Table 2 probabilities does not perform that elicitation:
the logit of a mean probability is not generally the mean logit. This API
therefore requires explicit priors and does not invent the paper's calibrated
normal means. The radiation example's standard deviation 6 alone is insufficient
to reproduce its prior.

The following small example uses an explicitly fixed independence model and
illustrative priors. Omit `association` to infer the paper's common correlation.
Increase the retained draws according to the precision of the quantities used
for a decision; these short chains demonstrate the interface.

```python
import numpy as np
from mdanderson_stats import fit_uaroet, uaroet_allocation, uaroet_parameter_names

names = uaroet_parameter_names(2, 2, 2)
print(names)  # Efficacy baseline/increment, then toxicity baseline/increment.
fit = fit_uaroet(
    counts=[[[4, 0], [2, 0]], [[1, 1], [3, 1]]],
    prior_mean=[0, 0.3, -1, 0.3],
    prior_sd=[0.7, 0.4, 0.7, 0.4],
    association=0,
    draws=128,
    warmup=64,
    chains=2,
    rng=np.random.default_rng(14),
)
allocation = uaroet_allocation(
    fit,
    utility=[[20, 0], [100, 40]],
    treated=[6, 6],
    toxicity_limit=0.5,
    p_L=0,
    p_U=0.9,
    utility_tolerance=100,
    good_utility_cutoff=40,
)
print(allocation.action, allocation.probabilities)
if allocation.action == "randomize":
    assigned_dose = np.random.default_rng(15).choice(2, p=allocation.probabilities)
    print(assigned_dose)
```

`fit.joint` has axes `(chain, draw, dose, efficacy, toxicity)`. Inference is
serial, using elliptical slice updates for the normal coordinates and a
uniform-proposal Metropolis update for correlation. With no observations,
independent prior samples replace posterior sampling. Limits are five doses,
four categories per endpoint, 10,000 total observations and two million retained
joint-probability cells. `max_evaluations` caps the combined number of likelihood
calls and Gaussian rectangle integrations at two million; the result records
both components separately. Exhausting the limit raises an error.

Independence log probabilities remain finite even when their exponentials
underflow. For a correlated fit, an observed cell whose probability cannot be
represented raises an error; it is not silently treated as impossible data.
An integration failure also stops the fit. Very diffuse priors or nearly singular
correlations can reach these limits.

## Allocation rules

The utility table has efficacy rows and toxicity columns. It must be nonnegative,
nondecreasing with efficacy and nonincreasing with toxicity. Set a utility
cutoff to define good outcomes; equality is included. Randomization weights are
the posterior predictive probabilities of those outcomes, normalized among
eligible doses. They are not proportional to posterior mean utility.

The paper's three statistical requirements are applied separately:

- Safety: exclude a dose if the posterior probability that its toxicity tail
  exceeds the specified toxicity limit is greater than the upper cutoff.
- Near-optimality: its posterior mean utility must be within `delta` of the
  global maximum across doses, with equality allowed.
- Minimality: its posterior probability of attaining the largest utility must
  be at least the lower cutoff. Every exact per-draw tie counts as being best,
  so these probabilities need not sum to one.

The last two comparisons use all doses before intersection with safety,
following equations 7–9. There is no additional marginal efficacy-futility
threshold in this method. If the resulting set is empty, stop without selecting
a dose. If eligible doses have zero probability of a good outcome, proportional
randomization is undefined and raises an error.

For interim assignment, counts of previously assigned patients additionally
prevent escalation beyond the next level above the highest tried dose. The
initial dose is explicitly supplied and follows the paper's physician-specified
assignment. Final selection removes the escalation restriction.

The paper's final global-maximum notation is ambiguous when the global best
dose fails the safety rule. The API exposes both interpretations. The default
`final_rule="acceptable"` selects the highest mean utility in the acceptable
set. `final_rule="paper_global"` follows the literal formula: when the acceptable
set is nonempty, it selects the global mean-utility maximum, which can lie
outside that set. Both stop if the acceptable set is empty and resolve exact
ties by the lowest dose. These are explicit rules, not a claim about the native
executable's undocumented convention.

Supply `delta` for the current analysis. Across sequential analyses, the paper
requires a nonincreasing sequence. Its radiation example uses utility units of
20 at the earlier analyses and 15 later; this is a calibrated study-specific
choice, not a universal default.

The API names `delta` as `utility_tolerance`. `bad_toxicity_level` is the first
category included in the toxicity tail. `best_dose` reports the mean-utility
maximizer among eligible doses during enrollment; use `probabilities` for the
randomized assignment. Set `final=True` for the final rule described above.

## Evidence and remaining work

Independent base-R fixtures integrate copula cells over a uniform quantile
coordinate. They cover the published radiation-therapy probabilities, multiple
outcome scales, positive/negative dependence, independence and the limiting
copulas. A one-dose binary case at fixed zero correlation reduces to two
independent logistic-normal posteriors, providing a direct integration reference
for posterior moments. See the [audit](../research/uaroet-audit.md) and
[source record](uaroet-sources.json) for validation results and conventions.

The original executable has not been run for parity. Native prior files,
automatic pseudo-trial/ESS calibration, adaptive posterior precision control,
delayed-outcome simulation and native report formats remain separate work. Original
vendor code, executable files and papers are not bundled.
