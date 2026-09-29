# TITE-BOIN12 Bayesian data augmentation

`tite_boin12_bda_posterior` implements the binary-outcome augmentation model
from Section 2.2.1 of the [TITE-BOIN12 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9199061/).
It complements the existing [approximate-likelihood method](tite-boin12.md).
Patient records use the same one-based dose labels and endpoint statuses:
`1` observed event, `0` completed non-event, and `-1` pending.

```python
import numpy as np
from mdanderson_stats import BOIN12Design, tite_boin12_bda_posterior

fit = tite_boin12_bda_posterior(
    BOIN12Design(0.35, 0.25, utilities=(100, 30, 65, 0)),
    doses=[1, 1, 1, 1],
    toxicity=[0, -1, 1, -1],
    efficacy=[1, 1, -1, -1],
    toxicity_followup=[1.0, 0.6, 0.1, 0.2],
    efficacy_followup=[1.0, 0.1, 0.25, 0.4],
    toxicity_window=1.0,
    efficacy_window=1.0,
    n_doses=1,
    prior_concentrations=[1.2, 0.8, 0.3, 0.7],
    draws=1000,
    warmup=1000,
    chains=4,
    rng=np.random.default_rng(152),
)
print(fit.posterior.utility_probability)
print(fit.diagnostics.boin12_metrics.batch_mean_mcse)
```

The example uses a **synthetic prior of total concentration three**, chosen
for an independent numerical reference. It is not the paper's prior or a
recommended clinical design. `prior_concentrations` is required and accepts
four positive concentrations shared across doses or a `(n_doses, 4)` array.
The cell order is always `(no toxicity/efficacy, no toxicity/no efficacy,
toxicity/efficacy, toxicity/no efficacy)`.

The paper gives a total prior concentration of one and marginal prior means
`Pr(toxicity)=0.5*toxicity_limit` and `Pr(efficacy)=efficacy_limit`. These
constraints leave the joint prior association unspecified. This API makes
that remaining choice explicit instead of inferring native defaults.

## Missing outcomes and posterior summaries

Each iteration imputes pending endpoint outcomes and then samples the
dose-specific joint probabilities from their conditional Dirichlet posterior.
Observed endpoints remain fixed. For a pending endpoint, the uniform
event-time model contributes a survival factor `1 - followup/window` when
the imputed outcome is an event, and a factor one otherwise.

When both endpoints are pending, multiply the four joint probabilities by
`[1-wE, 1, (1-wT)*(1-wE), 1-wT]` and normalize. With one endpoint observed,
restrict to the two compatible cells and apply the remaining endpoint's
survival factor. These formulas use the paper's working independence of
event times conditional on the endpoint outcomes. They preserve dependence
between the binary toxicity and efficacy outcomes through their joint model.

Two statistical layers are reported separately:

- `joint_probability_draws` and their diagnostics describe the Dirichlet
  imputation model, with axes `(chain, draw, dose, cell)`.
- `posterior` averages the ordinary BOIN12 complete-data overdose, futility
  and quasi-Beta utility summaries across the retained imputations. These
  utility probabilities are not tails of the direct Dirichlet-weighted utility.

The result also returns mean completed joint counts and admissibility.
With fully observed records, the BOIN12 layer reduces to its complete-data
calculation; joint probability draws still reflect the supplied Dirichlet prior.

## Diagnostics and scope

Chains run sequentially. The result includes split R-hat, batch-means MCSE
and other existing chain summaries for joint probabilities, completed counts,
and five BOIN12 metrics in this order: overdose probability, futility
probability, utility mean, utility desirability, utility quasi-event count. Constant completed-data
metrics can have undefined R-hat; completion alone does not certify convergence.

Preflight limits bound retained arrays, summary temporaries and total
imputation work. Patient-by-draw imputation histories are not retained.
Nonrepresentable conditional probabilities raise instead of substituting
arbitrary outcomes.

An [independent reference](../tools/reference_tite_boin12_bda.py) enumerates
all 16 compatible missing-outcome assignments in the example and integrates
each under the Dirichlet prior exactly. A seeded four-chain check matched all
13 joint, count and BOIN12 summaries within 1.38 estimated batch MCSEs. See the
[validation audit](../research/tite-boin12-bda-audit.md).

This entry point provides posterior calculations. The existing
`tite_boin12_decision` still uses approximate likelihood; an integrated BDA
conduct/calendar simulator is not yet supplied. Complete-outcome final
selection remains available through `tite_boin12_select_obd`.
