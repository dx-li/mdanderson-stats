# CRM prior effective sample size

`simulate_crm_prior_ess` implements the complete-outcome CRM calculation from
the Bayesian Effective Sample Size Calculator, catalog entry 154. Each
replication simulates an adaptive trial, then calculates the expected
information in random subsets of that completed trial. It uses the empiric
power model `p_j(beta) = skeleton_j ** exp(beta)` and a normal prior
`beta ~ Normal(0, beta_sd**2)`.

```python
from mdanderson_stats import simulate_crm_prior_ess

result = simulate_crm_prior_ess(
    true_toxicity=[.08, .18, .30, .45],
    skeleton=[.05, .15, .30, .50],
    target=.25, max_patients=12, replications=8, rng=154,
)
assert result.dose_indices.shape == (8, 12)
assert result.information_gap.shape == (13,)
print(result.continuous_ess, result.native_grid_ess, result.crossing_status)
```

The small example demonstrates the calculation, not sufficient simulation
precision for a trial-design decision. The result depends on the supplied
truth, skeleton, target and full-trial size as well as the prior variance.
It is an information-based prior ESS, not an MCMC effective sample size.

## Adaptive trial

The true toxicity probabilities generate binary outcomes; the ordered skeleton
defines the fitted model. These are distinct inputs and need not agree.
The native workflow starts at dose 1, the Python default. Dose indices are
one-based. Cohorts contain one patient and all outcomes are observed before
the next assignment.

At each update, the posterior mean of beta is inserted into the dose-response
function. The next recommendation is the probability closest to target,
preferring the lower dose for an exact tie. After a DLT it cannot exceed the
current dose; otherwise it cannot exceed the current dose plus one. Downward
moves are unrestricted. The final recommendation is closest to target without
these interim restrictions. No extra overdose-elimination or early-stopping
rule is added to the source workflow.

This is a different model from the package's logistic/BMA CRM trial engines.
It uses deterministic one-dimensional posterior integration rather than MCMC.

## Information path and matching conventions

The source repeatedly samples m patients without replacement from a completed
M-patient adaptive trial. Conditional on that trial, the expected sum of
information is exactly m/M times the full sum. The implementation calculates
this expectation directly and averages across simulated trials. It avoids
repeated fitting and subset-sampling noise while retaining variability between
trials.

`mean_subset_information` and `information_gap` index m=0,...,M. They describe
expected subsets of the full trial, **not the first m patients** or independent
trials ending at m. Information is the negative log-likelihood second
derivative at beta=0. The source gap is `1/beta_sd**2 - mean_information(m)`;
this CRM criterion does not subtract an epsilon-prior information term.

- `continuous_ess` is the first zero crossing of that path, or `None` if the
  simulated range does not reach it. The result never extrapolates past M.
- `native_grid_ess` is the nearest-to-zero gap on 50 equally spaced points
  in [0,M], using the original R matching convention. It can equal M even
  when there is no crossing.
- `crossing_status` makes this distinction explicit. `matching` selects which
  value appears as `ess_estimate`; both estimates remain available.

Using the native grid does not reproduce R's random streams or finite random
subset noise. For example, the saved reference's continuous estimate is
1.3059566036 and its grid estimate is 1.2244897959 for the same expected path.

## Posterior integration conventions

The default `posterior_moments="full"` integrates normalization and moments
over the entire real line. The original dfcrm routine integrates normalization
over the real line but moment numerators only over [-10,10]. Their difference
is negligible at its default `beta_sd=sqrt(1.34)` and material for diffuse
priors. `posterior_moments="native_truncated_numerator"` explicitly requests
that legacy convention; it is not a consistently normalized truncated prior.
Changing conventions can change subsequent adaptive dose assignments.

The [source audit](../research/crm-prior-ess-audit.md) records this discrepancy,
the native allocation rules and independent reference calculations. Very small
and near-one skeleton probabilities use stable likelihood-curvature formulas.

## Reproducibility and returned histories

Use `rng` for a local seed or NumPy generator. Alternatively, supply
`outcome_uniforms` with shape `(replications,max_patients)` and values in [0,1)
to replay outcomes through the adaptive design. An outcome is a DLT when its
uniform is below the true probability at the assigned dose. Supplied uniforms
replace random generation; they do not force particular dose assignments.

Readonly results retain dose/outcome/uniform histories, prepatient beta means,
final posterior moments and selections, and each trial's information total.
Replications run sequentially. Input, result-size and integration-work limits
are checked before random generation; unresolved numerical integration raises
an error rather than silently dropping a trial.

The calculation supports up to 20 doses, 200 patients per trial and 1,000
replications, subject to `max_work`. `beta_sd` must be in (0,100], with its
variance and inverse variance representable in float64. The default budget is 50 million
quadrature-node/dose evaluations, with an explicit maximum of one billion.
The preflight budget allows at most 4,200 integrand evaluations per integral
and one full-real integral per fit. The native convention reserves up to
16 integrals per fit because it splits the finite moment interval around the
posterior mode. `work_units` records this conservative reserved budget;
`quadrature_evaluations` records the actual integrand calls. Each integral
also enforces its evaluation ceiling while running.

TITE-CRM, unknown-mean variance ESS and native reports remain separate gaps
in entry 154. The TITE source uses enrollment times where follow-up weights
are needed, requiring a separate explicit convention and audit.
