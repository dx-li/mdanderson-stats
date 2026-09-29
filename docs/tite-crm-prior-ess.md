# TITE-CRM prior effective sample size

`simulate_tite_crm_prior_ess` simulates adaptive TITE-CRM trials and estimates
the prior effective sample size from expected subsets of their patient
histories. It uses the empirical power model
`p_j(beta) = skeleton_j ** exp(beta)` with a normal prior on beta.

The ESS criterion must be selected explicitly. `criterion="followup"` uses
the outcomes observed at the last enrollment plus `assessment_delay`.
`criterion="native_arrival"` reproduces BayesESS's legacy calculation, which
uses absolute enrollment times as weights and eventual toxicity outcomes.
These calculate different quantities.

```python
import numpy as np
from mdanderson_stats import simulate_tite_crm_prior_ess

inputs = dict(
    true_toxicity=[0.05, 0.15, 0.30, 0.45],
    skeleton=[0.05, 0.10, 0.20, 0.35],
    target=0.20,
    max_patients=8,
    replications=1,
    obswin=10,
    rate=2,
    accrual="fixed",
    outcome_uniforms=[[0.9, 0.01, 0.9, 0.9, 0.01, 0.9, 0.9, 0.01]],
    event_time_uniforms=[[0.0, 0.2, 0.0, 0.0, 0.7, 0.0, 0.0, 0.9]],
)
result = simulate_tite_crm_prior_ess(
    **inputs,
    criterion="followup",
    assessment_delay=3,
)
assert result.latent_outcomes[0, -1] == 1
assert result.observed_outcomes[0, -1] == 0
assert np.isclose(result.assessment_weights[0, -1], 0.3)
assert np.isclose(result.event_study_times[0, -1], 49)
legacy = simulate_tite_crm_prior_ess(**inputs, criterion="native_arrival")
assert legacy.observed_outcomes[0, -1] == 1
print(result.ess_estimate, result.crossing_status)
```

This single fixed trial illustrates replay and pending-outcome handling; it
does not establish simulation precision. For simulation, omit all replay tapes
and supply `rng` with a seed or NumPy generator, choosing a replication count
within the work budget. Arrays retain every trial's dose assignments, latent
and assessment outcomes, arrival and event times, weights and posterior means.

## Dose assignment and timing

The scalar starting dose defaults to one. Interim recommendations use only
events observed by the next enrollment and linear follow-up weights for
remaining patients. The posterior mean of beta determines the dose whose
estimated toxicity probability is closest to target, with lower-dose tie
breaking. Upward moves are capped at the current dose plus one, including
after a toxicity; downward moves are unrestricted. This is the source's
scalar-start TITE rule. Initial-design vectors and two-stage waiting rules
are outside this interface.

The first enrollment occurs after one accrual interval. Fixed intervals equal
`obswin/rate`; Poisson intervals have that mean. A latent toxicity occurs
uniformly within its patient's observation window. `event_delays` are measured
from enrollment, while `event_study_times` use the study calendar. Non-events
have infinite event times.

At the ESS assessment, a pending toxicity remains unobserved and receives
partial follow-up weight. Only observed toxicities receive weight one.
The returned final posterior and selected dose separately use all eventual
outcomes, as the native end-of-trial fit does; they are not interim estimates
at the earlier ESS assessment.

## Information and conventions

The signed log-likelihood second derivative is evaluated at beta zero. For
partially followed non-events it can be positive, so the corresponding
information can be negative. The calculation preserves this sign. Conditional
on a simulated M-patient trial, an m-patient random subset has expected
curvature equal to m/M times the full sum. The implementation uses that exact
expectation rather than repeatedly drawing subsets.

`information_gap` is prior precision plus the mean subset curvature for
m=0,...,M. `continuous_ess` is its zero crossing within that range, or `None`.
`native_grid_ess` selects the nearest-to-zero gap on the source's 50-point
grid. `matching` chooses the reported `ess_estimate`; `crossing_status` remains
explicit even when the nearest grid point exists without a crossing.

The default `posterior_moments="full"` uses full-real posterior moments.
`"native_truncated_numerator"` reproduces dfcrm's full-real normalization
with moment numerators restricted to [-10,10]. This legacy convention is not
a consistently truncated prior and can change decisions for diffuse priors.

Replay requires outcome and event-time uniforms for every patient, plus
arrival uniforms for Poisson accrual. Tapes and `rng` are mutually exclusive.
NumPy random streams and the exact subset expectation do not reproduce R's
finite Monte Carlo stream. BayesESS also hardcodes Poisson accrual; the fixed
option here follows dfcrm's documented scalar-start simulator.

Fits run sequentially with input, storage and quadrature-work limits.
The input ceilings are 20 doses, 200 patients and 1,000 replications, subject
to a default `max_work` of 50 million node-by-patient evaluations and a hard
maximum of one billion. Unresolved numerical refinement raises an error. See the
[source and numerical audit](../research/tite-crm-prior-ess-audit.md) for
reference scope and limitations.
