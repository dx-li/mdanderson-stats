# BF-BOIN simulation with categorical response profiles

`simulate_bf_boin` can generate response outcomes from an explicit BARD
categorical-logistic response model while keeping the BF-BOIN calendar engine.
The supplied `true_response` vector must exactly match the model's calibrated
population response margins. Each enrolled patient receives a profile drawn
from the model's supplied joint profile probabilities; the simulator retains
the zero-based profile row, one-based factor categories, marginal
`P(response | dose, profile)`, and the probability used for the response draw.

By default, toxicity and response are conditionally independent given dose
and profile. To specify dependence, pass
`joint_toxicity_response_probability[dose, profile] =
P(DLT and response | dose, profile)`. The simulator checks the Frechet bounds
against both marginal probabilities, samples DLT under the existing BF-BOIN
Weibull rule, then samples response conditional on that DLT status. This is an
explicit Python input convention; it does not infer an endpoint association
from the BARD examples.

```python
import numpy as np
from mdanderson_stats import BFBOINDesign, bard_response_model, simulate_bf_boin

model = bard_response_model(
    population_response=[0.35, 0.55],
    factor_profiles=[[1], [2]],
    profile_probabilities=[0.4, 0.6],
    response_odds_ratios=[[1.0, 2.0]],
)
result = simulate_bf_boin(
    BFBOINDesign(n_cap=8),
    true_toxicity=[0.08, 0.18],
    true_response=model.population_response,
    response_model=model,
    # Omitting joint_toxicity_response_probability selects conditional
    # independence given dose and profile.
    cohorts=3,
    cohort_size=2,
    trials=10,
    rng=2026,
)
trial_profiles = result.factor_history[0]
trial_responses = result.response_history[0]
print(np.unique(trial_profiles, axis=0), trial_responses)
```

With the profile model omitted, response generation follows the existing
per-dose Bernoulli path and all optional profile/elimination histories are
`None`. With the model supplied, `profile_index_history` is zero-based,
`factor_history` uses one-based source category labels, and
`response_probability_history` is the model's marginal conditional response
probability. `sampled_response_probability_history` records the actual
conditional response probability; it equals the marginal value under the
independence convention. `eliminated_history` stores the final cumulative
BF-BOIN exclusion mask for each trial.

Model and joint-probability inputs are checked before random draws. Profile
history retention is bounded to one million cells per request, in addition to
the simulator's existing patient-record caps. The random generator is NumPy's
generator; this feature does not promise R RNG stream parity.
