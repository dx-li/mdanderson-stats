# TOP two-endpoint calendar replay and simulation

These workflows apply [TOP two-endpoint monitoring](top-endpoints.md) to
patient enrollment and endpoint-specific outcome delays. They support both
co-primary efficacy and efficacy/toxicity designs.

## Replay an observed trial path

`run_top_multiendpoint_trial(design, interarrival, event_delays)` takes one
planned arrival gap per patient and a `(max_subjects, 2)` array of potential
endpoint event delays. A finite delay in the endpoint's assessment window
means an event; positive infinity means no event by window end. The latter
outcome becomes known only when the assessment window finishes.

Each endpoint resolves independently. At a scheduled look, only observed
outcomes and elapsed follow-up contribute to the decision. During suspension,
the clock advances to the next pending endpoint's event or completed window,
and the decision is reconsidered before enrolling anyone else. When accrual
resumes, the next planned arrival gap starts from the resumed time. These are
explicit Python calendar conventions; native scheduling and seed parity have
not been established.

The source's combination rules still apply at the final look. A co-primary
trial can succeed once one endpoint is definitively acceptable while the other
remains unresolved. An efficacy/toxicity trial can stop for a definitive adverse
endpoint while the other remains pending.

```python
import numpy as np
from mdanderson_stats import TOPMultiEndpointDesign, run_top_multiendpoint_trial

design = TOPMultiEndpointDesign(
    4,
    [0.25, 0.25, 0.25, 0.25],
    0.5,
    0,
    mode="coprimary",
    looks=[2, 4],
    windows=[1, 2],
)
trial = run_top_multiendpoint_trial(
    design,
    np.zeros(4),
    np.tile([0.5, np.inf], (4, 1)),
)
assert trial.decision == "success"
assert trial.final_time == 1
assert trial.pending.tolist() == [0, 4]
assert trial.interim_suspension_time == 0.5
assert trial.final_followup_time == 0.5
```

The result contains enrollment times, observed event times, observed 0/1/NaN
outcomes, endpoint counts and compact analysis steps. Pending outcomes remain
NaN; their potential future event times are not exposed as observed events.
Trial duration ends at the decision. Interim suspension and final waiting
are reported separately.

## Simulate operating characteristics

`simulate_top_multiendpoint` accepts four joint outcome probabilities in the
same order as the prior: `(1,1), (1,0), (0,1), (0,0)`. Zero-probability truth
cells are allowed. One categorical draw per patient preserves the specified
association between outcomes; drawing the two binary outcomes independently
would generally give different trial success probabilities.

Arrival gaps can be fixed or exponential, with mean `1/accrual_rate`, including
the first patient. Conditional on each patient's joint outcome, event times
are sampled independently by endpoint from mixture-uniform distributions over
the thirds of its assessment window. This conditional timing independence is
an explicit simulation assumption.

By default, truth timing matches `design.timing_probabilities`. Supply
`truth_timing_probabilities`, either a shared triple or two triples, to study
different event timing while retaining the design's analysis weights.

```python
from mdanderson_stats import simulate_top_multiendpoint

simulation = simulate_top_multiendpoint(
    design,
    [0.3, 0.2, 0.1, 0.4],
    accrual_rate=2,
    trials=200,
    truth_timing_probabilities=[0.2, 0.3, 0.5],
    rng=134,
)
print(simulation.success_probability, simulation.success_mcse)
print(simulation.duration.mean(), simulation.patients.mean())
```

Two hundred trials make a quick example, with appreciable Monte Carlo
uncertainty. The supplied C and gamma are design inputs; the simulator itself
does not calibrate frequentist error rates. See the
[finite-grid calibration guide](top-endpoints-calibration.md) for a search over
explicit joint null scenarios and independent validation.

Simulation is vectorized across trials and retains compact per-trial summaries
instead of analysis histories. It supports at most 100,000 trials and two
million endpoint-patient cells, with limits checked before simulation-array
allocation. The [source and numerical audit](../research/top-endpoints-audit.md)
records published rules and independent exact final-look references.
