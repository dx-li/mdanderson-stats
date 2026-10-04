# Generated BARD BF-BLRM stage one

`simulate_bard_blrm_stage_one` composes explicit stochastic input generation
with the existing `run_bard_blrm_trial` calendar and allocation engine. The
configuration captures all model, conduct, sampler, and budget settings, plus
an explicit maximum arrival count (at most 2,000), accrual rate, arrival law,
and DLT window. `max_escalation_patients` is required: the paper's calibrated
escalation cap is not recoverable from the available source.

```python
from mdanderson_stats.bard_blrm import BARDLogisticPrior
from mdanderson_stats.bard_blrm_generation import (
    BARDBLRMSimulationDesign,
    simulate_bard_blrm_stage_one,
)
from mdanderson_stats.bard_response import bard_response_model

response = bard_response_model([0.25, 0.40, 0.55], [[1], [2]], [0.7, 0.3], [[1.0, 2.0]])
design = BARDBLRMSimulationDesign(
    doses=[1, 2, 3],
    reference_dose=1,
    prior=BARDLogisticPrior([-3.0, 0.0], [0.0, 0.0]),
    target_interval=[0.15, 0.35],
    eta=0.30,
    cohort_size=1,
    max_escalation_patients=6,
    backfill_evaluable_cap=3,
    draws=8,
    warmup=0,
    chains=2,
    max_arrivals=20,
    arrival_distribution="exponential",
    accrual_rate=1.0,
    dlt_window=1.0,
)
generated = simulate_bard_blrm_stage_one(design, [0.05, 0.15, 0.30], response, rng=20261004)
trial = generated.trial
```

The point-mass prior is an illustrative configuration for this small example.
Use positive prior standard deviations to learn toxicity from observations,
with enough posterior draws to resolve decisions near the safety cutoff.
The [prior audit](../research/bard-blrm-audit.md) explains the separate
initial-screening discrepancy in the paper's printed simulation settings.

An integer or omitted seed is recorded in `generated.seed` and can be replayed;
a supplied NumPy `Generator` is advanced directly and has `seed=None`. The
result retains the raw calendar replay and immutable arrival-aligned outcome
tapes. For a patient in `trial.patients`, use `patient.arrival_index` and
`patient.dose - 1` to locate the generated profile and potential outcomes.
Declined arrivals remain in the tape and are never reassigned to a later
patient.

Arrival zero is at time zero. Subsequent gaps are independently Uniform on
`(0, 2/accrual_rate)` or Exponential with mean `1/accrual_rate`. At each
arrival, one factor profile is drawn from the response model's supplied **joint
profile distribution**. Outcomes are generated independently across dose
columns as an explicit set of potential-outcome conventions; only the
assigned-dose column is observed. DLT times use the shared calibrated Weibull
law with `F(window)=p` and `F(window/2)=p/2`; negative DLT outcomes are assessed
at the window. Responses are assessed at the window. If a joint DLT-response
probability is supplied, response is drawn conditional on the generated DLT
status; otherwise the endpoints are conditionally independent. Optional
titration grade-2 probabilities are dose-specific and generated conditional
on no DLT, with the supplied assessment delay.

This is an explicit Python generation and replay convention, not native
random-number, calendar, model-prior, or report parity. The source does not
fully specify the native arrival law or the calibrated escalation-patient
cap. See the [source audit](../research/bard-blrm-generation-audit.md).
