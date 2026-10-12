"""Saved PRT simulation, replay and unconditional batch reporting.

Illustrative narrow priors make this a quick workflow example, not a calibrated
trial design. Native arithmetic/calendar defects require explicit opt-in.
"""

import json
from dataclasses import replace

import numpy as np

from mdanderson_stats import (
    PRTStudy,
    read_prt_study,
    run_prt_simulation_calendar,
    simulate_prt_batch,
)

study = PRTStudy(
    3,
    2,
    interval_endpoints=(3, 7, 10),
    max_patients=6,
    true_toxicity=(0.1, 0.2),
    mean_interarrival=20,
    hazard_alpha=5,
    hazard_beta=1,
    repetitions=2,
    prior_mean=-1.5,
    prior_variance=0.005,
    draws=2048,
    warmup=1000,
    chains=4,
)
saved = read_prt_study(study.to_json())
trial = run_prt_simulation_calendar(
    saved,
    np.arange(20) * 20,
    np.full((20, 2), 11.0),
    rng=np.random.default_rng(123),
    profile="legacy_1_0",
)
assert trial.status == "selected" and trial.selected_dose == 0
assert trial.assigned == (3, 3)
assert trial.steps[-1].analysis.chain_decisions_agree
assert json.loads(trial.to_json())["status"] == "selected"
print("Supplied-calendar trial:", trial.status, trial.selected_dose, trial.duration)
print(trial.to_csv())

batch = simulate_prt_batch(saved, rng=np.random.default_rng(20261012), profile="legacy_1_0")
summary = batch.summary()
assert len(batch.trials) == saved.repetitions
assert (
    abs(
        sum(summary["selection_probability"])
        + summary["safety_stop_probability"]
        + summary["unresolved_probability"]
        - 1
    )
    < 1e-12
)
for generated in batch.trials:
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(generated.rng_state_json)
    replay = run_prt_simulation_calendar(
        saved, generated.arrival_times, generated.potential_delays, rng=rng, profile="legacy_1_0"
    )
    assert replay.to_json() == replace(generated, generation_state_json=None).to_json()
print(json.dumps(summary, indent=2))
print(batch.to_csv())
