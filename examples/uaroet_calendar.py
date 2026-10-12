"""Portable calendar jobs and reports; deliberately short demo fits, not design validation."""

from dataclasses import replace
from pathlib import Path

from mdanderson_stats import (
    UAROETDesktopScenario,
    merge_uaroet_desktop_batches,
    read_uaroet_desktop_batch,
    read_uaroet_desktop_study,
    simulate_uaroet_desktop,
)

# Run from the checkout, or copy this example beside uaroet-python-study.json.
fixture = Path(__file__).with_name("uaroet-python-study.json")
if not fixture.exists():
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/uaroet-python-study.json"
study = read_uaroet_desktop_study(fixture.read_text())
p = dict(study.trial_parameters)
p.update(NumPatients=1, CohortSize_First=1, NumTimesToShrinkUtilityCutoff=1)
study = replace(
    study,
    patients=(),
    trial_parameters=p,
    simulation_parameters={"NumChains": 2, "BurnInSamples": 0},
)
scenario = UAROETDesktopScenario(1, [[0.7, 0.3], [0.5, 0.5]], [[0.9, 0.1], [0.7, 0.3]], 0.2)
options = dict(repetitions=1, seed=12, target=0.99, maximum_rhat=1000, chunk_draws=32, max_draws=32)
a = simulate_uaroet_desktop(study, scenario, first_trial=0, **options)
b = simulate_uaroet_desktop(study, scenario, first_trial=1, **options)
combined = merge_uaroet_desktop_batches(a, b)
assert read_uaroet_desktop_batch(combined.to_json()) == combined
assert combined.trials == 2
print(combined.to_csv())
