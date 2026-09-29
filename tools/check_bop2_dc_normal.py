"""Check Normal BOP2-DC against independent base-R posterior and trial references.

Regenerate references with `Rscript tools/reference_bop2_dc_normal.R .`.
The fixed path tape uses NumPy default_rng(1560931), Normal(.6, 1.4), (64, 6).
"""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_normal import (
    bop2_dc_normal_design,
    run_bop2_dc_normal_trial,
    simulate_bop2_dc_normal,
)

root = Path(__file__).resolve().parents[1]
fixtures = root / "tests/fixtures"


def read(suffix):
    with (fixtures / f"bop2-dc-normal-{suffix}.csv").open() as stream:
        return list(csv.DictReader(stream))


start = time.monotonic()
cases = {row["case_id"]: row for row in read("cases")}
checked = 0
states = {}
for row in read("reference"):
    case = cases[row["case_id"]]
    settings = {
        key: float(value)
        for key, value in case.items()
        if key not in ("case_id", "observations", "looks", "max_subjects", "observation_offset")
    }
    offset = float(case["observation_offset"])
    for key in ("prior_mean", "theta_lrv", "theta_cmv"):
        settings[key] += offset
    design = bop2_dc_normal_design(
        int(case["max_subjects"]),
        looks=[int(n) for n in case["looks"].split(";")],
        **settings,
    )
    n = int(row["n"])
    state = design.monitor([float(y) + offset for y in case["observations"].split(";")][:n])
    states[(row["case_id"], n)] = state
    for field, reference in (
        ("posterior_location_centered", "posterior_centered_location"),
        ("location_offset", "location_offset"),
        ("posterior_location", "posterior_location"),
        ("posterior_df", "posterior_df"),
        ("posterior_scale", "posterior_t_scale"),
        ("posterior_lrv", "probability_lrv"),
        ("posterior_cmv", "probability_cmv"),
    ):
        absolute_tolerance = 2e-15 if field in ("posterior_lrv", "posterior_cmv") else 0
        assert_allclose(
            getattr(state, field), float(row[reference]), rtol=8e-14, atol=absolute_tolerance
        )
        checked += 1
    assert state.decision == row["decision"], (row["case_id"], state)

for n in (2, 4):
    for first, second in (("offset_base", "offset_1e15"), ("tiny_units", "huge_units")):
        a, b = states[(first, n)], states[(second, n)]
        assert_allclose(
            [a.posterior_lrv, a.posterior_cmv],
            [b.posterior_lrv, b.posterior_cmv],
            rtol=8e-14,
            atol=2e-15,
        )
        assert a.decision == b.decision
assert states[("strict_equality", 4)].posterior_lrv == 0.5
assert states[("strict_equality", 4)].decision == "final_consider"

design = bop2_dc_normal_design(
    6,
    0,
    1,
    prior_mean=0.15,
    prior_precision=0.7,
    prior_shape=2,
    prior_scale=1,
    lambda_lrv=0.75,
    lambda_cmv=0.4,
    gamma_lrv=0.5,
    gamma_cmv=0.5,
    looks=[2, 4, 6],
)
path_rows = read("paths")
paths = np.array([float(row["value"]) for row in path_rows]).reshape(64, 6)
assert_array_equal(paths, np.random.default_rng(1560931).normal(0.6, 1.4, size=(64, 6)))
replays = [run_bop2_dc_normal_trial(design, path) for path in paths]
reference = read("replay-reference")
assert sum(len(trial.states) for trial in replays) == len(reference)
for row in reference:
    trial = replays[int(row["trial"]) - 1]
    state = next(state for state in trial.states if state.sample_size == int(row["sample_size"]))
    for field, expected in (
        ("posterior_location", "posterior_location"),
        ("posterior_df", "posterior_df"),
        ("posterior_scale", "posterior_scale"),
        ("posterior_lrv", "probability_lrv"),
        ("posterior_cmv", "probability_cmv"),
    ):
        assert_allclose(getattr(state, field), float(row[expected]), rtol=5e-13, atol=2e-15)
        checked += 1
    assert state.decision == row["decision"]

simulation = simulate_bop2_dc_normal(design, 0.6, 1.4, n_trials=64, rng=1560931)
assert_array_equal(simulation.sample_size, [trial.enrolled for trial in replays])
assert_array_equal(simulation.decision, [trial.decision for trial in replays])
for row in read("replay-summary"):
    index = simulation.decision_labels.index(row["decision"])
    assert simulation.decision_count[index] == int(row["count"])
    for actual, expected in (
        (simulation.decision_probability[index], row["probability"]),
        (simulation.decision_mcse[index], row["binomial_mcse"]),
        (simulation.mean_enrollment, row["mean_enrollment"]),
        (simulation.enrollment_mcse, row["enrollment_mcse"]),
    ):
        assert_allclose(actual, float(expected), rtol=3e-14, atol=2e-15)
        checked += 1
assert np.count_nonzero(simulation.decision_count) == 4
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            cases=len(cases),
            posterior_looks=len(states),
            trial_paths=len(paths),
            reached_trial_looks=len(reference),
            checked_summaries=checked,
            decision_counts=dict(
                zip(simulation.decision_labels, simulation.decision_count.tolist())
            ),
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
