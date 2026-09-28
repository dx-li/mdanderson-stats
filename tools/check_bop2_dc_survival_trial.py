"""Check BOP2-DC replay against independent R and simulation against analytic OC."""

import csv
import json
import resource
import time
from pathlib import Path

from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design
from mdanderson_stats.bop2_dc_survival_trial import (
    run_bop2_dc_survival_trial,
    simulate_bop2_dc_survival,
)

root = Path(__file__).resolve().parents[1]


def rows(name):
    with (root / "tests/fixtures" / name).open() as stream:
        return list(csv.DictReader(stream))


inputs = rows("bop2-dc-survival-trial-inputs.csv")
references = rows("bop2-dc-survival-trial-reference.csv")
start = time.monotonic()
checked = 0
for name in dict.fromkeys(row["case"] for row in inputs):
    case = [row for row in inputs if row["case"] == name]
    expected = [row for row in references if row["case"] == name]
    p = case[0]
    design = bop2_dc_survival_design(
        len(case),
        **{
            key: float(p[key])
            for key in (
                "lrv",
                "cmv",
                "lambda_lrv",
                "lambda_cmv",
                "gamma_lrv",
                "gamma_cmv",
                "prior_shape",
                "prior_scale",
            )
        },
        looks=[int(n) for n in p["looks"].split(";")],
    )
    trial = run_bop2_dc_survival_trial(
        design,
        [float(r["enrollment"]) for r in case],
        [float(r["event"]) for r in case],
        final_followup=float(p["followup"]),
    )
    assert len(trial.states) == len(expected), name
    for index, (state, row) in enumerate(zip(trial.states, expected, strict=True)):
        assert int(state.sample_size) == int(row["enrolled"])
        assert int(state.events) == int(row["events"])
        assert str(state.decision.item()) == row["decision"]
        assert_allclose(float(state.total_time), float(row["total_time"]), rtol=3e-14, atol=0)
        assert_allclose(
            trial.calendar_times[index], float(row["analysis_time"]), rtol=3e-14, atol=0
        )
        assert_allclose(
            [float(state.posterior_lrv), float(state.posterior_cmv)],
            [float(row["posterior_lrv"]), float(row["posterior_cmv"])],
            rtol=3e-14,
            atol=2e-15,
        )
        checked += 1
    assert trial.decision == expected[-1]["decision"]
    assert trial.duration == trial.calendar_times[-1]

design = bop2_dc_survival_design(
    1,
    lrv=3,
    cmv=5,
    lambda_lrv=0.6,
    lambda_cmv=0.4,
    prior_shape=1,
    prior_scale=2,
    looks=[1],
)
result = simulate_bop2_dc_survival(
    design,
    6,
    accrual_rate=2,
    final_followup=3,
    n_trials=16000,
    rng=1569028,
)
analytic = {r["metric"]: float(r["value"]) for r in rows("bop2-dc-survival-analytic-oc.csv")}
errors = []
for label, probability, mcse in zip(
    result.decision_labels, result.decision_probability, result.decision_mcse, strict=True
):
    error = abs(probability - analytic[label])
    assert error <= 5 * mcse + 2e-14
    if mcse:
        errors.append(error / mcse)
for mean_name, error_name in (
    ("mean_enrollment", "enrollment_mcse"),
    ("mean_events", "events_mcse"),
    ("mean_total_time", "total_time_mcse"),
    ("mean_duration", "duration_mcse"),
):
    error = abs(getattr(result, mean_name) - analytic[mean_name])
    mcse = getattr(result, error_name)
    assert error <= 5 * mcse + 2e-14
    if mcse:
        errors.append(error / mcse)
replayed = simulate_bop2_dc_survival(
    design,
    6,
    accrual_rate=2,
    final_followup=3,
    n_trials=16000,
    rng=result.rng_seed,
)
assert_array_equal(result.decision, replayed.decision)
assert_array_equal(result.total_time, replayed.total_time)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            calendar_cases=8,
            looks=checked,
            analytic_scenarios=1,
            trials=result.trials,
            maximum_error_mcse=float(max(errors)),
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
