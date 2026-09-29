"""Verify randomized survival posteriors, calendar replay and simulated summaries."""

import csv
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_randomized_survival import (
    bop2_dc_randomized_survival_design,
    run_bop2_dc_randomized_survival_trial,
)
from mdanderson_stats.bop2_dc_randomized_survival_simulation import (
    simulate_bop2_dc_randomized_survival,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "tests/fixtures/bop2-dc-randomized-survival-"


def read(prefix, suffix):
    with Path(str(prefix) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


def make_design(row):
    return bop2_dc_randomized_survival_design(
        int(row["max_subjects"]),
        float(row["median_lrv"]),
        float(row["median_cmv"]),
        control_prior=[float(row["control_a"]), float(row["control_b"])],
        treatment_prior=[float(row["treatment_a"]), float(row["treatment_b"])],
        arm_assignments=[int(x) for x in row["assignments"]],
        looks=[int(x) for x in row["looks"].split(";")],
        graduate_at_interim=bool(int(row["graduate_at_interim"])),
        **{key: float(row[key]) for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")},
    )


checked = 0


def compare(actual, expected):
    global checked
    assert_allclose(actual, expected, rtol=3e-14, atol=1e-115)
    checked += np.asarray(actual).size


start = time.monotonic()
cases = read(PREFIX, "cases")
references = read(PREFIX, "calendar-replay")
replays = {
    row["case_id"]: run_bop2_dc_randomized_survival_trial(
        make_design(row),
        [float(x) for x in row["enrollment_times"].split(";")],
        [float(x) for x in row["event_durations"].split(";")],
        final_followup=float(row["final_followup"]),
    )
    for row in cases
}
assert sum(len(replay.states) for replay in replays.values()) == len(references)
maximum_discrepancy = 0.0
for row in references:
    replay = replays[row["case_id"]]
    index = next(
        i for i, state in enumerate(replay.states) if state.total_n.item() == int(row["look"])
    )
    state = replay.states[index]
    assert state.decision.item() == row["decision"], row
    for name in (
        "control_n",
        "treatment_n",
        "control_events",
        "treatment_events",
        "control_exposure",
        "treatment_exposure",
    ):
        compare(getattr(state, name), float(row[name]))
    compare(state.posterior_shape, [float(row["control_shape"]), float(row["treatment_shape"])])
    compare(state.posterior_scale, [float(row["control_scale"]), float(row["treatment_scale"])])
    compare(replay.calendar_times[index], float(row["analysis_time"]))
    for margin in ("lrv", "cmv"):
        p = float(getattr(state, "posterior_" + margin))
        error = float(getattr(state, "absolute_error_" + margin))
        discrepancy = abs(p - float(row["posterior_" + margin]))
        assert discrepancy <= error + float(row["error_" + margin]) + 2e-14, row
        maximum_discrepancy = max(maximum_discrepancy, discrepancy)
        checked += 1
for name, replay in replays.items():
    expected = [row for row in references if row["case_id"] == name]
    assert [state.total_n.item() for state in replay.states] == [
        int(row["look"]) for row in expected
    ]
    assert replay.decision == expected[-1]["decision"]
    assert replay.enrolled == int(expected[-1]["look"])
assert {replay.decision for replay in replays.values()} == {
    "stop_no_go",
    "graduate",
    "final_go",
    "final_consider",
    "final_no_go",
}
for name in ("scale_1e-100", "scale_1e100"):
    for a, b in zip(replays["final_go"].states, replays[name].states, strict=True):
        assert a.decision.item() == b.decision.item()
        assert_allclose(
            [a.posterior_lrv, a.posterior_cmv],
            [b.posterior_lrv, b.posterior_cmv],
            rtol=0,
            atol=2e-13,
        )
analytic = read(PREFIX, "analytic-checks")[0]
design = bop2_dc_randomized_survival_design(
    2,
    0,
    1,
    control_prior=[1, 1],
    treatment_prior=[1, 2],
    arm_assignments=[0, 1],
    looks=[2],
)
state = design.monitor(0, 0, 1, 0, 0, 1)
compare(state.posterior_lrv, 2 / 3)
compare(state.posterior_lrv, float(analytic["probability"]))

# The oracle receives exactly the generated tapes, then recomputes every as-of
# event/exposure summary and posterior decision independently in base R.
raw_prefix = ROOT / "research/raw/bop2-dc-randomized-survival-simulation-"
raw_prefix.parent.mkdir(parents=True, exist_ok=True)
generated = []
scenarios = []
for arrival in ("fixed", "poisson"):
    for graduate in (0, 1):
        template = dict(cases[0])
        template.update(
            median_lrv=0,
            median_cmv=0.5,
            lambda_lrv=0.5,
            lambda_cmv=0.5,
            control_a=2,
            control_b=1,
            treatment_a=2,
            treatment_b=1,
            graduate_at_interim=graduate,
        )
        if graduate:
            template.update(median_lrv=-4, median_cmv=-2, lambda_lrv=0.05, lambda_cmv=0.05)
        trials = 12
        seed = 419 + graduate + 2 * (arrival == "poisson")
        generator = np.random.default_rng(seed)
        prefix_id = f"{arrival}_{graduate}"
        for trial in range(trials):
            if arrival == "fixed":
                arrivals = np.arange(1, 5) / 4
            else:
                arrivals = np.cumsum(generator.exponential(0.25, size=4))
            times = generator.exponential(np.array([2, 3, 2, 3]) / np.log(2), size=4)
            generated.append(
                dict(
                    template,
                    case_id=f"{prefix_id}_{trial}",
                    enrollment_times=";".join(format(x, ".17g") for x in arrivals),
                    event_durations=";".join(format(x, ".17g") for x in times),
                )
            )
        scenarios.append((template, arrival, seed, trials, prefix_id))
with Path(str(raw_prefix) + "cases.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(generated)
subprocess.run(
    ["Rscript", str(ROOT / "tools/reference_bop2_dc_randomized_survival.R"), str(raw_prefix)],
    check=True,
    cwd=ROOT,
)
sim_reference = read(raw_prefix, "calendar-replay")
counts = []
for template, arrival, seed, trials, prefix_id in scenarios:
    result = simulate_bop2_dc_randomized_survival(
        make_design(template),
        2,
        3,
        accrual_rate=4,
        final_followup=0.5,
        n_trials=trials,
        arrival=arrival,
        rng=seed,
    )
    terminal = [
        [row for row in sim_reference if row["case_id"] == f"{prefix_id}_{trial}"][-1]
        for trial in range(trials)
    ]
    enrolled = np.array([int(row["look"]) for row in terminal])
    times = np.array([float(row["analysis_time"]) for row in terminal])
    decisions = [row["decision"] for row in terminal]
    compare(result.enrolled, enrolled)
    compare(result.duration, times)
    compare(result.analysis_time, times)
    assert_array_equal(result.decision, decisions)
    for field, columns in (
        ("arm_n", ("control_n", "treatment_n")),
        ("arm_events", ("control_events", "treatment_events")),
        ("arm_exposure", ("control_exposure", "treatment_exposure")),
    ):
        expected = np.array([[float(row[column]) for column in columns] for row in terminal])
        compare(getattr(result, field), expected)
        if field != "arm_n":
            suffix = "events" if field == "arm_events" else "exposure"
            compare(getattr(result, "mean_" + suffix), expected.mean(axis=0))
            compare(
                getattr(result, suffix + "_mcse"), expected.std(axis=0, ddof=1) / np.sqrt(trials)
            )
    expected_count = np.array([decisions.count(label) for label in result.decision_labels])
    expected_p = expected_count / trials
    compare(result.decision_count, expected_count)
    compare(result.decision_probability, expected_p)
    compare(result.decision_mcse, np.sqrt(expected_p * (1 - expected_p) / trials))
    compare(result.mean_enrollment, enrolled.mean())
    compare(result.enrollment_mcse, enrolled.std(ddof=1) / np.sqrt(trials))
    compare(result.mean_duration, times.mean())
    compare(result.duration_mcse, times.std(ddof=1) / np.sqrt(trials))
    assert result.rng_seed == seed
    assert expected_count.sum() == trials
    if template["graduate_at_interim"]:
        assert expected_count[result.decision_labels.index("graduate")] > 0
    counts.append(expected_count.tolist())

usage = resource.getrusage(resource.RUSAGE_SELF)
child = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    json.dumps(
        {
            "cases": len(cases),
            "reached_analyses": len(references),
            "simulation_trials": len(generated),
            "simulation_analyses": len(sim_reference),
            "numeric_summaries": checked,
            "maximum_posterior_discrepancy": maximum_discrepancy,
            "simulation_counts": counts,
            "elapsed_seconds_after_imports": time.monotonic() - start,
            "peak_mib": usage.ru_maxrss / 1024**2,
            "R_peak_mib": child.ru_maxrss / 1024**2,
            "swaps": usage.ru_nswap,
        },
        indent=2,
    )
)
