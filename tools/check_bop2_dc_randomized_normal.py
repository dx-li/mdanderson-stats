"""Check randomized Normal monitoring and simulated OCs against independent R."""

import csv
import json
import resource
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design
from mdanderson_stats.bop2_dc_randomized_normal_simulation import (
    simulate_bop2_dc_randomized_normal,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "tests/fixtures/bop2-dc-randomized-normal-"


def read(prefix, suffix):
    with Path(str(prefix) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


def make_design(row):
    shift = float(row.get("common_shift", 0))
    return bop2_dc_randomized_normal_design(
        int(row["max_subjects"]),
        float(row["theta_lrv"]),
        float(row["theta_cmv"]),
        control_prior=[
            float(row["c_mu"]) + shift,
            *[float(row["c_" + key]) for key in ("k", "a", "b")],
        ],
        treatment_prior=[
            float(row["t_mu"]) + shift,
            *[float(row["t_" + key]) for key in ("k", "a", "b")],
        ],
        arm_assignments=[int(x) for x in row["assignments"]],
        looks=[int(x) for x in row["looks"].split(";")],
        graduate_at_interim=bool(int(row["graduate_at_interim"])),
        **{key: float(row[key]) for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")},
    )


start = time.monotonic()
cases = read(PREFIX, "cases")
references = read(PREFIX, "monitor-replay")
replays = {
    row["case_id"]: make_design(row).replay(
        [float(x) + float(row["common_shift"]) for x in row["outcomes"].split(";")]
    )
    for row in cases
}
assert sum(len(replay.states) for replay in replays.values()) == len(references)
checked = 0
maximum_discrepancy = 0.0
for row in references:
    replay = replays[row["case_id"]]
    state = next(state for state in replay.states if state.total_n == int(row["look"]))
    assert state.decision == row["decision"], row
    for name in ("control_n", "treatment_n"):
        assert getattr(state, name) == int(row[name])
    for field, column in (
        ("control_location_centered", "control_location_centered"),
        ("treatment_location_centered", "treatment_location_centered"),
        ("control_df", "control_df"),
        ("treatment_df", "treatment_df"),
        ("control_scale", "control_t_scale"),
        ("treatment_scale", "treatment_t_scale"),
        ("difference_location", "difference_location"),
    ):
        assert_allclose(getattr(state, field), float(row[column]), rtol=2e-14, atol=1e-25)
        checked += 1
    for margin in ("lrv", "cmv"):
        p = getattr(state, "posterior_" + margin)
        error = getattr(state, "absolute_error_" + margin)
        discrepancy = abs(p - float(row["posterior_" + margin]))
        assert discrepancy <= error + float(row["error_" + margin]) + 2e-14, row
        maximum_discrepancy = max(maximum_discrepancy, discrepancy)
        checked += 1
for name, replay in replays.items():
    expected = [row for row in references if row["case_id"] == name]
    assert [state.total_n for state in replay.states] == [int(row["look"]) for row in expected]
    assert replay.terminal_decision == expected[-1]["decision"]
    assert replay.outcomes_observed.size == int(expected[-1]["look"])
assert {replay.terminal_decision for replay in replays.values()} == {
    "stop_no_go",
    "graduate",
    "final_go",
    "final_consider",
    "final_no_go",
}
base = replays["offset_base"]
for name in ("offset_plus_1e15", "scale_1e-8", "scale_1e8"):
    shifted = replays[name]
    for a, b in zip(base.states, shifted.states, strict=True):
        assert a.decision == b.decision
        assert_allclose(
            [a.posterior_lrv, a.posterior_cmv],
            [b.posterior_lrv, b.posterior_cmv],
            rtol=0,
            atol=2e-14,
        )

for row in read(PREFIX, "prior-only-cauchy"):
    margin = float(row["margin"])
    design = bop2_dc_randomized_normal_design(
        2,
        margin,
        margin + 1,
        control_prior=[-0.5, 2, 0.5, 1],
        treatment_prior=[1, 2, 0.5, 4],
        arm_assignments=[0, 1],
        looks=[2],
    )
    state = design.monitor([], [])
    analytic = float(row["analytic_probability"])
    assert abs(float(row["numerical_probability"]) - analytic) < 1e-12
    discrepancy = abs(state.posterior_lrv - analytic)
    assert discrepancy <= state.absolute_error_lrv + 2e-14
    maximum_discrepancy = max(maximum_discrepancy, discrepancy)
    checked += 1

# Supply the independent R oracle with the exact standardized latent draws used
# by the simulator. R performs every posterior update and stopping decision.
raw_prefix = ROOT / "research/raw/bop2-dc-randomized-normal-simulation-"
raw_prefix.parent.mkdir(parents=True, exist_ok=True)
generated = []
simulation_inputs = []
for graduate in (0, 1):
    template = dict(cases[0])
    template.update(
        theta_lrv=0,
        theta_cmv=0.5,
        lambda_lrv=0.5,
        lambda_cmv=0.5,
        c_mu=0.25,
        c_k=4,
        c_a=2,
        c_b=3,
        t_mu=0.25,
        t_k=4,
        t_a=2,
        t_b=3,
        graduate_at_interim=graduate,
    )
    if graduate:
        template.update(theta_cmv=0.25, lambda_lrv=0.25, lambda_cmv=0.25)
    seed = 815 + graduate
    trials = 24
    generator = np.random.default_rng(seed)
    centered = dict(template, c_mu=0, t_mu=0)
    for trial in range(trials):
        values = generator.standard_normal(4) + np.array([0, 0.25, 0, 0.25])
        generated.append(
            dict(
                centered,
                case_id=f"simulation_{graduate}_{trial}",
                outcomes=";".join(format(x, ".17g") for x in values),
            )
        )
    simulation_inputs.append((template, trials, seed))
with Path(str(raw_prefix) + "cases.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(generated)
subprocess.run(
    ["Rscript", str(ROOT / "tools/reference_bop2_dc_randomized_normal.R"), str(raw_prefix)],
    check=True,
    cwd=ROOT,
)
sim_reference = read(raw_prefix, "monitor-replay")
all_counts = []
for template, trials, seed in simulation_inputs:
    design = make_design(template)
    result = simulate_bop2_dc_randomized_normal(
        design,
        0.25,
        1,
        0.5,
        1,
        n_trials=trials,
        rng=seed,
    )
    count = np.zeros_like(result.look_decision_probability, dtype=np.int64)
    enrolled = []
    for trial in range(trials):
        name = f"simulation_{template['graduate_at_interim']}_{trial}"
        terminal = [row for row in sim_reference if row["case_id"] == name][-1]
        n = int(terminal["look"])
        enrolled.append(n)
        count[list(design.looks).index(n), result.decision_labels.index(terminal["decision"])] += 1
    p = count / trials
    decision_p = p.sum(axis=0)
    assert_array_equal(result.decision_count, count.sum(axis=0))
    assert_allclose(result.decision_probability, decision_p, rtol=0, atol=1e-15)
    assert_allclose(result.look_decision_probability, p, rtol=0, atol=1e-15)
    assert_allclose(result.decision_mcse, np.sqrt(decision_p * (1 - decision_p) / trials))
    assert_allclose(result.look_decision_mcse, np.sqrt(p * (1 - p) / trials))
    assert_allclose(result.sample_size_probability, p.sum(axis=1), rtol=0, atol=1e-15)
    assert_allclose(result.expected_sample_size, np.mean(enrolled))
    assert_allclose(result.enrollment_mcse, np.std(enrolled, ddof=1) / np.sqrt(trials))
    if template["graduate_at_interim"]:
        assert result.decision_count[result.decision_labels.index("graduate")] > 0
    assert result.rng_seed == seed
    assert 0 <= result.maximum_quadrature_error <= design.comparison_tolerance
    shifted_design = replace(
        design, control_prior=(1e15 + 0.25, 4, 2, 3), treatment_prior=(1e15 + 0.25, 4, 2, 3)
    )
    shifted = simulate_bop2_dc_randomized_normal(
        shifted_design,
        1e15 + 0.25,
        1,
        1e15 + 0.5,
        1,
        n_trials=trials,
        rng=seed,
    )
    assert_array_equal(shifted.look_decision_probability, result.look_decision_probability)
    assert shifted.expected_sample_size == result.expected_sample_size
    all_counts.append(result.decision_count.tolist())
    checked += 5 + 10 + 5 + 10 + 2 + 2

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
            "simulation_counts": all_counts,
            "elapsed_seconds_after_imports": time.monotonic() - start,
            "peak_mib": usage.ru_maxrss / 1024**2,
            "R_peak_mib": child.ru_maxrss / 1024**2,
            "swaps": usage.ru_nswap,
        },
        indent=2,
    )
)
