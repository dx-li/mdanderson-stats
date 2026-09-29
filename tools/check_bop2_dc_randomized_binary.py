"""Compare randomized binary posteriors and exact OCs with independent base R."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.beta_binomial import BetaBinomialPosterior
from mdanderson_stats.beta_comparison import compare_beta_difference
from mdanderson_stats.bop2_dc_randomized_binary import bop2_dc_randomized_binary_design

root = Path(__file__).resolve().parents[1]
prefix = root / "tests/fixtures/bop2-dc-randomized-binary-"


def read(suffix):
    with Path(str(prefix) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


start = time.monotonic()
designs = {}
for row in read("settings"):
    designs[row["config_id"]] = bop2_dc_randomized_binary_design(
        int(row["max_subjects"]),
        float(row["theta_lrv"]),
        float(row["theta_cmv"]),
        control_prior=[float(row["control_alpha"]), float(row["control_beta"])],
        treatment_prior=[float(row["treatment_alpha"]), float(row["treatment_beta"])],
        arm_assignments=[int(value) for value in row["arm_assignments"]],
        looks=[int(value) for value in row["looks"].split(";")],
        graduate_at_interim=bool(int(row["graduate_at_interim"])),
        **{key: float(row[key]) for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")},
    )

max_posterior_error = 0
checked = 0
for row in read("beta-tail-examples"):
    comparison = compare_beta_difference(
        BetaBinomialPosterior(float(row["control_alpha"]), float(row["control_beta"])),
        BetaBinomialPosterior(float(row["treatment_alpha"]), float(row["treatment_beta"])),
        float(row["margin"]),
    )
    expected = float(row["probability_above"])
    error = abs(float(comparison.above_margin) - expected)
    assert error <= float(comparison.absolute_error) + 2e-13
    if float(row["margin"]) == 0:
        assert_allclose(expected, 5 / 6, rtol=0, atol=3e-15)
    max_posterior_error = max(max_posterior_error, error)
    checked += 1

replays = {}
for row in read("path-decisions"):
    design = designs[row["config_id"]]
    path = [int(value) for value in row["path"]]
    trial = design.replay(path)
    replays[(row["config_id"], row["path"])] = trial
    assert trial.terminal_decision == row["terminal_decision"], row
    assert len(trial.responses_observed) == int(row["terminal_look"])
    assert_array_equal(trial.responses_observed, path[: int(row["terminal_look"])])

monitor_rows = read("monitor-paths")
assert sum(len(trial.states) for trial in replays.values()) == len(monitor_rows)
for row in monitor_rows:
    design = designs[row["config_id"]]
    trial = replays[(row["config_id"], row["path"])]
    state = next(state for state in trial.states if int(state.total_n) == int(row["look"]))
    for field, column in (
        ("control_n", "control_n"),
        ("treatment_n", "treatment_n"),
        ("control_responses", "control_y"),
        ("treatment_responses", "treatment_y"),
    ):
        assert int(getattr(state, field)) == int(row[column])
    for field, error_field in (
        ("posterior_lrv", "absolute_error_lrv"),
        ("posterior_cmv", "absolute_error_cmv"),
    ):
        error = abs(float(getattr(state, field)) - float(row[field]))
        assert error <= float(getattr(state, error_field)) + 2e-13
        max_posterior_error = max(max_posterior_error, error)
        checked += 1
    assert state.decision.item() == row["decision"]
    if row["graduate_cutoff_lrv"] != "NA":
        assert_allclose(
            design._gradient_cutoffs(int(row["look"])),
            [float(row["graduate_cutoff_lrv"]), float(row["graduate_cutoff_cmv"])],
            rtol=2e-14,
            atol=2e-15,
        )

truths = read("truths")
ocs = {
    key: design.operating_characteristics(
        [float(row["control_probability"]) for row in truths],
        [float(row["treatment_probability"]) for row in truths],
    )
    for key, design in designs.items()
}
for row in read("exact-oc"):
    i = [item["truth_id"] for item in truths].index(row["truth_id"])
    oc = ocs[row["config_id"]]
    for field in (
        "final_go",
        "final_consider",
        "final_no_go",
        "no_go_probability",
        "expected_sample_size",
    ):
        assert_allclose(getattr(oc, field)[i], float(row[field]), rtol=2e-14, atol=2e-15)
        checked += 1
    for field, values in (
        ("stop_no_go", [float(row["stop_look_2"]), 0]),
        ("graduate", [float(row["graduate_look_2"]), 0]),
        ("sample_size_probability", [float(row["sample_size_2"]), float(row["sample_size_4"])]),
    ):
        assert_allclose(getattr(oc, field)[i], values, rtol=2e-14, atol=2e-15)
        checked += 2
    assert_allclose(oc.sample_size_probability[i].sum(), 1, rtol=0, atol=2e-15)
assert np.any(ocs["base_grad"].graduate > 0)
assert np.all(ocs["base_no_grad"].graduate == 0)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            configurations=len(designs),
            truth_pairs=len(truths),
            response_paths=len(replays),
            reached_looks=len(monitor_rows),
            checked_summaries=checked,
            maximum_posterior_difference=max_posterior_error,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
