"""Compare exact randomized binary calibration with independent R path enumeration."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_randomized_binary_optimization import (
    BOP2DCRandomizedBinaryInfeasibleError,
    optimize_bop2_dc_randomized_binary,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "tests/fixtures/bop2-dc-randomized-binary-calibration-"


def read(suffix):
    with Path(str(PREFIX) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


start = time.monotonic()
settings, truths = read("settings"), read("truths")
metric_rows, decision_rows, selected_rows = (
    read("candidate-metrics"),
    read("candidate-decisions"),
    read("selected"),
)
results = {}
selected = {}
checked = 0
candidate_total = 0
for row in settings:
    config = row["config_id"]
    truth_pairs = {
        item["scenario"]: [float(item["control_probability"]), float(item["treatment_probability"])]
        for item in truths
        if item["config_id"] == config
    }
    kwargs = dict(
        max_subjects=int(row["max_subjects"]),
        theta_lrv=float(row["theta_lrv"]),
        theta_cmv=float(row["theta_cmv"]),
        futile_truth=truth_pairs["futile"],
        effective_truth=truth_pairs["effective"],
        control_prior=[float(row["control_alpha"]), float(row["control_beta"])],
        treatment_prior=[float(row["treatment_alpha"]), float(row["treatment_beta"])],
        arm_assignments=[int(x) for x in row["arm_assignments"]],
        looks=[int(x) for x in row["looks"].split(";")],
        objective=row["objective"],
        graduate_at_interim=bool(int(row["graduate_at_interim"])),
        false_go_limit=float(row["false_go_limit"]),
        false_no_go_limit=float(row["false_no_go_limit"]),
        false_consider_limit=None
        if row["false_consider_limit"] == "NA"
        else float(row["false_consider_limit"]),
        **{
            key: [float(x) for x in row[key].split(";")]
            for key in ("lambda_lrv_grid", "lambda_cmv_grid", "gamma_lrv_grid", "gamma_cmv_grid")
        },
    )
    reference_selection = next(item for item in selected_rows if item["config_id"] == config)
    impossible = reference_selection["no_feasible_candidate"] == "TRUE"
    if impossible:
        try:
            optimize_bop2_dc_randomized_binary(**kwargs)
        except BOP2DCRandomizedBinaryInfeasibleError:
            pass
        else:
            raise AssertionError("infeasible grid unexpectedly returned a design")
        # The error intentionally has no evidence payload. Relax limits only to
        # inspect the same candidates' unconstrained OCs, then check the original
        # constraint flags independently below.
        result = optimize_bop2_dc_randomized_binary(
            **dict(kwargs, false_go_limit=1, false_no_go_limit=1, false_consider_limit=1)
        )
    else:
        result = optimize_bop2_dc_randomized_binary(**kwargs)
        assert result.selected_index == int(reference_selection["selected_index"]), config
        selected[config] = result.selected_index
        assert_array_equal(
            result.candidates.parameters[result.selected_index],
            [
                float(reference_selection["selected_" + key])
                for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
            ],
        )
        assert_array_equal(
            result.candidates.parameters[result.selected_index],
            [
                getattr(result.design, key)
                for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
            ],
        )
    results[config] = result
    evidence = result.candidates
    candidate_total += len(evidence.parameters)
    feasibility = (evidence.false_go_rate <= kwargs["false_go_limit"]) & (
        evidence.false_no_go_rate <= kwargs["false_no_go_limit"]
    )
    if kwargs["false_consider_limit"] is not None:
        feasibility &= evidence.false_consider_rate <= kwargs["false_consider_limit"]
    expected_rows = [item for item in metric_rows if item["config_id"] == config]
    assert len(expected_rows) == len(evidence.parameters)
    for item in expected_rows:
        index = int(item["candidate_index"])
        assert_array_equal(
            evidence.parameters[index],
            [float(item[key]) for key in ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")],
        )
        for field in (
            "false_go_rate",
            "false_no_go_rate",
            "correct_go_rate",
            "false_consider_rate",
        ):
            assert_allclose(getattr(evidence, field)[index], float(item[field]), rtol=0, atol=2e-14)
            checked += 1
        assert_allclose(
            evidence.expected_sample_size[:, index],
            [float(item["expected_n_futile"]), float(item["expected_n_effective"])],
            rtol=0,
            atol=2e-14,
        )
        checked += 2
        assert bool(feasibility[index]) == (item["feasible"] == "TRUE")
    if not impossible:
        assert_array_equal(feasibility, evidence.feasible)
    assert np.count_nonzero(feasibility) == int(reference_selection["feasible_count"])
    assert np.all(evidence.maximum_comparison_error <= result.design.comparison_tolerance)
    assert not evidence.decision_probability.flags.writeable
    assert not evidence.parameters.flags.writeable

for row in decision_rows:
    result = results[row["config_id"]]
    scenario = result.scenarios.index(row["scenario"])
    candidate = int(row["candidate_index"])
    look = list(result.design.looks).index(int(row["look"]))
    decision = result.decision_labels.index(row["decision"])
    actual = result.candidates.decision_probability[scenario, candidate, look, decision]
    assert_allclose(actual, float(row["probability"]), rtol=0, atol=2e-14)
    checked += 1
    if decision == 0:
        actual_size = result.candidates.sample_size_probability[scenario, candidate, look]
        assert_allclose(actual_size, float(row["sample_size_probability"]), rtol=0, atol=2e-14)
        checked += 1
assert selected["cgr_grad"] == 0 and selected["ess_grad"] == 4
assert selected["tie_first"] == 0
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        {
            "configurations": len(settings),
            "candidate_configurations": candidate_total,
            "numeric_summaries": checked,
            "selected_indices": selected,
            "elapsed_seconds_after_imports": time.monotonic() - start,
            "peak_mib": usage.ru_maxrss / 1024**2,
            "swaps": usage.ru_nswap,
        },
        indent=2,
    )
)
