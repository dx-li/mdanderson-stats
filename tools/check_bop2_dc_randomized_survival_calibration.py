"""Verify randomized survival calibration against independent R calendar/tail calculations."""

import csv
import itertools
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_randomized_survival import bop2_dc_randomized_survival_design
from mdanderson_stats.bop2_dc_randomized_survival_optimization import (
    optimize_bop2_dc_randomized_survival,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "research/raw/bop2-dc-randomized-survival-calibration-"


def write(suffix, rows):
    with Path(str(PREFIX) + suffix + ".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read(suffix):
    with Path(str(PREFIX) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


start = time.monotonic()
settings, grids, paths = [], [], []
results = {}
T, V, N = 24, 18, 6
LOG2 = np.log(2)
seed, rate, followup = 1560930, 2, 1.5
names = ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
values = ((0.35, 0.65, 0.85), (0.2, 0.45), (0, 0.5), (0.5,))
parameters = list(itertools.product(*values))
truths = ((1, 1), (1.5, 3.5))
arms = np.array([0, 1, 0, 1, 0, 1])
for config, objective, graduate, arrival, lrv in (
    ("cgr_fixed", "cgr", True, "fixed", 0),
    ("ess_fixed", "ess_futile", True, "fixed", 0),
    ("poisson", "cgr", False, "poisson", -0.25),
):
    design = bop2_dc_randomized_survival_design(
        N,
        median_lrv=lrv,
        median_cmv=0.5,
        control_prior=(2, 1.5),
        treatment_prior=(1.5, 1),
        arm_assignments=arms,
        looks=(2, 4, 6),
        graduate_at_interim=graduate,
    )
    kwargs = dict(
        futile_truth=truths[0],
        effective_truth=truths[1],
        **{key + "_grid": val for key, val in zip(names, values, strict=True)},
        accrual_rate=rate,
        final_followup=followup,
        arrival=arrival,
        false_go_limit=0.5,
        false_no_go_limit=0.84,
        false_consider_limit=0.7,
        n_trials=T,
        n_validation=V,
        rng=seed,
        objective=objective,
    )
    result = optimize_bop2_dc_randomized_survival(design, **kwargs)
    results[config] = result
    settings.append(
        dict(
            config_id=config,
            max_subjects=N,
            arm_assignments="010101",
            looks="2;4;6",
            median_lrv=lrv,
            median_cmv=0.5,
            control_prior_shape=2,
            control_prior_rate=1.5,
            treatment_prior_shape=1.5,
            treatment_prior_rate=1,
            accrual_rate=rate,
            final_followup=followup,
            arrival=arrival,
            futile_control_median=truths[0][0],
            futile_treatment_median=truths[0][1],
            effective_control_median=truths[1][0],
            effective_treatment_median=truths[1][1],
            false_go_limit=kwargs["false_go_limit"],
            false_no_go_limit=kwargs["false_no_go_limit"],
            false_consider_limit=kwargs["false_consider_limit"],
            objective=objective,
            graduate_at_interim=int(graduate),
        )
    )
    grids.extend(
        dict(config_id=config, candidate_index=i, **dict(zip(names, p, strict=True)))
        for i, p in enumerate(parameters)
    )
    assert result.calibration_seed != result.validation_seed
    # This small calibration fits one bounded path chunk. Reconstruct random
    # inputs only; R independently computes all posterior and decision outcomes.
    generator = np.random.default_rng(result.calibration_seed)
    enrolled = (
        np.broadcast_to(np.arange(1, N + 1) / rate, (T, N))
        if arrival == "fixed"
        else generator.exponential(1 / rate, (T, N)).cumsum(axis=1)
    )
    standard = generator.exponential(1 / LOG2, (T, N)) * LOG2
    for scenario, truth in zip(("futile", "effective"), truths, strict=True):
        durations = standard * (np.asarray(truth)[arms] / LOG2)
        for trial in range(T):
            paths.extend(
                dict(
                    config_id=config,
                    phase="calibration",
                    scenario=scenario,
                    trial=trial + 1,
                    patient=p + 1,
                    arrival_time=float(enrolled[trial, p]),
                    event_duration=float(durations[trial, p]),
                )
                for p in range(N)
            )
    # The public holdout simulator draws arrivals and then event durations for
    # each trial. Its stage is independent; both truths reuse its seed.
    for scenario, truth in zip(("futile", "effective"), truths, strict=True):
        generator = np.random.default_rng(result.validation_seed)
        means = np.asarray(truth)[arms] / LOG2
        for trial in range(V):
            enrolled_one = (
                np.arange(1, N + 1) / rate
                if arrival == "fixed"
                else generator.exponential(1 / rate, N).cumsum()
            )
            durations = generator.exponential(means, N)
            paths.extend(
                dict(
                    config_id=config,
                    phase="validation",
                    scenario=scenario,
                    trial=trial + 1,
                    patient=p + 1,
                    arrival_time=float(enrolled_one[p]),
                    event_duration=float(durations[p]),
                )
                for p in range(N)
            )
write("settings", settings)
write("grid", grids)
write("paths", paths)
subprocess.run(
    [
        "Rscript",
        str(ROOT / "tools/reference_bop2_dc_randomized_survival_calibration.R"),
        str(PREFIX),
    ],
    check=True,
    cwd=ROOT,
)
metrics, decisions = read("candidate-metrics"), read("calibration-decisions")
validation, selections = read("validation-decisions"), read("selected-validation")
checked = 0
for config, result in results.items():
    selected = next(r for r in selections if r["config_id"] == config)
    assert result.selected_index == int(selected["selected_index"]), config
    assert result.candidate_count == len(parameters)
    assert int(result.feasible.sum()) == int(selected["feasible_candidates"])
    assert not result.feasible.flags.writeable
    assert_array_equal(
        [getattr(result.design, key) for key in names], parameters[result.selected_index]
    )
    for row in (r for r in metrics if r["config_id"] == config):
        c = int(row["candidate_index"])
        for field in (
            "false_go_rate",
            "false_go_mcse",
            "false_no_go_rate",
            "false_no_go_mcse",
            "correct_go_rate",
            "correct_go_mcse",
            "false_consider_rate",
        ):
            assert_allclose(getattr(result, field)[c], float(row[field]), rtol=2e-14, atol=2e-15)
            checked += 1
        assert bool(result.feasible[c]) == (row["feasible"] == "TRUE")
        for scenario in ("futile", "effective"):
            for field, reference in (
                ("expected_sample_size_" + scenario, "expected_n_" + scenario),
                ("sample_size_" + scenario + "_mcse", "expected_n_" + scenario + "_mcse"),
                (scenario + "_final_consider_rate", "false_consider_" + scenario + "_rate"),
                (scenario + "_final_consider_mcse", "false_consider_" + scenario + "_mcse"),
            ):
                assert_allclose(
                    getattr(result, field)[c], float(row[reference]), rtol=2e-14, atol=2e-15
                )
                checked += 1
            for d, label in enumerate(result.decision_labels):
                rows = [
                    r
                    for r in decisions
                    if r["config_id"] == config
                    and int(r["candidate_index"]) == c
                    and r["scenario"] == scenario
                    and r["decision"] == label
                ]
                count = sum(int(r["count"]) for r in rows)
                probability = count / T
                assert getattr(result, scenario + "_decision_count")[c, d] == count
                assert_allclose(
                    getattr(result, scenario + "_decision_probability")[c, d],
                    probability,
                    rtol=0,
                    atol=2e-15,
                )
                assert_allclose(
                    getattr(result, scenario + "_decision_mcse")[c, d],
                    np.sqrt(probability * (1 - probability) / T),
                    rtol=2e-14,
                    atol=2e-15,
                )
                checked += 3
    for scenario in ("futile", "effective"):
        oc = getattr(result, "validation_" + scenario)
        for d, label in enumerate(result.decision_labels):
            count = sum(
                int(r["count"])
                for r in validation
                if r["config_id"] == config and r["scenario"] == scenario and r["decision"] == label
            )
            p = count / V
            assert oc.decision_count[d] == count
            assert_allclose(oc.decision_probability[d], p, rtol=0, atol=2e-15)
            assert_allclose(oc.decision_mcse[d], np.sqrt(p * (1 - p) / V), rtol=2e-14, atol=2e-15)
            checked += 3
        for field, reference in (
            ("mean_enrollment", "validation_expected_n_" + scenario),
            ("enrollment_mcse", "validation_expected_n_" + scenario + "_mcse"),
        ):
            assert_allclose(getattr(oc, field), float(selected[reference]), rtol=2e-14, atol=2e-15)
            checked += 1
        for a, arm in enumerate(("control", "treatment")):
            for metric in ("events", "exposure"):
                for field, ending in (("mean_" + metric, ""), (metric + "_mcse", "_mcse")):
                    reference = "validation_" + scenario + "_mean_" + arm + "_" + metric + ending
                    assert_allclose(
                        getattr(oc, field)[a], float(selected[reference]), rtol=2e-14, atol=2e-15
                    )
                    checked += 1
    for field in (
        "validation_false_go_rate",
        "validation_false_go_mcse",
        "validation_false_no_go_rate",
        "validation_false_no_go_mcse",
        "validation_correct_go_rate",
        "validation_correct_go_mcse",
        "validation_false_consider_rate",
    ):
        assert_allclose(getattr(result, field), float(selected[field]), rtol=2e-14, atol=2e-15)
        checked += 1
    assert result.validation_feasible == (selected["validation_feasible"] == "TRUE")
assert results["cgr_fixed"].selected_index == 4
assert results["ess_fixed"].selected_index == 6
usage = resource.getrusage(resource.RUSAGE_SELF)
child = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    json.dumps(
        dict(
            configurations=len(results),
            candidates=len(parameters),
            numeric_summaries=checked,
            selected={k: v.selected_index for k, v in results.items()},
            feasible={k: int(v.feasible.sum()) for k, v in results.items()},
            holdout_feasible={k: v.validation_feasible for k, v in results.items()},
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            reference_peak_mib=child.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap + child.ru_nswap,
        ),
        indent=2,
    )
)
