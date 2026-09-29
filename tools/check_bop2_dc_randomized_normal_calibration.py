"""Check randomized Normal grid selection against independent R density integration."""

import csv
import itertools
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design
from mdanderson_stats.bop2_dc_randomized_normal_optimization import (
    optimize_bop2_dc_randomized_normal,
)

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "research/raw/bop2-dc-randomized-normal-calibration-"


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
T, V, N = 24, 18, 4
seed = 1560929
names = ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
values = ((0.4, 0.65, 0.85), (0.2, 0.45), (0, 0.5), (0.5,))
parameters = list(itertools.product(*values))
truths = ((0.25, 0.25), (-0.25, 1.0))
sds = ((0.75, 1.25), (0.5, 1.0))
for config, objective, graduate in (
    ("cgr", "cgr", True),
    ("ess", "ess_futile", True),
    ("no_graduation", "cgr", False),
):
    design = bop2_dc_randomized_normal_design(
        N,
        theta_lrv=0,
        theta_cmv=0.5,
        control_prior=(0, 1, 2, 1),
        treatment_prior=(0.25, 0.75, 2.5, 1.5),
        arm_assignments=(0, 1, 0, 1),
        looks=(2, 4),
        graduate_at_interim=graduate,
    )
    kwargs = dict(
        futile_truth=truths[0],
        effective_truth=truths[1],
        futile_truth_sd=sds[0],
        effective_truth_sd=sds[1],
        **{key + "_grid": val for key, val in zip(names, values, strict=True)},
        false_go_limit=0.75,
        false_no_go_limit=0.13,
        false_consider_limit=0.6,
        n_trials=T,
        n_validation=V,
        rng=seed,
        objective=objective,
    )
    result = optimize_bop2_dc_randomized_normal(design, **kwargs)
    results[config] = result
    row = dict(
        config_id=config,
        max_subjects=N,
        arm_assignments="0101",
        looks="2;4",
        theta_lrv=0,
        theta_cmv=0.5,
        control_prior_mean=0,
        control_prior_precision=1,
        control_prior_shape=2,
        control_prior_scale=1,
        treatment_prior_mean=0.25,
        treatment_prior_precision=0.75,
        treatment_prior_shape=2.5,
        treatment_prior_scale=1.5,
        futile_control_mean=truths[0][0],
        futile_treatment_mean=truths[0][1],
        futile_control_sd=sds[0][0],
        futile_treatment_sd=sds[0][1],
        effective_control_mean=truths[1][0],
        effective_treatment_mean=truths[1][1],
        effective_control_sd=sds[1][0],
        effective_treatment_sd=sds[1][1],
        false_go_limit=kwargs["false_go_limit"],
        false_no_go_limit=kwargs["false_no_go_limit"],
        false_consider_limit=kwargs["false_consider_limit"],
        objective=objective,
        graduate_at_interim=int(graduate),
    )
    settings.append(row)
    grids.extend(
        dict(config_id=config, candidate_index=i, **dict(zip(names, p, strict=True)))
        for i, p in enumerate(parameters)
    )
    assert_array_equal(result.candidates.parameters, parameters)
    assert result.calibration_oc.rng_seed != result.validation_oc.rng_seed
    for phase, stage in (
        ("calibration", result.calibration_oc),
        ("validation", result.validation_oc),
    ):
        z = np.random.default_rng(stage.rng_seed).standard_normal((stage.n_trials, N, 2))
        for scenario, truth, sd in zip(("futile", "effective"), truths, sds, strict=True):
            observations = np.empty((stage.n_trials, N))
            observations[:, ::2] = z[:, ::2, 0] * sd[0]
            observations[:, 1::2] = (truth[1] - truth[0]) + z[:, 1::2, 1] * sd[1]
            for trial, tape in enumerate(observations, 1):
                paths.extend(
                    dict(
                        config_id=config,
                        phase=phase,
                        scenario=scenario,
                        trial=trial,
                        patient=j,
                        value=float(y),
                    )
                    for j, y in enumerate(tape, 1)
                )
write("settings", settings)
write("grid", grids)
write("paths", paths)
subprocess.run(
    ["Rscript", str(ROOT / "tools/reference_bop2_dc_randomized_normal_calibration.R"), str(PREFIX)],
    check=True,
    cwd=ROOT,
)
metrics, decisions = read("candidate-metrics"), read("calibration-decisions")
validation, selections = read("validation-decisions"), read("selected-validation")
checked = 0
for config, result in results.items():
    evidence = result.candidates
    selected = next(r for r in selections if r["config_id"] == config)
    assert result.selected_index == int(selected["selected_index"]), config
    assert sum(evidence.feasible) == int(selected["feasible_candidates"])
    assert not evidence.parameters.flags.writeable
    assert np.all(evidence.maximum_quadrature_error <= result.design.comparison_tolerance)
    for row in (r for r in metrics if r["config_id"] == config):
        c = int(row["candidate_index"])
        for field in (
            "false_go_rate",
            "false_no_go_rate",
            "correct_go_rate",
            "false_consider_rate",
        ):
            assert_allclose(getattr(evidence, field)[c], float(row[field]), rtol=0, atol=2e-15)
            checked += 1
        assert bool(evidence.feasible[c]) == (row["feasible"] == "TRUE")
        for s, scenario in enumerate(("futile", "effective")):
            for field, reference in (
                ("expected_sample_size", "expected_n_" + scenario),
                ("enrollment_mcse", "expected_n_" + scenario + "_mcse"),
            ):
                assert_allclose(
                    getattr(evidence, field)[s, c], float(row[reference]), rtol=2e-14, atol=2e-15
                )
                checked += 1
            for d, label in enumerate(result.calibration_oc.decision_labels):
                probability = sum(
                    float(r["probability"])
                    for r in decisions
                    if r["config_id"] == config
                    and int(r["candidate_index"]) == c
                    and r["scenario"] == scenario
                    and r["decision"] == label
                )
                assert_allclose(
                    evidence.decision_probability[s, c, d], probability, rtol=0, atol=2e-15
                )
                assert_allclose(
                    evidence.decision_mcse[s, c, d],
                    np.sqrt(probability * (1 - probability) / T),
                    rtol=2e-14,
                    atol=2e-15,
                )
                checked += 2
    for s, scenario in enumerate(("futile", "effective")):
        for d, label in enumerate(result.validation_oc.decision_labels):
            probability = sum(
                float(r["probability"])
                for r in validation
                if r["config_id"] == config and r["scenario"] == scenario and r["decision"] == label
            )
            assert_allclose(
                result.validation_oc.decision_probability[s, d], probability, rtol=0, atol=2e-15
            )
            assert_allclose(
                result.validation_oc.decision_mcse[s, d],
                np.sqrt(probability * (1 - probability) / V),
                rtol=2e-14,
                atol=2e-15,
            )
            checked += 2
    for field in (
        "validation_false_go_rate",
        "validation_false_no_go_rate",
        "validation_correct_go_rate",
        "validation_false_consider_rate",
    ):
        assert_allclose(getattr(result, field), float(selected[field]), rtol=0, atol=2e-15)
        checked += 1
    assert result.validation_feasible == (selected["validation_feasible"] == "TRUE")
    assert_array_equal(
        result.calibration_oc.decision_probability,
        evidence.decision_probability[:, result.selected_index],
    )
    for s, scenario in enumerate(("futile", "effective")):
        for field, reference in (
            ("expected_sample_size", "validation_expected_n_" + scenario),
            ("enrollment_mcse", "validation_expected_n_" + scenario + "_mcse"),
        ):
            if reference in selected:
                assert_allclose(
                    getattr(result.validation_oc, field)[s],
                    float(selected[reference]),
                    rtol=2e-14,
                    atol=2e-15,
                )
                checked += 1
assert results["cgr"].selected_index == 0
assert results["ess"].selected_index == 6
assert not results["ess"].validation_feasible
usage = resource.getrusage(resource.RUSAGE_SELF)
child = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    json.dumps(
        dict(
            configurations=len(results),
            candidates=len(parameters),
            numeric_summaries=checked,
            selected={k: v.selected_index for k, v in results.items()},
            feasible={k: int(v.candidates.feasible.sum()) for k, v in results.items()},
            holdout_feasible={k: v.validation_feasible for k, v in results.items()},
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            reference_peak_mib=child.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap + child.ru_nswap,
        ),
        indent=2,
    )
)
