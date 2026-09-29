"""Verify Normal calibration and its holdout against independent base-R replay."""

import csv
import itertools
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_normal_optimization import optimize_bop2_dc_normal

root = Path(__file__).resolve().parents[1]
prefix = root / "research/raw/bop2-dc-normal-calibration-"


def write(suffix, rows):
    with Path(str(prefix) + suffix + ".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read(suffix):
    with Path(str(prefix) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


settings = dict(
    max_subjects=8,
    theta_lrv=0,
    theta_cmv=0.5,
    theta_futile=-0.5,
    theta_effective=1.5,
    truth_sd=1.3,
    prior_mean=0,
    prior_precision=0.5,
    prior_shape=1.5,
    prior_scale=0.75,
    lambda_lrv_grid=[0.6, 0.8, 0.95],
    lambda_cmv_grid=[0.2, 0.5, 0.8],
    gamma_lrv_grid=[0, 0.5],
    gamma_cmv_grid=[0.5],
    looks=[2, 4, 8],
    false_go_limit=0.04,
    false_no_go_limit=0.2,
    false_consider_limit=0.8,
    n_trials=64,
    n_validation=48,
    rng=1560932,
)
start = time.monotonic()
results = {
    objective: optimize_bop2_dc_normal(**settings, objective=objective)
    for objective in ("cgr", "ess_futile")
}
base = results["cgr"]
grid_names = ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
grid = [
    dict(zip(grid_names, values, strict=True))
    for values in itertools.product(*(settings[name + "_grid"] for name in grid_names))
]
assert_array_equal(base.candidates.parameters, [list(row.values()) for row in grid])
write("grid", grid)
write(
    "settings",
    [
        {
            key: ";".join(map(str, value)) if key == "looks" else value
            for key, value in settings.items()
            if not key.endswith("_grid")
        }
    ],
)
paths = []
assert base.calibration_oc.rng_seed != base.validation_oc.rng_seed
for phase, seed, trials in (
    ("calibration", base.calibration_oc.rng_seed, settings["n_trials"]),
    ("validation", base.validation_oc.rng_seed, settings["n_validation"]),
):
    centered = settings["truth_sd"] * np.random.default_rng(seed).standard_normal(
        (trials, settings["max_subjects"])
    )
    for truth in ("futile", "effective"):
        for trial, outcomes in enumerate(centered):
            for patient, value in enumerate(outcomes, 1):
                paths.append(
                    dict(phase=phase, truth=truth, trial=trial, patient=patient, value=float(value))
                )
write("paths", paths)
subprocess.run(
    ["Rscript", str(root / "tools/reference_bop2_dc_normal_calibration.R"), str(prefix)],
    cwd=root,
    check=True,
)
reference, metrics = read("reference"), read("metrics")
selection = {row["objective"]: row for row in read("selection")}
checked = 0
for objective, result in results.items():
    selected = selection[objective]
    assert result.selected_index == int(selected["selected_index"])
    for row in reference:
        scenario = ("futile", "effective").index(row["truth"])
        phase, metric = row["phase"], row["metric"]
        if phase == "calibration":
            candidate = int(row["candidate"])
            source = result.candidates
            index = (scenario, candidate)
            count = result.calibration_oc.n_trials
        elif phase == "validation_" + objective:
            source = result.validation_oc
            index = scenario
            count = source.n_trials
        else:
            continue
        if metric == "mean_enrollment":
            actual = source.expected_sample_size[index]
        elif metric == "enrollment_mcse":
            actual = source.enrollment_mcse[index]
        elif metric == "trials":
            actual = count
        else:
            is_mcse = metric.startswith("mcse_")
            label = metric.removeprefix("mcse_") if is_mcse else metric.removeprefix("p_")
            field = "decision_mcse" if is_mcse else "decision_probability"
            actual = getattr(source, field)[index][
                result.calibration_oc.decision_labels.index(label)
            ]
        assert_allclose(actual, float(row["value"]), rtol=2e-14, atol=2e-15)
        checked += 1
    for row in metrics:
        i = int(row["candidate"])
        for field in ("false_go_rate", "false_no_go_rate", "correct_go_rate"):
            assert_allclose(
                getattr(result.candidates, field)[i], float(row[field]), rtol=0, atol=2e-15
            )
        fc = max(float(row["false_consider_futile"]), float(row["false_consider_effective"]))
        assert_allclose(result.candidates.false_consider_rate[i], fc, rtol=0, atol=2e-15)
        assert bool(result.candidates.feasible[i]) == (row["feasible"] == "TRUE")
    assert_array_equal(
        result.calibration_oc.decision_probability,
        result.candidates.decision_probability[:, result.selected_index],
    )
    for field in (
        "validation_false_go_rate",
        "validation_false_no_go_rate",
        "validation_false_consider_rate",
        "validation_correct_go_rate",
    ):
        assert_allclose(getattr(result, field), float(selected[field]), rtol=0, atol=2e-15)
    assert result.validation_feasible == (selected["validation_feasible"] == "TRUE")

# The same latent draws and model differences must give identical choices after
# adding a large exactly representable offset to all location inputs.
shifted_settings = {**settings}
for name in ("theta_lrv", "theta_cmv", "theta_futile", "theta_effective", "prior_mean"):
    shifted_settings[name] += 1e15
shifted = optimize_bop2_dc_normal(**shifted_settings)
assert_array_equal(shifted.candidates.decision_probability, base.candidates.decision_probability)
assert_array_equal(shifted.candidates.expected_sample_size, base.candidates.expected_sample_size)
assert_array_equal(
    shifted.validation_oc.decision_probability, base.validation_oc.decision_probability
)
assert shifted.selected_index == base.selected_index
assert not base.validation_feasible
assert np.any(base.candidates.feasible) and not np.all(base.candidates.feasible)
usage = resource.getrusage(resource.RUSAGE_SELF)
child = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    json.dumps(
        dict(
            candidates=len(grid),
            checked_summaries=checked,
            calibration_trials=settings["n_trials"],
            validation_trials=settings["n_validation"],
            selected={key: value.selected_index for key, value in results.items()},
            validation_feasible={key: value.validation_feasible for key, value in results.items()},
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            reference_peak_mib=child.ru_maxrss / 1024**2,
            combined_peak_upper_bound_mib=(usage.ru_maxrss + child.ru_maxrss) / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
