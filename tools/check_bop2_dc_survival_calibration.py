"""Compare complete survival calibration grids with independent base-R replay.

Writes small deterministic path tapes under research/raw, then runs the R
oracle on those inputs. No reference MCMC or parallel numerical work is used.
"""

import csv
import itertools
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats.bop2_dc_survival_optimization import optimize_bop2_dc_survival
from mdanderson_stats.bop2_survival_trial import _survival_paths

root = Path(__file__).resolve().parents[1]
prefix = root / "research/raw/bop2-dc-survival-calibration-"


def write_rows(suffix, rows):
    with Path(str(prefix) + suffix + ".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(suffix):
    with Path(str(prefix) + suffix + ".csv").open() as stream:
        return list(csv.DictReader(stream))


start = time.monotonic()
settings = dict(
    max_subjects=6,
    lrv=3,
    cmv=5,
    theta_futile=1.5,
    theta_effective=8,
    lambda_lrv_grid=[0.4, 0.7, 0.9],
    lambda_cmv_grid=[0.1, 0.3, 0.5],
    gamma_lrv_grid=[0, 0.5],
    gamma_cmv_grid=[0.5],
    prior_shape=1,
    prior_scale=2,
    looks=[2, 4, 6],
    accrual_rate=1,
    final_followup=4,
    false_go_limit=0.04,
    false_no_go_limit=0.6,
    false_consider_limit=0.8,
    n_trials=64,
    n_validation=48,
    arrival="poisson",
    rng=1560930,
)
results = {
    objective: optimize_bop2_dc_survival(**settings, objective=objective)
    for objective in ("cgr", "ess_futile")
}
base = results["cgr"]
for value in results.values():
    assert value.calibration_seed == base.calibration_seed
    assert value.validation_seed == base.validation_seed
    assert value.calibration_seed != value.validation_seed
grid_names = ("lambda_lrv", "lambda_cmv", "gamma_lrv", "gamma_cmv")
grid = [
    dict(zip(grid_names, candidate, strict=True))
    for candidate in itertools.product(*(settings[name + "_grid"] for name in grid_names))
]
write_rows("grid", grid)
write_rows(
    "settings",
    [
        {
            key: ";".join(map(str, value)) if key == "looks" else value
            for key, value in settings.items()
            if key
            in (
                "max_subjects",
                "lrv",
                "cmv",
                "prior_shape",
                "prior_scale",
                "looks",
                "final_followup",
                "false_go_limit",
                "false_no_go_limit",
                "false_consider_limit",
            )
        }
    ],
)
paths = []
for phase, seed, trials in (
    ("calibration", base.calibration_seed, settings["n_trials"]),
    ("validation", base.validation_seed, settings["n_validation"]),
):
    for truth, median in (
        ("futile", settings["theta_futile"]),
        ("effective", settings["theta_effective"]),
    ):
        for start_row, enrollment, events, _ in _survival_paths(
            settings["max_subjects"],
            median,
            settings["accrual_rate"],
            settings["final_followup"],
            trials,
            settings["arrival"],
            np.random.default_rng(seed),
        ):
            for trial in range(enrollment.shape[0]):
                for patient in range(settings["max_subjects"]):
                    paths.append(
                        dict(
                            phase=phase,
                            truth=truth,
                            trial=start_row + trial,
                            patient=patient,
                            enrollment=float(enrollment[trial, patient]),
                            event=float(events[trial, patient]),
                        )
                    )
write_rows("paths", paths)
subprocess.run(
    ["Rscript", str(root / "tools/reference_bop2_dc_survival_calibration.R"), str(prefix)],
    check=True,
    cwd=root,
    capture_output=True,
    text=True,
)
reference = read_rows("reference")
metric_rows = read_rows("metrics")
selection = {row["objective"]: int(row["selected_index"]) for row in read_rows("selection")}
labels = base.validation_futile.decision_labels
checked = 0
for objective, result in results.items():
    assert result.selected_index == selection[objective]
    for row in reference:
        phase, truth, metric = row["phase"], row["truth"], row["metric"]
        if phase == "calibration":
            index = int(row["candidate"])
            if metric == "expected_sample_size":
                actual = getattr(result, "expected_sample_size_" + truth)[index]
            elif metric == "sample_size_mcse":
                actual = getattr(result, "sample_size_" + truth + "_mcse")[index]
            else:
                is_error = metric.endswith("_mcse")
                label = metric.removesuffix("_mcse")
                actual = getattr(
                    result, truth + "_decision_" + ("mcse" if is_error else "probability")
                )[index, labels.index(label)]
        elif phase == "validation_" + objective:
            validation = getattr(result, "validation_" + truth)
            if metric == "expected_sample_size":
                actual = validation.mean_sample_size
            elif metric == "sample_size_mcse":
                actual = validation.sample_size_mcse
            else:
                is_error = metric.endswith("_mcse")
                actual = getattr(validation, "decision_" + ("mcse" if is_error else "probability"))[
                    labels.index(metric.removesuffix("_mcse"))
                ]
        else:
            continue
        assert_allclose(actual, float(row["value"]), rtol=2e-14, atol=2e-15)
        checked += 1
    for row in metric_rows:
        index = int(row["candidate"])
        for field in (
            "false_go_rate",
            "false_no_go_rate",
            "correct_go_rate",
            "false_consider_rate",
        ):
            assert_allclose(getattr(result, field)[index], float(row[field]), rtol=0, atol=2e-15)
        assert bool(result.feasible[index]) == (row["feasible"] == "TRUE")
    fg = result.validation_futile.decision_probability[labels.index("final_go")]
    fn = sum(
        result.validation_effective.decision_probability[labels.index(label)]
        for label in ("stop_no_go", "final_no_go")
    )
    fc = max(
        getattr(result, "validation_" + truth).decision_probability[labels.index("final_consider")]
        for truth in ("futile", "effective")
    )
    assert result.validation_feasible == (
        fg <= settings["false_go_limit"]
        and fn <= settings["false_no_go_limit"]
        and fc <= settings["false_consider_limit"]
    )
    assert result.validation_false_go_rate == fg
    assert result.validation_false_no_go_rate == fn
replayed = optimize_bop2_dc_survival(**{**settings, "rng": base.rng_seed}, objective="cgr")
assert_array_equal(replayed.futile_decision_probability, base.futile_decision_probability)
assert_array_equal(
    replayed.validation_effective.decision_probability,
    base.validation_effective.decision_probability,
)
assert not base.validation_feasible
assert results["ess_futile"].validation_feasible
usage = resource.getrusage(resource.RUSAGE_SELF)
reference_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
print(
    json.dumps(
        dict(
            candidates=base.candidate_count,
            calibration_trials_per_truth=settings["n_trials"],
            holdout_trials_per_truth=settings["n_validation"],
            checked_summaries=checked,
            selected=selection,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            reference_process_peak_mib=reference_usage.ru_maxrss / 1024**2,
            combined_peak_upper_bound_mib=(usage.ru_maxrss + reference_usage.ru_maxrss) / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
