"""Bounded independent R comparison for full-data proportional-density GOF."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from mdanderson_stats.proportional_density_full_bootstrap import (
    ProportionalDensityFullBootstrapTape,
    proportional_density_full_bootstrap,
)
from numpy.testing import assert_allclose

from mdanderson_stats.proportional_density import proportional_density

root = Path(__file__).resolve().parents[1]
fixtures = root / "tests/fixtures"


def read(name):
    with (fixtures / name).open() as stream:
        return list(csv.DictReader(stream))


rows = read("proportional-density-full-inputs.csv")
times = np.array([float(row["time"]) for row in rows])
events = np.array([int(row["event"]) for row in rows])
arms = np.array([int(row["treatment"]) for row in rows])
indices = read("proportional-density-full-resamples.csv")
components = ["failure0", "failure1", "censor0", "censor1"]
arrays = []
for component in components:
    selected = [row for row in indices if row["component"] == component]
    width = 1 + max(int(row["position"]) for row in selected)
    tape = np.empty((5, width), dtype=int)
    for row in selected:
        tape[int(row["replicate"]), int(row["position"])] = int(row["index"])
    arrays.append(tape)
tape = ProportionalDensityFullBootstrapTape(*arrays)
references = read("proportional-density-full-reference.csv")
curves = read("proportional-density-full-curves.csv")
support = np.unique(times[events == 1])
censors = [times[(arms == arm) & (events == 0)] for arm in (0, 1)]
start = time.monotonic()
max_curve_error = max_statistic_error = 0.0
checked_curves = checked_statistics = 0
for pooled in [False, True]:
    expected = [row for row in references if row["equal_censoring"] == str(pooled).upper()]
    result = proportional_density_full_bootstrap(
        times, events, arms, replicates=5, resample_tape=tape, equal_censoring=pooled
    )
    original = float(expected[0]["statistic"])
    assert_allclose(result.statistic, original, atol=2e-11, rtol=0)
    bootstrap_expected = np.array(
        [float(row["statistic"]) if row["statistic"] else np.nan for row in expected[1:]]
    )
    assert_allclose(
        result.bootstrap_statistics, bootstrap_expected, atol=2e-11, rtol=0, equal_nan=True
    )
    good = np.isfinite(bootstrap_expected)
    max_statistic_error = max(
        max_statistic_error,
        abs(result.statistic - original),
        float(np.max(np.abs(result.bootstrap_statistics[good] - bootstrap_expected[good]))),
    )
    failed = int((~good).sum())
    exceed = int(np.sum(bootstrap_expected >= original))
    assert result.failed_replicates == failed
    assert result.pvalue_lower == (1 + exceed) / 6
    assert result.pvalue_upper == (1 + exceed + failed) / 6
    checked_statistics += 1 + int(good.sum())
    for row in expected:
        if row["failure"]:
            continue
        replicate = int(row["replicate"])
        if replicate == -1:
            t, d, z = times, events, arms
        else:
            f0, f1, c0, c1 = [array[replicate] for array in arrays]
            t = np.r_[support[f0], support[f1], censors[0][c0], censors[1][c1]]
            d = np.r_[np.ones(12), np.zeros(8)]
            z = np.r_[np.zeros(6), np.ones(6), np.zeros(4), np.ones(4)]
        fit = proportional_density(t, d, z, equal_censoring=pooled)
        curve = [
            x
            for x in curves
            if x["equal_censoring"] == str(pooled).upper() and int(x["replicate"]) == replicate
        ]
        expected_curves = np.array(
            [[float(x["time"]), float(x["fitted"]), float(x["nonparametric"])] for x in curve]
        )
        actual = np.column_stack(
            [fit.time, fit.disease_survival[:, 0], fit.nonparametric_survival[:, 0]]
        )
        assert_allclose(actual, expected_curves, atol=2e-11, rtol=0)
        max_curve_error = max(
            max_curve_error, float(np.max(np.abs(actual[:, 1:] - expected_curves[:, 1:])))
        )
        checked_curves += len(curve)
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            statistics=checked_statistics,
            curve_rows=checked_curves,
            max_statistic_error=max_statistic_error,
            max_curve_error=max_curve_error,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
