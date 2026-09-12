"""Calibration checked against exhaustive, independently enumerated R paths."""

import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.bop2_dc_optimization import BOP2DCInfeasibleError, optimize_bop2_dc


def _optimize(**options):
    return optimize_bop2_dc(
        4,
        0.2,
        0.5,
        0.2,
        0.6,
        prior=(1, 1),
        looks=[2, 4],
        lambda_lrv_grid=[0.7, 0.8, 0.9],
        lambda_cmv_grid=[0.4, 0.5],
        gamma_lrv_grid=[0.5],
        gamma_cmv_grid=[0.5],
        false_go_limit=0.2,
        **options,
    )


@pytest.mark.parametrize("objective", ["cgr", "ess_futile"])
@pytest.mark.parametrize("consider_limit", [0.3, 0.6])
def test_optimum_against_exhaustive_r_grid(objective, consider_limit):
    with (Path(__file__).parent / "fixtures" / "bop2-dc-grid.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    feasible = []
    for i in range(0, len(rows), 2):
        futile, effective = rows[i], rows[i + 1]
        metrics = (
            float(futile["go"]),
            float(effective["no_go"]),
            float(effective["go"]),
            max(float(futile["consider"]), float(effective["consider"])),
            float(futile["expected_sample_size"]),
        )
        if metrics[0] <= 0.2 and metrics[1] <= 0.3 and metrics[3] <= consider_limit:
            primary = (-metrics[2], metrics[4]) if objective == "cgr" else (metrics[4], -metrics[2])
            key = (*primary, float(futile["lambda_lrv"]), float(futile["lambda_cmv"]))
            feasible.append((key, metrics))
    expected = min(feasible)[1]
    result = _optimize(
        objective=objective, false_no_go_limit=0.3, false_consider_limit=consider_limit
    )
    actual = (
        result.false_go_rate,
        result.false_no_go_rate,
        result.correct_go_rate,
        result.false_consider_rate,
        result.futile_expected_sample_size,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    assert result.candidate_count == 6


def test_no_candidate_can_meet_all_three_constraints():
    with pytest.raises(BOP2DCInfeasibleError):
        _optimize(false_no_go_limit=0.1, false_consider_limit=0.3)
