import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats._cdflib import _freeze
from mdanderson_stats.random_survival_forest import (
    RandomSurvivalForestFit,
    RandomSurvivalForestOOB,
    _forest_fingerprint,
)
from mdanderson_stats.random_survival_forest_brier import (
    _trapz_over_source_grid,
    random_survival_forest_oob_brier_score,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _read_csv(name: str) -> list[dict[str, str]]:
    with (_FIXTURES / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value: str) -> float:
    return float("nan") if value in ("", "NA", "NaN") else float(value)


def _fixture_fit(
    case: str, rows: list[dict[str, str]]
) -> tuple[RandomSurvivalForestFit, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["case"] == case]
    n = max(int(row["observation"]) for row in selected)
    m = max(int(row["grid_index"]) for row in selected)
    time = np.empty(n, dtype=float)
    event = np.empty(n, dtype=float)
    grid = np.empty(m, dtype=float)
    survival = np.full((n, m), np.nan)
    for row in selected:
        i = int(row["observation"]) - 1
        j = int(row["grid_index"]) - 1
        time[i] = float(row["time"])
        event[i] = float(row["event"])
        grid[j] = float(row["grid_time"])
        survival[i, j] = _float(row["oob_survival"])
    valid = np.all(np.isfinite(survival), axis=1)
    contributors = valid.astype(np.int64)
    x = np.empty((n, 0), dtype=float)
    oob = RandomSurvivalForestOOB(
        time_grid=_freeze(grid),
        survival=_freeze(survival),
        cumulative_hazard=_freeze(np.zeros_like(survival)),
        contributor_count=np.frombuffer(contributors.tobytes(), dtype=np.int64),
        mortality=_freeze(np.zeros(n)),
        concordance_error=float("nan"),
        comparable_pairs=0,
    )
    fit = RandomSurvivalForestFit(
        time_grid=_freeze(grid),
        covariate_mean=_freeze(np.empty(0)),
        covariate_count=0,
        trees=(),
        n_trees=1,
        mtry=1,
        nodesize=1,
        nsplit=0,
        sample_fraction=0.632,
        replace=False,
        random_state=1,
        sampled_rows=n,
        leaf_event_records=0,
        node_count=1,
        split_work=0,
        max_depth=0,
        inbag_membership=np.zeros((1, (n + 7) // 8), dtype=np.uint8),
        oob=oob,
        training_fingerprint=_forest_fingerprint(time, event, x),
    )
    return fit, time, event


def test_oob_brier_and_crps_match_independent_base_r_fixtures():
    inputs = _read_csv("random-survival-oob-brier-input.csv")
    contributions = _read_csv("random-survival-oob-brier-contributions.csv")
    curves = _read_csv("random-survival-oob-brier-curve.csv")
    cases = sorted({row["case"] for row in inputs})

    for case in cases:
        fit, time, event = _fixture_fit(case, inputs)
        result = random_survival_forest_oob_brier_score(fit, time, event)
        expected_curve = [row for row in curves if row["case"] == case]
        expected_rows = [row for row in contributions if row["case"] == case]
        assert_allclose(result.time_grid, [float(row["time"]) for row in expected_curve])
        assert_allclose(
            result.censor_survival,
            [float(row["censor_survival"]) for row in expected_curve],
            rtol=1e-13,
            atol=1e-14,
        )
        assert_allclose(
            result.score,
            [float(row["brier_score"]) for row in expected_curve],
            rtol=1e-13,
            atol=1e-14,
        )
        expected_crps = float(expected_curve[0]["crps"])
        expected_standardized = float(expected_curve[0]["crps_std"])
        assert_allclose(result.crps, expected_crps, rtol=1e-13, atol=1e-14)
        assert_allclose(result.crps_standardized, expected_standardized, rtol=1e-13, atol=1e-14)
        expected_matrix = np.full(result.brier.shape, np.nan)
        # The source fixture is observation-major; grid_index in the input
        # table disambiguates reduced/uneven evaluation grids.
        input_rows = [row for row in inputs if row["case"] == case]
        for row in expected_rows:
            i = int(row["observation"]) - 1
            candidates = [
                item
                for item in input_rows
                if int(item["observation"]) == int(row["observation"])
                and float(item["grid_time"]) == float(row["time"])
            ]
            assert len(candidates) == 1
            j = int(candidates[0]["grid_index"]) - 1
            expected_matrix[i, j] = _float(row["brier_contribution"])
        assert_allclose(result.brier, expected_matrix, rtol=1e-13, atol=1e-14, equal_nan=True)
        assert result.valid_row_count == int(np.isfinite(fit.oob.survival[:, 0]).sum())
        assert not result.brier.flags.writeable
        assert not result.score.flags.writeable


def test_oob_brier_requires_matching_full_training_data_and_oob_fit():
    inputs = _read_csv("random-survival-oob-brier-input.csv")
    fit, time, event = _fixture_fit("ties_and_between_censors", inputs)
    with pytest.raises(ValueError, match="training data values or row order"):
        random_survival_forest_oob_brier_score(fit, time[::-1], event[::-1])
    with pytest.raises(ValueError, match="compute_oob=True"):
        random_survival_forest_oob_brier_score(replace(fit, oob=None), time, event)


def test_oob_brier_preflights_output_and_singleton_grid_policy():
    inputs = _read_csv("random-survival-oob-brier-input.csv")
    fit, time, event = _fixture_fit("ties_and_between_censors", inputs)
    with pytest.raises(ValueError, match="max_cells"):
        random_survival_forest_oob_brier_score(fit, time, event, max_cells=1)
    with pytest.raises(ValueError, match="max_work"):
        random_survival_forest_oob_brier_score(fit, time, event, max_work=1)
    area, standardized = _trapz_over_source_grid(np.array([0.0]), np.array([0.4]))
    assert area == 0
    assert np.isnan(standardized)
    area, standardized = _trapz_over_source_grid(np.array([2.0]), np.array([0.4]))
    assert area == 0
    assert standardized == 0


def test_crps_integration_preserves_close_gaps_at_large_time_scale():
    first = 1e300
    second = np.nextafter(first, np.inf)
    third = np.nextafter(second, np.inf)
    grid = np.array([first, second, third])
    score = np.full(3, 0.25)
    area, standardized = _trapz_over_source_grid(grid, score)
    expected = 0.25 * (third - first)
    assert area > 0
    assert_allclose(area, expected, rtol=1e-14)
    assert_allclose(standardized, expected / third, rtol=1e-14)
