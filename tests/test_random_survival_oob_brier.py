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
    fit_random_survival_forest,
)
from mdanderson_stats.random_survival_forest_brier import (
    _project_censor_survival,
    _rfsrc_brier_components,
    _source_nodesize,
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


def test_rfsrc_censor_projection_and_literal_ipcw_match_base_r_fixtures():
    inputs = _read_csv("random-survival-cens-rfsrc/random-survival-cens-rfsrc-input.csv")
    projections = _read_csv("random-survival-cens-rfsrc/random-survival-cens-rfsrc-projection.csv")
    predicted = _read_csv(
        "random-survival-cens-rfsrc/random-survival-cens-rfsrc-predicted-curves.csv"
    )
    terms = _read_csv("random-survival-cens-rfsrc/random-survival-cens-rfsrc-terms.csv")
    scores = _read_csv("random-survival-cens-rfsrc/random-survival-cens-rfsrc-score.csv")
    calls = _read_csv("random-survival-cens-rfsrc/random-survival-cens-rfsrc-calls.csv")
    assert all(row["ntree"] == "50" and row["nsplit"] == "1" for row in calls)
    assert all(row["splitrule"] == "random" and row["perf_type"] == "none" for row in calls)
    assert [_source_nodesize(6, 2), _source_nodesize(301, 2), _source_nodesize(2001, 2)] == [
        5,
        10,
        10,
    ]

    for case in sorted({row["case"] for row in inputs}):
        source_rows = [row for row in inputs if row["case"] == case]
        observation_ids = sorted({int(row["observation"]) for row in source_rows})
        grid_ids = sorted({int(row["event_grid_index"]) for row in source_rows})
        times = np.array(
            [
                float(next(row["time"] for row in source_rows if int(row["observation"]) == i))
                for i in observation_ids
            ]
        )
        events = np.array(
            [
                float(next(row["event"] for row in source_rows if int(row["observation"]) == i))
                for i in observation_ids
            ]
        )
        grid = np.array(
            [
                float(
                    next(
                        row["event_grid_time"]
                        for row in source_rows
                        if int(row["event_grid_index"]) == j
                    )
                )
                for j in grid_ids
            ]
        )
        survival = np.full((len(observation_ids), len(grid_ids)), np.nan)
        for row in source_rows:
            survival[int(row["observation"]) - 1, int(row["event_grid_index"]) - 1] = _float(
                row["forest_survival"]
            )
        valid = np.all(np.isfinite(survival), axis=1)

        prediction_rows = [row for row in predicted if row["case"] == case]
        if prediction_rows:
            censor_times = sorted({float(row["censor_grid_time"]) for row in prediction_rows})
            censor_by_row = np.full((len(observation_ids), len(censor_times)), np.nan)
            for row in prediction_rows:
                censor_by_row[
                    int(row["prediction_row"]) - 1,
                    censor_times.index(float(row["censor_grid_time"])),
                ] = float(row["censor_survival"])
            projected_check = _project_censor_survival(
                np.asarray(censor_times), censor_by_row, grid
            )
        else:
            projected_check = np.ones((len(observation_ids), len(grid)))
        expected_projection = np.full_like(projected_check, np.nan)
        for row in projections:
            if row["case"] == case:
                expected_projection[
                    int(row["observation"]) - 1, int(row["event_grid_index"]) - 1
                ] = float(row["censor_survival"])
        assert_allclose(projected_check, expected_projection, rtol=1e-14, atol=1e-14)

        brier, score, counts = _rfsrc_brier_components(
            times, events, grid, survival, expected_projection, valid
        )
        expected_brier = np.full_like(brier, np.nan)
        for row in terms:
            if row["case"] == case:
                expected_brier[int(row["observation"]) - 1, int(row["event_grid_index"]) - 1] = (
                    _float(row["reference_contribution"])
                )
        assert_allclose(brier, expected_brier, rtol=1e-13, atol=1e-14, equal_nan=True)
        expected_scores = [row for row in scores if row["case"] == case]
        assert_allclose(score, [_float(row["reference_score"]) for row in expected_scores])
        assert counts.tolist() == [int(row["finite_contributors"]) for row in expected_scores]
        reference_crps = _float(expected_scores[0]["reference_crps"])
        reference_std = _float(expected_scores[0]["reference_crps_std"])
        crps, standardized = _trapz_over_source_grid(grid, score)
        assert_allclose(crps, reference_crps, rtol=1e-13, atol=1e-14)
        assert_allclose(standardized, reference_std, rtol=1e-13, atol=1e-14)


def test_rfsrc_censor_forest_end_to_end_is_seeded_and_keeps_categories():
    n = 60
    time = np.arange(1, n + 1, dtype=float)
    event = np.where(np.arange(n) % 3 == 0, 0.0, 1.0)
    covariates = np.column_stack((np.sin(time / 4), np.arange(n) % 3)).astype(float)
    fit = fit_random_survival_forest(
        time,
        event,
        covariates,
        categorical_features=[1],
        n_trees=24,
        nodesize=1,
        compute_oob=True,
        random_state=91,
    )
    first = random_survival_forest_oob_brier_score(
        fit,
        time,
        event,
        covariates,
        censor_model="rfsrc",
        censor_random_state=37,
    )
    second = random_survival_forest_oob_brier_score(
        fit,
        time,
        event,
        covariates,
        censor_model="rfsrc",
        censor_random_state=37,
    )
    assert first.censor_forest_fit is not None
    assert first.censor_forest_fit.n_trees == 50
    assert first.censor_forest_fit.nodesize == 5
    assert first.censor_forest_fit.nsplit == 1
    assert first.censor_forest_fit.split_rule == "random"
    assert first.censor_forest_fit.random_state == 37
    assert first.censor_forest_fit.categorical_levels[1] is not None
    assert first.censor_random_state == 37
    assert first.censor_model == "rfsrc"
    assert first.censor_survival.shape == first.brier.shape == (n, first.time_grid.size)
    assert_allclose(first.censor_survival, second.censor_survival)
    assert_allclose(first.brier, second.brier, equal_nan=True)
    assert first.score_row_count is not None
    assert first.score_row_count.shape == first.time_grid.shape
    assert np.all(first.score_row_count <= first.valid_row_count)


def test_rfsrc_no_censor_uses_g_one_without_fitting_and_preflights_extra_matrix():
    time = np.arange(1, 13, dtype=float)
    event = np.ones(time.size)
    x = np.arange(time.size, dtype=float)[:, None]
    fit = fit_random_survival_forest(
        time, event, x, n_trees=8, nodesize=1, compute_oob=True, random_state=3
    )
    result = random_survival_forest_oob_brier_score(
        fit, time, event, x, censor_model="rfsrc", censor_random_state=9
    )
    assert result.censor_forest_fit is None
    assert result.censor_random_state == 9
    assert np.all(result.censor_survival == 1)
    legacy_cells = (
        3 * time.size * result.time_grid.size
        + 10 * time.size
        + 5 * result.time_grid.size
    )
    with pytest.raises(ValueError, match="max_cells"):
        random_survival_forest_oob_brier_score(
            fit, time, event, x, censor_model="rfsrc", max_cells=legacy_cells
        )
