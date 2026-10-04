"""Global-censor OOB Brier support for predictor-only imputation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import fit_random_survival_forest
from mdanderson_stats.random_survival_forest_brier import (
    _censor_survival,
    random_survival_forest_oob_brier_score,
)
from mdanderson_stats.random_survival_forest_vimp import (
    permutation_random_survival_forest_importance,
)


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 36
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 4 != 0).astype(float)
    categorical = (np.arange(n) % 3).astype(float)
    numeric = np.sin(np.arange(n) / 3.0)
    numeric[[4, 19, 31]] = np.nan
    return time, event, np.column_stack((categorical, numeric))


def _fit(time: np.ndarray, event: np.ndarray, x: np.ndarray):
    return fit_random_survival_forest(
        time,
        event,
        x,
        categorical_features=(0,),
        na_action="impute",
        n_trees=18,
        mtry=2,
        nodesize=2,
        nsplit=0,
        ntime=0,
        random_state=731,
        compute_oob=True,
    )


def test_predictor_only_imputed_oob_brier_matches_direct_global_ipcw_equations() -> None:
    time, event, x = _case()
    fit = _fit(time, event, x)
    assert fit.imputation_performed
    assert fit.oob is not None and fit.oob.concordance_available

    result = random_survival_forest_oob_brier_score(fit, time, event, x)
    assert result.censor_model == "km"
    np.testing.assert_array_equal(result.row_indices, np.arange(time.size))
    np.testing.assert_array_equal(
        result.censor_survival, _censor_survival(time, event, result.time_grid)
    )

    oob = fit.oob
    assert oob is not None
    wrong_oob = replace(oob, row_indices=np.arange(time.size)[::-1])
    with pytest.raises(ValueError, match="OOB row map"):
        random_survival_forest_oob_brier_score(replace(fit, oob=wrong_oob), time, event, x)
    valid = oob.contributor_count > 0
    expected = np.full_like(result.brier, np.nan)
    for i in np.flatnonzero(valid):
        event_index = np.searchsorted(result.time_grid, time[i], side="right") - 1
        event_weight = 1.0 if event_index < 0 else result.censor_survival[event_index]
        for j, evaluation_time in enumerate(result.time_grid):
            if time[i] > evaluation_time:
                expected[i, j] = (1.0 - oob.survival[i, j]) ** 2 / result.censor_survival[j]
            elif event[i] == 1:
                expected[i, j] = oob.survival[i, j] ** 2 / event_weight
            else:
                expected[i, j] = 0.0
    np.testing.assert_allclose(result.brier, expected, rtol=1e-13, atol=1e-14, equal_nan=True)
    assert np.any(result.brier[valid] > 0)
    np.testing.assert_array_equal(
        result.score_row_count, np.full(result.time_grid.size, valid.sum())
    )
    np.testing.assert_allclose(
        result.score, np.mean(expected[valid], axis=0), rtol=1e-13, atol=1e-14
    )

    changed = x.copy()
    changed[4, 0] = 2.0
    with pytest.raises(ValueError, match="original training values"):
        random_survival_forest_oob_brier_score(fit, time, event, changed)

    with pytest.raises(ValueError, match="censor_model='rfsrc'"):
        random_survival_forest_oob_brier_score(
            fit, time, event, x, censor_model="rfsrc", censor_random_state=2
        )
    with pytest.raises(ValueError, match="does not yet define missing-data semantics"):
        permutation_random_survival_forest_importance(
            fit, time, event, x, feature_indices=[1], random_state=3
        )


def test_global_censor_weights_are_positive_on_event_interest_grid() -> None:
    time, event, _ = _case()
    event_grid = np.unique(time[event == 1])
    censor_survival = _censor_survival(time, event, event_grid)
    # At every event time the risk set contains at least one uncensored event
    # record, so the finite-sample global estimator cannot reach zero there.
    assert np.all(np.isfinite(censor_survival))
    assert np.all(censor_survival > 0.0)


def test_all_missing_row_is_excluded_by_the_fit_row_map() -> None:
    time, event, x = _case()
    time = time.copy()
    event = event.copy()
    x = x.copy()
    time[7] = np.nan
    event[7] = np.nan
    x[7] = np.nan
    fit = _fit(time, event, x)
    result = random_survival_forest_oob_brier_score(fit, time, event, x)
    expected_rows = np.delete(np.arange(time.size), 7)
    np.testing.assert_array_equal(result.row_indices, expected_rows)
    np.testing.assert_array_equal(
        result.censor_survival,
        _censor_survival(time[expected_rows], event[expected_rows], result.time_grid),
    )


def test_missing_retained_outcomes_remain_unsupported_for_imputed_brier() -> None:
    time, event, x = _case()
    time = time.copy()
    event = event.copy()
    time[7] = np.nan
    fit = _fit(time, event, x)
    with pytest.raises(ValueError, match="missing predictors among rows retained"):
        random_survival_forest_oob_brier_score(fit, time, event, x)
