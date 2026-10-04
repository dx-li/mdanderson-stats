"""Alignment and refusal rules for downstream missing-data adapters."""

from __future__ import annotations

import numpy as np
import pytest

from mdanderson_stats.random_survival_forest import fit_random_survival_forest
from mdanderson_stats.random_survival_forest_brier import (
    random_survival_forest_oob_brier_score,
)
from mdanderson_stats.random_survival_forest_vimp import (
    permutation_random_survival_forest_importance,
)


def _survival_data(*, missing: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 28
    time = np.arange(1.0, n + 1.0)
    event = (np.arange(n) % 3 != 0).astype(float)
    x = np.column_stack((np.sin(np.arange(n)), np.cos(np.arange(n))))
    if missing == "omit":
        x[[3, 18], 0] = np.nan
    elif missing == "impute":
        x[[4, 20], 1] = np.nan
    return time, event, x


def _fit(time: np.ndarray, event: np.ndarray, x: np.ndarray, na_action: str):
    return fit_random_survival_forest(
        time,
        event,
        x,
        na_action=na_action,
        n_trees=16,
        mtry=2,
        nodesize=2,
        nsplit=0,
        ntime=0,
        random_state=731,
        compute_oob=True,
    )


def test_omitted_brier_rows_keep_original_mapping_and_verify_raw_input() -> None:
    time, event, x = _survival_data(missing="omit")
    fit = _fit(time, event, x, "omit")
    result = random_survival_forest_oob_brier_score(fit, time, event, x)

    expected = np.array([i for i in range(time.size) if i not in (3, 18)])
    np.testing.assert_array_equal(fit.training_row_indices, expected)
    np.testing.assert_array_equal(result.row_indices, expected)
    assert result.brier.shape[0] == expected.size

    changed = x.copy()
    changed[3, 1] += 0.25
    with pytest.raises(ValueError, match="original training values"):
        random_survival_forest_oob_brier_score(fit, time, event, changed)

    reordered = np.arange(time.size)[::-1]
    with pytest.raises(ValueError, match="original training values"):
        permutation_random_survival_forest_importance(
            fit,
            time[reordered],
            event[reordered],
            x[reordered],
            feature_indices=[0],
            random_state=37,
        )


def test_imputed_oob_diagnostics_reject_undefined_missing_semantics() -> None:
    time, event, x = _survival_data(missing="impute")
    fit = _fit(time, event, x, "impute")
    with pytest.raises(ValueError, match="does not yet define missing-data semantics"):
        random_survival_forest_oob_brier_score(fit, time, event, x)
    with pytest.raises(ValueError, match="does not yet define missing-data semantics"):
        permutation_random_survival_forest_importance(
            fit, time, event, x, feature_indices=[0], random_state=37
        )
