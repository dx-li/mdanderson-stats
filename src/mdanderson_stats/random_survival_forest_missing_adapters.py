"""Input alignment for adapters operating on a fitted survival forest."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .random_survival_forest import (
    RandomSurvivalForestFit,
    _forest_data,
    _forest_data_allow_missing,
    _forest_original_fingerprint,
)


def _adapter_training_data(
    fit: RandomSurvivalForestFit,
    time: ArrayLike,
    event: ArrayLike,
    covariates: ArrayLike | None,
    *,
    adapter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate raw inputs, then return the exact rows represented by an OOB fit."""
    if fit.imputation_performed:
        raise ValueError(
            f"{adapter} does not yet define missing-data semantics for imputed forests"
        )
    if fit.na_action not in ("raise", "omit", "impute"):
        raise ValueError("fit contains invalid missing-data metadata")
    if fit.na_action == "raise":
        t, e, x = _forest_data(time, event, covariates)
        if x.shape[1] != fit.covariate_count:
            raise ValueError("training covariate count does not match the OOB fit")
        if fit.training_row_indices is not None:
            identity = np.arange(t.size, dtype=np.int64)
            if not np.array_equal(fit.training_row_indices, identity):
                raise ValueError("fit row map is inconsistent with na_action='raise'")
        row_indices = np.arange(t.size, dtype=np.int64)
        return t, e, x, row_indices

    original_time, original_event, original_x = _forest_data_allow_missing(
        time, event, covariates
    )
    if original_x.shape[1] != fit.covariate_count:
        raise ValueError("training covariate count does not match the OOB fit")
    levels = fit.categorical_levels
    if levels and len(levels) != original_x.shape[1]:
        raise ValueError("fit contains inconsistent categorical level metadata")
    categorical = tuple(i for i, column_levels in enumerate(levels) if column_levels is not None)
    fingerprint = _forest_original_fingerprint(
        original_time, original_event, original_x, categorical
    )
    if fit.original_training_fingerprint is None:
        if fit.na_action != "raise":
            raise ValueError("fit is missing the original training fingerprint")
    elif fingerprint != fit.original_training_fingerprint:
        raise ValueError("original training values, missingness or row order do not match the fit")

    if fit.training_row_indices is None:
        raise ValueError("fit is missing its original training row map")
    row_indices = np.asarray(fit.training_row_indices, dtype=np.int64)
    if (
        row_indices.ndim != 1
        or row_indices.size == 0
        or np.any(row_indices < 0)
        or np.any(row_indices >= original_time.size)
        or np.unique(row_indices).size != row_indices.size
    ):
        raise ValueError("fit contains an invalid original training row map")
    if row_indices.size == original_time.size and np.array_equal(
        row_indices, np.arange(original_time.size)
    ):
        retained_time, retained_event, retained_x = (
            original_time,
            original_event,
            original_x,
        )
    else:
        retained_time = original_time[row_indices]
        retained_event = original_event[row_indices]
        retained_x = original_x[row_indices]
    if any(
        np.any(~np.isfinite(values))
        for values in (retained_time, retained_event, retained_x)
    ):
        raise ValueError("fit row map retains values that are missing in the original input")
    t, e, x = _forest_data(retained_time, retained_event, retained_x)
    return t, e, x, row_indices
