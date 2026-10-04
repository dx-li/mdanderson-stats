"""Source-defined patient-allocation risk summaries for dose-finding simulations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar

_MAX_DOSES = 100
_MAX_CELLS = 1_000_000
_MAX_COUNT = 2**53


@dataclass(frozen=True, slots=True)
class DoseAllocationRisks:
    """Monte Carlo risks that more than a fixed share of planned enrollment is above target."""

    trials: int
    has_exact_target: bool
    unavailable_reason: str | None
    overdose60_trials: int | None
    overdose80_trials: int | None
    overdose60_probability: float | None
    overdose80_probability: float | None
    overdose60_mcse: float | None
    overdose80_mcse: float | None


def dose_allocation_risks(
    patients: ArrayLike,
    true_toxicity: ArrayLike,
    *,
    target: float,
    planned_patients: ArrayLike,
) -> DoseAllocationRisks:
    """Summarize BOIN/Keyboard overdosing risks from trial-by-dose count rows.

    A dose is above target exactly when its supplied true toxicity probability is
    strictly greater than ``target``. For each trial, the event is that the
    total allocated above-target patients is strictly greater than 60% or 80%
    of *planned* enrollment. Integer cross-products implement both strict
    comparisons without floating-point threshold rounding.

    The source applications report these metrics only if at least one true dose
    probability equals the target. If none does, event counts and risks are
    unavailable with an explicit reason. Multiple exact-target doses are
    handled by the same probability-based mask; this avoids the native R
    single-index condition error while preserving its stated calculation.
    Probabilities are fractions; multiply by 100 for native-style percentages.
    MCSE is the ordinary Bernoulli Monte Carlo standard error.
    """
    matrix = _count_matrix(patients)
    probability = _probability_vector(true_toxicity)
    threshold = scalar(target, "target")
    if not 0 <= threshold <= 1:
        raise ValueError("target must lie in [0, 1]")
    if probability.shape != (matrix.shape[1],):
        raise ValueError("true_toxicity must have one entry per dose column")
    planned = _planned_counts(planned_patients, matrix.shape[0])
    actual = matrix.sum(axis=1, dtype=np.int64)
    if np.any(planned < actual):
        raise ValueError("planned_patients must be at least each trial's realized enrollment")

    if not np.any(probability == threshold):
        return DoseAllocationRisks(
            trials=matrix.shape[0],
            has_exact_target=False,
            unavailable_reason=(
                "source summary is defined only when at least one true dose probability "
                "equals target"
            ),
            overdose60_trials=None,
            overdose80_trials=None,
            overdose60_probability=None,
            overdose80_probability=None,
            overdose60_mcse=None,
            overdose80_mcse=None,
        )

    above_target = matrix @ (probability > threshold).astype(np.int64)
    count60 = int(np.count_nonzero(5 * above_target > 3 * planned))
    count80 = int(np.count_nonzero(5 * above_target > 4 * planned))
    p60 = count60 / matrix.shape[0]
    p80 = count80 / matrix.shape[0]
    return DoseAllocationRisks(
        trials=matrix.shape[0],
        has_exact_target=True,
        unavailable_reason=None,
        overdose60_trials=count60,
        overdose80_trials=count80,
        overdose60_probability=p60,
        overdose80_probability=p80,
        overdose60_mcse=float(np.sqrt(p60 * (1 - p60) / matrix.shape[0])),
        overdose80_mcse=float(np.sqrt(p80 * (1 - p80) / matrix.shape[0])),
    )


def _count_matrix(value: ArrayLike) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or value.dtype.kind not in "iuf":
            raise ValueError("patients must be a real numeric matrix")
        if value.ndim != 2:
            raise ValueError("patients must be a two-dimensional trial-by-dose matrix")
        rows, array_columns = value.shape
        if rows < 1 or not 2 <= array_columns <= _MAX_DOSES or value.size > _MAX_CELLS:
            raise ValueError("patients must have 1..1000000 cells and 2..100 dose columns")
        raw = value
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if rows < 1 or rows > _MAX_CELLS:
            raise ValueError("patients must contain 1..1000000 trial rows")
        n_columns: int | None = None
        cells = 0
        for row in value:
            if isinstance(row, np.ndarray):
                if np.iscomplexobj(row) or row.ndim != 1:
                    raise ValueError("patients must be a flat trial-by-dose matrix")
                row_width = int(row.size)
                if np.issubdtype(row.dtype, np.bool_):
                    raise ValueError("patients must be real integer counts")
            elif isinstance(row, (list, tuple)):
                row_width = len(row)
                if any(isinstance(item, (list, tuple, np.ndarray)) for item in row):
                    raise ValueError("patients must be a flat trial-by-dose matrix")
                if any(
                    isinstance(item, (complex, np.complexfloating, bool, np.bool_)) for item in row
                ):
                    raise ValueError("patients must be real integer counts")
            else:
                raise ValueError("patients must be a two-dimensional trial-by-dose matrix")
            if n_columns is None:
                n_columns = row_width
            if row_width != n_columns:
                raise ValueError("patients must have the same number of doses in every row")
            cells += row_width
            if cells > _MAX_CELLS:
                raise ValueError("patients must contain at most 1000000 count cells")
        if n_columns is None or not 2 <= n_columns <= _MAX_DOSES:
            raise ValueError("patients must have 2..100 dose columns")
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or raw.dtype.kind == "b":
            raise ValueError("patients must be real integer counts")
    else:
        raise ValueError("patients must be a bounded two-dimensional matrix")

    counts = finite(raw, "patients")
    if np.any((counts < 0) | (counts != np.floor(counts)) | (counts >= _MAX_COUNT)):
        raise ValueError("patients must contain nonnegative integer counts below 2**53")
    return counts.astype(np.int64)


def _probability_vector(value: ArrayLike) -> FloatArray:
    if isinstance(value, np.ndarray):
        if (
            np.iscomplexobj(value)
            or value.dtype.kind == "b"
            or value.ndim != 1
            or not 2 <= value.size <= _MAX_DOSES
        ):
            raise ValueError("true_toxicity must be a real vector of length 2..100")
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= _MAX_DOSES:
            raise ValueError("true_toxicity must be a vector of length 2..100")
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            raise ValueError("true_toxicity must be one-dimensional")
        if any(isinstance(item, (complex, np.complexfloating, bool, np.bool_)) for item in value):
            raise ValueError("true_toxicity must be real")
    else:
        raise ValueError("true_toxicity must be a bounded one-dimensional vector")
    probability = finite(value, "true_toxicity")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("true_toxicity entries must lie in [0, 1]")
    return probability


def _planned_counts(value: ArrayLike, rows: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value) or value.dtype.kind == "b" or value.ndim not in (0, 1):
            raise ValueError("planned_patients must be an integer or one value per trial")
        if value.ndim == 1 and value.size not in (1, rows):
            raise ValueError("planned_patients must be scalar or match the trial count")
    elif isinstance(value, (list, tuple)):
        if not value or len(value) not in (1, rows):
            raise ValueError("planned_patients must be scalar or match the trial count")
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
            raise ValueError("planned_patients must be one-dimensional")
        if any(isinstance(item, (complex, np.complexfloating, bool, np.bool_)) for item in value):
            raise ValueError("planned_patients must be real counts")
    elif isinstance(value, (complex, np.complexfloating, bool, np.bool_)):
        raise ValueError("planned_patients must be real integer counts")
    counts = finite(value, "planned_patients")
    if counts.ndim == 0:
        counts = counts.reshape(1)
    if np.any((counts < 0) | (counts != np.floor(counts)) | (counts >= _MAX_COUNT)):
        raise ValueError("planned_patients must contain nonnegative integer counts below 2**53")
    if counts.size not in (1, rows):
        raise ValueError("planned_patients must be scalar or match the trial count")
    return np.broadcast_to(counts.astype(np.int64), (rows,))
