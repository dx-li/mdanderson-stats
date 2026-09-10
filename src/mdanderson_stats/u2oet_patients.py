"""Native U2OET patient snapshots and next-patient cohort decisions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import _real
from .u2oet_decision import U2OETPosterior, _integer, u2oet_allocation
from .u2oet_scenario import _rows


@dataclass(frozen=True)
class U2OETPatients:
    """Records retain native IDs/one-based doses and zero-based outcomes (-1 pending)."""

    records: FloatArray
    treated: FloatArray
    complete: FloatArray
    toxicity_only: FloatArray
    ignored_outcomes: int
    last_dose: tuple[int, int] | None
    trailing_treatments: int


def u2oet_patients(
    records: ArrayLike, *, dose_counts: tuple[int, int], efficacy_levels: int, toxicity_levels: int
) -> U2OETPatients:
    """Aggregate all assignments; use partial information only when toxicity is known."""
    if len(dose_counts) != 2:
        raise ValueError("dose_counts must contain two entries")
    dims = tuple(_integer(v, "dose count", 2, 5) for v in dose_counts)
    le = _integer(efficacy_levels, "efficacy_levels", 2, 4)
    lt = _integer(toxicity_levels, "toxicity_levels", 2, 4)
    rows = _real(records, "records")
    if rows.shape == (0,):
        rows = rows.reshape(0, 5)
    if (
        rows.ndim != 2
        or rows.shape[1] != 5
        or rows.shape[0] > 2500
        or np.any(rows != np.floor(rows))
    ):
        raise ValueError("require at most 2500 rows of five integer patient fields")
    if rows.size and (
        rows[0, 0] != 1 or np.any(np.diff(rows[:, 0]) <= 0) or np.any(rows[:, 0] >= 2**53)
    ):
        raise ValueError("patient IDs must start at 1 and strictly increase below 2**53")
    for column, low, high in ((1, 1, dims[0]), (2, 1, dims[1]), (3, -1, le - 1), (4, -1, lt - 1)):
        if np.any((rows[:, column] < low) | (rows[:, column] > high)):
            raise ValueError("patient dose or outcome index outside the configured range")
    treated = np.zeros(dims)
    complete = np.zeros((*dims, le, lt))
    tox = np.zeros((*dims, lt))
    last = None
    trailing = 0
    ignored = 0
    for _, a, b, efficacy, toxicity in rows.astype(np.int64):
        pair = (int(a - 1), int(b - 1))
        treated[pair] += 1
        if toxicity < 0:
            ignored += 1
        elif efficacy < 0:
            tox[*pair, toxicity] += 1
        else:
            complete[*pair, efficacy, toxicity] += 1
        trailing = trailing + 1 if pair == last else 1
        last = pair
    return U2OETPatients(
        _freeze(rows), _freeze(treated), _freeze(complete), _freeze(tox), ignored, last, trailing
    )


def read_u2oet_patients(
    path: str | Path, *, dose_counts: tuple[int, int], efficacy_levels: int, toxicity_levels: int
) -> U2OETPatients:
    """Read five numeric columns per patient, as specified by guide section 3.1."""
    return u2oet_patients(
        _rows(path),
        dose_counts=dose_counts,
        efficacy_levels=efficacy_levels,
        toxicity_levels=toxicity_levels,
    )


@dataclass(frozen=True)
class U2OETPatientDecision:
    probabilities: FloatArray
    continuing_cohort: bool
    cohort_position: int
    closed_unacceptable_cohort: bool
    reason: str


def u2oet_next_patient(
    posterior: U2OETPosterior,
    patients: U2OETPatients,
    *,
    cohort_size: int = 3,
    max_patients: int = 60,
    surplus: int | None = None,
    top: int | None = 2,
    initial: tuple[int, int] | None = None,
    greedy: bool = False,
) -> U2OETPatientDecision:
    """Guide section 3: continue an acceptable open cohort, otherwise allocate anew.

    A trailing run modulo cohort_size determines whether a cohort is open.
    Enrollment caps stop new assignments but do not choose a final recommendation.
    The caller must recompute the posterior using the latest patient snapshot.
    """
    if not isinstance(patients, U2OETPatients) or not isinstance(posterior, U2OETPosterior):
        raise ValueError("patients and posterior must be U2OET result objects")
    size = _integer(cohort_size, "cohort_size", 1, 2500)
    maximum = _integer(max_patients, "max_patients", 1, 2500)
    if posterior.acceptable.shape != patients.treated.shape or posterior.acceptable.dtype != bool:
        raise ValueError("posterior and patient dose grids must match")
    weights = np.zeros(patients.treated.shape)
    if patients.records.shape[0] >= maximum:
        return U2OETPatientDecision(_freeze(weights), False, 0, False, "enrollment limit reached")
    occupied = patients.trailing_treatments % size
    last = patients.last_dose
    if occupied and last is not None and posterior.acceptable[last]:
        weights[last] = 1
        return U2OETPatientDecision(
            _freeze(weights), True, occupied + 1, False, "continue open cohort"
        )
    allocation = u2oet_allocation(
        posterior,
        patients.treated,
        surplus=size if surplus is None else surplus,
        top=top,
        initial=initial,
        greedy=greedy,
    )
    return U2OETPatientDecision(
        allocation.probabilities,
        False,
        1 if allocation.best is not None else 0,
        bool(occupied and last is not None),
        allocation.reason,
    )
