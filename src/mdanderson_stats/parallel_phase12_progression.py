"""Six-dose C++ phase-I progression and outcome-dependent accrual readiness."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite
from .parallel_phase12_decision import _mask
from .parallel_phase12_model import phase12_snapshot


@dataclass(frozen=True)
class Phase12PhaseOne:
    opened: np.ndarray
    closed: np.ndarray
    admissible: np.ndarray
    probability: FloatArray
    done: bool
    trial_closed: bool


def phase12_phase_one(
    enrolled: ArrayLike,
    toxicities: ArrayLike,
    *,
    current_dose: int = 0,
    opened: ArrayLike = (1, 0, 0, 0, 0, 0),
    closed: ArrayLike = (0,) * 6,
    admissible: ArrayLike = (0,) * 6,
) -> Phase12PhaseOne:
    """One archived DF3Plus3 transition, marginalizing fair-coin assignment.

    Pass the updated masks and the actual next assigned dose at the next call.
    enrolled counts every assignment; toxicities counts observed positive DLTs.
    Call only when the separate accrual-readiness condition is satisfied.
    `done` ends phase I; `trial_closed` specifically marks lowest-dose rejection.
    """
    n, t = count(enrolled, "enrolled"), count(toxicities, "toxicities")
    if n.shape != (6,) or t.shape != (6,) or np.any(t > n) or np.any(n > 6):
        raise ValueError(
            "enrolled/toxicities must be six counts with 0 <= toxicity <= enrolled <= 6"
        )
    current = count(current_dose, "current_dose")
    if current.ndim or current > 5:
        raise ValueError("current_dose must be an integer from 0 to 5")
    dose = int(current)
    op, shut, adm = (
        _mask(opened, "opened"),
        _mask(closed, "closed"),
        _mask(admissible, "admissible"),
    )
    if np.any(shut & adm) or np.any((shut | adm) & ~op):
        raise ValueError("closed/admissible doses must have been opened and cannot overlap")
    if not op[dose] or shut[dose] or adm[dose]:
        raise ValueError("current dose must be open, not closed, and not yet declared admissible")
    p = np.zeros(6)
    done = trial_closed = False
    number, events = int(n[dose]), int(t[dose])
    if events >= 2:
        shut[dose] = True
        if dose in (0, 5):
            done = True
            trial_closed = dose == 0
        elif dose in (1, 2):
            done = bool((shut[1] & shut[2]) or (shut[2] & adm[1]))
        else:
            done = bool(shut[3] & shut[4])
    elif number in (3, 6):
        if number == 3 and events == 1:
            p[dose] = 1
        else:
            adm[dose] = True
            done = dose == 5
    elif dose == 0:
        p[0] = 1
    # If not returning the same dose or ending phase I, reproduce OpenDoses
    # and Randomize. Intermediate coin flips yield equal final weights for
    # the two still-active doses in a level; their random stream is not copied.
    if not done and not np.any(p):
        level: tuple[int, ...] = (dose,)
        # OpenDoses is called after resolving a dose, not mid-cohort.
        if shut[dose] or adm[dose]:
            if dose == 0:
                op[1:3] = True
                level = (1, 2)
            elif dose in (1, 2):
                if adm[2] and (shut[1] or adm[1]):
                    op[4] = True
                    level = (3, 4)
                if adm[1] and adm[2]:
                    op[3] = True
                    level = (3, 4)
            elif dose in (3, 4) and adm[3] and adm[4]:
                op[5] = True
                level = (5,)
        if level == (dose,):
            level = (1, 2) if dose in (1, 2) else (3, 4) if dose in (3, 4) else (dose,)
        active = [i for i in level if op[i] and not shut[i] and not adm[i]]
        if active:
            p[active] = 1 / len(active)
        else:
            done = True
    masks = [np.frombuffer(x.tobytes(), dtype=bool) for x in (op, shut, adm)]
    return Phase12PhaseOne(masks[0], masks[1], masks[2], _freeze(p), done, trial_closed)


def phase12_accrual_ready(
    records: ArrayLike,
    *,
    time: float,
    phase_two_start: int | None = None,
    trial_closed: bool = False,
    max_patients: int = 80,
) -> bool:
    """C++ CanAddPatient/IsCohortObserved for chronologically entered patients.

    Records use phase12_snapshot's six columns. In phase I, each three-patient
    boundary waits for all three toxicity outcomes. In phase II, each five-
    patient boundary waits for all five responses. phase_two_start is the number
    of patients enrolled before phase II, not a calendar time. This does not
    generate the next rounded-exponential arrival attempt used by the simulator.
    """
    phase12_snapshot(records, time=time)  # shared endpoint/time validation
    x = finite(records, "records")
    if x.size == 0:
        x = np.empty((0, 6))
    if np.any(np.diff(x[:, 1]) < 0) or np.any(x[:, 1] > time):
        raise ValueError("records must be chronological and already entered at analysis time")
    maximum = count(max_patients, "max_patients")
    if maximum.ndim or maximum < 1:
        raise ValueError("max_patients must be a positive integer")
    if not isinstance(trial_closed, (bool, np.bool_)):
        raise ValueError("trial_closed must be boolean")
    if trial_closed or len(x) >= maximum:
        return False
    if phase_two_start is None:
        size, column, enrolled = 3, 5, len(x)
    else:
        start = count(phase_two_start, "phase_two_start")
        if start.ndim or start > len(x):
            raise ValueError("phase_two_start must be an integer from zero to enrollment")
        size, column, enrolled = 5, 3, len(x) - int(start)
    return bool(enrolled < size or enrolled % size or np.all(x[-size:, column] <= time))
