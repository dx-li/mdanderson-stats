"""MERIT trial replay and simulation with persistent arm-specific stopping."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

from ._validation import FloatArray, count
from .boin import _owned
from .merit import MERITDesign, _integer, _isotonic
from .merit_interims import MERITInterims
from .merit_simulation import _correlation, _draw, _rates


@dataclass(frozen=True)
class MERITTrialResult:
    patients: NDArray[np.int64]
    toxicity: NDArray[np.int64]
    efficacy: NDArray[np.int64]
    stopped_toxicity: NDArray[np.bool_]
    stopped_futility: NDArray[np.bool_]
    admissible: NDArray[np.bool_]
    mean_patients: FloatArray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    any_selection_probability: float
    power_one: float | None
    power_two: float | None


def _run(
    design: MERITDesign,
    interims: MERITInterims,
    trials: int,
    next_outcomes: Callable[[int], tuple[np.ndarray, np.ndarray]],
    truly_admissible: ArrayLike | None,
) -> MERITTrialResult:
    table = interims.boundaries(design.patients_per_arm)
    truth = None if truly_admissible is None else np.asarray(truly_admissible)
    if truth is not None and (truth.shape != (design.doses,) or truth.dtype != bool):
        raise ValueError("truly_admissible must be a boolean dose vector")
    shape = (trials, design.doses)
    patients, toxicity, efficacy = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    stop_t, stop_e = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    active = np.ones(shape, dtype=bool)
    boundary_at = {int(n): i for i, n in enumerate(table.patients)}
    for n in range(1, design.patients_per_arm + 1):
        t, e = next_outcomes(n)
        patients += active
        toxicity += t & active
        efficacy += e & active
        if n in boundary_at:
            k = boundary_at[n]
            stop_t |= active & table.assess_toxicity[k] & (toxicity >= table.toxicity_stop_min[k])
            stop_e |= active & table.assess_efficacy[k] & (efficacy <= table.efficacy_stop_max[k])
            active &= ~(stop_t | stop_e)
        if not active.any():
            break
    selected = np.zeros(shape, dtype=bool)
    # All surviving arms have the same final sample size. Pool only survivors;
    # stopped arms remain excluded and contribute no pseudo/future observations.
    patterns = (active * (1 << np.arange(design.doses))).sum(1)
    for mask in np.unique(patterns):
        if mask == 0:
            continue
        rows = np.flatnonzero(patterns == mask)
        doses = np.flatnonzero(mask & (1 << np.arange(design.doses)))
        cells = np.ix_(rows, doses)
        tx, ex = toxicity[cells].astype(float), efficacy[cells].astype(float)
        if design.isotonic_toxicity:
            tx = _isotonic(tx)
        if design.isotonic_efficacy:
            ex = _isotonic(ex)
        selected[cells] = (tx <= design.toxicity_max) & (ex >= design.efficacy_min)
    probability = selected.mean(0)
    any_selection = selected.any(1)
    p1 = p2 = None
    if truth is not None:
        p1 = float((any_selection & ~selected[:, ~truth].any(1)).mean())
        p2 = float(selected[:, truth].any(1).mean())
    return MERITTrialResult(
        _owned(patients),
        _owned(toxicity),
        _owned(efficacy),
        _owned(stop_t),
        _owned(stop_e),
        _owned(selected),
        _owned(patients.mean(0)),
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        float(any_selection.mean()),
        p1,
        p2,
    )


def simulate_merit_interims(
    design: MERITDesign,
    interims: MERITInterims,
    toxicity_rates: ArrayLike,
    efficacy_rates: ArrayLike,
    *,
    correlation: float = 0.5,
    trials: int = 10000,
    truly_admissible: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> MERITTrialResult:
    """Complete-endpoint cohort progression; stop each arm permanently on either rule.

    Interim monitoring uses each arm's raw counts. Final isotonic pooling uses
    surviving arms only. These explicit conventions are not verified native parity.
    No calendar-time accrual, pending outcomes or sample-size reallocation.
    """
    trials = _integer(trials, "trials", 1, 1000000)
    rho = _correlation(correlation)
    qt = ndtri(_rates(toxicity_rates, design.doses, "toxicity_rates"))
    qe = ndtri(_rates(efficacy_rates, design.doses, "efficacy_rates"))
    generator = np.random.default_rng(rng)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        zt, ze = _draw(generator, trials, design.doses, rho)
        return zt <= qt, ze <= qe

    return _run(design, interims, trials, draw, truly_admissible)


def run_merit_trial(
    design: MERITDesign,
    interims: MERITInterims,
    toxicity_outcomes: ArrayLike,
    efficacy_outcomes: ArrayLike,
) -> MERITTrialResult:
    """Replay a complete trial from outcome matrices (patient position, dose).

    Matrices may end early if every arm has stopped. Entries after an arm stops are
    ignored and may be filled with zeros; they are never included as observations.
    Result arrays have one trial row. Both endpoints must be binary and complete
    for every actually enrolled patient. This is not a pending-outcome API.
    """
    t, e = (
        count(toxicity_outcomes, "toxicity_outcomes"),
        count(efficacy_outcomes, "efficacy_outcomes"),
    )
    if (
        t.ndim != 2
        or t.shape != e.shape
        or t.shape[1] != design.doses
        or not 1 <= t.shape[0] <= design.patients_per_arm
        or np.any(t > 1)
        or np.any(e > 1)
    ):
        raise ValueError(
            "outcomes require binary matrices with 1..patients_per_arm rows and doses columns"
        )

    def outcome(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n > t.shape[0]:
            raise ValueError("outcomes ended before all arms stopped or completed")
        return t[None, n - 1, :].astype(bool), e[None, n - 1, :].astype(bool)

    return _run(design, interims, 1, outcome, None)
