"""Conventional 3+3 simulation and the BOIN site's enrollment comparisons."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, count, finite
from .boin import BOINDesign, _owned
from .boin_simulation import BOINSimulation, simulate_boin


@dataclass(frozen=True)
class ThreePlusThreeSimulation:
    """One-based selections, with zero for no MTD; count axes are (trial, dose).

    Patients/toxicities include any comparison expansion. Dose-finding counts
    exclude expansion, whose outcomes do not revise the original MTD selection.
    Selection probability/MCSE bins are [no MTD, dose 1, ..., dose J].
    """

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    dose_finding_patients: NDArray[np.int64]
    dose_finding_toxicities: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray


def _summary(
    n: np.ndarray, y: np.ndarray, selected: np.ndarray, finding_n: np.ndarray, finding_y: np.ndarray
) -> ThreePlusThreeSimulation:
    frequency = np.bincount(selected, minlength=n.shape[1] + 1) / len(n)
    return ThreePlusThreeSimulation(
        _owned(n),
        _owned(y),
        _owned(finding_n),
        _owned(finding_y),
        _owned(selected),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / len(n))),
        _owned(n.mean(axis=0)),
        _owned(y.mean(axis=0)),
    )


def simulate_three_plus_three(
    true_toxicity: ArrayLike,
    *,
    trials: int = 1000,
    start_dose: int = 1,
    rng: int | np.random.Generator | None = None,
) -> ThreePlusThreeSimulation:
    """Simulate the site's 3+3 rule, confirming a selected dose with six patients.

    Zero DLTs in three escalates; one expands to six; at least two eliminates
    that and higher doses. A lower candidate with only three patients is expanded
    before selection. The highest dose is also expanded to six before selection.
    Independent trials are advanced together with vectorized binomial draws.
    """
    p = finite(true_toxicity, "true_toxicity")
    sizes = count([trials, start_dose], "simulation sizes")
    if p.ndim != 1 or not 2 <= p.size <= 100 or np.any((p < 0) | (p > 1)):
        raise ValueError("true_toxicity must contain 2..100 probabilities in [0,1]")
    if sizes.shape != (2,) or not 1 <= sizes[0] <= 1_000_000 or not 1 <= sizes[1] <= p.size:
        raise ValueError("require 1..1000000 trials and a valid one-based start dose")
    repetitions, start = map(int, sizes)
    generator = np.random.default_rng(rng)
    n = np.zeros((repetitions, p.size), dtype=np.int64)
    y = np.zeros_like(n)
    dose = np.full(repetitions, start - 1)
    ceiling = np.full(repetitions, p.size)
    selected = np.zeros(repetitions, dtype=np.int64)
    active = np.ones(repetitions, dtype=bool)
    while np.any(active):
        rows = np.flatnonzero(active)
        j = dose[rows]
        n[rows, j] += 3
        y[rows, j] += generator.binomial(3, p[j])
        unsafe = y[rows, j] >= 2
        down = rows[unsafe]
        ceiling[down] = dose[down]
        dose[down] -= 1
        active[down[dose[down] < 0]] = False
        down = down[dose[down] >= 0]
        confirmed = down[n[down, dose[down]] == 6]
        selected[confirmed] = dose[confirmed] + 1
        active[confirmed] = False
        safe = rows[~unsafe]
        j = dose[safe]
        at_top = j + 1 == ceiling[safe]
        confirmed = safe[at_top & (n[safe, j] == 6)]
        selected[confirmed] = dose[confirmed] + 1
        active[confirmed] = False
        # Remaining safe trials either complete the current cohort or escalate.
        escalate = safe[~at_top & ((n[safe, j] == 6) | (y[safe, j] == 0))]
        dose[escalate] += 1
    return _summary(n, y, selected, n, y)


@dataclass(frozen=True)
class BOINThreePlusThreeComparison:
    boin: BOINSimulation
    three_plus_three: ThreePlusThreeSimulation
    matching: str
    boin_max_patients: NDArray[np.int64]
    expansion_patients: NDArray[np.int64]


def compare_boin_three_plus_three(
    design: BOINDesign,
    true_toxicity: ArrayLike,
    *,
    matching: str = "none",
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    titration: bool = False,
    titration_cap: int | None = None,
    moderate_toxicity: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> BOINThreePlusThreeComparison:
    """Compare independently simulated trials under the site's two matching modes.

    'expand_three_plus_three' adds patients at its selected MTD only when both
    designs select an MTD and BOIN enrolled more patients. 'match_boin' sets each
    BOIN cap to ceil(realized 3+3 enrollment / cohort_size) full cohorts. Actual
    enrollment can differ due to early stopping or cohort rounding.
    """
    if matching not in ("none", "expand_three_plus_three", "match_boin"):
        raise ValueError("unknown sample-size matching mode")
    requested = count([cohorts, cohort_size], "cohort settings")
    if requested.shape != (2,) or np.any(requested < 1) or np.prod(requested) > 100000:
        raise ValueError("require positive cohort settings with at most 100000 patients")
    size = int(requested[1])
    generator = np.random.default_rng(rng)
    conventional = simulate_three_plus_three(
        true_toxicity, trials=trials, start_dose=start_dose, rng=generator
    )
    caps = (
        (conventional.patients.sum(axis=1) + size - 1) // size
        if matching == "match_boin"
        else np.full(len(conventional.patients), int(requested[0]))
    )
    boin = simulate_boin(
        design,
        true_toxicity,
        cohorts=caps,
        cohort_size=size,
        trials=trials,
        start_dose=start_dose,
        titration=titration,
        titration_cap=titration_cap,
        moderate_toxicity=moderate_toxicity,
        rng=generator,
    )
    expansion = np.zeros(len(caps), dtype=np.int64)
    if matching == "expand_three_plus_three":
        eligible = (boin.selected_dose > 0) & (conventional.selected_dose > 0)
        expansion = np.where(
            eligible,
            np.maximum(0, boin.patients.sum(axis=1) - conventional.patients.sum(axis=1)),
            0,
        )
        n, y = conventional.patients.copy(), conventional.toxicities.copy()
        rows = np.flatnonzero(expansion)
        j = conventional.selected_dose[rows] - 1
        n[rows, j] += expansion[rows]
        y[rows, j] += generator.binomial(expansion[rows], finite(true_toxicity, "true_toxicity")[j])
        conventional = _summary(
            n,
            y,
            conventional.selected_dose,
            conventional.dose_finding_patients,
            conventional.dose_finding_toxicities,
        )
    return BOINThreePlusThreeComparison(
        boin, conventional, matching, _owned(caps * size), _owned(expansion)
    )
