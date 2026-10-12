"""Bounded readers for original ToxFinder patient and simulation CSV records."""

import csv
import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .toxfinder_model import ToxFinderFit, ToxFinderPrior, fit_toxfinder
from .toxfinder_native_decision import toxfinder_native_allocation, toxfinder_native_stage1
from .toxfinder_native_information import _frozen
from .toxfinder_native_trial import _integer


def _rows(text: str, maximum: int) -> list[list[str]]:
    if not isinstance(text, str) or len(text) > 5_000_000:
        raise ValueError("CSV text must contain at most five million characters")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0].lstrip().startswith("#"):
            continue
        while row and not row[-1].strip():
            row.pop()
        rows.append(row)
        if len(rows) > maximum:
            raise ValueError("CSV exceeds record limit")
    return rows


def _scale(reference_doses: ArrayLike) -> FloatArray:
    scale = finite(reference_doses, "reference_doses")
    if scale.shape != (2,) or np.any(scale <= 0):
        raise ValueError("reference_doses must contain two positive scales")
    return scale


@dataclass(frozen=True)
class ToxFinderPatientRecord:
    """Dates/times remain literal identifiers; the model is not time-dependent."""

    patient_id: str
    identifier: str
    year: str
    month: str
    day: str
    time: str
    cohort_id: str
    standardized_dose: FloatArray
    toxicity: int  # 0=no, 1=yes, 2=unset


def read_toxfinder_subject_csv(
    text: str, *, reference_doses: ArrayLike
) -> tuple[ToxFinderPatientRecord, ...]:
    scale = _scale(reference_doses)
    records = []
    ids: set[str] = set()
    for row in _rows(text, 100):
        if len(row) != 10:
            raise ValueError("subject rows require ten fields")
        dose = finite([float(row[7]), float(row[8])], "dose") / scale
        if np.any(dose < 0) or np.any(dose > 1):
            raise ValueError("subject doses must standardize into [0,1]")
        if row[9].strip() not in ("0", "1", "2"):
            raise ValueError("toxicity must be 0 (no), 1 (yes), or 2 (unset)")
        if not row[0] or row[0] in ids:
            raise ValueError("patient IDs must be nonempty and unique")
        ids.add(row[0])
        outcome = int(row[9])
        if outcome == 1 and not np.any(dose):
            raise ValueError("toxicity is impossible at the origin")
        records.append(
            ToxFinderPatientRecord(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6], _frozen(dose), outcome
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class ToxFinderObservedRecommendation:
    records: tuple[ToxFinderPatientRecord, ...]
    fit: ToxFinderFit
    dose: FloatArray | None
    recommended_doses: FloatArray
    selected_index: int | None
    observed_patients: int
    pending_patients: int
    reason: str


def recommend_toxfinder_native(
    records: tuple[ToxFinderPatientRecord, ...],
    dose_levels: ArrayLike,
    *,
    prior: ToxFinderPrior,
    target: float,
    stage1_patients: int,
    stage2_patients: int,
    cohort_size: int,
    seed: int,
    start_index: int = 0,
    potency: float = 1.0,
    mode: Literal["fi_ckp", "fi"] = "fi_ckp",
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
) -> ToxFinderObservedRecommendation:
    """Fit observed patients only; observed counts control stage; admitted counts control capacity.

    Unlike simulation's paired-cohort optimization, each request refits available
    observed data. Unset outcomes remain in the returned records and never enter
    the likelihood as non-toxic observations.
    """
    from .toxfinder_native_decision import _levels

    levels = _levels(dose_levels)
    n1 = _integer(stage1_patients, "stage1_patients", 1, 100)
    n2 = _integer(stage2_patients, "stage2_patients", 0, 100)
    cohort = _integer(cohort_size, "cohort_size", 1, 100)
    start = _integer(start_index, "start_index", 0, len(levels) - 1)
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    if n1 % cohort or n2 % cohort or n1 + n2 > 100 or len(records) > n1 + n2:
        raise ValueError("stage sizes must be cohort multiples, capacity at most 100")
    if not all(isinstance(r, ToxFinderPatientRecord) for r in records):
        raise TypeError("records must be ToxFinderPatientRecord instances")
    # Validate manually constructed records as strictly as imported records.
    if len({r.patient_id for r in records}) != len(records):
        raise ValueError("patient IDs must be unique")
    for r in records:
        if (
            r.toxicity not in (0, 1, 2)
            or r.standardized_dose.shape != (2,)
            or not np.all(np.isfinite(r.standardized_dose))
            or np.any(r.standardized_dose < 0)
            or np.any(r.standardized_dose > 1)
        ):
            raise ValueError("invalid patient record")
    observed = [r for r in records if r.toxicity != 2]
    history = [r.standardized_dose for r in observed]
    outcomes = [r.toxicity for r in observed]
    fitted = fit_toxfinder(
        history if history else levels[start : start + 1],
        outcomes if history else [0],
        np.ones(len(history)) if history else [0],
        prior=prior,
        rng=np.random.default_rng(seed),
        draws=draws,
        warmup=warmup,
        chains=chains,
    )
    selected: int | None
    dose: FloatArray | None
    capacity = len(records) >= n1 + n2
    if len(observed) < n1:
        dose = toxfinder_native_stage1(
            levels,
            fitted.log_parameters,
            history,
            outcomes,
            target=target,
            start_index=start,
            log_parameters=True,
        )
        pairs, selected, reason = _frozen([dose]), 0, "stage_one"
    else:
        if mode not in ("fi", "fi_ckp"):
            raise ValueError("unsupported allocation mode")
        allocation = toxfinder_native_allocation(
            levels,
            fitted.log_parameters,
            target=target,
            potency=potency,
            mode=mode,
            doses=history,
            toxicities=outcomes,
            subjects=np.ones(len(history)),
            prior=prior,
            log_parameters=True,
            allocation=not capacity,
        )
        selected = 2 if ((len(observed) - n1) // cohort) % 2 == 0 else 1
        pairs, dose, reason = allocation.doses, allocation.doses[selected], "stage_two"
    if capacity:
        selected, dose, reason = None, None, "capacity_reached"
    return ToxFinderObservedRecommendation(
        tuple(records),
        fitted,
        dose,
        pairs,
        selected,
        len(observed),
        len(records) - len(observed),
        reason,
    )


@dataclass(frozen=True)
class ToxFinderArchivedSimulation:
    trial_id: int
    standardized_doses: FloatArray
    seeds: tuple[int, int]
    toxicities: int
    subjects: int
    mean_true_exposure: float
    parameters: FloatArray


def read_toxfinder_simulations_csv(
    text: str, *, reference_doses: ArrayLike, retained_draws: int
) -> tuple[ToxFinderArchivedSimulation, ...]:
    """Read native archived summaries, converting native to public parameter order.

    Draw count and physical scales are explicit: the historical dims.csv header
    is insufficient to safely infer every field. No native RNG parity is implied.
    """
    scale = _scale(reference_doses)
    draws = _integer(retained_draws, "retained_draws", 1, 10_000)
    result = []
    cells = 0
    for row in _rows(text, 10_000):
        if len(row) != 12 + 6 * draws:
            raise ValueError("simulation row length does not match retained_draws")
        cells += 6 * draws
        if cells > 200_000:
            raise ValueError("archived draws exceed 200,000 parameter values")
        values = finite([float(v) for v in row], "simulation values")
        pairs = values[1:7].reshape((3, 2)) / scale
        parameters = values[12:].reshape((draws, 6))[:, [0, 3, 1, 4, 2, 5]]
        if np.any(pairs < 0) or np.any(pairs > 1) or np.any(parameters <= 0):
            raise ValueError("invalid standardized doses or positive parameters")
        ints = values[[0, 7, 8, 9, 10]]
        if np.any(ints < 0) or np.any(ints != np.floor(ints)) or not 0 <= values[9] <= values[10]:
            raise ValueError("invalid trial identifier, seeds, or counts")
        if values[10] < 1 or values[10] > 10_000 or not 0 <= values[11] <= 1:
            raise ValueError("invalid subject count or true exposure")
        result.append(
            ToxFinderArchivedSimulation(
                int(values[0]),
                _frozen(pairs),
                (int(values[7]), int(values[8])),
                int(values[9]),
                int(values[10]),
                float(values[11]),
                _frozen(parameters),
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class ToxFinderNativeStudy:
    """Recovered execution settings; original prior/MCMC metadata is preserved."""

    settings: tuple[tuple[str, tuple[str, ...]], ...]
    reference_doses: FloatArray
    dose_levels: FloatArray
    true_surface: FloatArray
    target: float
    potency: float
    start_index: int
    stage1_patients: int
    stage2_patients: int
    cohort_size: int
    mode: Literal["fi_ckp", "fi"]

    def to_tmml(self) -> str:
        """Semantic settings round-trip; comments and formatting are not retained."""
        import shlex

        return "\n".join(shlex.join([key, *values]) for key, values in self.settings) + "\n"


def read_toxfinder_tmml(text: str) -> ToxFinderNativeStudy:
    """Read original design/surface settings without inferring a fitted prior.

    Supply a separately validated ToxFinderPrior to simulation. Native elicitation
    and MCMC controls remain metadata; they are never silently mapped to Python
    solver or sampler settings. No shell commands or expressions are executed.
    """
    import shlex

    from .toxfinder_native_decision import _goal, _levels
    from .toxfinder_native_trial import toxfinder_native_surface

    if not isinstance(text, str) or len(text) > 1_000_000:
        raise ValueError("TMML must contain at most one million characters")
    settings: dict[str, tuple[str, ...]] = {}
    for line in text.splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens:
            if tokens[0] in settings:
                raise ValueError("duplicate TMML setting")
            settings[tokens[0]] = tuple(tokens[1:])

    def numbers(key: str, length: int) -> FloatArray:
        if key not in settings or len(settings[key]) != length:
            raise ValueError(f"TMML requires {key} with {length} values")
        return finite([float(v) for v in settings[key]], key)

    def integer(key: str, low: int, high: int) -> int:
        value = float(numbers(key, 1)[0])
        if value != int(value):
            raise ValueError(f"{key} must be an integer")
        return _integer(int(value), key, low, high)

    scale = _scale(numbers("tl-d_star", 2))
    size = integer("tl-nD", 2, 50)
    levels = _levels(numbers("tl-D1", 2 * size).reshape((size, 2)) / scale)
    dimensions = numbers("sim-num-points", 2)
    if (
        np.any(dimensions != np.floor(dimensions))
        or np.any(dimensions < 2)
        or np.any(dimensions > 100)
    ):
        raise ValueError("surface dimensions must be integers in [2,100]")
    nx, ny = (int(v) for v in dimensions)
    surface = numbers("sim-surface", nx * ny).reshape((ny, nx))
    toxfinder_native_surface(surface, levels)
    n1, n2 = integer("num-stage1", 1, 100), integer("num-stage2", 0, 100)
    cohort = integer("cohort", 1, 100)
    if n1 % cohort or n2 % cohort or n1 + n2 > 100:
        raise ValueError("stage sizes must be cohort multiples, capacity at most 100")
    mode = integer("tl-alloc", 1, 4)
    if mode not in (1, 4):
        raise ValueError("only advertised FI+CKP (1) and FI-only (4) modes are supported")
    potency = float(numbers("lambda", 1)[0])
    if potency <= 0:
        raise ValueError("potency must be positive")
    return ToxFinderNativeStudy(
        tuple(settings.items()),
        _frozen(scale),
        _frozen(levels),
        _frozen(surface),
        _goal(float(numbers("pi_star", 1)[0])),
        potency,
        integer("nD-start", 1, size) - 1,
        n1,
        n2,
        cohort,
        "fi_ckp" if mode == 1 else "fi",
    )
