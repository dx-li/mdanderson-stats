"""Immutable, replayable input specifications for BARD BF-BOIN scenarios."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import ArrayLike

from ._validation import scalar
from .bard import bard_select_obd
from .bard_bf_boin_simulation import BARDBFBOINSimulation, simulate_bard_bf_boin
from .bard_bf_boin_trial import BARDStageTwoDesign
from .bard_response import BARDResponseModel, bard_response_model
from .bf_boin import BFBOINDesign
from .bf_boin_simulation import _validate_bard_response_truth, _weibull_endpoint

_FORMAT_VERSION = 1
_MAX_JSON_BYTES = 1_048_576
_MAX_SERIALIZED_CELLS = 100_000
_MAX_LABEL = 128
_MAX_TRIALS = 1_000_000
_MAX_PATIENT_WORK = 100_000_000
_MAX_DOSES = 100
_MAX_PROFILES = 100_000
_MAX_FACTORS = 5
_DESIGN_KEYS = {
    "target",
    "n_cap",
    "n_stop",
    "elimination_probability",
    "extra_safe",
    "safety_offset",
    "bound_mtd",
    "stay_at_one_of_three",
    "deescalate_at_two_of_six",
}
_STAGE_TWO_KEYS = {
    "total_target",
    "eligible_profiles",
    "prior",
    "safety_weights",
    "toxicity_limit",
    "efficacy_limit",
    "safety_cutoff",
    "efficacy_cutoff",
    "utilities",
    "margin",
    "tie_arm",
    "stage_two_accrual_rate",
    "dose_pair",
    "stage_two_profile_probabilities",
    "stage_two_joint_toxicity_response_probability",
    "allocation_probability",
    "tie_probability",
    "balanced_factors",
    "arrival_distribution",
}


def _real_cell(value: object) -> TypeGuard[Real]:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _number(value: object, name: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a finite real number")
        raw = float(value.item())
    elif _real_cell(value):
        raw = float(value)
    else:
        raise ValueError(f"{name} must be a finite real number")
    number = scalar(raw, name)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value: ArrayLike, name: str, maximum: int, *, minimum: int = 1) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not minimum <= value.size <= maximum or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real array")
    elif isinstance(value, (list, tuple)):
        if not minimum <= len(value) <= maximum or any(not _real_cell(item) for item in value):
            raise ValueError(f"{name} must be a bounded one-dimensional real sequence")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional real sequence")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real values")
    return tuple(float(item) for item in result)


def _matrix(
    value: ArrayLike,
    name: str,
    *,
    max_rows: int,
    max_cols: int,
    integer: bool = False,
) -> tuple[tuple[int, ...], ...] | tuple[tuple[float, ...], ...]:
    if isinstance(value, np.ndarray):
        if (
            value.ndim != 2
            or not 1 <= value.shape[0] <= max_rows
            or not 1 <= value.shape[1] <= max_cols
            or value.size > 1_000_000
            or value.dtype.kind not in ("iu" if integer else "iuf")
        ):
            raise ValueError(f"{name} must be a bounded two-dimensional real array")
        rows, cols = value.shape
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if not 1 <= rows <= max_rows:
            raise ValueError(f"{name} row count is outside its supported bounds")
        first = value[0]
        if not isinstance(first, (list, tuple, np.ndarray)) or (
            isinstance(first, np.ndarray) and (first.ndim != 1 or first.dtype.kind not in "iuf")
        ):
            raise ValueError(f"{name} must be a rectangular matrix")
        cols = len(first)
        if not 1 <= cols <= max_cols or rows * cols > 1_000_000:
            raise ValueError(f"{name} exceeds the bounded matrix dimensions")
        for row in value:
            if (
                not isinstance(row, (list, tuple, np.ndarray))
                or (isinstance(row, np.ndarray) and (row.ndim != 1 or row.dtype.kind not in "iuf"))
                or len(row) != cols
                or any(not _real_cell(item) for item in row)
            ):
                raise ValueError(f"{name} must contain only real scalar cells in a rectangle")
    else:
        raise ValueError(f"{name} must be a bounded matrix")
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real values")
    if integer:
        if np.any(array != np.floor(array)):
            raise ValueError(f"{name} must contain integer labels")
        return tuple(tuple(int(item) for item in row) for row in array)
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return tuple(tuple(float(item) for item in row) for row in result)


def _reject_unknown(mapping: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or len(mapping) > len(keys) or set(mapping) != keys:
        raise ValueError(f"{name} has unknown or missing fields")
    return mapping


def _duplicate_safe_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _atomic_write(path: str | os.PathLike[str], text: str) -> Path:
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise
    return destination


def _design_dict(design: BFBOINDesign) -> dict[str, object]:
    return {
        "target": design.target,
        "n_cap": design.n_cap,
        "n_stop": design.n_stop,
        "elimination_probability": design.elimination_probability,
        "extra_safe": design.extra_safe,
        "safety_offset": design.safety_offset,
        "bound_mtd": design.bound_mtd,
        "stay_at_one_of_three": design.stay_at_one_of_three,
        "deescalate_at_two_of_six": design.deescalate_at_two_of_six,
    }


def _stage_two_dict(stage: BARDStageTwoDesign) -> dict[str, object]:
    return {
        "total_target": stage.total_target,
        "eligible_profiles": np.asarray(stage.eligible_profiles).tolist(),
        "prior": np.asarray(stage.prior).tolist(),
        "safety_weights": np.asarray(stage.safety_weights).tolist(),
        "toxicity_limit": stage.toxicity_limit,
        "efficacy_limit": stage.efficacy_limit,
        "safety_cutoff": stage.safety_cutoff,
        "efficacy_cutoff": stage.efficacy_cutoff,
        "utilities": np.asarray(stage.utilities).tolist(),
        "margin": stage.margin,
        "tie_arm": stage.tie_arm,
        "stage_two_accrual_rate": stage.stage_two_accrual_rate,
        "dose_pair": stage.dose_pair,
        "stage_two_profile_probabilities": (
            None
            if stage.stage_two_profile_probabilities is None
            else np.asarray(stage.stage_two_profile_probabilities).tolist()
        ),
        "stage_two_joint_toxicity_response_probability": (
            None
            if stage.stage_two_joint_toxicity_response_probability is None
            else np.asarray(stage.stage_two_joint_toxicity_response_probability).tolist()
        ),
        "allocation_probability": stage.allocation_probability,
        "tie_probability": stage.tie_probability,
        "balanced_factors": (
            None if stage.balanced_factors is None else list(stage.balanced_factors)
        ),
        "arrival_distribution": stage.arrival_distribution,
    }


@dataclass(frozen=True, slots=True)
class BARDStudySpecification:
    """Complete explicit inputs for replaying one BARD BF-BOIN scenario."""

    label: str
    design: BFBOINDesign
    true_toxicity: ArrayLike
    population_response: ArrayLike
    factor_profiles: ArrayLike
    profile_probabilities: ArrayLike
    response_odds_ratios: ArrayLike
    stage_two: BARDStageTwoDesign
    true_obd_noninferiority: int
    true_obd_utility: int
    seed: int
    trials: int = 1_000
    scenario_label: str | None = None
    cohorts: int = 10
    cohort_size: int = 3
    start_dose: int = 1
    stage_one_accrual_rate: float = 1.0
    dlt_window: float = 1.0
    stage_one_arrival_distribution: str = "uniform"
    joint_toxicity_response_probability: ArrayLike | None = None
    expand_after_escalation: bool = False
    accelerated_titration: bool = False
    titration_cap: int | None = None
    true_grade2: ArrayLike | None = None
    grade2_assessment_delay: float | None = None
    max_patient_work: int = _MAX_PATIENT_WORK

    def __post_init__(self) -> None:
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > _MAX_LABEL
        ):
            raise ValueError(f"label must be nonempty and at most {_MAX_LABEL} characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in self.label):
            raise ValueError("label cannot contain control characters")
        if self.scenario_label is not None and (
            not isinstance(self.scenario_label, str)
            or not self.scenario_label.strip()
            or len(self.scenario_label) > _MAX_LABEL
            or any(ord(char) < 32 or ord(char) == 127 for char in self.scenario_label)
        ):
            raise ValueError("scenario_label must be nonempty, printable, and bounded")
        if not isinstance(self.design, BFBOINDesign):
            raise TypeError("design must be a BFBOINDesign")
        if not isinstance(self.stage_two, BARDStageTwoDesign):
            raise TypeError("stage_two must be a BARDStageTwoDesign")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an explicit nonnegative 64-bit integer")
        seed = int(self.seed)
        if not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an explicit nonnegative 64-bit integer")
        trials = _positive_integer(self.trials, "trials", _MAX_TRIALS)
        cohorts = _positive_integer(self.cohorts, "cohorts", 100_000)
        cohort_size = _positive_integer(self.cohort_size, "cohort_size", 100_000)
        start_dose = _positive_integer(self.start_dose, "start_dose", _MAX_DOSES)
        max_patient_work = _positive_integer(
            self.max_patient_work, "max_patient_work", _MAX_PATIENT_WORK
        )
        toxicity = _vector(self.true_toxicity, "true_toxicity", _MAX_DOSES, minimum=2)
        response = _vector(self.population_response, "population_response", _MAX_DOSES, minimum=2)
        if len(toxicity) != len(response) or any(not 0 <= x <= 1 for x in toxicity + response):
            raise ValueError("toxicity and population response must align and lie in [0,1]")
        profiles = _matrix(
            self.factor_profiles,
            "factor_profiles",
            max_rows=_MAX_PROFILES,
            max_cols=_MAX_FACTORS,
            integer=True,
        )
        probabilities = _vector(self.profile_probabilities, "profile_probabilities", _MAX_PROFILES)
        odds = _matrix(
            self.response_odds_ratios,
            "response_odds_ratios",
            max_rows=_MAX_FACTORS,
            max_cols=20,
        )
        model = bard_response_model(response, profiles, probabilities, odds)
        n_doses = len(toxicity)
        profile_array = np.asarray(profiles, dtype=np.int64)
        stage_two_eligible = np.asarray(self.stage_two.eligible_profiles, dtype=bool)
        if profile_array.shape[0] != stage_two_eligible.size:
            raise ValueError("stage-two eligibility must match the factor-profile rows")
        if model.factor_profiles.shape[1] != np.asarray(odds).shape[0]:
            raise ValueError("response-odds rows must match factor count")
        stage_two_weights = (
            np.asarray(model.profile_probabilities)
            if self.stage_two.stage_two_profile_probabilities is None
            else np.asarray(self.stage_two.stage_two_profile_probabilities)
        )
        if stage_two_weights.shape != (profile_array.shape[0],):
            raise ValueError("stage-two profile probabilities must match factor-profile rows")
        eligible_weights = stage_two_weights[stage_two_eligible]
        if (
            np.any(stage_two_weights < 0)
            or abs(float(stage_two_weights.sum()) - 1.0) > 1e-12
            or float(eligible_weights.sum()) <= 0
        ):
            raise ValueError("stage-two profile weights must sum to one and have eligible mass")
        stage_two_joint = self.stage_two.stage_two_joint_toxicity_response_probability
        if stage_two_joint is not None and np.asarray(stage_two_joint).shape != (
            n_doses,
            profile_array.shape[0],
        ):
            raise ValueError("stage-two joint endpoint probabilities must match doses and profiles")
        if self.stage_two.dose_pair is not None:
            pair_array = np.asarray(self.stage_two.dose_pair)
            if pair_array.shape != (2,) or int(pair_array[1]) > n_doses:
                raise ValueError("stage-two dose_pair is outside the dose grid")
        if (
            self.stage_two.balanced_factors is not None
            and max(self.stage_two.balanced_factors) >= model.factor_profiles.shape[1]
        ):
            raise ValueError("balanced_factors references a missing factor column")
        true_ni = _positive_integer(
            self.true_obd_noninferiority, "true_obd_noninferiority", n_doses
        )
        true_utility = _positive_integer(self.true_obd_utility, "true_obd_utility", n_doses)
        if start_dose > n_doses:
            raise ValueError("start_dose is outside the dose grid")
        stage_one_rate = _number(self.stage_one_accrual_rate, "stage_one_accrual_rate")
        dlt_window = _number(self.dlt_window, "dlt_window")
        if stage_one_rate <= 0 or dlt_window <= 0:
            raise ValueError("stage-one accrual rate and DLT window must be positive")
        for probability in toxicity:
            _weibull_endpoint(probability, dlt_window)
        if self.stage_one_arrival_distribution not in ("uniform", "exponential"):
            raise ValueError("stage_one_arrival_distribution must be uniform or exponential")
        for name, flag in (
            ("expand_after_escalation", self.expand_after_escalation),
            ("accelerated_titration", self.accelerated_titration),
        ):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError(f"{name} must be Boolean")
        if self.accelerated_titration:
            if self.true_grade2 is None or self.grade2_assessment_delay is None:
                raise ValueError(
                    "accelerated titration requires true_grade2 and grade2_assessment_delay"
                )
            if self.titration_cap is not None:
                cap = _positive_integer(self.titration_cap, "titration_cap", n_doses)
                if cap < start_dose:
                    raise ValueError("titration_cap cannot be below start_dose")
            grade2 = _vector(self.true_grade2, "true_grade2", _MAX_DOSES, minimum=2)
            if len(grade2) != n_doses or any(not 0 <= x <= 1 for x in grade2):
                raise ValueError("true_grade2 must align with doses and lie in [0,1]")
            delay = _number(self.grade2_assessment_delay, "grade2_assessment_delay")
            if delay <= 0:
                raise ValueError("grade2_assessment_delay must be positive")
        elif any(
            x is not None
            for x in (self.titration_cap, self.true_grade2, self.grade2_assessment_delay)
        ):
            raise ValueError(
                "grade-2 and titration-cap settings require accelerated_titration=True"
            )
        joint: tuple[tuple[float, ...], ...] | None = None
        if self.joint_toxicity_response_probability is not None:
            joint_array = _probability_matrix(
                self.joint_toxicity_response_probability, "joint_toxicity_response_probability"
            )
            if joint_array.shape != (n_doses, len(profiles)):
                raise ValueError(
                    "joint toxicity-response probabilities must match doses and profiles"
                )
            joint = tuple(tuple(float(item) for item in row) for row in joint_array)
        # Validate both endpoint joint distributions through the simulator's
        # shared marginal/Frechet contract, without sampling or allocating trials.
        if joint is not None:
            _validate_bard_response_truth(
                model,
                np.asarray(response, dtype=np.float64),
                np.asarray(toxicity, dtype=np.float64),
                np.asarray(joint, dtype=np.float64),
            )
        if stage_two_joint is not None:
            _validate_bard_response_truth(
                model,
                np.asarray(response, dtype=np.float64),
                np.asarray(toxicity, dtype=np.float64),
                np.asarray(stage_two_joint, dtype=np.float64),
            )
        for method in ("utility", "noninferiority"):
            bard_select_obd(
                np.ones((2, 4), dtype=np.int64),
                prior=np.asarray(self.stage_two.prior),
                safety_weights=np.asarray(self.stage_two.safety_weights),
                toxicity_limit=self.stage_two.toxicity_limit,
                efficacy_limit=self.stage_two.efficacy_limit,
                safety_cutoff=self.stage_two.safety_cutoff,
                efficacy_cutoff=self.stage_two.efficacy_cutoff,
                method=method,
                utilities=np.asarray(self.stage_two.utilities),
                margin=self.stage_two.margin,
                tie_arm=self.stage_two.tie_arm,
            )
        serialized_cells = (
            len(response)
            + profile_array.size
            + len(probabilities)
            + len(odds) * len(odds[0])
            + stage_two_eligible.size
            + np.asarray(self.stage_two.prior).size
            + np.asarray(self.stage_two.safety_weights).size
            + np.asarray(self.stage_two.utilities).size
        )
        if self.stage_two.stage_two_profile_probabilities is not None:
            serialized_cells += np.asarray(self.stage_two.stage_two_profile_probabilities).size
        if stage_two_joint is not None:
            serialized_cells += np.asarray(stage_two_joint).size
        if joint is not None:
            serialized_cells += np.asarray(joint).size
        if serialized_cells > _MAX_SERIALIZED_CELLS:
            raise ValueError("BARD study exceeds the 100,000-cell serialization limit")
        stage_one_bound = cohorts * cohort_size + n_doses * self.design.n_cap
        if self.accelerated_titration:
            cap = n_doses if self.titration_cap is None else int(self.titration_cap)
            stage_one_bound += cap - start_dose + 1 + cohort_size
        if stage_one_bound > 1_000:
            raise ValueError("a stage-one trial may enroll at most 1000 patients")
        patient_work = trials * (stage_one_bound + self.stage_two.total_target)
        if patient_work > max_patient_work:
            raise ValueError(
                f"patient_work_bound {patient_work} exceeds max_patient_work={max_patient_work}"
            )
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "cohorts", cohorts)
        object.__setattr__(self, "cohort_size", cohort_size)
        object.__setattr__(self, "start_dose", start_dose)
        object.__setattr__(self, "max_patient_work", max_patient_work)
        object.__setattr__(self, "true_obd_noninferiority", true_ni)
        object.__setattr__(self, "true_obd_utility", true_utility)
        object.__setattr__(self, "true_toxicity", toxicity)
        object.__setattr__(self, "population_response", response)
        object.__setattr__(self, "factor_profiles", profiles)
        object.__setattr__(self, "profile_probabilities", probabilities)
        object.__setattr__(self, "response_odds_ratios", odds)
        object.__setattr__(self, "stage_one_accrual_rate", stage_one_rate)
        object.__setattr__(self, "dlt_window", dlt_window)
        object.__setattr__(self, "joint_toxicity_response_probability", joint)
        captured_design = self.design
        object.__setattr__(
            self,
            "design",
            BFBOINDesign(
                target=captured_design.target,
                n_cap=captured_design.n_cap,
                n_stop=captured_design.n_stop,
                elimination_probability=captured_design.elimination_probability,
                extra_safe=captured_design.extra_safe,
                safety_offset=captured_design.safety_offset,
                bound_mtd=captured_design.bound_mtd,
                stay_at_one_of_three=captured_design.stay_at_one_of_three,
                deescalate_at_two_of_six=captured_design.deescalate_at_two_of_six,
            ),
        )
        object.__setattr__(self, "stage_two", _freeze_stage_two(self.stage_two))
        object.__setattr__(self, "expand_after_escalation", bool(self.expand_after_escalation))
        object.__setattr__(self, "accelerated_titration", bool(self.accelerated_titration))
        object.__setattr__(
            self,
            "titration_cap",
            None if self.titration_cap is None else int(self.titration_cap),
        )
        if self.true_grade2 is not None:
            object.__setattr__(
                self, "true_grade2", _vector(self.true_grade2, "true_grade2", _MAX_DOSES)
            )
        if self.grade2_assessment_delay is not None:
            object.__setattr__(
                self,
                "grade2_assessment_delay",
                _number(self.grade2_assessment_delay, "grade2_assessment_delay"),
            )

    @property
    def patient_work_bound(self) -> int:
        n_doses = np.asarray(self.true_toxicity).size
        stage_one = self.cohorts * self.cohort_size + n_doses * self.design.n_cap
        if self.accelerated_titration:
            cap = n_doses if self.titration_cap is None else int(self.titration_cap)
            stage_one += cap - self.start_dose + 1 + self.cohort_size
        return self.trials * (stage_one + self.stage_two.total_target)

    @property
    def response_model(self) -> BARDResponseModel:
        return bard_response_model(
            self.population_response,
            self.factor_profiles,
            self.profile_probabilities,
            self.response_odds_ratios,
        )

    def validate(self) -> BARDStudySpecification:
        """Return self after the full structural, model, and work preflight."""
        # Construction performs this preflight and stores only immutable snapshots.
        return self

    def run(self) -> BARDBFBOINSimulation:
        """Run the captured scenario with its independent explicit seed."""
        self.validate()
        return simulate_bard_bf_boin(
            self.design,
            self.true_toxicity,
            self.response_model,
            self.stage_two,
            true_obd_noninferiority=self.true_obd_noninferiority,
            true_obd_utility=self.true_obd_utility,
            trials=self.trials,
            scenario_label=self.scenario_label,
            cohorts=self.cohorts,
            cohort_size=self.cohort_size,
            start_dose=self.start_dose,
            stage_one_accrual_rate=self.stage_one_accrual_rate,
            dlt_window=self.dlt_window,
            stage_one_arrival_distribution=self.stage_one_arrival_distribution,
            joint_toxicity_response_probability=self.joint_toxicity_response_probability,
            expand_after_escalation=self.expand_after_escalation,
            accelerated_titration=self.accelerated_titration,
            titration_cap=self.titration_cap,
            true_grade2=self.true_grade2,
            grade2_assessment_delay=self.grade2_assessment_delay,
            rng=self.seed,
            max_patient_work=self.max_patient_work,
        )

    def to_dict(self) -> dict[str, object]:
        model_cells = (
            np.asarray(self.population_response).size
            + np.asarray(self.factor_profiles).size
            + np.asarray(self.profile_probabilities).size
            + np.asarray(self.response_odds_ratios).size
        )
        stage_cells = (
            np.asarray(self.stage_two.eligible_profiles).size
            + np.asarray(self.stage_two.prior).size
            + np.asarray(self.stage_two.safety_weights).size
            + np.asarray(self.stage_two.utilities).size
        )
        if self.stage_two.stage_two_profile_probabilities is not None:
            stage_cells += np.asarray(self.stage_two.stage_two_profile_probabilities).size
        if self.stage_two.stage_two_joint_toxicity_response_probability is not None:
            stage_cells += np.asarray(
                self.stage_two.stage_two_joint_toxicity_response_probability
            ).size
        if self.joint_toxicity_response_probability is not None:
            stage_cells += np.asarray(self.joint_toxicity_response_probability).size
        if model_cells + stage_cells > _MAX_SERIALIZED_CELLS:
            raise ValueError("BARD study exceeds the 100,000-cell serialization limit")
        result: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "label": self.label,
            "design": _design_dict(self.design),
            "true_toxicity": self.true_toxicity,
            "response_model": {
                "population_response": self.population_response,
                "factor_profiles": self.factor_profiles,
                "profile_probabilities": self.profile_probabilities,
                "response_odds_ratios": self.response_odds_ratios,
            },
            "stage_two": _stage_two_dict(self.stage_two),
            "true_obd_noninferiority": self.true_obd_noninferiority,
            "true_obd_utility": self.true_obd_utility,
            "seed": self.seed,
            "trials": self.trials,
            "scenario_label": self.scenario_label,
            "cohorts": self.cohorts,
            "cohort_size": self.cohort_size,
            "start_dose": self.start_dose,
            "stage_one_accrual_rate": self.stage_one_accrual_rate,
            "dlt_window": self.dlt_window,
            "stage_one_arrival_distribution": self.stage_one_arrival_distribution,
            "joint_toxicity_response_probability": self.joint_toxicity_response_probability,
            "expand_after_escalation": self.expand_after_escalation,
            "accelerated_titration": self.accelerated_titration,
            "titration_cap": self.titration_cap,
            "true_grade2": self.true_grade2,
            "grade2_assessment_delay": self.grade2_assessment_delay,
            "max_patient_work": self.max_patient_work,
        }
        text = json.dumps(result, allow_nan=False, separators=(",", ":"))
        if len(text.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("BARD study JSON exceeds the 1 MiB limit")
        return result

    def to_json(self) -> str:
        text = json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("BARD study JSON exceeds the 1 MiB limit")
        return text

    @classmethod
    def from_dict(cls, value: object) -> BARDStudySpecification:
        expected = {
            "format_version",
            "label",
            "design",
            "true_toxicity",
            "response_model",
            "stage_two",
            "true_obd_noninferiority",
            "true_obd_utility",
            "seed",
            "trials",
            "scenario_label",
            "cohorts",
            "cohort_size",
            "start_dose",
            "stage_one_accrual_rate",
            "dlt_window",
            "stage_one_arrival_distribution",
            "joint_toxicity_response_probability",
            "expand_after_escalation",
            "accelerated_titration",
            "titration_cap",
            "true_grade2",
            "grade2_assessment_delay",
            "max_patient_work",
        }
        data = _reject_unknown(value, expected, "BARD study")
        if (
            isinstance(data["format_version"], bool)
            or not isinstance(data["format_version"], int)
            or data["format_version"] != _FORMAT_VERSION
        ):
            raise ValueError("unsupported BARD study format_version")
        design_raw = _reject_unknown(data["design"], _DESIGN_KEYS, "BF-BOIN design")
        design = BFBOINDesign(
            target=design_raw["target"],
            n_cap=design_raw["n_cap"],
            n_stop=design_raw["n_stop"],
            elimination_probability=design_raw["elimination_probability"],
            extra_safe=design_raw["extra_safe"],
            safety_offset=design_raw["safety_offset"],
            bound_mtd=design_raw["bound_mtd"],
            stay_at_one_of_three=design_raw["stay_at_one_of_three"],
            deescalate_at_two_of_six=design_raw["deescalate_at_two_of_six"],
        )
        model = _reject_unknown(
            data["response_model"],
            {
                "population_response",
                "factor_profiles",
                "profile_probabilities",
                "response_odds_ratios",
            },
            "response model inputs",
        )
        stage_raw = dict(
            _reject_unknown(data["stage_two"], _STAGE_TWO_KEYS, "BARD stage-two design")
        )
        if stage_raw["balanced_factors"] is not None:
            factors = stage_raw["balanced_factors"]
            if not isinstance(factors, (list, tuple)) or len(factors) > _MAX_FACTORS:
                raise ValueError("balanced_factors must be a bounded JSON list")
            stage_raw["balanced_factors"] = tuple(factors)
        stage_two = BARDStageTwoDesign(**stage_raw)
        return cls(
            label=data["label"],
            design=design,
            true_toxicity=data["true_toxicity"],
            population_response=model["population_response"],
            factor_profiles=model["factor_profiles"],
            profile_probabilities=model["profile_probabilities"],
            response_odds_ratios=model["response_odds_ratios"],
            stage_two=stage_two,
            true_obd_noninferiority=data["true_obd_noninferiority"],
            true_obd_utility=data["true_obd_utility"],
            seed=data["seed"],
            trials=data["trials"],
            scenario_label=data["scenario_label"],
            cohorts=data["cohorts"],
            cohort_size=data["cohort_size"],
            start_dose=data["start_dose"],
            stage_one_accrual_rate=data["stage_one_accrual_rate"],
            dlt_window=data["dlt_window"],
            stage_one_arrival_distribution=data["stage_one_arrival_distribution"],
            joint_toxicity_response_probability=data["joint_toxicity_response_probability"],
            expand_after_escalation=data["expand_after_escalation"],
            accelerated_titration=data["accelerated_titration"],
            titration_cap=data["titration_cap"],
            true_grade2=data["true_grade2"],
            grade2_assessment_delay=data["grade2_assessment_delay"],
            max_patient_work=data["max_patient_work"],
        )

    @classmethod
    def from_json(cls, text: str) -> BARDStudySpecification:
        if not isinstance(text, str) or len(text) > _MAX_JSON_BYTES:
            raise ValueError("BARD study JSON must be text no larger than 1 MiB")
        if len(text.encode("utf-8")) > _MAX_JSON_BYTES:
            raise ValueError("BARD study JSON must be text no larger than 1 MiB")
        value = json.loads(
            text, object_pairs_hook=_duplicate_safe_object, parse_constant=_reject_json_constant
        )
        return cls.from_dict(value)

    def write_json(self, path: str | os.PathLike[str]) -> Path:
        return _atomic_write(path, self.to_json())

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> BARDStudySpecification:
        source = Path(path)
        with source.open("rb") as stream:
            content = stream.read(_MAX_JSON_BYTES + 1)
        if len(content) > _MAX_JSON_BYTES:
            raise ValueError("BARD study JSON file exceeds the 1 MiB limit")
        return cls.from_json(content.decode("utf-8"))


def _positive_integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} must be in 1..{maximum}")
    return result


def _probability_matrix(value: ArrayLike, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.size > 1_000_000 or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a bounded probability matrix")
    elif isinstance(value, (list, tuple)):
        rows = len(value)
        if not 1 <= rows <= _MAX_DOSES:
            raise ValueError(f"{name} row count is outside its supported bounds")
        first = value[0]
        if not isinstance(first, (list, tuple, np.ndarray)):
            raise ValueError(f"{name} must be a rectangular matrix")
        cols = len(first)
        if not 1 <= cols <= _MAX_PROFILES or rows * cols > 1_000_000:
            raise ValueError(f"{name} exceeds the bounded matrix dimensions")
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != cols:
                raise ValueError(f"{name} must be a rectangular matrix")
            if isinstance(row, np.ndarray) and (row.ndim != 1 or row.dtype.kind not in "iuf"):
                raise ValueError(f"{name} must be a real matrix")
            if any(not _real_cell(item) for item in row):
                raise ValueError(f"{name} must contain real scalar cells")
    else:
        raise ValueError(f"{name} must be a bounded probability matrix")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return result


def _freeze_stage_two(stage: BARDStageTwoDesign) -> BARDStageTwoDesign:
    clone = BARDStageTwoDesign(
        total_target=stage.total_target,
        eligible_profiles=stage.eligible_profiles,
        prior=stage.prior,
        safety_weights=stage.safety_weights,
        toxicity_limit=stage.toxicity_limit,
        efficacy_limit=stage.efficacy_limit,
        safety_cutoff=stage.safety_cutoff,
        efficacy_cutoff=stage.efficacy_cutoff,
        utilities=stage.utilities,
        margin=stage.margin,
        tie_arm=stage.tie_arm,
        stage_two_accrual_rate=stage.stage_two_accrual_rate,
        dose_pair=stage.dose_pair,
        stage_two_profile_probabilities=stage.stage_two_profile_probabilities,
        stage_two_joint_toxicity_response_probability=stage.stage_two_joint_toxicity_response_probability,
        allocation_probability=stage.allocation_probability,
        tie_probability=stage.tie_probability,
        balanced_factors=stage.balanced_factors,
        arrival_distribution=stage.arrival_distribution,
    )
    for name in (
        "eligible_profiles",
        "prior",
        "safety_weights",
        "utilities",
        "stage_two_profile_probabilities",
        "stage_two_joint_toxicity_response_probability",
    ):
        value = getattr(clone, name)
        if isinstance(value, np.ndarray):
            frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(clone, name, frozen)
    return clone
