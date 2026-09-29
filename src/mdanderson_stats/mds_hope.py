"""MDS-HOPE published Eq. S1 relative-risk score and explicit risk grouping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray

_CONTRIBUTION_NAMES = (
    "age_years",
    "anc_10e9_l",
    "hemoglobin_g_dl",
    "platelets_10e9_l",
    "marrow_blast_percent",
    "cytogenetic_risk_score",
    "sf3b1_mutation",
    "ezh2_mutation",
    "tp53_hit_count",
    "kras_mutation",
    "ptpn11_mutation",
)
_COEFFICIENTS = np.asarray(
    [0.025, 0.067, -0.160, -0.0021, 0.051, 0.284, -0.294, 0.347, 0.345, 0.681, 1.094],
    dtype=float,
)
_RISK_GROUP_LABELS = (
    "very low",
    "low",
    "intermediate low",
    "intermediate high",
    "high",
    "very high",
)
_RISK_CUTOFFS = np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5])
_MAX_BATCH_PATIENTS = 250_000


@dataclass(frozen=True)
class MDSHopeCovariates:
    """Inputs for Eq. S1; labs use raw publisher units and cytogenetics is pre-encoded.

    ``cytogenetic_risk_score`` must already be a numeric score. The supplement
    does not state the five-category numeric mapping. Gene indicators are 0/1;
    ``tp53_hit_count`` is 0 (wild type), 1 (single hit), or 2 (multi-hit).
    """

    age_years: ArrayLike
    anc_10e9_l: ArrayLike
    hemoglobin_g_dl: ArrayLike
    platelets_10e9_l: ArrayLike
    marrow_blast_percent: ArrayLike
    cytogenetic_risk_score: ArrayLike
    sf3b1_mutation: ArrayLike
    ezh2_mutation: ArrayLike
    tp53_hit_count: ArrayLike
    kras_mutation: ArrayLike
    ptpn11_mutation: ArrayLike


@dataclass(frozen=True)
class MDSHopeScore:
    """Immutable raw Cox linear predictors and their eleven additive terms."""

    linear_predictor: FloatArray
    contributions: FloatArray
    contribution_names: tuple[str, ...]
    log_hazard_ratio: FloatArray | None = None
    relative_hazard: FloatArray | None = None


@dataclass(frozen=True)
class MDSHopeRiskClassification:
    """Six-group classification using explicit standardization supplied by caller."""

    standardized_score: FloatArray
    group_code: NDArray[np.int8]
    group_labels: tuple[str, ...]


def _numeric_vector(value: ArrayLike, name: str, *, allow_bool: bool = False) -> FloatArray:
    raw = np.asarray(value)
    valid_kind = "iufb" if allow_bool else "iuf"
    if raw.ndim > 1 or raw.size == 0 or raw.dtype.kind not in valid_kind:
        raise ValueError(f"{name} must be a finite real scalar or one-dimensional array")
    with np.errstate(over="ignore", invalid="ignore"):
        numeric = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    return numeric.reshape(1) if numeric.ndim == 0 else numeric


def _preflight_patient_count(values: tuple[tuple[str, ArrayLike], ...]) -> int:
    lengths: set[int] = set()
    for name, value in values:
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                raise ValueError(f"{name} must be a finite real scalar or one-dimensional array")
            length = int(value.size) if value.ndim == 1 else 1
        elif isinstance(value, (list, tuple)):
            if len(value) > _MAX_BATCH_PATIENTS:
                raise ValueError(f"batch exceeds {_MAX_BATCH_PATIENTS} patients")
            if any(isinstance(item, (list, tuple, np.ndarray)) for item in value):
                raise ValueError(f"{name} must be one-dimensional")
            length = len(value)
        else:
            array = np.asarray(value)
            if array.ndim > 1:
                raise ValueError(f"{name} must be a finite real scalar or one-dimensional array")
            length = int(array.size) if array.ndim == 1 else 1
        if length == 0:
            raise ValueError(f"{name} must not be empty")
        if length > _MAX_BATCH_PATIENTS:
            raise ValueError(f"batch exceeds {_MAX_BATCH_PATIENTS} patients")
        if length > 1:
            lengths.add(length)
    if len(lengths) > 1:
        raise ValueError("non-scalar covariate arrays must have the same length")
    return next(iter(lengths), 1)


def _covariate_matrix(covariates: MDSHopeCovariates) -> FloatArray:
    if not isinstance(covariates, MDSHopeCovariates):
        raise ValueError("covariates must be an MDSHopeCovariates instance")
    names = _CONTRIBUTION_NAMES
    raw_values = tuple((name, getattr(covariates, name)) for name in names)
    _preflight_patient_count(raw_values)
    values = [
        _numeric_vector(
            getattr(covariates, name),
            name,
            allow_bool=name
            in {"sf3b1_mutation", "ezh2_mutation", "kras_mutation", "ptpn11_mutation"},
        )
        for name in names
    ]
    lengths = {value.size for value in values if value.size != 1}
    if len(lengths) > 1:
        raise ValueError("non-scalar covariate arrays must have the same length")
    patients = next(iter(lengths), 1)
    matrix = np.column_stack(
        [np.full(patients, float(value[0])) if value.size == 1 else value for value in values]
    )
    age, anc, hemoglobin, platelets, blasts = matrix[:, :5].T
    if np.any(age < 0):
        raise ValueError("age_years must be nonnegative")
    if np.any(anc < 0):
        raise ValueError("anc_10e9_l must be nonnegative")
    if np.any(hemoglobin < 0):
        raise ValueError("hemoglobin_g_dl must be nonnegative")
    if np.any(platelets < 0):
        raise ValueError("platelets_10e9_l must be nonnegative")
    if np.any((blasts < 0) | (blasts > 100)):
        raise ValueError("marrow_blast_percent must be in [0,100]")
    for column, name in zip(
        (6, 7, 9, 10),
        ("sf3b1_mutation", "ezh2_mutation", "kras_mutation", "ptpn11_mutation"),
        strict=True,
    ):
        if np.any((matrix[:, column] != 0) & (matrix[:, column] != 1)):
            raise ValueError(f"{name} must contain only 0 or 1")
    if np.any(~np.isin(matrix[:, 8], (0, 1, 2))) or np.any(matrix[:, 8] != np.floor(matrix[:, 8])):
        raise ValueError("tp53_hit_count must contain only 0, 1, or 2")
    return matrix


def mds_hope_score(
    covariates: MDSHopeCovariates,
    *,
    reference: MDSHopeCovariates | None = None,
) -> MDSHopeScore:
    """Calculate Eq. S1 raw eta and contributions; optionally compare with one profile.

    Laboratory inputs are raw ANC (10^9/L), hemoglobin (g/dL), platelets
    (10^9/L), marrow blasts (percent points), and age (years). The cytogenetic
    predictor is an explicitly encoded numeric score. No missing-value
    imputation, category mapping, centering, or baseline-survival estimate is
    applied. Batches are limited to 250,000 patients before score matrices are
    allocated. A supplied reference returns log-HR from raw covariate
    differences; relative hazard is exponentiated only if it does not overflow.
    Underflow to zero is allowed while the finite log-HR remains available.
    """
    matrix = _covariate_matrix(covariates)
    # Raw-unit signs follow Eq. S1 after substituting neutropenia=-ANC,
    # anemia=-hemoglobin, and thrombocytopenia=-platelets/10.
    with np.errstate(over="ignore", invalid="ignore"):
        contributions = matrix * _COEFFICIENTS
        predictor = np.sum(contributions, axis=1)
    if not np.all(np.isfinite(predictor)) or not np.all(np.isfinite(contributions)):
        raise ArithmeticError("MDS-HOPE linear predictor is not representable")
    log_hazard_ratio: FloatArray | None = None
    relative_hazard: FloatArray | None = None
    if reference is not None:
        reference_values = _covariate_matrix(reference)
        if reference_values.shape[0] != 1:
            raise ValueError("reference must describe exactly one patient profile")
        with np.errstate(over="ignore", invalid="ignore"):
            difference = matrix - reference_values
            difference_contributions = difference * _COEFFICIENTS
            log_hazard_ratio = np.sum(difference_contributions, axis=1)
        if not np.all(np.isfinite(log_hazard_ratio)):
            raise ArithmeticError("log hazard ratio is not representable")
        maximum_log = np.log(np.finfo(float).max)
        if np.any(log_hazard_ratio > maximum_log):
            raise ArithmeticError("relative hazard overflows floating-point range")
        with np.errstate(under="ignore"):
            hazard_ratio = np.exp(log_hazard_ratio)
        if np.any(~np.isfinite(hazard_ratio)):
            raise ArithmeticError("relative hazard is not representable")
        log_hazard_ratio = _freeze(log_hazard_ratio)
        relative_hazard = _freeze(hazard_ratio)
    return MDSHopeScore(
        _freeze(predictor),
        _freeze(contributions),
        _CONTRIBUTION_NAMES,
        log_hazard_ratio,
        relative_hazard,
    )


def _standardized_vector(value: ArrayLike, name: str) -> FloatArray:
    return _numeric_vector(value, name)


def mds_hope_risk_groups(standardized_score: ArrayLike) -> NDArray[np.int8]:
    """Assign published six-group cutoffs to caller-standardized scores.

    Codes map to ``(very low, low, intermediate low, intermediate high, high,
    very high)``. Cutoffs are inclusive at -1.5, -0.5, 0, 0.5, and 1.5 on the
    lower group, matching the supplement's intervals. The input must already
    be standardized with externally supplied calibration parameters.
    """
    score = _standardized_vector(standardized_score, "standardized_score")
    codes = np.searchsorted(_RISK_CUTOFFS, score, side="left").astype(np.int8)
    return np.frombuffer(codes.tobytes(), dtype=np.int8).reshape(codes.shape)


def mds_hope_standardized_risk_groups(
    raw_score: ArrayLike,
    *,
    reference_center: float,
    reference_sd: float,
) -> MDSHopeRiskClassification:
    """Standardize raw eta with explicit external center/SD, then apply Eq. S1 groups.

    The supplement gives cutoffs but not its training-score mean or SD. These
    values must come from an explicit reference supplied by the caller; the
    prediction cohort is never used to infer them.
    """
    score = _standardized_vector(raw_score, "raw_score")
    center = _numeric_vector(reference_center, "reference_center")
    scale = _numeric_vector(reference_sd, "reference_sd")
    if center.size != 1 or scale.size != 1 or not np.isfinite(scale[0]) or scale[0] <= 0:
        raise ValueError("reference_center must be scalar and reference_sd must be positive scalar")
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = (score - center[0]) / scale[0]
    if not np.all(np.isfinite(standardized)):
        raise ArithmeticError("standardized MDS-HOPE score is not representable")
    return MDSHopeRiskClassification(
        _freeze(standardized),
        mds_hope_risk_groups(standardized),
        _RISK_GROUP_LABELS,
    )
