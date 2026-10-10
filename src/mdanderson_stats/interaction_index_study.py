"""Bounded source-defined three-drug simulation study for interaction indices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from scipy.stats import norm, t

from ._validation import FloatArray
from .boin import _owned
from .interaction_index_pooled import interaction_index_pooled_error
from .median_effect import MedianEffectFit, fit_median_effect
from .multc_study import _atomic_write

_TRUE_MEDIAN_DOSES = (1.0, 2.0, 4.0)
_DOSE_COUNT = 6
_OBSERVATIONS_PER_TRIAL = 19
_MAX_VECTOR = 25
_MAX_TOTAL_REPLICATES = 50_000
_TRUE_COEFFICIENTS = (
    (0.0, -1.0),
    (float(np.log(2.0)), -1.0),
    (float(np.log(4.0)), -1.0),
)

type FloatPair = tuple[float, float]


@dataclass(frozen=True)
class InteractionIndexStudyCell:
    """Operating characteristics for one true index and transformed-error SD."""

    true_interaction_index: float
    true_combination_effect: float
    error_sd: float
    mean_estimated_index: float
    raw_ci_coverage: float
    log_ci_coverage: float
    mean_raw_ci_length: float
    mean_log_ci_length_on_index_scale: float
    fraction_log_ci_below_one: float
    fraction_log_ci_contains_one: float
    fraction_log_ci_above_one: float


@dataclass(frozen=True)
class InteractionIndexStudySummary:
    """Immutable inputs and streamed summaries from Scenario 1."""

    interaction_indices: tuple[float, ...]
    error_sd: tuple[float, ...]
    replicates: int
    seed: int
    confidence: float
    error_distribution: str
    error_scale: str
    median_doses: tuple[float, float, float]
    true_coefficients: tuple[FloatPair, FloatPair, FloatPair]
    single_agent_doses: tuple[tuple[float, ...], ...]
    combination_doses: tuple[float, float, float]
    observations_per_trial: int
    cells: tuple[InteractionIndexStudyCell, ...]
    estimated_indices: FloatArray | None = field(default=None, compare=False, repr=False)

    @property
    def simulated_trials(self) -> int:
        return len(self.interaction_indices) * len(self.error_sd) * self.replicates

    @property
    def cell_count(self) -> int:
        return len(self.interaction_indices) * len(self.error_sd)

    def to_json(self) -> str:
        """Capture settings, summaries and optionally the bounded QQ samples."""
        value = {
            key: item
            for key, item in self.__dict__.items()
            if key not in ("cells", "estimated_indices")
        }
        value["format_version"] = 1
        value["cells"] = [asdict(cell) for cell in self.cells]
        value["estimated_indices"] = (
            None if self.estimated_indices is None else self.estimated_indices.tolist()
        )
        return json.dumps(value, indent=2, allow_nan=False) + "\n"

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_json())

    def plot_qq(self, cell: int = 0) -> Any:
        """Normal QQ panels for the index and log index, as in original Scenario 1."""
        if self.estimated_indices is None:
            raise ValueError("QQ plots require retain_samples=True when running the study")
        if isinstance(cell, (bool, np.bool_)) or not isinstance(cell, (int, np.integer)):
            raise ValueError("cell must be an integer within the study cells")
        if not 0 <= cell < len(self.cells):
            raise ValueError("cell must be an integer within the study cells")
        import matplotlib.pyplot as plt

        ordered = np.sort(self.estimated_indices[cell])
        n = ordered.size
        # R/S-Plus qqnorm uses ppoints, rather than scipy.probplot's Filliben positions.
        a = 0.375 if n <= 10 else 0.5
        quantiles = norm.ppf((np.arange(1, n + 1) - a) / (n + 1 - 2 * a))
        figure, axes = plt.subplots(1, 2)
        for axis, values, label in zip(
            axes,
            (ordered, np.log(ordered)),
            ("Interaction index", "Log interaction index"),
            strict=True,
        ):
            axis.scatter(quantiles, values)
            axis.set(xlabel="Normal quantile", ylabel=label)
        settings = self.cells[cell]
        figure.suptitle(
            f"True index {settings.true_interaction_index:g}; error SD {settings.error_sd:g}"
        )
        figure.tight_layout()
        return figure


def _bounded_numeric_tuple(value: ArrayLike, name: str, *, maximum: int) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not 1 <= value.size <= maximum:
            raise ValueError(f"{name} must be a one-dimensional vector of 1..{maximum} values")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain finite real values")
        if value.dtype.kind == "b":
            raise ValueError(f"{name} must contain real numeric values, not booleans")
        raw = tuple(value)
    elif isinstance(value, (list, tuple)):
        if not 1 <= len(value) <= maximum:
            raise ValueError(f"{name} must contain 1..{maximum} values")
        if any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must contain scalar values")
        if any(isinstance(item, (bool, np.bool_)) for item in value):
            raise ValueError(f"{name} must contain real numeric values, not booleans")
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain finite real values")
        raw = tuple(value)
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional array, list, or tuple")
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not all(np.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite real values")
    return values


def _validate_seed(rng: int | None) -> int:
    if rng is not None and (
        isinstance(rng, (bool, np.bool_))
        or not isinstance(rng, (int, np.integer))
        or int(rng) < 0
        or int(rng).bit_length() > 64
    ):
        raise ValueError("rng must be None or a nonnegative integer seed no larger than uint64")
    if rng is not None:
        return int(rng)
    sequence = np.random.SeedSequence()
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _update_mean(current: float, value: float, count: int) -> float:
    updated = current + (value - current) / count
    if not np.isfinite(updated):
        raise ArithmeticError("summary mean is not representable")
    return updated


def simulate_interaction_index_three_drug_study(
    *,
    interaction_indices: ArrayLike = (0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.67, 2.5, 5.0),
    error_sd: ArrayLike = (0.1, 0.4),
    replicates: int = 1000,
    rng: int | None = None,
    retain_samples: bool = False,
) -> InteractionIndexStudySummary:
    """Run the source-defined three-drug, single-combination-dose study.

    Independent normal errors are generated on the logit-effect scale for the
    six observations per drug and the one fixed-combination observation. The
    source's Section 2 pooled-error interval is applied to each generated
    dataset. Trials are streamed serially; total candidate-index × error-SD ×
    replicate cells are capped at 50,000. A generated effect that rounds to 0
    or 1, or any unrepresentable fit/interval, raises with its cell and
    replicate context. No responses are clipped, retried, or discarded.

    ``rng`` is an integer seed or ``None``. The returned ``seed`` records the
    uint64 seed used by the generator, including when it was generated automatically.
    The printed source candidate 1.67 is used literally; pass ``5/3`` if the
    unrounded value that implies the printed effect 0.625 is desired.
    """
    tau_values = _bounded_numeric_tuple(
        interaction_indices, "interaction_indices", maximum=_MAX_VECTOR
    )
    sd_values = _bounded_numeric_tuple(error_sd, "error_sd", maximum=_MAX_VECTOR)
    if any(not 1e-6 <= value <= 1e6 for value in tau_values) or len(set(tau_values)) != len(
        tau_values
    ):
        raise ValueError("interaction indices must be unique and in [1e-6, 1e6]")
    if any(not 0 < value <= 5 for value in sd_values) or len(set(sd_values)) != len(sd_values):
        raise ValueError("error_sd values must be unique and in (0, 5]")
    if (
        isinstance(replicates, (bool, np.bool_))
        or not isinstance(replicates, (int, np.integer))
        or not 1 <= int(replicates) <= _MAX_TOTAL_REPLICATES
    ):
        raise ValueError(f"replicates must be an integer in [1, {_MAX_TOTAL_REPLICATES}]")
    repetitions = int(replicates)
    total_trials = len(tau_values) * len(sd_values) * repetitions
    if total_trials > _MAX_TOTAL_REPLICATES:
        raise ValueError(
            f"total study cells ({total_trials}) exceed the {_MAX_TOTAL_REPLICATES} replicate cap"
        )
    if not isinstance(retain_samples, (bool, np.bool_)):
        raise ValueError("retain_samples must be boolean")
    estimates = (
        np.empty((len(tau_values) * len(sd_values), repetitions)) if retain_samples else None
    )
    seed = _validate_seed(rng)
    generator = np.random.default_rng(seed)

    dose_grids: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] = (
        tuple(float(value) for value in np.linspace(0.1, 3.0 * _TRUE_MEDIAN_DOSES[0], _DOSE_COUNT)),
        tuple(float(value) for value in np.linspace(0.1, 3.0 * _TRUE_MEDIAN_DOSES[1], _DOSE_COUNT)),
        tuple(float(value) for value in np.linspace(0.1, 3.0 * _TRUE_MEDIAN_DOSES[2], _DOSE_COUNT)),
    )
    combination_doses: tuple[float, float, float] = (
        _TRUE_MEDIAN_DOSES[0] / 3.0,
        _TRUE_MEDIAN_DOSES[1] / 3.0,
        _TRUE_MEDIAN_DOSES[2] / 3.0,
    )
    single_dose_arrays = tuple(np.asarray(grid, dtype=float) for grid in dose_grids)
    combo_dose_array = np.asarray(combination_doses, dtype=float)
    true_log_medians = np.log(np.asarray(_TRUE_MEDIAN_DOSES))
    confidence = 0.95
    cells: list[InteractionIndexStudyCell] = []

    for tau in tau_values:
        true_log_index = float(np.log(tau))
        for sigma in sd_values:
            means = np.zeros(3, dtype=float)
            coverage = np.zeros(2, dtype=np.int64)
            classification = np.zeros(3, dtype=np.int64)
            for replicate in range(1, repetitions + 1):
                errors = generator.normal(0.0, sigma, size=_OBSERVATIONS_PER_TRIAL)
                observed_curves: list[np.ndarray] = []
                try:
                    for drug, doses in enumerate(single_dose_arrays):
                        fitted_logit = (
                            true_log_medians[drug]
                            - np.log(doses)
                            + errors[drug * _DOSE_COUNT : (drug + 1) * _DOSE_COUNT]
                        )
                        observed_curves.append(expit(fitted_logit))
                    combo_logit = true_log_index + errors[-1]
                    combo_effect = float(expit(combo_logit))
                    models: tuple[MedianEffectFit, ...] = tuple(
                        fit_median_effect(doses, effects)
                        for doses, effects in zip(single_dose_arrays, observed_curves, strict=True)
                    )
                    result = interaction_index_pooled_error(
                        models, combo_dose_array, combo_effect, confidence=confidence
                    )
                    estimate = float(result.index)
                    log_se = float(result.log_standard_error)
                    log_limits = np.asarray(result.log_interval, dtype=float)
                    index_limits = np.asarray(result.interval, dtype=float)
                    critical = float(t.isf((1.0 - confidence) / 2.0, result.degrees_of_freedom))
                    raw_se = estimate * log_se
                    raw_lower, raw_upper = (
                        estimate - critical * raw_se,
                        estimate + critical * raw_se,
                    )
                    raw_length = raw_upper - raw_lower
                    log_length = float(index_limits[1] - index_limits[0])
                    if not all(
                        np.isfinite(value)
                        for value in (
                            estimate,
                            raw_se,
                            raw_lower,
                            raw_upper,
                            raw_length,
                            log_length,
                        )
                    ):
                        raise ArithmeticError(
                            "raw or exponentiated log interval is not representable"
                        )
                except (ValueError, ArithmeticError, FloatingPointError, OverflowError) as exc:
                    raise RuntimeError(
                        f"study failed at interaction_index={tau:g}, error_sd={sigma:g}, "
                        f"replicate={replicate}/{repetitions}: {exc}"
                    ) from exc

                if estimates is not None:
                    estimates[len(cells), replicate - 1] = estimate
                means[0] = _update_mean(means[0], estimate, replicate)
                means[1] = _update_mean(means[1], raw_length, replicate)
                means[2] = _update_mean(means[2], log_length, replicate)
                coverage[0] += int(raw_lower <= tau <= raw_upper)
                coverage[1] += int(log_limits[0] <= true_log_index <= log_limits[1])
                classification[0] += int(log_limits[1] < 0.0)
                classification[1] += int(log_limits[0] <= 0.0 <= log_limits[1])
                classification[2] += int(log_limits[0] > 0.0)

            cells.append(
                InteractionIndexStudyCell(
                    tau,
                    tau / (1.0 + tau),
                    sigma,
                    float(means[0]),
                    float(coverage[0] / repetitions),
                    float(coverage[1] / repetitions),
                    float(means[1]),
                    float(means[2]),
                    float(classification[0] / repetitions),
                    float(classification[1] / repetitions),
                    float(classification[2] / repetitions),
                )
            )

    return InteractionIndexStudySummary(
        tau_values,
        sd_values,
        repetitions,
        seed,
        confidence,
        "normal",
        "logit_effect",
        _TRUE_MEDIAN_DOSES,
        _TRUE_COEFFICIENTS,
        dose_grids,
        combination_doses,
        _OBSERVATIONS_PER_TRIAL,
        tuple(cells),
        None if estimates is None else _owned(estimates),
    )
