"""Recovered Lee–Kong Scenario 2: fixed-ratio dose-response comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from ._validation import FloatArray
from .boin import _owned
from .interaction_index import InteractionIndex, interaction_index_ray
from .interaction_index_pooled import interaction_index_pooled_error
from .interaction_index_study import _bounded_numeric_tuple, _validate_seed
from .interaction_monte_carlo import InteractionMonteCarlo, interaction_index_monte_carlo
from .median_effect import MedianEffectFit, fit_median_effect
from .multc_study import _atomic_write

INTERACTION_INDEX_SOURCE_SCENARIOS = (0.2, 0.4, 0.6, 0.8, 1.0, 1 / 0.8, 1 / 0.6, 1 / 0.4, 1 / 0.2)
_SOURCE_DOSES = (
    tuple(map(float, np.linspace(0.1, 3, 5))),
    tuple(map(float, np.linspace(0.1, 6, 5))),
    tuple(map(float, np.linspace(0.5, 4.5, 5))),
)
_SOURCE_COEFFICIENTS = ((0.0, -1.0), (float(np.log(2)), -1.0), (float(2 * np.log(1.5)), -2.0))
_SOURCE_EFFECTS = tuple(float(0.1 + 0.02 * i) for i in range(43))


def _integer(value: object, name: str, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim or raw.dtype.kind not in "iuf" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a positive integer")
    number = float(raw)
    if not 1 <= number <= maximum or number != int(number):
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return int(number)


@dataclass(frozen=True)
class InteractionFixedRayReplicate:
    """One generated dataset and the three source-defined pointwise comparisons."""

    error_sd: float
    replicate: int
    seed: int
    observed_effects: FloatArray
    models: tuple[MedianEffectFit, MedianEffectFit, MedianEffectFit]
    delta: InteractionIndex
    monte_carlo: InteractionMonteCarlo
    source_monte_carlo_interval: FloatArray
    observed: InteractionIndex
    observed_monte_carlo_interval: FloatArray
    observed_length_ratio: FloatArray


@dataclass(frozen=True)
class InteractionFixedRayStudy:
    """Immutable source controls and replayable Python datasets/results."""

    seed: int
    error_sd: tuple[float, ...]
    replicates: int
    samples: int
    confidence: float
    effects: tuple[float, ...]
    dose_ratio: float
    dose_grids: tuple[tuple[float, ...], ...]
    true_coefficients: tuple[tuple[float, float], ...]
    true_index: FloatArray
    results: tuple[InteractionFixedRayReplicate, ...]

    def to_json(self) -> str:
        """Captured inputs plus numerical results, with no executable objects."""
        value: dict[str, Any] = {
            "format_version": 1,
            "seed": self.seed,
            "error_sd": self.error_sd,
            "replicates": self.replicates,
            "samples": self.samples,
            "confidence": self.confidence,
            "effects": self.effects,
            "dose_ratio": self.dose_ratio,
            "dose_grids": self.dose_grids,
            "true_coefficients": self.true_coefficients,
            "true_index": self.true_index.tolist(),
            "source_monte_carlo_lower_floor": 0.0001,
            "results": [],
        }
        for result in self.results:
            value["results"].append(
                {
                    "error_sd": result.error_sd,
                    "replicate": result.replicate,
                    "seed": result.seed,
                    "observed_effects": result.observed_effects.tolist(),
                    "delta_index": result.delta.index.tolist(),
                    "delta_interval": result.delta.interval.tolist(),
                    "monte_carlo_standard_error": result.monte_carlo.standard_error.tolist(),
                    "monte_carlo_raw_interval": result.monte_carlo.interval.tolist(),
                    "source_monte_carlo_interval": result.source_monte_carlo_interval.tolist(),
                    "slope_reversal_fraction": result.monte_carlo.slope_reversal_fraction.tolist(),
                    "observed_index": result.observed.index.tolist(),
                    "observed_interval": result.observed.interval.tolist(),
                    "observed_monte_carlo_interval": result.observed_monte_carlo_interval.tolist(),
                    "observed_length_ratio": result.observed_length_ratio.tolist(),
                }
            )
        return json.dumps(value, indent=2, allow_nan=False) + "\n"

    def write_json(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_json())

    def plot(self, result: int = 0) -> Any:
        """Source-style comparison panel; ordinary Figure.savefig exports it."""
        raw_index = np.asarray(result)
        if (
            raw_index.ndim
            or raw_index.dtype.kind not in "iu"
            or not 0 <= result < len(self.results)
        ):
            raise ValueError("result index must be an integer within the recorded results")
        index = int(result)
        row = self.results[index]
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        axis.plot(self.effects, self.true_index, label="True index")
        for interval, style, name in (
            (row.delta.interval, "--", "Log delta"),
            (row.source_monte_carlo_interval[: len(self.effects)], ":", "Normal coefficient MC"),
        ):
            axis.plot(self.effects, interval[:, 0], style, label=name)
            axis.plot(self.effects, interval[:, 1], style)
        observed = row.observed.interval
        effects = row.observed_effects[2]
        axis.vlines(effects, observed[:, 0], observed[:, 1], label="Observed-combination log delta")
        axis.scatter(effects, row.observed.index)
        axis.axhline(1, color="gray", linestyle="-.")
        axis.set(
            xlabel="Effect",
            ylabel="Interaction index",
            yscale="log",
            title=f"Error SD {row.error_sd:g}; replicate {row.replicate}",
        )
        axis.legend()
        return figure


def simulate_interaction_index_fixed_ray_study(
    *,
    error_sd: tuple[float, ...] = (0.2, 0.4),
    replicates: int = 7,
    samples: int = 500,
    confidence: float = 0.95,
    effects: tuple[float, ...] = _SOURCE_EFFECTS,
    rng: int | None = 399,
    max_total_coefficient_draws: int = 2_000_000,
    max_total_work: int = 50_000_000,
    max_storage_bytes: int = 64_000_000,
) -> InteractionFixedRayStudy:
    """Run the recovered Scenario 2, with ratio d2/d1=2 and total mixture doses.

    Normal errors are on logit effects. Each dataset fits three separate curves,
    compares fixed-ray log-delta/normal-coefficient intervals over the grid plus
    observed mixture effects, and reports observed-to-MC interval-length ratios.
    Native reporting floors the MC lower endpoint at .0001; raw intervals are
    retained separately. PCG64 streams differ from S-Plus/R. Failures are explicit,
    never dropped or retried. Aggregate limits are checked before RNG creation.
    """
    sigmas = _bounded_numeric_tuple(error_sd, "error_sd", maximum=25)
    if any(not 0 < v <= 5 for v in sigmas):
        raise ValueError("error_sd must lie in (0,5]")
    effect_values = _bounded_numeric_tuple(effects, "effects", maximum=1000)
    if any(not 0 < v < 1 for v in effect_values) or any(
        b <= a for a, b in zip(effect_values, effect_values[1:])
    ):
        raise ValueError("effects must be strictly increasing and lie in (0,1)")
    count = _integer(replicates, "replicates", 100)
    draws = _integer(samples, "samples", 100_000)
    if draws < 2:
        raise ValueError("samples must be at least two")
    conf = np.asarray(confidence)
    if conf.ndim or conf.dtype.kind not in "iuf" or not 0 < float(conf) < 1:
        raise ValueError("confidence must be a finite probability strictly between zero and one")
    level = float(conf)
    coefficient_budget = _integer(
        max_total_coefficient_draws, "max_total_coefficient_draws", 20_000_000
    )
    work_budget = _integer(max_total_work, "max_total_work", 1_000_000_000)
    storage_budget = _integer(max_storage_bytes, "max_storage_bytes", 1_000_000_000)
    total = len(sigmas) * count
    coefficient_cells = total * draws * 6
    if coefficient_cells > coefficient_budget:
        raise ValueError("study exceeds max_total_coefficient_draws")
    n_effect = len(effect_values) + 5
    if total * draws * n_effect * 3 > work_budget:
        raise ValueError("study exceeds max_total_work")
    if (
        coefficient_cells * 8 + total * n_effect * 160 + draws * 32 * 3 * 8 + 131072
        > storage_budget
    ):
        raise ValueError("study exceeds max_storage_bytes")
    seed = _validate_seed(rng)
    dose_arrays = [np.array(d) for d in _SOURCE_DOSES]
    truths = tuple(MedianEffectFit(a, b, np.zeros((2, 2)), 5, 0) for a, b in _SOURCE_COEFFICIENTS)
    truth = interaction_index_ray(truths[:2], truths[2], (1, 2), effect_values).index
    generator = np.random.Generator(np.random.PCG64(seed))
    child_seeds = generator.integers(0, 2**64 - 1, size=total, dtype=np.uint64, endpoint=True)
    results = []
    for cell, sigma in enumerate(sigmas):
        for replicate in range(count):
            child = int(child_seeds[cell * count + replicate])
            stream = np.random.Generator(np.random.PCG64(child))
            observed = np.stack(
                [
                    expit(a + b * np.log(d) + stream.normal(0, sigma, len(d)))
                    for d, (a, b) in zip(dose_arrays, _SOURCE_COEFFICIENTS, strict=True)
                ]
            )
            try:
                fits = tuple(
                    fit_median_effect(d, y) for d, y in zip(dose_arrays, observed, strict=True)
                )
                delta = interaction_index_ray(
                    fits[:2], fits[2], (1, 2), effect_values, confidence=level
                )
                grid = np.r_[effect_values, observed[2]]
                mc = interaction_index_monte_carlo(
                    fits[:2], fits[2], (1, 2), grid, samples=draws, confidence=level, rng=stream
                )
                reported = mc.interval.copy()
                reported[:, 0] = np.maximum(reported[:, 0], 0.0001)
                mixture = dose_arrays[2][:, None] * np.array([1 / 3, 2 / 3])[None, :]
                known = interaction_index_pooled_error(
                    fits[:2], mixture, observed[2], confidence=level
                )
                reported_observed = reported[-5:]
                lengths = reported_observed[:, 1] - reported_observed[:, 0]
                if np.any(lengths <= 0):
                    raise ArithmeticError(
                        "native MC lower-floor reporting has nonpositive interval length"
                    )
                ratio = np.diff(known.interval, axis=1)[:, 0] / lengths
                if not np.isfinite(ratio).all():
                    raise ArithmeticError("interval-length ratio is not representable")
            except (ValueError, ArithmeticError, FloatingPointError, OverflowError) as error:
                raise RuntimeError(
                    f"fixed-ray study failed at error_sd={sigma:g}, "
                    f"replicate={replicate + 1}: {error}"
                ) from error
            results.append(
                InteractionFixedRayReplicate(
                    sigma,
                    replicate + 1,
                    child,
                    _owned(observed),
                    (fits[0], fits[1], fits[2]),
                    delta,
                    mc,
                    _owned(reported),
                    known,
                    _owned(reported_observed),
                    _owned(ratio),
                )
            )
    return InteractionFixedRayStudy(
        seed,
        sigmas,
        count,
        draws,
        level,
        effect_values,
        2.0,
        _SOURCE_DOSES,
        _SOURCE_COEFFICIENTS,
        truth,
        tuple(results),
    )
