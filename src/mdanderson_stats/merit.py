"""MERIT admissible-dose decisions and Bayesian interim monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import betainc, betaincc

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    x = scalar(value, name)
    if x != int(x) or not low <= x <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(x)


def _isotonic(values: FloatArray) -> FloatArray:
    # Equal-weight min-max characterization of PAVA; batched over trials.
    # With 2..4 doses, this avoids a Python/SciPy call for every trial.
    cumulative = np.concatenate([np.zeros((*values.shape[:-1], 1)), values.cumsum(-1)], -1)
    result = np.empty_like(values)
    doses = values.shape[-1]
    for j in range(doses):
        lower = np.full(values.shape[:-1], -np.inf)
        for start in range(j + 1):
            upper = np.full(values.shape[:-1], np.inf)
            for end in range(j, doses):
                mean = (cumulative[..., end + 1] - cumulative[..., start]) / (end - start + 1)
                upper = np.minimum(upper, mean)
            lower = np.maximum(lower, upper)
        result[..., j] = lower
    return result


def _counts(
    toxicity: ArrayLike, efficacy: ArrayLike, n: int, doses: int
) -> tuple[FloatArray, FloatArray]:
    t, e = count(toxicity, "toxicity"), count(efficacy, "efficacy")
    if t.shape != e.shape or t.ndim < 1 or t.shape[-1] != doses or t.size > 4000000:
        raise ValueError(
            "matching count arrays must have doses on the last axis and <=4 million cells"
        )
    if np.any(t > n) or np.any(e > n):
        raise ValueError("counts cannot exceed patients per arm")
    return t, e


@dataclass(frozen=True)
class MERITSelection:
    toxicity: FloatArray
    efficacy: FloatArray
    admissible: NDArray[np.bool_]


@dataclass(frozen=True)
class MERITDesign:
    patients_per_arm: int
    toxicity_max: int
    efficacy_min: int
    doses: int = 2
    isotonic_toxicity: bool = True
    isotonic_efficacy: bool = True

    def __post_init__(self) -> None:
        n = _integer(self.patients_per_arm, "patients_per_arm", 1, 500)
        _integer(self.toxicity_max, "toxicity_max", 0, n)
        _integer(self.efficacy_min, "efficacy_min", 0, n)
        _integer(self.doses, "doses", 2, 4)
        for flag in (self.isotonic_toxicity, self.isotonic_efficacy):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError("isotonic flags must be boolean")

    def select(self, toxicity: ArrayLike, efficacy: ArrayLike) -> MERITSelection:
        """Return the admissible set, not a single final optimal biological dose."""
        t, e = _counts(toxicity, efficacy, self.patients_per_arm, self.doses)
        t = _isotonic(t) if self.isotonic_toxicity else t
        e = _isotonic(e) if self.isotonic_efficacy else e
        return MERITSelection(
            _owned(t), _owned(e), _owned((t <= self.toxicity_max) & (e >= self.efficacy_min))
        )


@dataclass(frozen=True)
class MERITMonitoring:
    toxicity_probability: FloatArray
    futility_probability: FloatArray
    stop_toxicity: NDArray[np.bool_]
    stop_futility: NDArray[np.bool_]


def merit_monitor(
    patients_per_arm: int,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    *,
    toxicity_target: float,
    efficacy_target: float,
    toxicity_cutoff: float = 0.95,
    efficacy_cutoff: float = 0.95,
    prior: ArrayLike = (0.1, 0.1),
    isotonic_toxicity: bool = True,
    isotonic_efficacy: bool = True,
) -> MERITMonitoring:
    """Paper section 2.5: strict posterior stopping at an equal-size interim.

    Targets are the acceptable alternative rates. Optional isotonic pooling is
    on counts before the beta posterior calculation. No cross-look state is inferred.
    """
    n = _integer(patients_per_arm, "patients_per_arm", 1, 500)
    raw = finite(toxicity, "toxicity")
    if raw.ndim < 1:
        raise ValueError("counts require a dose axis")
    design = MERITDesign(n, n, 0, raw.shape[-1], isotonic_toxicity, isotonic_efficacy)
    transformed = design.select(toxicity, efficacy)
    rates = finite(
        [toxicity_target, efficacy_target, toxicity_cutoff, efficacy_cutoff], "thresholds"
    )
    shapes = finite(prior, "prior")
    if (
        rates.shape != (4,)
        or np.any((rates <= 0) | (rates >= 1))
        or shapes.shape != (2,)
        or np.any(shapes <= 0)
        or shapes.sum() > 1e6
    ):
        raise ValueError(
            "thresholds must be in (0,1); prior needs two positive shapes totaling <=1e6"
        )
    t, e = transformed.toxicity, transformed.efficacy
    pt = betaincc(shapes[0] + t, shapes[1] + n - t, toxicity_target)
    pe = betainc(shapes[0] + e, shapes[1] + n - e, efficacy_target)
    return MERITMonitoring(
        _owned(pt), _owned(pe), _owned(pt > toxicity_cutoff), _owned(pe > efficacy_cutoff)
    )
