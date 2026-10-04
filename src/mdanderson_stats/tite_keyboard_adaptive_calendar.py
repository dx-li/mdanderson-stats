"""Settings and compact per-step diagnostics for adaptive calendar TITE-Keyboard."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from ._tite_calendar import CalendarStep, CalendarTrial
from .tite_keyboard import TITEKeyboardDecision

_MAX_FIT_WORK = 50_000_000
_MAX_TRIAL_WORK = 1_000_000_000
_MAX_CHAINS = 8
_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000


@dataclass(frozen=True)
class TITEKeyboardAdaptiveSettings:
    """Explicit timing priors, sampler settings, and aggregate trial-work limit."""

    lambda_prior: tuple[float, float]
    gamma_prior: tuple[float, float]
    chains: int = 4
    draws: int = 4_000
    warmup: int = 1_000
    max_fit_work: int = _MAX_FIT_WORK
    max_total_work: int = 100_000_000
    max_split_rhat: float = 1.1
    max_weight_mcse: float = 0.025

    def __post_init__(self) -> None:
        for name, prior in (("lambda_prior", self.lambda_prior), ("gamma_prior", self.gamma_prior)):
            if (
                not isinstance(prior, tuple)
                or len(prior) != 2
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in prior
                )
                or any(not isfinite(float(value)) or float(value) <= 0 for value in prior)
            ):
                raise ValueError(f"{name} must be a positive finite (shape, rate) pair")
        integer_bounds = (
            ("chains", self.chains, 2, _MAX_CHAINS),
            ("draws", self.draws, 8, _MAX_DRAWS),
            ("warmup", self.warmup, 0, _MAX_WARMUP),
            ("max_fit_work", self.max_fit_work, 1, _MAX_FIT_WORK),
            ("max_total_work", self.max_total_work, 1, _MAX_TRIAL_WORK),
        )
        for name, integer_value, minimum, maximum in integer_bounds:
            if not isinstance(integer_value, (int, np.integer)) or isinstance(
                integer_value, (bool, np.bool_)
            ):
                raise ValueError(f"{name} must be an integer")
            if not minimum <= int(integer_value) <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        for name, scalar_value, lower in (
            ("max_split_rhat", self.max_split_rhat, 1.0),
            ("max_weight_mcse", self.max_weight_mcse, 0.0),
        ):
            if not isinstance(scalar_value, (int, float)) or isinstance(scalar_value, bool):
                raise ValueError(f"{name} must be a positive finite scalar")
            number = float(scalar_value)
            if not isfinite(number) or number <= lower:
                raise ValueError(f"{name} must be finite and greater than {lower}")


@dataclass(frozen=True)
class TITEKeyboardAdaptiveFitDiagnostics:
    """Compact diagnostics retained for one adaptive calendar decision."""

    log_shape_mean: NDArray[np.float64]
    log_shape_mcse: NDArray[np.float64]
    log_shape_split_rhat: NDArray[np.float64]
    acceptance_rate: NDArray[np.float64]
    max_log_shape_split_rhat: float
    max_pending_weight_mcse: float
    max_pending_weight_split_rhat: float
    observed_dlt_count: int
    pending_count: int
    draws: int
    warmup: int
    evaluations: int
    work_units: int
    prior_only: bool
    diagnostics_passed: bool


@dataclass(frozen=True)
class TITEKeyboardAdaptiveStep(CalendarStep[TITEKeyboardDecision]):
    adaptive_fit: TITEKeyboardAdaptiveFitDiagnostics | None


@dataclass(frozen=True)
class TITEKeyboardAdaptiveTrial(CalendarTrial[TITEKeyboardAdaptiveStep]):
    adaptive_work_units: int
    adaptive_fit_count: int
