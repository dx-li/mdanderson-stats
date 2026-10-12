"""Explicit Python posterior service for current EffTox controller workflows."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .efftox_desktop import EffToxDesktopStudy
from .efftox_desktop_decision import (
    EffToxDesktopDecision,
    _toxicity_summary,
    efftox_desktop_decision,
)
from .efftox_model import _integer, fit_efftox


@dataclass(frozen=True)
class EffToxDesktopFitDiagnostic:
    observed_patients: int
    selected_dose_index: int | None
    likelihood_evaluations: int
    max_split_rhat: float
    max_batch_mean_mcse: float
    mean_efficacy: tuple[float, ...]
    mean_toxicity: tuple[float, ...]


class EffToxDesktopInference:
    """Deterministic count-keyed MCMC service with explicit prior interpretation.

    The required ``monotone_toxicity`` flag chooses a positive-truncated or
    untruncated Gaussian toxicity slope. The latter matches the executed native
    integrand; native numerical posterior identity is not asserted. Saved
    elicitation inputs must be calibrated before use.
    Same counts/seed/settings reproduce the same analysis regardless of pending
    enumeration order. Diagnostics are retained for every invocation; no chain
    convergence or clinical certification is implied.

    ``max_work`` bounds nominal dose/coordinate/sweep work, not actual likelihood
    evaluations or wall time. Each individual fit retains its own work bounds.
    """

    def __init__(
        self,
        study: EffToxDesktopStudy,
        *,
        monotone_toxicity: bool,
        seed: int,
        draws: int = 1000,
        warmup: int = 500,
        chains: int = 4,
        max_work: int = 20_000_000,
        toxicity_summary_profile: Literal["model", "desktop_legacy"] = "model",
    ) -> None:
        if toxicity_summary_profile not in {"model", "desktop_legacy"}:
            raise ValueError("unknown toxicity_summary_profile")
        self.toxicity_summary_profile = toxicity_summary_profile
        self.study = study
        self.prior = study.saved_prior(monotone_toxicity=monotone_toxicity)
        metadata = ET.fromstring(study.xml).find("PythonEffToxCalibration")
        if (
            metadata is not None
            and metadata.get("monotone_toxicity") != str(bool(monotone_toxicity)).lower()
        ):
            raise ValueError("toxicity-slope profile differs from the saved Python calibration")
        self.seed = _integer(seed, "seed", 0, 2**63 - 1)
        self.draws = _integer(draws, "draws", 8, 10_000)
        self.warmup = _integer(warmup, "warmup", 0, 10_000)
        self.chains = _integer(chains, "chains", 2, 4)
        self.max_work = _integer(max_work, "max_work", 1, 200_000_000)
        self.fit_work = len(study.doses) * self.chains * (self.draws + self.warmup) * 6
        if self.fit_work > 2_000_000 or len(study.doses) * self.chains * self.draws > 200_000:
            raise ValueError("requested fit exceeds EffTox sampler work or storage bounds")
        design = ET.fromstring(study.xml).find("DoseFinderParams")
        assert design is not None
        self.settings = {
            "efficacy_limit": float(design.findtext("ResponseCutoff", "nan")),
            "toxicity_limit": float(design.findtext("ToxicityCutoff", "nan")),
            "efficacy_probability": float(design.findtext("AcceptableProbOfHighRes", "nan")),
            "toxicity_probability": float(design.findtext("AcceptableProbOfLowTox", "nan")),
        }
        self._diagnostics: list[EffToxDesktopFitDiagnostic] = []

    def check_study(self, study: EffToxDesktopStudy) -> None:
        """Reject accidental service reuse with different active design settings."""
        for name in (
            "doses",
            "prior_means",
            "prior_sds",
            "contour_points",
            "starting_dose_index",
            "newest_hyperparameters",
        ):
            if getattr(study, name) != getattr(self.study, name):
                raise ValueError("inference service belongs to different study settings")
        design = ET.fromstring(study.xml).find("DoseFinderParams")
        original = ET.fromstring(self.study.xml).find("DoseFinderParams")
        assert design is not None and original is not None
        for name in (
            "ResponseCutoff",
            "ToxicityCutoff",
            "AcceptableProbOfLowTox",
            "AcceptableProbOfHighRes",
        ):
            if float(design.findtext(name, "nan")) != float(original.findtext(name, "nan")):
                raise ValueError("inference service belongs to different study settings")

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        """Actual sampler and compatibility settings for reproducible reports."""
        return tuple(
            (key, str(value))
            for key, value in {
                "monotone_toxicity": self.prior.monotone_toxicity,
                "toxicity_summary_profile": self.toxicity_summary_profile,
                "sampler_seed": self.seed,
                "draws": self.draws,
                "warmup": self.warmup,
                "chains": self.chains,
                "max_work": self.max_work,
            }.items()
        )

    @property
    def diagnostics(self) -> tuple[EffToxDesktopFitDiagnostic, ...]:
        return tuple(self._diagnostics)

    def require_analyses(self, quantity: int) -> None:
        """Preflight a complete workflow before any partial result is produced."""
        number = _integer(quantity, "quantity", 0, 100_000)
        if (number + len(self._diagnostics)) * self.fit_work > self.max_work:
            raise ValueError(
                "workflow exceeds max_work; increase the explicit budget or batch runs"
            )

    def __call__(self, counts: NDArray[np.int64]) -> EffToxDesktopDecision:
        self.require_analyses(1)
        # Canonical int64 bytes have stable endianness and no randomized Python hash.
        raw = np.asarray(counts)
        if raw.shape != (len(self.study.doses), 2, 2) or not np.isfinite(raw).all():
            raise ValueError("counts must be finite dose-by-efficacy-by-toxicity cells")
        if np.any(raw < 0) or np.any(raw != np.floor(raw)) or raw.sum() > 10_000:
            raise ValueError("counts must be nonnegative integers totaling at most 10000")
        key = hashlib.sha256(np.asarray(raw, dtype="<i8").tobytes()).digest()
        entropy = [self.seed, *np.frombuffer(key, dtype="<u4").tolist()]
        fit = fit_efftox(
            self.study.doses,
            raw,
            prior=self.prior,
            draws=self.draws,
            warmup=self.warmup,
            chains=self.chains,
            rng=np.random.default_rng(np.random.SeedSequence(entropy)),
        )
        result = efftox_desktop_decision(
            fit,
            self.study.contour(),
            starting_dose_index=self.study.starting_dose_index,
            toxicity_summary_profile=self.toxicity_summary_profile,
            **self.settings,
        )

        def maximum(values: NDArray) -> float:
            valid = np.asarray(values)[~np.isnan(values)]
            return float(valid.max()) if valid.size else float("nan")

        self._diagnostics.append(
            EffToxDesktopFitDiagnostic(
                int(raw.sum()),
                result.dose_index,
                fit.likelihood_evaluations,
                maximum(fit.summary.split_rhat),
                maximum(fit.summary.batch_mean_mcse),
                tuple(fit.efficacy_probabilities.mean(axis=(0, 1))),
                tuple(_toxicity_summary(fit, self.toxicity_summary_profile)[1].mean(axis=(0, 1))),
            )
        )
        return result
