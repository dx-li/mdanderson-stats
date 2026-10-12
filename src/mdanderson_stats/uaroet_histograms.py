"""Bounded streaming posterior histograms with explicit guide/native bin conventions."""

import csv
import io

import numpy as np
from numpy.typing import NDArray

from .uaroet import _integer
from .uaroet_desktop_rules import _profile
from .uaroet_desktop_study import UAROETDesktopStudy
from .uaroet_fit import UAROETFit


class UAROETPosteriorHistograms:
    """Callback for fit/recommend on_chunk; stores100 bins per dose and quantity.

    Safe bins follow guide lower limits[0,1),...,[99,100]; legacy_1_8 instead
    reproduces trunc(value+.5), capped at99, as recovered from native arithmetic.
    Probabilities are percentages. Acceptable efficacy/unacceptable toxicity are
    summed within each draw BEFORE binning, not summed histogram columns.
    Partial output is diagnostic until validate_complete passes on the fit.
    """

    def __init__(self, study: UAROETDesktopStudy, *, profile: str = "safe"):
        _profile(profile)
        if not isinstance(study, UAROETDesktopStudy):
            raise ValueError("require a UAROETDesktopStudy")
        self.study = study
        self.profile = profile
        self.quantities = (
            *(f"PE{k}" for k in range(study.efficacy_levels)),
            *(f"PT{k}" for k in range(study.toxicity_levels)),
            "PUtil",
            "PEff",
            "PTox",
        )
        self._counts = np.zeros((study.dose_count, 100, len(self.quantities)), dtype=np.int64)
        self._next_draw: dict[int, int] = {}

    @property
    def counts(self) -> NDArray[np.int64]:
        """Immutable snapshot indexed(dose, bin, quantity)."""
        return np.frombuffer(self._counts.tobytes(), dtype=np.int64).reshape(self._counts.shape)

    def __call__(self, chain: int, first_draw: int, fit: UAROETFit) -> None:
        c = _integer(chain, "chain", 0, 19)
        first = _integer(first_draw, "first_draw", 0, 1_000_000)
        s = self.study
        if (
            not isinstance(fit, UAROETFit)
            or fit.joint.ndim != 5
            or fit.joint.shape[0] != 1
            or fit.joint.shape[2:] != (s.dose_count, s.efficacy_levels, s.toxicity_levels)
        ):
            raise ValueError("chunk must contain one chain on the study grid")
        if first != self._next_draw.get(c, 0):
            raise ValueError("posterior chunks must be contiguous and nonduplicated per chain")
        joint = fit.joint[0]
        end = first + len(joint)
        if len(joint) < 1 or end > 1_000_000:
            raise ValueError("histogram retained-draw budget exceeded")
        efficacy, toxicity = joint.sum(-1), joint.sum(-2)
        values = np.concatenate(
            (
                100 * efficacy,
                100 * toxicity,
                np.einsum("ndet,et->nd", joint, s.utility)[..., None],
                100
                * efficacy[..., int(s.trial_parameters["LowestAcceptableEffLevel"]) :].sum(-1)[
                    ..., None
                ],
                100
                * toxicity[..., int(s.trial_parameters["LowestUnacceptableToxLevel"]) :].sum(-1)[
                    ..., None
                ],
            ),
            axis=-1,
        )
        if np.any(~np.isfinite(values)) or np.any((values < -1e-9) | (values > 100 + 1e-9)):
            raise ArithmeticError("posterior histogram quantities leave[0,100]")
        # Remove only floating-point endpoint roundoff, never invalid probability mass.
        bins = np.minimum(
            99, np.floor(np.clip(values, 0, 100) + (0.5 if self.profile == "legacy_1_8" else 0))
        ).astype(np.int64)
        for d in range(s.dose_count):
            for q in range(len(self.quantities)):
                self._counts[d, :, q] += np.bincount(bins[:, d, q], minlength=100)
        self._next_draw[c] = end

    def validate_complete(self, *, chains: int, draws_per_chain: int) -> None:
        n = _integer(chains, "chains", 1, 20)
        draws = _integer(draws_per_chain, "draws_per_chain", 1, 1_000_000)
        if self._next_draw != dict.fromkeys(range(n), draws) or not np.all(
            self._counts.sum(1) == n * draws
        ):
            raise ValueError("posterior histogram coverage is incomplete")

    def to_csv(self, dose: int) -> str:
        """One dose's counts; zero-based input dose, explicit bin-rule metadata column."""
        d = _integer(dose, "dose", 0, self.study.dose_count - 1)
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["bin", *self.quantities, "profile"])
        for b, row in enumerate(self._counts[d]):
            writer.writerow([b, *row, self.profile])
        return stream.getvalue()
