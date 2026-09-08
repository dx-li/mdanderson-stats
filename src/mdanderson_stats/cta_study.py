"""Reusable CTA analysis choices and independent table-study snapshots."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .cta import ContingencyChiSquare, contingency_chi_square
from .cta_binomial import BinomialComparison, binomial_comparison
from .cta_diagnostic import DiagnosticAccuracy, diagnostic_accuracy
from .cta_fisher import FisherExact, fisher_exact
from .cta_kappa import CohenKappa, cohen_kappa
from .cta_mcnemar import McNemarAnalysis, mcnemar_analysis
from .cta_odds import OddsRatio, odds_ratio


@dataclass(frozen=True)
class CTAStudySpecification:
    chi_square: bool = True
    kappa: bool = False
    mcnemar: bool = False
    diagnostic: bool = False
    odds: bool = False
    binomial: bool = False
    fisher: Literal["auto", "always", "never"] = "auto"
    expected_threshold: float = 5
    fisher_threshold: float = 10
    fisher_alternative: Literal["two-sided", "less", "greater", "source"] | None = None
    standard: Literal["rows", "columns"] = "columns"
    positive_index: int = 0
    risk_factor: Literal["rows", "columns"] = "columns"
    response_index: int = 0
    alpha: float = 0.05
    groups: Literal["rows", "columns"] = "rows"
    event_index: int | None = None
    legacy: bool = False

    def __post_init__(self) -> None:
        for name in ("chi_square", "kappa", "mcnemar", "diagnostic", "odds", "binomial", "legacy"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.fisher, str) or self.fisher not in ("auto", "always", "never"):
            raise ValueError("fisher must be auto, always or never")
        if self.fisher_alternative is not None and (
            not isinstance(self.fisher_alternative, str)
            or self.fisher_alternative not in ("two-sided", "less", "greater", "source")
        ):
            raise ValueError("invalid fisher_alternative")
        if self.legacy and self.fisher_alternative not in (None, "source"):
            raise ValueError("legacy requires the source Fisher alternative")
        for name in ("standard", "risk_factor", "groups"):
            if not isinstance(getattr(self, name), str) or getattr(self, name) not in (
                "rows",
                "columns",
            ):
                raise ValueError(f"{name} must be rows or columns")
        for name in ("positive_index", "response_index", "event_index"):
            value = getattr(self, name)
            if name == "event_index" and value is None:
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value not in (0, 1)
            ):
                raise ValueError(f"{name} must be zero or one")
        for name in ("expected_threshold", "fisher_threshold", "alpha"):
            value = scalar(getattr(self, name), name)
            if (name == "alpha" and not 0 < value < 1) or (name != "alpha" and value < 0):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)

    def run(self, observed: ArrayLike) -> "CTAStudy":
        """Analyze one table; reuse this immutable specification for later tables.

        Explicitly selected analyses fail if their input requirements are not
        met. Auto Fisher follows CTA's 2x2 small-expected-cell trigger and records
        why it was skipped. Core analysis functions separately accept batches.
        """
        ob = finite(observed, "observed").copy()
        if ob.ndim != 2 or min(ob.shape) < 2 or np.any(ob < 0):
            raise ValueError(
                "observed must be one nonnegative table with at least two rows and columns"
            )
        with np.errstate(over="ignore"):
            total = ob.sum()
        if not np.isfinite(total):
            raise ValueError("table total must be finite")
        chi = (
            contingency_chi_square(
                ob, expected_threshold=self.expected_threshold, legacy=self.legacy
            )
            if self.chi_square
            else None
        )
        fisher = None
        note = "disabled"
        if self.fisher == "always":
            fisher = fisher_exact(ob, alternative=self.fisher_alternative, legacy=self.legacy)
            note = "explicitly requested"
        elif self.fisher == "auto":
            if chi is None:
                note = "auto requires chi-square analysis"
            elif ob.shape != (2, 2):
                note = "auto requires a 2x2 table"
            elif chi.minimum_expected >= self.fisher_threshold:
                note = "minimum expected count is not below the Fisher threshold"
            elif total > 50_000:
                note = "auto skipped: table total exceeds 50,000"
            elif np.any(ob != np.floor(ob)):
                note = "auto skipped: Fisher probabilities require integer counts"
            else:
                fisher = fisher_exact(ob, alternative=self.fisher_alternative, legacy=self.legacy)
                note = "auto: minimum expected count below the Fisher threshold"
        kappa = cohen_kappa(ob, legacy=self.legacy) if self.kappa else None
        mcnemar = mcnemar_analysis(ob) if self.mcnemar else None
        diagnostic = (
            diagnostic_accuracy(
                ob, standard=self.standard, positive_index=self.positive_index, legacy=self.legacy
            )
            if self.diagnostic
            else None
        )
        odds = (
            odds_ratio(
                ob,
                risk_factor=self.risk_factor,
                response_index=self.response_index,
                alpha=self.alpha,
                legacy=self.legacy,
            )
            if self.odds
            else None
        )
        binomial = (
            binomial_comparison(
                ob, groups=self.groups, event_index=self.event_index, legacy=self.legacy
            )
            if self.binomial
            else None
        )
        ob.flags.writeable = False
        return CTAStudy(ob, self, chi, fisher, kappa, mcnemar, diagnostic, odds, binomial, note)


@dataclass(frozen=True)
class CTAStudy:
    observed: FloatArray
    specification: CTAStudySpecification
    chi_square: ContingencyChiSquare | None
    fisher: FisherExact | None
    kappa: CohenKappa | None
    mcnemar: McNemarAnalysis | None
    diagnostic: DiagnosticAccuracy | None
    odds: OddsRatio | None
    binomial: BinomialComparison | None
    fisher_note: str

    def report(self, *, digits: int = 6) -> str:
        """Return settings, observations, margins and every selected result."""
        from .cta_report import format_cta

        return format_cta(self, digits=digits)

    def write_report(self, path: str | Path, *, digits: int = 6) -> None:
        """Write the complete UTF-8 summary, replacing an existing destination."""
        Path(path).write_text(self.report(digits=digits), encoding="utf-8")
