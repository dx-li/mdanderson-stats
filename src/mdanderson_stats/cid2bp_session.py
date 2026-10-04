"""Repeatable CID2BP comparisons and cumulative plain-text reporting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._validation import scalar
from .bayesian_monitoring import _integer
from .cid2bp import BinomialDifferenceInterval, cid2bp_interval

_METHODS = frozenset(
    {
        "wald",
        "continuity_corrected",
        "yates",
        "peskun_native",
        "cox_snell",
        "weighted_mid_p",
        "weighted_likelihood",
        "auto",
        "exact",
    }
)
_MAX_CALCULATIONS = 1_000


@dataclass(frozen=True, slots=True)
class CID2BPRequest:
    """One comparison and its ordered method choices.

    ``count1`` and ``count2`` are either trial totals or failure counts as
    selected by ``entry``. Repeated method names are retained and calculated.
    """

    successes1: int
    count1: int
    successes2: int
    count2: int
    entry: Literal["trials", "failures"] = "trials"
    confidence: float = 0.95
    methods: tuple[str, ...] = ("cox_snell",)

    def __post_init__(self) -> None:
        if self.entry not in ("trials", "failures"):
            raise ValueError("entry must be 'trials' or 'failures'")
        if not isinstance(self.methods, tuple) or not self.methods:
            raise ValueError("methods must be a nonempty tuple of method names")
        if len(self.methods) > _MAX_CALCULATIONS:
            raise ValueError(f"a request may contain at most {_MAX_CALCULATIONS} methods")
        if any(not isinstance(method, str) or method not in _METHODS for method in self.methods):
            raise ValueError("methods contains an unknown CID2BP method")


@dataclass(frozen=True, slots=True)
class CID2BPRecord:
    """One calculation, retaining entered counts and the resolved method."""

    comparison_index: int
    entry: Literal["trials", "failures"]
    successes1: int
    count1: int
    successes2: int
    count2: int
    trials1: int
    failures1: int
    trials2: int
    failures2: int
    confidence: float
    requested_method: str
    result: BinomialDifferenceInterval


@dataclass(frozen=True, slots=True)
class CID2BPSession:
    """Ordered results for a bounded series of CID2BP comparisons."""

    records: tuple[CID2BPRecord, ...]

    def report(self, *, digits: int = 8) -> str:
        """Return a tab-separated report containing one row per calculation."""
        _validate_digits(digits)
        header = [
            "Comparison",
            "Entry",
            "Successes 1",
            "Failures 1",
            "Trials 1",
            "Rate 1",
            "Successes 2",
            "Failures 2",
            "Trials 2",
            "Rate 2",
            "Confidence",
            "Estimate (p1-p2)",
            "Requested method",
            "Method",
            "Lower",
            "Upper",
        ]
        rows = ["CID2BP session", "\t".join(header)]
        for item in self.records:
            values = [
                str(item.comparison_index),
                item.entry,
                str(item.successes1),
                str(item.failures1),
                str(item.trials1),
                _format(item.successes1 / item.trials1, digits),
                str(item.successes2),
                str(item.failures2),
                str(item.trials2),
                _format(item.successes2 / item.trials2, digits),
                _format(item.confidence, digits),
                _format(item.result.estimate, digits),
                item.requested_method,
                item.result.method,
                _format(item.result.lower, digits),
                _format(item.result.upper, digits),
            ]
            rows.append("\t".join(values))
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 8) -> Path:
        """Render the complete report before writing the requested destination."""
        content = self.report(digits=digits)
        destination = Path(path)
        destination.write_text(content, encoding="utf-8")
        return destination


def cid2bp_session(requests: Sequence[CID2BPRequest]) -> CID2BPSession:
    """Run ordered comparisons and methods through :func:`cid2bp_interval`.

    All request structure, count ranges, confidence levels, and the aggregate
    1,000-calculation limit are checked before any interval calculation begins.
    The function is serial and accepts a sized sequence so oversized iterables
    are rejected before being copied or evaluated.
    """
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise ValueError("requests must be a sized sequence of CID2BPRequest values")
    if not 1 <= len(requests) <= _MAX_CALCULATIONS:
        raise ValueError(f"requests must contain 1..{_MAX_CALCULATIONS} comparisons")
    if any(not isinstance(request, CID2BPRequest) for request in requests):
        raise ValueError("every request must be a CID2BPRequest")

    prepared: list[tuple[CID2BPRequest, int, int, float]] = []
    calculation_count = 0
    for request in requests:
        calculation_count += len(request.methods)
        if calculation_count > _MAX_CALCULATIONS:
            raise ValueError(f"session exceeds the {_MAX_CALCULATIONS}-calculation limit")
        x1 = _integer(request.successes1, "successes1")
        entered1 = _integer(request.count1, "count1")
        x2 = _integer(request.successes2, "successes2")
        entered2 = _integer(request.count2, "count2")
        if request.entry == "trials":
            n1, n2 = entered1, entered2
        else:
            n1, n2 = x1 + entered1, x2 + entered2
        if not (1 <= n1 <= 1_000_000 and 1 <= n2 <= 1_000_000):
            raise ValueError("each sample must have 1..1000000 trials")
        if not (0 <= x1 <= n1 and 0 <= x2 <= n2):
            raise ValueError("successes must lie between zero and the trial count")
        confidence = scalar(request.confidence, "confidence")
        if not 1e-6 <= confidence <= 1 - 1e-12:
            raise ValueError("confidence must be in [1e-6,1-1e-12]")
        if "continuity_corrected" in request.methods and min(n1, n2) < 2:
            raise ValueError("continuity_corrected requires at least two trials in each group")
        if "weighted_likelihood" in request.methods or "weighted_mid_p" in request.methods:
            if n1 + n2 > 200:
                raise ValueError(
                    "weighted likelihood methods permit at most 200 total observations"
                )
        if "exact" in request.methods and max(n1, n2) > 100:
            raise ValueError("exact permits at most 100 trials per sample")
        if "peskun_native" in request.methods and (n1 + 1) * (n2 + 1) > 2_000_000:
            raise ValueError("peskun_native permits at most 2 million grid points")
        prepared.append((request, n1, n2, confidence))

    records: list[CID2BPRecord] = []
    for comparison_index, (request, n1, n2, confidence) in enumerate(prepared, start=1):
        x1, x2 = int(request.successes1), int(request.successes2)
        failures1, failures2 = n1 - x1, n2 - x2
        for method in request.methods:
            result = cid2bp_interval(
                x1,
                n1,
                x2,
                n2,
                confidence=confidence,
                method=method,
            )
            records.append(
                CID2BPRecord(
                    comparison_index,
                    request.entry,
                    x1,
                    request.count1,
                    x2,
                    request.count2,
                    n1,
                    failures1,
                    n2,
                    failures2,
                    confidence,
                    method,
                    result,
                )
            )
    return CID2BPSession(tuple(records))


def _validate_digits(digits: int) -> None:
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
        raise ValueError("digits must be an integer from 1 to 17")


def _format(value: float, digits: int) -> str:
    return format(float(value), f".{digits}g")
