"""Combined CUMINC curves, group comparisons and numerical summaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .cumulative_incidence import CumulativeIncidence, IncidenceSummary, cumulative_incidence
from .gray_test import GrayTest, _labels, gray_test

Label = str | float


@dataclass(frozen=True)
class CumIncStudy:
    """Cause/group keys refer to original labels; curve event codes are internal.

    Curves and tests are immutable mappings. Strata affect tests, not the
    descriptive group-specific curves. n_dropped records complete-case deletion.
    """

    curves: Mapping[tuple[Label, Label], CumulativeIncidence]
    tests: Mapping[Label, GrayTest]
    causes: tuple[Label, ...]
    groups: tuple[Label, ...]
    strata: tuple[Label, ...]
    group_counts: Mapping[Label, int]
    n_dropped: int
    confidence: float

    def summaries(
        self,
        times: ArrayLike | None = None,
        *,
        causes: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> dict[tuple[Label, Label], IncidenceSummary]:
        """Select exact cause/group labels and return their numerical summaries."""
        selected_causes = _select(causes, self.causes, "causes")
        selected_groups = _select(groups, self.groups, "groups")
        return {
            (cause, group): self.curves[cause, group].summary(times, confidence=self.confidence)
            for cause in selected_causes
            for group in selected_groups
        }

    def report(self, *, times: ArrayLike | None = None, digits: int = 6) -> str:
        """Counts, Gray tests and all curve summaries in a text report."""
        # Validate precision even for an all-censored study with no curves.
        IncidenceSummary(np.empty((0, 5)), self.confidence).report(digits=digits)
        lines = [
            f"CUMINC\nDropped observations: {self.n_dropped}",
            "\nGroup counts",
            "group\tobservations" + "".join(f"\tevents {c}" for c in self.causes),
        ]
        for group in self.groups:
            values = [str(group), str(self.group_counts[group])]
            values.extend(str(self.curves[c, group].n_events) for c in self.causes)
            lines.append("\t".join(values))
        lines.extend(["\nGray tests", "cause\tstatistic\tdf\tpvalue\trank"])
        for cause, test in self.tests.items():
            statistic = "NA" if test.statistic is None else f"{test.statistic:.{digits}g}"
            pvalue = "NA" if test.pvalue is None else f"{test.pvalue:.{digits}g}"
            lines.append(f"{cause}\t{statistic}\t{test.degrees_freedom}\t{pvalue}\t{test.rank}")
        for (cause, group), summary in self.summaries(times).items():
            lines.append(f"\nCause = {cause}, Group = {group}")
            lines.append(summary.report(digits=digits).rstrip())
        return "\n".join(lines) + "\n"

    def write_report(
        self, path: str | Path, *, times: ArrayLike | None = None, digits: int = 6
    ) -> None:
        """Write a UTF-8 report, replacing an existing destination."""
        Path(path).write_text(self.report(times=times, digits=digits), encoding="utf-8")


def _select(values: ArrayLike | None, available: tuple[Label, ...], name: str) -> tuple:
    if values is None:
        return available
    array = np.asarray(values, dtype=object)
    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or vector of exact labels")
    selected = tuple(array.reshape(-1).tolist())
    if any(x not in available for x in selected) or len(set(selected)) != len(selected):
        raise ValueError(f"{name} must contain unique existing labels")
    return selected


def cuminc(
    time: ArrayLike,
    cause: ArrayLike,
    group: ArrayLike | None = None,
    *,
    strata: ArrayLike | None = None,
    censor: Label = 0,
    rho: float = 0,
    confidence: float = 0.95,
    missing: Literal["raise", "drop"] = "raise",
) -> CumIncStudy:
    """Analyze every observed cause and group, with optional stratified Gray tests.

    Each label vector must contain all strings or all finite numbers. Censor
    identifies the censoring label; all others are distinct competing causes.
    Missing time/cause/group/stratum values are rejected unless missing='drop'.
    Complete-case deletion is then applied jointly and counted in the result.
    """
    if missing not in ("raise", "drop"):
        raise ValueError("missing must be 'raise' or 'drop'")
    level, power = scalar(confidence, "confidence"), scalar(rho, "rho")
    if not 0 < level < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not isinstance(censor, str) and (
        not isinstance(censor, Real)
        or isinstance(censor, (bool, np.bool_))
        or not np.isfinite(float(censor))
    ):
        raise ValueError("censor must be a string or finite numeric label")
    t, event = np.asarray(time, dtype=object), np.asarray(cause, dtype=object)
    g = np.ones(t.size, dtype=object) if group is None else np.asarray(group, dtype=object)
    st = np.zeros(t.size, dtype=object) if strata is None else np.asarray(strata, dtype=object)
    if t.ndim != 1 or t.size == 0 or any(a.shape != t.shape for a in [event, g, st]):
        raise ValueError("Require nonempty matching time/cause/group/strata vectors")
    absent = np.zeros(t.size, dtype=bool)
    for array in [t, event, g, st]:
        absent |= np.array([x is None or isinstance(x, Real) and np.isnan(float(x)) for x in array])
    dropped = int(absent.sum())
    if dropped and missing == "raise":
        raise ValueError("Missing observations require missing='drop'")
    if np.all(absent):
        raise ValueError("No observations remain after removing missing values")
    t, event, g, st = (a[~absent] for a in [t, event, g, st])
    times = finite(t, "time")
    if np.any(times < 0):
        raise ValueError("time must be nonnegative")
    labels, event_index = _labels(event, times.shape, "cause")
    groups, group_index = _labels(g, times.shape, "group")
    levels, _ = _labels(st, times.shape, "strata")
    causes = tuple(c for c in labels if c != censor)
    # Keep zero reserved for censoring, including datasets without censoring.
    code_map = np.array([0 if c == censor else causes.index(c) + 1 for c in labels])
    codes = code_map[event_index]
    curves, tests = {}, {}
    for number, label in enumerate(causes, start=1):
        for index, name in enumerate(groups):
            selected = group_index == index
            curves[label, name] = cumulative_incidence(
                times[selected], codes[selected], event_of_interest=number
            )
        if len(groups) > 1:
            # Original labels are retained in GrayTest; the numeric cause is internal.
            tests[label] = gray_test(
                times, codes, g, strata=st, event_of_interest=number, rho=power
            )
    counts = np.bincount(group_index, minlength=len(groups))
    return CumIncStudy(
        MappingProxyType(curves),
        MappingProxyType(tests),
        causes,
        groups,
        levels,
        MappingProxyType(dict(zip(groups, map(int, counts), strict=True))),
        dropped,
        level,
    )
