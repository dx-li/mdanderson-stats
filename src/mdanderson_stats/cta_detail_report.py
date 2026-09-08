"""Bounded per-cell and per-probability listings for CTA studies."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import binom

from .cta import _yates_contributions
from .cta_fisher import _fisher_support
from .cta_report import _value

if TYPE_CHECKING:
    from .cta_study import CTAStudy


def detail_lines(study: "CTAStudy", digits: int, max_terms: int) -> list[str]:
    chi, fisher, comparison = study.chi_square, study.fisher, study.binomial
    required = study.observed.size if chi is not None else 0
    if fisher is not None:
        required += int(fisher.terms)
    if comparison is not None:
        events = list(map(int, comparison.events))
        required += 2 * (min(events) + 1) if comparison.legacy else sum(events) + 2
    if required > max_terms:
        raise ValueError(
            f"detailed report requires {required} rows; increase max_terms={max_terms}"
        )
    lines: list[str] = []
    if chi is not None:
        lines.extend(
            ["\nCell details", "row\tcolumn\tobserved\texpected\tPearson\tYates\trow%\tcolumn%"]
        )
        yates = (
            _yates_contributions(np.abs(study.observed - chi.expected), chi.expected, chi.legacy)
            if chi.yates_statistic is not None
            else np.full_like(chi.expected, np.nan)
        )
        for i, j in np.ndindex(study.observed.shape):
            values = [
                i,
                j,
                study.observed[i, j],
                chi.expected[i, j],
                chi.contributions[i, j],
                yates[i, j],
                chi.row_percent[i, j],
                chi.column_percent[i, j],
            ]
            lines.append("\t".join(_value(value, digits) for value in values))
    if fisher is not None:
        support, probabilities, selected, _ = _fisher_support(
            study.observed.ravel(), fisher.alternative, fisher.legacy
        )
        a, b, c, d = map(int, study.observed.ravel())
        row, col, total = a + b, a + c, a + b + c + d
        lines.extend(["\nFisher terms", "a\tb\tc\td\tprobability\tcumulative"])
        cumulative = np.cumsum(probabilities[selected])
        for at, running in zip(selected, cumulative, strict=True):
            x = int(support[at])
            values = [x, row - x, col - x, total - row - col + x, probabilities[at], running]
            lines.append("\t".join(_value(value, digits) for value in values))
    if comparison is not None:
        events = list(map(int, comparison.events))
        n = sum(events)
        lower_group = (0 if events[0] < events[1] else 1) if comparison.legacy else 0
        upper_group = 1 - lower_group if comparison.legacy else 0
        for label, group, lower in [("lower", lower_group, True), ("upper", upper_group, False)]:
            counts = np.arange(events[group] + 1) if lower else np.arange(n, events[group] - 1, -1)
            p = comparison.group_sizes[group] / comparison.group_sizes.sum()
            probabilities = binom.pmf(counts, n, p)
            if not np.all(np.isfinite(probabilities)):
                raise RuntimeError("nonfinite binomial detail probabilities")
            lines.extend([f"\nBinomial {label} terms", "group\tcount\tprobability\tcumulative"])
            for k, probability, running in zip(
                counts, probabilities, np.cumsum(probabilities), strict=True
            ):
                values = [group, k, probability, running]
                lines.append("\t".join(_value(value, digits) for value in values))
    return lines
