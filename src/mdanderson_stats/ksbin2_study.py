"""KSBIN2 study-level null scans, paired-hypothesis summaries and reports."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .ksbin2_multistage import KStageTwoSampleBinomial, KSTwoSampleOperatingCharacteristics


@dataclass(frozen=True)
class KSTwoSampleStudy:
    design: KStageTwoSampleBinomial
    probability1: FloatArray
    probability2: FloatArray
    null_probability: FloatArray
    null_grid: FloatArray
    null_scan: KSTwoSampleOperatingCharacteristics
    null: KSTwoSampleOperatingCharacteristics
    alternative: KSTwoSampleOperatingCharacteristics
    cumulative_grid_significance: FloatArray
    maximizing_null_probability: FloatArray

    def scan(
        self, probability1: ArrayLike, probability2: ArrayLike
    ) -> KSTwoSampleOperatingCharacteristics:
        """Evaluate arbitrary broadcast probability pairs with the fixed design."""
        return self.design.operating_characteristics(probability1, probability2)

    def revise(
        self,
        *,
        design: KStageTwoSampleBinomial | None = None,
        probability1: ArrayLike | None = None,
        probability2: ArrayLike | None = None,
        null_probability: ArrayLike | None = None,
        null_grid: ArrayLike | None = None,
    ) -> "KSTwoSampleStudy":
        """Recompute supplied changes, retaining unspecified inputs including the null.

        To select the default mean null for new alternatives, call ksbin2_study
        anew; revision deliberately retains the previously evaluated null rate.
        """
        return ksbin2_study(
            self.design if design is None else design,
            self.probability1 if probability1 is None else probability1,
            self.probability2 if probability2 is None else probability2,
            null_probability=self.null_probability
            if null_probability is None
            else null_probability,
            null_grid=self.null_grid if null_grid is None else null_grid,
        )

    def report(self, *, digits: int = 6) -> str:
        """TSV sections with actual null probabilities separate from grid maxima."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        rows = [
            "KSBIN2 study",
            f"Direction\t{self.design.alternative}",
            "Criteria\t" + " ".join(map(str, self.design.criteria)),
        ]

        def row(*values: str | int | float) -> None:
            rows.append(
                "\t".join(
                    v if isinstance(v, str) else format(float(v), f".{digits}g") for v in values
                )
            )

        rows.append("Ordinary null-grid significance")
        row(
            "Stage",
            "Group 1 cumulative trials",
            "Group 2 cumulative trials",
            "Reject group",
            "Quit group",
            "Reject score",
            "Quit score",
            "Cumulative grid maximum",
            "Maximizing grid probability",
        )
        for i, (n1, n2) in enumerate(self.design.cumulative_trials):
            rg = self.design.reject_group[i]
            qg = self.design.quit_group[i] if i < len(self.design.quit_group) else -1
            ordering = self.design.orderings[i]
            reject_score = "disabled" if rg == -1 else float(ordering.score[ordering.group_end[rg]])
            quit_score = (
                "disabled"
                if qg == -1
                else float(ordering.score[0 if qg == 0 else ordering.group_end[qg - 1] + 1])
            )
            row(
                i + 1,
                n1,
                n2,
                rg,
                qg,
                reject_score,
                quit_score,
                self.cumulative_grid_significance[i],
                self.maximizing_null_probability[i],
            )
        for case, index in enumerate(np.ndindex(self.probability1.shape), start=1):
            rows.append(f"Case {case}")
            row(
                "Alternative probabilities",
                float(self.probability1[index]),
                float(self.probability2[index]),
            )
            row("Common null probability", float(self.null_probability[index]))
            for label, result in [("Null", self.null), ("Alternative", self.alternative)]:
                rows.append(label + " operating characteristics")
                row("Stage", "Reject", "Cumulative reject", "Quit", "Cumulative quit", "Continue")
                rejection = result.rejection[index]
                quitting = result.quitting[index]
                cumulative_r = np.cumsum(rejection)
                cumulative_q = np.cumsum(quitting)
                for i in range(len(rejection)):
                    row(
                        i + 1,
                        rejection[i],
                        cumulative_r[i],
                        quitting[i],
                        cumulative_q[i],
                        result.continuation[index][i],
                    )
                row("Expected sample sizes", *result.expected_sample_size[index])
        rows.append(
            "Significance maxima cover the specified null grid only; they are not continuum bounds."
        )
        rows.append(
            "Operating characteristics and expected sizes use actual supplied probabilities, "
            "without mid-p adjustment."
        )
        rows.append(
            "Group indices are zero-based and inclusive; -1 disables a boundary. "
            "Final nonrejections quit."
        )
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 6) -> Path:
        """Write UTF-8 output, explicitly replacing path; propagate I/O errors."""
        content = self.report(digits=digits)
        path = Path(path)
        path.write_text(content, encoding="utf-8")
        return path


def ksbin2_study(
    design: KStageTwoSampleBinomial,
    probability1: ArrayLike,
    probability2: ArrayLike,
    *,
    null_probability: ArrayLike | None = None,
    null_grid: ArrayLike | None = None,
) -> KSTwoSampleStudy:
    """Summarize a fixed design, including cumulative null-grid maxima by stage.

    The default null rate is the mean of the alternative rates, matching PRH0 in
    the original. That rate affects actual null operating characteristics, while
    significance uses a separate explicit scan grid. Probability cases broadcast.
    """
    if not isinstance(design, KStageTwoSampleBinomial):
        raise ValueError("design must be a KStageTwoSampleBinomial")
    p1, p2 = np.broadcast_arrays(
        finite(probability1, "probability1"), finite(probability2, "probability2")
    )
    if np.any((p1 < 0) | (p1 > 1) | (p2 < 0) | (p2 > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    p0 = (p1 + p2) / 2 if null_probability is None else finite(null_probability, "null_probability")
    p1, p2, p0 = np.broadcast_arrays(p1, p2, p0)
    alternative = design.operating_characteristics(p1, p2)
    null = design.operating_characteristics(p0, p0)
    grid = (
        np.arange(51, dtype=np.float64) * (0.01 if design.alternative == "two-sided" else 0.02)
        if null_grid is None
        else finite(null_grid, "null_grid")
    )
    if (
        grid.ndim != 1
        or grid.size == 0
        or np.any((grid < 0) | (grid > 1))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("null_grid must be a nonempty strictly increasing vector in [0,1]")
    scan = design.operating_characteristics(grid, grid)
    cumulative = np.cumsum(scan.rejection, axis=-1)
    return KSTwoSampleStudy(
        design,
        p1,
        p2,
        p0,
        grid,
        scan,
        null,
        alternative,
        cumulative.max(axis=0),
        grid[np.argmax(cumulative, axis=0)],
    )
