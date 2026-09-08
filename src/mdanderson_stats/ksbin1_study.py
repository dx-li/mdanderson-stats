"""KSBIN1 study comparison, design revision, and readable/file output."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .ksbin1 import KSBinomialOperatingCharacteristics, ksbin1_operating_characteristics
from .ksbin1_table import KSBinomialBoundaryTable, ksbin1_boundary_table


@dataclass(frozen=True)
class KSBinomialStudy:
    """A study evaluated under paired null/alternative probabilities.

    characteristics has a leading hypothesis axis (null, alternative), followed
    by the broadcast case axes. Per-stage quantities have a final stage axis.
    Revision returns a freshly validated study without changing this result.
    """

    characteristics: KSBinomialOperatingCharacteristics
    null_probability: FloatArray
    alternative_probability: FloatArray
    single_stage_critical: int
    single_stage_significance: FloatArray
    single_stage_power: FloatArray

    @property
    def significance(self) -> FloatArray:
        return self.characteristics.rejection_probability[0]

    @property
    def power(self) -> FloatArray:
        return self.characteristics.rejection_probability[1]

    @property
    def expected_sample_savings(self) -> FloatArray:
        """Reference fixed sample size minus expected sample size, under H0/Ha."""
        c = self.characteristics
        return c.design.cumulative_trials[-1] - c.expected_sample_size

    def boundary_table(self, stage: int) -> KSBinomialBoundaryTable:
        c = self.characteristics
        return ksbin1_boundary_table(
            c.design,
            stage,
            self.null_probability,
            self.alternative_probability,
            self.single_stage_critical,
            alternative=c.alternative,
        )

    def revise(
        self,
        *,
        cumulative_trials: ArrayLike | None = None,
        critical: ArrayLike | None = None,
        quit: ArrayLike | None = None,
        null_probability: ArrayLike | None = None,
        alternative_probability: ArrayLike | None = None,
        single_stage_critical: int | None = None,
        alternative: str | None = None,
    ) -> "KSBinomialStudy":
        """Recalculate with supplied changes; unspecified inputs retain their values.

        When changing stage count, supply matching critical and quit arrays.
        Reversing the alternative requires explicit direction and suitable cutoffs.
        """
        c = self.characteristics
        old_quit = c.design.high if c.alternative == "less" else c.design.low
        return ksbin1_study(
            c.design.cumulative_trials if cumulative_trials is None else cumulative_trials,
            c.critical if critical is None else critical,
            old_quit if quit is None else quit,
            self.null_probability if null_probability is None else null_probability,
            self.alternative_probability
            if alternative_probability is None
            else alternative_probability,
            self.single_stage_critical if single_stage_critical is None else single_stage_critical,
            alternative=c.alternative if alternative is None else alternative,
        )

    def design_text(self) -> str:
        """Original KSBIN1 export: stage count, then increment/quit/critical rows.

        Final quit is -1, as in the source. This format carries no probabilities
        or direction and is not a complete study serialization.
        """
        c = self.characteristics
        quits = c.design.high if c.alternative == "less" else c.design.low
        rows = [str(len(c.critical))]
        previous = 0
        for n, quit, critical in zip(
            c.design.cumulative_trials, (*quits, -1), c.critical, strict=True
        ):
            rows.append(f"{n - previous} {quit} {critical}")
            previous = n
        return "\n".join(rows) + "\n"

    def write_design(self, path: str | Path) -> Path:
        """Write the original numeric design format, replacing path explicitly."""
        path = Path(path)
        path.write_text(self.design_text(), encoding="utf-8")
        return path

    def report(self, *, digits: int = 6, include_tables: bool = False) -> str:
        """Readable TSV sections; broadcast cases appear in C order.

        Reports include stage and cumulative decisions, the single-stage
        comparison, and the original correct-decision/overall expectations.
        Undefined conditional expectations are printed as NaN.
        """
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        if not isinstance(include_tables, bool):
            raise ValueError("include_tables must be boolean")
        c = self.characteristics
        stages = len(c.critical)
        quits = c.design.high if c.alternative == "less" else c.design.low
        tables = [self.boundary_table(i + 1) for i in range(stages)] if include_tables else []
        savings = self.expected_sample_savings
        rows = ["KSBIN1 study", f"Alternative\t{c.alternative}"]

        def row(*values: str | int | float) -> None:
            rows.append(
                "\t".join(
                    v if isinstance(v, str) else format(float(v), f".{digits}g") for v in values
                )
            )

        for case, index in enumerate(np.ndindex(self.null_probability.shape), start=1):
            rows.append(f"Case {case}")
            row("Null probability", float(self.null_probability[index]))
            row("Alternative probability", float(self.alternative_probability[index]))
            rows.append("Stage decisions")
            row(
                "Stage",
                "Cumulative trials",
                "Reject cutoff",
                "Quit cutoff",
                "H0 reject",
                "Ha reject",
                "H0 quit",
                "Ha quit",
            )
            rejection = c.rejection[(slice(None), *index)]
            quitting = c.quitting[(slice(None), *index)]
            continuation = c.continuation[(slice(None), *index)]
            for i, n in enumerate(c.design.cumulative_trials):
                row(i + 1, n, c.critical[i], (*quits, -1)[i], *rejection[:, i], *quitting[:, i])
            row("Total", "", "", "", *rejection.sum(axis=-1), *quitting.sum(axis=-1))
            rows.append("Single-stage comparison")
            row("Trials", "Reject cutoff", "Significance", "Power")
            row(
                c.design.cumulative_trials[-1],
                self.single_stage_critical,
                float(self.single_stage_significance[index]),
                float(self.single_stage_power[index]),
            )
            row(
                "Multistage minus single-stage",
                "",
                float(self.significance[index] - self.single_stage_significance[index]),
                float(self.power[index] - self.single_stage_power[index]),
            )
            rows.append("Cumulative decisions")
            row(
                "Stage",
                "H0 reject",
                "H0 quit",
                "H0 continue",
                "Ha reject",
                "Ha quit",
                "Ha continue",
            )
            cumrej, cumquit = np.cumsum(rejection, axis=-1), np.cumsum(quitting, axis=-1)
            for i in range(stages):
                row(
                    i + 1,
                    cumrej[0, i],
                    cumquit[0, i],
                    continuation[0, i],
                    cumrej[1, i],
                    cumquit[1, i],
                    continuation[1, i],
                )
            rows.append("Expected observations")
            row(
                "Hypothesis",
                "Given correct decision",
                "Overall",
                "Overall minus correct",
                "Savings versus single-stage",
            )
            correct = [
                c.expected_given_quitting[(0, *index)],
                c.expected_given_rejection[(1, *index)],
            ]
            for h, label in enumerate(["H0", "Ha"]):
                overall = c.expected_sample_size[(h, *index)]
                row(
                    label,
                    correct[h],
                    overall,
                    overall - correct[h],
                    savings[(h, *index)],
                )
            for table in tables:
                rows.append(f"Boundary assistance: stage {table.stage}")
                row("Events", "Stage significance", "Stage power", "Power loss if quit")
                for k in table.events:
                    row(
                        k,
                        table.significance[(*index, k)],
                        table.power[(*index, k)],
                        table.power_loss[(*index, k)],
                    )
        rows.append("Cutoffs are inclusive; -1 disables a boundary. Final nonrejections quit.")
        rows.append("NaN denotes an undefined expectation given a zero-probability decision.")
        if include_tables:
            rows.append(
                "Power loss is absolute reference-completion probability, ignoring future stopping."
            )
        return "\n".join(rows) + "\n"

    def write_report(
        self, path: str | Path, *, digits: int = 6, include_tables: bool = False
    ) -> Path:
        """Write UTF-8 report, replacing path explicitly; propagate I/O errors."""
        content = self.report(digits=digits, include_tables=include_tables)
        path = Path(path)
        path.write_text(content, encoding="utf-8")
        return path


def ksbin1_study(
    cumulative_trials: ArrayLike,
    critical: ArrayLike,
    quit: ArrayLike,
    null_probability: ArrayLike,
    alternative_probability: ArrayLike,
    single_stage_critical: int,
    *,
    alternative: str = "less",
) -> KSBinomialStudy:
    """Compare a multistage design to an explicit fixed-size rejection cutoff.

    The single-stage comparator uses the final planned sample size. Probability
    pairs broadcast; direction and all design boundaries are shared across cases.
    See ksbin1_operating_characteristics for boundary and design constraints.
    """
    p0, pa = np.broadcast_arrays(
        finite(null_probability, "null_probability"),
        finite(alternative_probability, "alternative_probability"),
    )
    c = ksbin1_operating_characteristics(
        cumulative_trials,
        critical,
        quit,
        np.stack([p0, pa]),
        alternative=alternative,
    )
    reference = ksbin1_boundary_table(
        c.design, 1, p0, pa, single_stage_critical, alternative=alternative
    )
    return KSBinomialStudy(
        c,
        p0,
        pa,
        reference.single_stage_critical,
        reference.single_stage_significance,
        reference.single_stage_power,
    )
