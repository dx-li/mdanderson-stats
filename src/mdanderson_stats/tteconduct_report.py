"""Static, portable HTML reports for TTEConduct boundary analyses."""

import os
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path

from numpy.typing import ArrayLike

from .tteconduct import (
    TTEConductBoundaryTable,
    TTEConductDesign,
    tteconduct_boundary_table,
    tteconduct_design,
)


def _label(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _number(value: float) -> str:
    """Format finite values with enough digits for float64 round trips."""
    return format(value, ".17g")


@dataclass(frozen=True)
class TTEConductReport:
    """Immutable design-and-boundary snapshot for static HTML output.

    The report is a rendered result, not an editable analysis session. Opening
    the saved HTML displays this snapshot and does not rerun its calculations.
    """

    design: TTEConductDesign
    time_unit: str
    title: str
    boundaries: TTEConductBoundaryTable

    def to_html(self) -> str:
        """Return a self-contained HTML report with full-precision values."""
        unit_label = _label(self.time_unit, "time_unit", 40)
        time_unit = escape(unit_label)
        title = escape(_label(self.title, "title", 120))
        design = self.design
        if not isinstance(design, TTEConductDesign):
            raise ValueError("design must be a TTEConductDesign")
        if not isinstance(self.boundaries, TTEConductBoundaryTable):
            raise ValueError("boundaries must be a TTEConductBoundaryTable")

        parameters = (
            ("Standard prior shape αS", design.alpha_standard),
            ("Standard prior scale βS", design.beta_standard),
            ("Experimental prior shape αE", design.alpha_experimental),
            ("Experimental prior scale βE", design.beta_experimental),
            ("Required additive improvement δ", design.delta),
            ("Futility cutoff c", design.cutoff),
            ("Maximum patients m", design.max_patients),
            ("Boundary-search cap", design.max_total_time),
            ("Absolute quadrature tolerance", design.absolute_tolerance),
        )
        parameter_rows = "\n".join(
            f'<tr><th scope="row">{escape(label)}</th><td>{_number(float(value))}</td></tr>'
            for label, value in parameters
        )
        boundary_rows: list[str] = []
        for row in self.boundaries.boundaries:
            if row.beyond_cap:
                boundary_text = "Not resolved within search cap"
            else:
                boundary_text = f"{_number(row.minimum_total_time)} {unit_label}"
            meaning = (
                "Continuation criterion met at zero exposure"
                if not row.beyond_cap and row.minimum_total_time == 0
                else (
                    "Not resolved within cap; this does not show continuation is impossible"
                    if row.beyond_cap
                    else "Minimum continuous total time on test to continue"
                )
            )
            boundary_rows.append(
                "<tr>"
                f"<td>{row.events:d}</td>"
                f"<td>{escape(boundary_text)}</td>"
                f"<td>{_number(row.probability)}</td>"
                f"<td>{_number(row.probability_error)}</td>"
                f"<td>{_number(row.residual)}</td>"
                f"<td>{escape(meaning)}</td>"
                "</tr>"
            )
        table_body = "\n".join(boundary_rows)
        return (
            "<!doctype html>\n"
            '<html lang="en"><head><meta charset="utf-8">'
            f"<title>{title}</title>"
            "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2em auto;"
            "padding:0 1em;color:#202124}table{border-collapse:collapse;margin:1em 0 2em}"
            "th,td{border:1px solid #b8bec5;padding:.45em .65em;text-align:left}"
            "caption{text-align:left;font-weight:700;margin-bottom:.5em}</style></head><body>\n"
            f"<h1>{title}</h1>\n"
            "<p>Exponential survival model with inverse-gamma priors on mean survival. "
            "The futility rule is the strict comparison "
            "P(μE &gt; μS + δ | data) &lt; c; reaching the maximum patient count also "
            "stops accrual. The values below are continuous Python boundaries in the "
            f"caller’s time unit ({time_unit}); no conversion to days or rounding is applied.</p>\n"
            "<table><caption>Design inputs and Python search settings</caption>"
            "<tbody>" + parameter_rows + "</tbody></table>\n"
            "<table><caption>Minimum total time on test by event count</caption>"
            '<thead><tr><th scope="col">Observed events</th>'
            '<th scope="col">Boundary</th><th scope="col">Probability</th>'
            '<th scope="col">Quadrature error estimate</th>'
            '<th scope="col">Probability residual (probability − c)</th>'
            '<th scope="col">Interpretation</th></tr></thead>'
            f"<tbody>{table_body}</tbody></table>\n"
            "<p>A zero boundary means the continuation criterion is met at zero exposure "
            "for that event count. A capped row reports the probability and residual at "
            "the search cap; it is unresolved within that cap, not proof continuation is "
            "impossible. The probability comparison is strict, and the maximum-patient "
            "rule can stop a trial even when a boundary is met.</p>\n"
            "<p>This is a static report snapshot from the independent Python implementation. "
            "Reopening the file displays the saved report; it does not restore an editable "
            "design session or recompute the boundaries.</p>\n"
            "</body></html>\n"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically write the rendered report as UTF-8 HTML."""
        destination = Path(path)
        content = self.to_html()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination


def tteconduct_report(
    design: TTEConductDesign,
    *,
    time_unit: str,
    events: int | ArrayLike | None = None,
    title: str = "TTEConduct stopping boundaries",
) -> TTEConductReport:
    """Create a report by calculating boundaries from one validated design.

    ``time_unit`` is a display label only; it does not transform inputs or
    results. The Python-only search cap and quadrature tolerance are reported
    alongside the seven inputs named in the TTEConduct guide.
    """
    if not isinstance(design, TTEConductDesign):
        raise TypeError("design must be a TTEConductDesign")
    unit_label = _label(time_unit, "time_unit", 40)
    title_label = _label(title, "title", 120)
    validated_design = tteconduct_design(
        design.alpha_standard,
        design.beta_standard,
        design.alpha_experimental,
        design.beta_experimental,
        design.delta,
        design.cutoff,
        design.max_patients,
        max_total_time=design.max_total_time,
        absolute_tolerance=design.absolute_tolerance,
    )
    boundaries = tteconduct_boundary_table(validated_design, events=events)
    return TTEConductReport(validated_design, unit_label, title_label, boundaries)
