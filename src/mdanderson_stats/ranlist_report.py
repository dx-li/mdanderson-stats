"""Printable RANLIST summaries and paginated treatment lists."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count
from .ranlist_session import RanlistSession


def ranlist_summary(session: RanlistSession) -> str:
    """Describe the list and enrolled counts without generating assignments."""
    if not isinstance(session, RanlistSession):
        raise ValueError("session must be a RanlistSession")
    spec = session.specification
    lines = [*spec.title, "", "RANLIST parameters", f"Seeds: {spec.seed[0]}, {spec.seed[1]}"]
    if spec.phrase:
        lines.append(f"Seed phrase (metadata): {spec.phrase}")
    lines.extend(
        [
            f"Randomization: {'restricted' if spec.restricted else 'unrestricted'}",
            f"Algorithm: {'archived Fortran' if spec.legacy else 'modern'}",
        ]
    )
    if spec.restricted:
        low, high = spec.balance
        lines.append(f"Natural block size: {int(sum(spec.weights))}")
        lines.append(
            f"Balance multiplier: {low}" if low == high else f"Balance multiplier: {low}..{high}"
        )
        if low != high:
            lines.append(
                "Multiplier selected once per stratum"
                if spec.legacy
                else "Multiplier redrawn at each refill"
            )
    lines.extend(
        [
            "",
            "Treatment  Name                            Count"
            if spec.restricted
            else "Treatment  Name                            Relative weight",
        ]
    )
    for i, (name, weight) in enumerate(zip(spec.treatments, spec.weights, strict=True), 1):
        lines.append(f"{i:9d}  {name:30}  {weight:.17g}")
    lines.extend(["", "Stratum  Name                            Enrolled"])
    for i, (name, enrolled) in enumerate(
        zip(spec.strata, session.current_patients, strict=True), 1
    ):
        lines.append(f"{i:7d}  {name:30}  {enrolled}")
    return "\n".join(lines) + "\n"


def ranlist_report(
    session: RanlistSession, patients: ArrayLike | None = None, *, max_rows: int = 100_000
) -> str:
    """Print enrolled patients, or specified counts per stratum, without enrollment.

    A scalar count applies to every stratum; a vector specifies each separately.
    Zero omits that stratum's assignment pages but retains it in the summary.
    Form feeds separate pages. Each page has 27 minus title-line-count rows,
    matching GENLST's 60-line pagination. Labels and numbers are not truncated.
    The total row cap is checked before generating or formatting assignments.
    """
    summary = ranlist_summary(session)
    if (
        isinstance(max_rows, (bool, np.bool_))
        or not isinstance(max_rows, (int, np.integer))
        or max_rows < 1
    ):
        raise ValueError("max_rows must be a positive integer")
    spec = session.specification
    counts = count(session.current_patients if patients is None else patients, "patients")
    if counts.ndim == 0:
        counts = np.full(len(spec.strata), counts)
    if counts.shape != (len(spec.strata),):
        raise ValueError("patients must be a scalar or one count per stratum")
    sizes = tuple(map(int, counts))
    if sum(sizes) > max_rows:
        raise ValueError("requested report exceeds max_rows")
    lines = [summary.rstrip(), "", "Listed patients per stratum: " + ", ".join(map(str, sizes))]
    page_size = 27 - len(spec.title)
    for stream, size in enumerate(sizes, 1):
        if size == 0:
            continue
        assignments = spec.allocate(np.arange(1, size + 1), strata=stream)
        for offset in range(0, size, page_size):
            page = offset // page_size + 1
            name = spec.strata[stream - 1]
            lines.extend(
                [
                    "\f",
                    *spec.title,
                    "",
                    f"STRATUM {stream}" + (f" ({name})" if name else "") + f"    PAGE {page}",
                    "",
                    "PATIENT   TREATMENT  TREATMENT NAME                  PATIENT INFORMATION",
                    "",
                ]
            )
            for index in range(offset, min(offset + page_size, size)):
                treatment = int(assignments.treatments[index])
                lines.extend(
                    [
                        f"{index + 1:9d} {treatment:9d}  {spec.treatments[treatment - 1]:30}  "
                        + "." * 29,
                        "",
                    ]
                )
    return "\n".join(lines) + "\n"
