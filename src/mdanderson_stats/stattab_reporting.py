"""Readable STATTAB help and bounded text tables."""

import numpy as np

from .cdflib_console import _positive
from .stattab_results import STATTAB_DISTRIBUTIONS, STATTABResult

_DESCRIPTIONS = {
    "beta": (
        "Beta",
        "CDF = integral from 0 to x of t^(a-1)*(1-t)^(b-1)/B(a,b).",
        "x: coordinate; cx: 1-x; a,b: positive shape parameters.",
    ),
    "binomial": (
        "Binomial",
        "At integer counts: CDF = sum from k=0 to s of choose(n,k)*pr^k*cpr^(n-k).",
        "s: successes; n: trials; pr: success chance; cpr: failure chance.",
    ),
    "neg_binomial": (
        "Negative binomial",
        "At integer counts: term = choose(f+s-1,f)*pr^s*cpr^f.",
        "f: failures before s successes; s: required successes; pr/cpr: success/failure chance.",
    ),
    "chisq": (
        "Chi-square",
        "CDF = P(df/2,x/2), where P is the regularized lower incomplete gamma.",
        "x: nonnegative coordinate; df: degrees of freedom.",
    ),
    "nc_chisq": (
        "Noncentral chi-square",
        "CDF = Poisson(pnonc/2) mixture of chi-square CDFs with df+2*j degrees.",
        "x: nonnegative coordinate; df: degrees of freedom; pnonc: noncentrality.",
    ),
    "f": (
        "F",
        "F = (U/dfn)/(V/dfd), for independent central chi-square U and V.",
        "f: nonnegative coordinate; dfn/dfd: numerator/denominator degrees of freedom.",
    ),
    "nc_f": (
        "Noncentral F",
        "F = (U/dfn)/(V/dfd); U is noncentral chi-square with noncentrality pnonc.",
        "f: coordinate; dfn/dfd: numerator/denominator degrees; V is central chi-square.",
    ),
    "gamma": (
        "Gamma",
        "Density = rate^shape*x^(shape-1)*exp(-rate*x)/Gamma(shape).",
        "x: nonnegative coordinate; rate: inverse scale (source A); shape: source B.",
    ),
    "normal": (
        "Normal",
        "CDF = Phi((x-mean)/sd). Two-sided p = 2*min(cum,ccum).",
        "x: coordinate; mean: location; sd: positive standard deviation.",
    ),
    "poisson": (
        "Poisson",
        "At integer s: term = exp(-mean)*mean^s/s!.",
        "s: event count; mean: expected number of events; zero mean is allowed for forward calls.",
    ),
    "t": (
        "Student t",
        "T = Z/sqrt(V/df), with standard normal Z and independent chi-square V.",
        "t: coordinate; df: degrees of freedom. Two-sided p = 2*min(cum,ccum).",
    ),
    "nc_t": (
        "Noncentral t",
        "T = (Z+pnonc)/sqrt(V/df), with standard normal Z and independent chi-square V.",
        "t: coordinate; df: degrees of freedom; pnonc: nonnegative normal displacement.",
    ),
}
_GRAMMAR = (
    "Enter one value per input position; use exactly one ? for the computed parameter.\n"
    "Each complementary pair needs one .; T selects a list; = reuses the last completed row.\n"
    "Numbers accept E/D exponents. Spaces, tabs and commas separate fields; # starts a comment.\n"
    "HELP displays this help; a blank line returns to the distribution menu.\n"
    "cum is the lower tail; ccum is its directly evaluated upper complement."
)


def stattab_help(distribution: str | None = None) -> str:
    """Return application grammar, or formula/parameter help for one distribution."""
    if distribution is None:
        menu = "\n".join(
            f"{d.menu:2d} {_DESCRIPTIONS[name][0]}" for name, d in STATTAB_DISTRIBUTIONS.items()
        )
        return "STATTAB statistical tables\n" + menu + "\n 0 Exit\n\n" + _GRAMMAR
    if not isinstance(distribution, str) or distribution not in STATTAB_DISTRIBUTIONS:
        raise ValueError("unknown STATTAB distribution")
    descriptor = STATTAB_DISTRIBUTIONS[distribution]
    title, formula, parameters = _DESCRIPTIONS[distribution]
    lines = [
        title,
        "Input order: " + " ".join(descriptor.parameters),
        formula,
        parameters,
        "Computed groups: " + ", ".join("/".join(g) for g in descriptor.groups),
        _GRAMMAR,
    ]
    if distribution in ("binomial", "neg_binomial", "poisson"):
        lines.append(
            "CDFs use continuous count extensions. Forward terms truncate counts; "
            "count inverses show floor/floor+1 rows."
        )
    if distribution in ("f", "nc_f"):
        lines.append(
            "This application does not invert F degrees of freedom. "
            "ccum is the many-sided probability column."
        )
    if distribution == "neg_binomial":
        lines.append("Zero required successes give unit mass at zero failures, including pr=0.")
    if distribution == "poisson":
        lines.append("At zero mean, the mass is one at zero events and zero at positive counts.")
    if distribution == "chisq":
        lines.append("ccum is the many-sided probability column.")
    if distribution == "gamma":
        lines.append("Input is rate before shape; output is shape before rate. Rate is not scale.")
    if distribution == "nc_t":
        lines.append(
            "DF inversion can have multiple roots. "
            "STATTABSession df_bracket can select a sign-changing interval."
        )
    return "\n".join(lines)


def format_stattab_result(
    result: STATTABResult,
    *,
    precision: int = 12,
    max_rows: int = 10000,
    max_output: int = 1000000,
) -> str:
    """Format primary and neighbor rows with explicit unavailable candidates.

    Row numbers are one-based flattened input indices. precision is significant
    digits (1..17); text is presentation, while result arrays retain float64.
    Limits include unavailable neighbor rows and the complete returned text.
    """
    if not isinstance(result, STATTABResult):
        raise ValueError("result must be a STATTABResult")
    _positive(precision, "precision")
    if precision > 17:
        raise ValueError("precision must not exceed 17")
    _positive(max_rows, "max_rows")
    _positive(max_output, "max_output")
    values = result.values.reshape(-1, len(result.columns))
    if len(values) * (1 + len(result.neighbors)) > max_rows:
        raise ValueError("formatted table exceeds max_rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot format nonfinite STATTAB values")
    lines: list[str] = []
    length = 0

    def append(line: str) -> None:
        nonlocal length
        length += len(line) + (1 if lines else 0)
        if length > max_output:
            raise ValueError("formatted table exceeds max_output")
        lines.append(line)

    append(f"{result.distribution}: computed {'/'.join(result.computed)}")
    width = max(precision + 7, max(map(len, result.columns)))
    append("row kind           " + " ".join(f"{key:>{width}}" for key in result.columns))
    lookups = [
        dict(zip(n.source_indices, range(len(n.source_indices)), strict=True))
        for n in result.neighbors
    ]
    for i, row in enumerate(values):
        append(f"{i + 1:3d} {'result':14s} " + " ".join(f"{v:>{width}.{precision}g}" for v in row))
        for neighbor, lookup in zip(result.neighbors, lookups, strict=True):
            if i in lookup:
                row = neighbor.values[lookup[i]]
                append(
                    f"{i + 1:3d} {neighbor.kind:14s} "
                    + " ".join(f"{v:>{width}.{precision}g}" for v in row)
                )
            else:
                append(
                    f"{i + 1:3d} {neighbor.kind}: unavailable "
                    "(outside count domain or next integer not representable)"
                )
    if not len(values):
        append("(empty table)")
    return "\n".join(lines)
