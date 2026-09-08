"""ONESAMPLE's four menu calculations as vectorized calls with readable output."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite
from .intervals import binomial_interval, poisson_interval
from .onesample import binomial_test, poisson_test

_CALCULATIONS = (
    "binomial_confidence",
    "binomial_test",
    "poisson_confidence",
    "poisson_test",
)


@dataclass(frozen=True)
class OneSampleResult:
    calculation: str
    events: FloatArray
    trials: FloatArray | None
    exposure: FloatArray
    parameter: FloatArray
    estimate: FloatArray
    lower: FloatArray | None
    upper: FloatArray | None
    p_less: FloatArray | None
    p_greater: FloatArray | None
    legacy_cutoffs: bool

    def report(self, *, digits: int = 6) -> str:
        """Return a TSV report; broadcast cases are flattened in C order."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        columns = {"Events": self.events}
        if self.trials is not None:
            columns.update({"Non-events": self.trials - self.events, "Trials": self.trials})
        else:
            columns["Exposure"] = self.exposure
        label = (
            "Confidence"
            if self.lower is not None
            else ("Null probability" if self.trials is not None else "Null rate")
        )
        columns[label] = self.parameter
        columns["Estimate"] = self.estimate
        if self.lower is not None and self.upper is not None:
            columns.update({"Lower": self.lower, "Upper": self.upper})
        elif self.p_less is not None and self.p_greater is not None:
            columns.update({"P (less)": self.p_less, "P (greater)": self.p_greater})
        rows = [f"ONESAMPLE {self.calculation}", "\t".join(columns)]
        rows += [
            "\t".join(format(float(value), f".{digits}g") for value in row)
            for row in zip(*(a.ravel() for a in columns.values()), strict=True)
        ]
        if self.p_less is not None:
            rows.append(
                "One-sided tails include the observed count; no two-sided p-value is implied."
            )
            if self.legacy_cutoffs:
                rows.append("Original small-probability cutoffs enabled.")
        return "\n".join(rows) + "\n"

    def write_report(self, path: str | Path, *, digits: int = 6) -> Path:
        """Write the UTF-8 report, replacing path explicitly; propagate I/O errors."""
        path = Path(path)
        content = self.report(digits=digits)
        path.write_text(content, encoding="utf-8")
        return path


def one_sample(
    calculation: str,
    events: ArrayLike,
    parameter: ArrayLike,
    *,
    second: ArrayLike | None = None,
    entry: str = "failures",
    exposure: ArrayLike = 1,
    legacy_cutoffs: bool = False,
) -> OneSampleResult:
    """Run binomial_confidence/binomial_test/poisson_confidence/poisson_test.

    parameter is the confidence fraction for confidence calculations, otherwise
    the hypothesized probability or rate. Binomial second is failures by default,
    or total trials with entry='trials'. The original interface's input limits
    apply; invalid totals and fractional counts raise instead of being truncated.
    Independent cases broadcast. Correct mathematical tails are the default.
    """
    if calculation not in _CALCULATIONS:
        raise ValueError(f"calculation must be one of {_CALCULATIONS}")
    if entry not in ("failures", "trials") or not isinstance(legacy_cutoffs, bool):
        raise ValueError("entry must be failures/trials and legacy_cutoffs must be boolean")
    binomial, confidence = calculation.startswith("binomial"), calculation.endswith("confidence")
    if confidence and legacy_cutoffs:
        raise ValueError("legacy_cutoffs applies only to test calculations")
    k, value, time = (
        count(events, "events"),
        finite(parameter, "parameter"),
        finite(exposure, "exposure"),
    )
    if np.any(k > 1e9):
        raise ValueError("Entered counts must not exceed 1e9")
    maximum = 1 - 1e-10 if confidence or binomial else 1e9
    if np.any((value < 1e-10) | (value > maximum)):
        raise ValueError(f"parameter must lie in [1e-10,{maximum}]")
    n = None
    if binomial:
        if second is None:
            raise ValueError("Binomial calculations require a second count")
        second = count(second, "second")
        if np.any(second > 1e9):
            raise ValueError("Entered counts must not exceed 1e9")
        if np.any(time != 1):
            raise ValueError("Exposure applies only to Poisson calculations")
        k, second, value, time = np.broadcast_arrays(k, second, value, time)
        n = k + second if entry == "failures" else second
        if np.any((n <= 0) | (k > n)):
            raise ValueError("Require a positive total at least as large as successes")
    else:
        if second is not None or entry != "failures":
            raise ValueError(
                "Poisson calculations do not use a second count or binomial entry mode"
            )
        if np.any((time < 1e-10) | (time > 1e9)):
            raise ValueError("exposure must lie in [1e-10,1e9]")
        k, value, time = np.broadcast_arrays(k, value, time)
    lower = upper = less = greater = None
    estimate = k / n if n is not None else k / time
    if confidence:
        if n is not None:
            lower, upper = binomial_interval(k, n, value)
        else:
            lower, upper = poisson_interval(k, value, time)
    else:
        result = (
            binomial_test(k, n, value, legacy_cutoffs=legacy_cutoffs)
            if n is not None
            else poisson_test(k, value, exposure=time, legacy_cutoffs=legacy_cutoffs)
        )
        less, greater = result.p_less, result.p_greater
    return OneSampleResult(
        calculation, k, n, time, value, estimate, lower, upper, less, greater, legacy_cutoffs
    )
