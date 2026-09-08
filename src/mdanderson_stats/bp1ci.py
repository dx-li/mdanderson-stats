"""BP1CI's percentage/entry modes and readable interval output."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite
from .intervals import bp1ci_binomial_interval, bp1ci_poisson_interval


@dataclass(frozen=True)
class BP1CIResult:
    distribution: str
    confidence_percent: FloatArray
    events: FloatArray
    failures: FloatArray | None
    trials: FloatArray | None
    exposure: FloatArray
    estimate: FloatArray
    lower: FloatArray
    upper: FloatArray

    def report(self, *, digits: int = 6) -> str:
        """Return a tab-separated table with one row per broadcast result."""
        if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
            raise ValueError("digits must be an integer from 1 to 17")
        columns = {"Confidence (%)": self.confidence_percent, "Events": self.events}
        if self.trials is not None and self.failures is not None:
            columns.update({"Failures": self.failures, "Trials": self.trials})
        else:
            columns["Exposure"] = self.exposure
        columns.update({"Lower": self.lower, "Estimate": self.estimate, "Upper": self.upper})
        rows = [f"BP1CI {self.distribution}", "\t".join(columns)]
        rows += [
            "\t".join(format(float(v), f".{digits}g") for v in row)
            for row in zip(*(value.ravel() for value in columns.values()), strict=True)
        ]
        if self.distribution == "poisson":
            rows.append("Uses the original BP1CI lower-bound formula; not the Garwood interval.")
        return "\n".join(rows) + "\n"


def bp1ci(
    events: ArrayLike,
    second: ArrayLike | None = None,
    *,
    distribution: str = "binomial",
    entry: str = "failures",
    confidence_percent: ArrayLike = 95,
    exposure: ArrayLike = 1,
) -> BP1CIResult:
    """Calculate with BP1CI's input conventions, broadcasting independent cases.

    Binomial second means failures (default) or total trials with entry='trials'.
    Each entered binomial number lies in [0,1e10], as in the program. Confidence
    percentages lie in [0.1,99.9999]. Fractional data are allowed, but a zero
    binomial total or successes>trials raises instead of printing undefined data.
    Poisson uses the original lower-tail formula and permits exposure scaling.
    """
    if distribution not in ("binomial", "poisson") or entry not in ("failures", "trials"):
        raise ValueError("distribution must be binomial/poisson; entry must be failures/trials")
    level = finite(confidence_percent, "confidence_percent")
    if np.any((level < 0.1) | (level > 99.9999)):
        raise ValueError("BP1CI confidence_percent must lie in [0.1,99.9999]")
    events, time = finite(events, "events"), finite(exposure, "exposure")
    if np.any(time <= 0):
        raise ValueError("exposure must be positive")
    failures = trials = None
    if distribution == "binomial":
        if second is None:
            raise ValueError("Binomial input requires a second value")
        second = finite(second, "second")
        if np.any((events < 0) | (events > 1e10)) or np.any((second < 0) | (second > 1e10)):
            raise ValueError("Each binomial input must lie in [0,1e10]")
        if np.any(time != 1):
            raise ValueError("Exposure scaling applies only to Poisson input")
        events, second, level, time = np.broadcast_arrays(events, second, level, time)
        trials = events + second if entry == "failures" else second
        failures = trials - events
        if np.any((trials <= 0) | (failures < 0)):
            raise ValueError("Binomial total must be positive and at least successes")
        lower, upper = bp1ci_binomial_interval(events, trials, level / 100)
        estimate = events / trials
    else:
        if second is not None:
            raise ValueError("Poisson input does not take a second count")
        events, level, time = np.broadcast_arrays(events, level, time)
        lower, upper = bp1ci_poisson_interval(events, level / 100, time)
        estimate = events / time
    return BP1CIResult(distribution, level, events, failures, trials, time, estimate, lower, upper)
