"""Independent integer/Fraction oracle for native x87 calendar fixtures.

This reference checker is not a public patient-conduct or simulation API.
"""

import math
from fractions import Fraction

import numpy as np


def add_x87(a: float, b: float) -> Fraction:
    """Round sum of finite nonnegative binary64 inputs to startup x87's 53-bit significand."""
    value = Fraction(a) + Fraction(b)
    if not value:
        return value
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    unit = exponent - 52
    scale = Fraction(2**unit) if unit >= 0 else Fraction(1, 2**-unit)
    quotient = value / scale
    q, r = divmod(quotient.numerator, quotient.denominator)
    if 2 * r > quotient.denominator or (2 * r == quotient.denominator and q % 2):
        q += 1
    return q * scale


def observe(arrivals, delays, endpoints, clock):
    window = endpoints[-1]
    observed = []
    censored = []
    interval = []
    for a, t in zip(arrivals, delays):
        event = Fraction(clock) >= add_x87(a, t) and t <= window
        elapsed = t if event else clock - a
        daily = (
            window + 1 if elapsed > window else math.floor(elapsed) if 1 <= elapsed < window else 0
        )
        observed.append(elapsed)
        censored.append(int(not event))
        interval.append(int(np.searchsorted(endpoints, daily, side="right")) + int(event))
    return observed, censored, interval


def final_clock(arrivals, delays, window):
    return float(max(add_x87(a, min(t, window)) for a, t in zip(arrivals[-3:], delays[-3:])))


if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tests/fixtures"
    cases = json.loads((root / "prt-desktop-calendar-precision.json").read_text())["cases"]
    for case in cases:
        observed, censored, _ = observe(
            [case["arrival"]], [case["delay"]], [25.0, 50.0, 100.0], case["clock"]
        )
        assert (observed[0], censored[0]) == (case["observed_time"], case["censored"])
    paths = json.loads((root / "prt-desktop-enrollment.json").read_text())["cases"]
    steps = 0
    for case in paths:
        for row in case["result"]:
            delays = case["delays"][: row["enrolled"]]
            assert observe(row["arrivals"], delays, case["endpoints"], row["clock"]) == (
                row["observed_time"],
                row["censored"],
                row["interval"],
            )
            if row["enrolled"] == case["maximum"]:
                assert final_clock(row["arrivals"], delays, case["endpoints"][-1]) == row["clock"]
            steps += 1
    print(f"{len(cases)} native x87 precision cases and {steps} enrollment steps match")
