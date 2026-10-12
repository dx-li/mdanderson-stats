"""Independent exact-sum check for the supplied native enrollment trace grid.

This checks these fixtures only; it does not claim exact-sum comparison reproduces
x87 rounding for every possible floating-point calendar input.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np

x = json.loads(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/prt-desktop-enrollment.json").read_text()
)
steps = 0
for case in x["cases"]:
    enrolled = []
    dose_list = []
    index = 0
    cohort = 3
    for (suspend, dose), expected in zip(case["decisions"], case["result"]):
        if suspend:
            index += 1
        else:
            enrolled.extend(case["arrivals"][index : index + cohort])
            dose_list.extend([dose] * cohort)
            index += cohort
        n = len(enrolled)
        clock = (
            case["arrivals"][index]
            if n < case["maximum"]
            else max(a + min(t, 10.0) for a, t in zip(enrolled[-3:], case["delays"][n - 3 : n]))
        )
        observed = []
        censored = []
        intervals = []
        for a, t in zip(enrolled, case["delays"]):
            event = Fraction(clock) >= Fraction(a) + Fraction(t) and t <= 10
            observed.append(t if event else clock - a)
            censored.append(int(not event))
            o = observed[-1]
            daily = 11 if o > 10 else int(o) if 1 <= o < 10 else 0
            intervals.append(
                int(np.searchsorted(case["endpoints"], daily, side="right")) + int(event)
            )
        assert (n, index, clock, enrolled, dose_list, censored, intervals) == (
            expected["enrolled"],
            expected["candidate_index"],
            expected["clock"],
            expected["arrivals"],
            expected["doses"],
            expected["censored"],
            expected["interval"],
        ), (case, expected)
        assert observed == expected["observed_time"]
        if cohort == 3 and case["maximum"] - n < 3:
            cohort = case["maximum"] - n
        steps += 1
print(len(x["cases"]), "cases", steps, "trace steps independently matched")
