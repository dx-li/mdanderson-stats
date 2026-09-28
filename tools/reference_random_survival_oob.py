"""Native RF-SRC concordance fixtures, using the pinned unchanged C kernel.

This calls the existing small reference harness, not a native forest engine.
The original source and compiled library remain in ignored research/raw.
"""

from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

from reference_random_survival_forest import PIN, doubles, library

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    started = time.perf_counter()
    lib = library()
    cases = [
        ("ordered", [1, 2, 3], [1, 1, 1], [3, 2, 1], [1, 1, 1]),
        ("reversed", [1, 2, 3], [1, 1, 1], [1, 2, 3], [1, 1, 1]),
        ("mortality_ties", [1, 2, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ("event_censor_tie", [1, 1, 2], [1, 0, 0], [2, 1, 0], [1, 1, 1]),
        ("event_tie_equal", [1, 1], [1, 1], [1, 1], [1, 1]),
        ("event_tie_unequal", [1, 1], [1, 1], [1, 2], [1, 1]),
        ("event_tie_risk_exact_epsilon", [1, 1], [1, 1], [0, 1e-9], [1, 1]),
        ("event_tie_risk_below_epsilon", [1, 1], [1, 1], [0, 0.999e-9], [1, 1]),
        ("ordinary_risk_exact_epsilon", [1, 2], [1, 1], [1e-9, 0], [1, 1]),
        ("ordinary_risk_above_epsilon", [1, 2], [1, 1], [1.001e-9, 0], [1, 1]),
        ("time_exact_epsilon", [0, 1e-9], [1, 1], [2, 1], [1, 1]),
        ("time_above_epsilon", [0, 1.001e-9], [1, 1], [2, 1], [1, 1]),
        ("zero_contributor", [1, 2, 3], [1, 1, 1], [3, 99, 1], [1, 0, 1]),
        ("all_censored", [1, 2, 3], [0, 0, 0], [3, 2, 1], [1, 1, 1]),
        ("no_contributors", [1, 2, 3], [1, 1, 1], [3, 2, 1], [0, 0, 0]),
        ("later_failure_only", [1, 2], [0, 1], [3, 1], [1, 1]),
    ]
    results = []
    for name, times, events, risk, count in cases:
        value = lib.getConcordanceIndex(
            1, len(times), doubles(times), doubles(events), doubles(risk), doubles(count)
        )
        results.append(
            {
                "case": name,
                "time": times,
                "event": events,
                "mortality": risk,
                "contributors": count,
                "concordance_error": None if math.isnan(value) else value,
            }
        )
    target = ROOT / "tests/fixtures/random-survival-oob-concordance.json"
    target.write_text(json.dumps({"native_revision": PIN, "cases": results}, indent=2) + "\n")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "cases": len(results),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_mib": usage.ru_maxrss / 1024**2,
                "swaps": usage.ru_nswap,
            }
        )
    )


if __name__ == "__main__":
    main()
