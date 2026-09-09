"""Measure independent reverse-communication searches and callback work."""

import json
import math
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

from mdanderson_stats import dcdflib_support as legacy


def main():
    cases = []
    for mode in ["interval", "step"]:
        for count in [100, 1000]:
            times = []
            counts = []
            for _ in range(3):
                evaluations = 0
                start = perf_counter()
                for i in range(count):
                    target = 0.1 + 3.8 * (i + 0.5) / count
                    if mode == "interval":
                        state = legacy.dstzr(0, 2, 1e-12, 0)
                        advance = legacy.dzror
                        result = advance(state)
                    else:
                        state = legacy.dstinv(0, 2, 0.1, 0.1, 5, 1e-12, 0)
                        advance = legacy.dinvr
                        result = advance(state, initial=1)
                    while result.status == 1:
                        result = advance(state, result.x * result.x - target)
                    if result.status != 0 or abs(result.x - math.sqrt(target)) > 1e-12:
                        raise AssertionError("Benchmark root failed verification")
                    evaluations += result.evaluations
                times.append(perf_counter() - start)
                counts.append(evaluations)
            assert len(set(counts)) == 1
            cases.append(
                dict(mode=mode, searches=count, seconds=median(times), evaluations=counts[0])
            )
    report = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        repetitions=3,
        workload=(
            "Independent scalar reverse-communication searches; not a native speedup comparison"
        ),
        cases=cases,
    )
    Path("docs/dcdflib-root-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
