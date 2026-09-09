"""Measure validated lexer and ASCII translation throughput without native I/O."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

from mdanderson_stats import lower_case_string, qlex


def main():
    cases = []
    for size in (1000, 10000, 100000):
        inputs = [
            ("identifier", "a" * size, 1),
            ("commands", "x=-1.25; " * size, 4 * size),
            ("case_conversion", "AbC ß " * size, None),
        ]
        for operation, text, count in inputs:
            times = []
            for _ in range(3):
                start = perf_counter()
                result = lower_case_string(text) if count is None else tuple(qlex(text))
                times.append(perf_counter() - start)
                if count is None:
                    assert result == "abc ß " * size
                else:
                    assert len(result) == count
                    if operation == "identifier":
                        assert result[0].text == text
                    else:
                        assert [(t.kind, t.text) for t in result[:4]] == [
                            ("ID", "x"),
                            ("OP", "="),
                            ("RL", "-1.25"),
                            ("DL", ";"),
                        ]
            elapsed = median(times)
            cases.append(
                dict(
                    operation=operation,
                    characters=len(text),
                    tokens=count,
                    seconds=elapsed,
                    characters_per_second=len(text) / elapsed,
                )
            )
    data = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        repetitions=3,
        comparison="Validated end-to-end throughput; no native speedup claim",
        cases=cases,
    )
    Path("docs/cdflib-strings-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
