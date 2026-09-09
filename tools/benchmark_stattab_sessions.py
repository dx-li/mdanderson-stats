"""Measure a STATTAB table request against repeated scalar session requests."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import STATTABResult, STATTABSession


def main():
    records = []
    for name, template in [("normal", "{} 0 1 ? ."), ("poisson", "{} 3 ? .")]:
        for size in [100, 1000]:
            data = np.linspace(0, 3, size)
            batch_times = []
            scalar_times = []
            for _ in range(3):
                batched = STATTABSession(name, max_table_size=size)
                scalar = STATTABSession(name)
                start = time.perf_counter()
                output = batched.execute(template.format("T"), table=data)
                batch_times.append(time.perf_counter() - start)
                assert isinstance(output, STATTABResult)
                start = time.perf_counter()
                rows = []
                for value in data:
                    result = scalar.execute(template.format(repr(float(value))))
                    assert isinstance(result, STATTABResult)
                    rows.append(result.values)
                scalar_times.append(time.perf_counter() - start)
                np.testing.assert_array_equal(output.values, np.stack(rows))
                assert batched.previous == scalar.previous
            a, b = statistics.median(batch_times), statistics.median(scalar_times)
            records.append(
                dict(distribution=name, rows=size, batch_seconds=a, scalar_seconds=b, speedup=b / a)
            )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope="Three-run medians; same-Python session batching, not native Fortran",
        cases=records,
    )
    Path("docs/stattab-sessions-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
