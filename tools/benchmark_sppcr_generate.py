"""Compare batched SPPCR generation with repeated single-experiment calls."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import sppcr_detection_probabilities, sppcr_generate


def main():
    p = sppcr_detection_probabilities([0.5, 1, 2], [0.1, 0.2, 0.3, 0.4])
    records = []
    for size in [32, 1024]:
        batch_times, separate_times = [], []
        for _ in range(3):
            a, b = np.random.default_rng(321), np.random.default_rng(321)
            start = time.perf_counter()
            batch = sppcr_generate(p, [20, 40, 80], rng=a, replicates=size)
            batch_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            separate = [sppcr_generate(p, [20, 40, 80], rng=b) for _ in range(size)]
            separate_times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(batch.seen, np.concatenate([s.seen for s in separate]))
            np.testing.assert_array_equal(
                batch.unseen, np.concatenate([s.unseen for s in separate])
            )
            assert a.bit_generator.state == b.bit_generator.state
        bt, st = statistics.median(batch_times), statistics.median(separate_times)
        records.append(
            dict(
                replicates=size,
                levels=3,
                alleles=4,
                batch_seconds=bt,
                scalar_seconds=st,
                speedup=st / bt,
            )
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope=(
            "Three-run medians, same Python API and identical samples/RNG state; "
            "not native Fortran speedup"
        ),
        cases=records,
    )
    Path("docs/sppcr-generation-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
