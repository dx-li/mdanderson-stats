"""Measure batched MULTI calculations and a large, log-domain Rom recurrence."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy

from mdanderson_stats import multiple_testing, rom_critical_values


def main() -> None:
    pvalues = np.random.default_rng(821).uniform(size=(2000, 100))
    batch_times, scalar_times, rom_times = [], [], []
    for _ in range(3):
        start = time.perf_counter()
        batch = multiple_testing(pvalues, "holm").adjusted_pvalues
        batch_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        rows = np.stack([multiple_testing(p, "holm").adjusted_pvalues for p in pvalues])
        scalar_times.append(time.perf_counter() - start)
        np.testing.assert_allclose(batch, rows, rtol=0, atol=0)
        start = time.perf_counter()
        thresholds = rom_critical_values(2000)
        rom_times.append(time.perf_counter() - start)
        assert np.all(np.isfinite(thresholds))
        assert np.all((thresholds > 0) & (thresholds <= 0.05))
        assert np.all(np.diff(thresholds) >= 0)
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "holm": {
            "workload": "2000 independent testing families, 100 p-values each, NumPy seed 821",
            "batch_seconds": statistics.median(batch_times),
            "row_seconds": statistics.median(scalar_times),
            "speedup": statistics.median(scalar_times) / statistics.median(batch_times),
            "comparison": "One batched call versus 2000 calls to the same Python API",
        },
        "rom": {
            "workload": "2000-test threshold vector at alpha=0.05",
            "seconds": statistics.median(rom_times),
            "smallest_threshold": float(thresholds[0]),
        },
    }
    Path("docs/multiplicity-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
