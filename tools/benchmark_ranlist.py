"""Compare batched RANLIST allocations with repeated identical Python calls."""

import json
import platform
from pathlib import Path

import numpy as np
from benchmark_numerics import measure

from mdanderson_stats import ranlist_restricted, ranlist_unrestricted


def main():
    results = {}
    for name, function, size in [
        ("unrestricted", ranlist_unrestricted, 10000),
        ("restricted_legacy", lambda p, w: ranlist_restricted(p, w, legacy=True), 2000),
        ("restricted_modern", ranlist_restricted, 500),
    ]:
        patients = np.arange(1, size + 1)
        results[name] = measure(
            lambda: function(patients, [1, 2, 3]).treatments,
            lambda: np.array([function(int(p), [1, 2, 3]).treatments for p in patients]),
            f"{size} consecutive patients; treatment counts/weights 1:2:3; default seeds",
        )
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "comparison": "Same Python API batched versus repeated scalar calls; "
        "not native Fortran or I/O",
        "results": results,
    }
    Path("docs/ranlist-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
