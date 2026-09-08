"""Compare vectorized RANDLIB draws with repeated calls to the same Python API."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from benchmark_numerics import measure

from mdanderson_stats import RandlibGenerator


def samples(name: str, size: int, batch: bool) -> np.ndarray:
    bank = RandlibGenerator()
    method = getattr(bank, name)
    return method(size) if batch else np.array([method()[0] for _ in range(size)])


def main():
    results = {}
    for name in ["normal", "exponential"]:
        results[name] = measure(
            lambda: samples(name, 10000, True),
            lambda: samples(name, 10000, False),
            "10,000 consecutive default-mode draws; default seeds and parameters",
        )
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison="Same Python API batched versus repeated scalar calls; "
        "not native C/Fortran or legacy sampling",
        results=results,
    )
    Path("docs/randlib-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
