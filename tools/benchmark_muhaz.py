"""MUHAZ batching benchmark with numerical agreement checks."""

import json
import platform
from pathlib import Path

import numpy as np
from benchmark_numerics import measure

from mdanderson_stats import muhaz_fixed, muhaz_neighbor_bandwidths


def main() -> None:
    times = np.linspace(0.1, 10, 2000)
    status = (np.arange(times.size) % 4 != 0).astype(int)
    grid = np.linspace(0, 10, 501)
    results = {}
    results["fixed_hazard"] = measure(
        lambda: muhaz_fixed(times, status, bandwidth=0.7, grid=grid).hazard,
        lambda: np.array(
            [muhaz_fixed(times, status, bandwidth=0.7, grid=[z]).hazard[0] for z in grid]
        ),
        "2000 observations, 25% censoring, 501 hazard grid points",
    )
    for method in ["failures", "survival"]:

        def radii(z):
            return muhaz_neighbor_bandwidths(
                times, status, neighbors=[5, 10, 20], grid=z, method=method
            ).bandwidth

        results[method] = measure(
            lambda: radii(grid),
            lambda: np.column_stack([radii([z])[:, 0] for z in grid]),
            "2000 observations, 25% censoring, counts 5/10/20, 501 query points",
        )
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison=(
            "One batched call versus 501 single-point calls to the same Python API; "
            "not a Fortran comparison or full selection benchmark"
        ),
        results=results,
    )
    Path("docs/muhaz-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
