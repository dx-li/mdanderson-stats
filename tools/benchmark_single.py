"""Measure SINGLE batched parameter evaluation against repeated scalar calls."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from benchmark_numerics import measure

from mdanderson_stats import single_design_precision, single_two_sample_precision


def main() -> None:
    grid = np.linspace(-1, 1, 1000)
    one = np.column_stack([grid, 1 + 0.2 * grid])
    two = np.column_stack([grid, 1 + 0.2 * grid, grid + 0.3])
    doses = np.array([-2.0, -0.5, 0.5, 2.0])
    counts = np.array([30.0, 20.0, 20.0, 30.0])
    results = {
        "one_sample": measure(
            lambda: single_design_precision(doses, counts, one).quantile_variance,
            lambda: np.array(
                [single_design_precision(doses, counts, p).quantile_variance for p in one]
            ),
            "1000 logistic linear parameter nodes, four fixed doses, 5% quantile variance",
        ),
        "two_sample": measure(
            lambda: single_two_sample_precision((doses, doses), (counts, counts), two).variance,
            lambda: np.array(
                [
                    single_two_sample_precision((doses, doses), (counts, counts), p).variance
                    for p in two
                ]
            ),
            "1000 logistic linear parameter nodes, four doses per group, "
            "location contrast variance",
        ),
    }
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison="Same Python API, one batched call versus 1000 scalar calls; "
        "not a comparison with archived executables or full design search",
        results=results,
    )
    Path("docs/single-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
