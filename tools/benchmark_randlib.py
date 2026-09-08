"""Compare vectorized RANDLIB draws with repeated calls to the same Python API."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from benchmark_numerics import measure

from mdanderson_stats import RandlibGenerator, RandlibMultivariateNormal

PARAMETERS = {
    "multivariate_normal": {
        "mean": [1, -2, 0.5],
        "covariance": [[4, -1, 0.2], [-1, 2, 0.5], [0.2, 0.5, 1]],
    },
    "multinomial": {"n": 100, "p": [0.2, 0.3, 0.5]},
    "negative_binomial": {"n": 10, "p": 0.3},
    "poisson": {"mu": 20.0},
    "binomial": {"n": 1000, "p": 0.3},
    "beta": {"a": 2.0, "b": 3.0},
    "gamma": {"shape": 2.5, "rate": 1.7},
    "chi_square": {"df": 5},
    "noncentral_chi_square": {"df": 5, "noncentrality": 2.3},
    "f": {"dfn": 5, "dfd": 12},
    "noncentral_f": {"dfn": 5, "dfd": 12, "noncentrality": 2.3},
}


def samples(name: str, size: int, batch: bool) -> np.ndarray:
    bank = RandlibGenerator()
    method = getattr(bank, name)
    options = PARAMETERS.get(name, {})
    if name == "multivariate_normal":
        parameters = RandlibMultivariateNormal(**options)
        return (
            bank.multivariate_normal(parameters, size)
            if batch
            else np.array([bank.multivariate_normal(parameters)[0] for _ in range(size)])
        )
    return (
        method(size, **options) if batch else np.array([method(**options)[0] for _ in range(size)])
    )


def main():
    results = {}
    for name in ["normal", "exponential", *PARAMETERS]:
        results[name] = measure(
            lambda: samples(name, 10000, True),
            lambda: samples(name, 10000, False),
            "10,000 consecutive default-mode draws; default seeds; "
            + str(PARAMETERS.get(name, "default parameters")),
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
