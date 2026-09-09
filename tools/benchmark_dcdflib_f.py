"""Compare legacy F batching with repeated scalar calls to the same Python API."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdff


def main():
    cases = []
    for size in (64, 256):
        nn = np.linspace(2, 10, size)
        r = cdff(f=1, dfn=nn, dfd=10)
        for mode in (1, 3, 4):

            def evaluate(index):
                if mode == 1:
                    return cdff(f=1, dfn=nn[index], dfd=10).p
                if mode == 3:
                    return cdff(
                        3,
                        p=r.p[index],
                        q=r.q[index],
                        f=1,
                        dfd=10,
                        df_bracket=(nn[index] * 0.99, nn[index] * 1.01),
                    ).dfn
                return cdff(
                    4, p=r.p[index], q=r.q[index], f=1, dfn=nn[index], df_bracket=(8, 12)
                ).dfd

            batch_times, scalar_times = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                batch_times.append(perf_counter() - start)
                start = perf_counter()
                scalar = np.array([float(evaluate(i)) for i in range(size)])
                scalar_times.append(perf_counter() - start)
                np.testing.assert_allclose(batch, scalar, rtol=1e-11, atol=0)
                np.testing.assert_allclose(
                    batch, r.p if mode == 1 else nn if mode == 3 else 10, rtol=1e-11, atol=0
                )
            bt, st = median(batch_times), median(scalar_times)
            cases.append(
                {
                    "operation": {1: "tails", 3: "dfn", 4: "dfd"}[mode],
                    "size": size,
                    "batch_seconds": bt,
                    "scalar_seconds": st,
                    "speedup": st / bt,
                }
            )
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "repetitions": 3,
        "inputs": (
            "f=1, dfn=linspace(2,10,n), dfd=10; "
            "dfn bracket .99..1.01 times input; dfd bracket 8..12"
        ),
        "comparison": (
            "One broadcast call versus n scalar calls to the same Python API; not native C/Fortran"
        ),
        "cases": cases,
    }
    Path("docs/dcdflib-f-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
