"""Validate and time batched legacy normal calls versus repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdfnor


def main():
    cases = []
    for size in (256, 2048):
        x = np.linspace(0.2, 5, size)
        forward = cdfnor(x=x, mean=-1, sd=2)
        for mode in (1, 2, 3, 4):
            name = {1: "p", 2: "x", 3: "mean", 4: "sd"}[mode]

            def evaluate(index):
                kwargs = dict(x=x[index], mean=-1, sd=2)
                if mode != 1:
                    kwargs.pop(name)
                    kwargs.update(p=forward.p[index], q=forward.q[index])
                return getattr(cdfnor(mode, **kwargs), name)

            bt, st = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                bt.append(perf_counter() - start)
                start = perf_counter()
                scalar = np.array([float(evaluate(i)) for i in range(size)])
                st.append(perf_counter() - start)
                np.testing.assert_allclose(batch, scalar, rtol=3e-13)
                np.testing.assert_allclose(batch, getattr(forward, name), rtol=3e-13)
            batch, scalar = median(bt), median(st)
            cases.append(
                dict(
                    operation=name,
                    size=size,
                    batch_seconds=batch,
                    scalar_seconds=scalar,
                    speedup=scalar / batch,
                )
            )
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        repetitions=3,
        inputs="x=linspace(.2,5,n), mean=-1, sd=2",
        comparison=(
            "One broadcast call versus n scalar calls to the same Python API; not native C/Fortran"
        ),
        cases=cases,
    )
    Path("docs/dcdflib-normal-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
