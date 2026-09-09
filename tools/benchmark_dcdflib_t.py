"""Validate and time batched legacy Student t calls versus repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdft


def main():
    cases = []
    for size in (64, 256):
        df = np.linspace(0.2, 20, size)
        forward = cdft(t=2, df=df)
        for mode in (1, 2, 3):
            name = {1: "p", 2: "t", 3: "df"}[mode]

            def evaluate(index):
                kwargs = dict(t=2, df=df[index])
                if mode != 1:
                    kwargs.pop(name)
                    kwargs.update(p=forward.p[index], q=forward.q[index])
                return getattr(cdft(mode, **kwargs), name)

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
        inputs="t=2, df=linspace(.2,20,n)",
        comparison=(
            "One broadcast call versus n scalar calls to the same Python API; not native C/Fortran"
        ),
        cases=cases,
    )
    Path("docs/dcdflib-t-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
