"""Validate and time legacy chi-square batches against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdfchi


def main():
    cases = []
    for size in (64, 256):
        a = np.linspace(0.2, 20, size)
        tiny_a = np.geomspace(1e-308, 4e-308, size)
        forward = cdfchi(x=2, df=a)
        tiny_forward = cdfchi(x=2, df=tiny_a)
        for mode in (1, 2, 3, 5):
            name = {1: "p", 2: "x", 3: "df", 5: "tiny_df_quantile"}[mode]

            def evaluate(index):
                if mode == 5:
                    return cdfchi(
                        2, p=tiny_forward.p[index], q=tiny_forward.q[index], df=tiny_a[index]
                    ).x
                kwargs = dict(x=2, df=a[index])
                if mode != 1:
                    kwargs.pop(name)
                    kwargs.update(p=forward.p[index], q=forward.q[index])
                return getattr(cdfchi(mode, **kwargs), name)

            bt, st = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                bt.append(perf_counter() - start)
                start = perf_counter()
                scalar = np.array([float(evaluate(i)) for i in range(size)])
                st.append(perf_counter() - start)
                np.testing.assert_allclose(batch, scalar, rtol=3e-12)
                np.testing.assert_allclose(
                    batch, 2 if mode == 5 else getattr(forward, name), rtol=3e-12
                )
            b, s = median(bt), median(st)
            cases.append(
                dict(operation=name, size=size, batch_seconds=b, scalar_seconds=s, speedup=s / b)
            )
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        repetitions=3,
        inputs="x=2, df=linspace(.2,20,n); tiny df=geomspace(1e-308,4e-308,n)",
        comparison=(
            "One broadcast call versus n scalar calls to the same Python API; not native C/Fortran"
        ),
        cases=cases,
    )
    Path("docs/dcdflib-chisq-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
