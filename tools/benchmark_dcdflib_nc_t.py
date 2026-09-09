"""Measure checked legacy noncentral t batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdftnc


def main():
    cases = []
    for size in (64, 256):
        t = np.linspace(-2, -0.2, size)
        forward = cdftnc(t=t, df=2, pnonc=3)
        for mode, name in [(1, "p"), (2, "t"), (3, "df"), (4, "pnonc")]:

            def evaluate(index):
                kwargs = dict(t=t[index], df=2, pnonc=3)
                if mode != 1:
                    kwargs.pop(name)
                    kwargs.update(p=forward.p[index], q=forward.q[index])
                if mode == 3:
                    kwargs["df_bracket"] = (1.9, 2.1)
                return getattr(cdftnc(mode, **kwargs), name)

            bt, st = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                bt.append(perf_counter() - start)
                start = perf_counter()
                scalar = np.array([float(evaluate(i)) for i in range(size)])
                st.append(perf_counter() - start)
                np.testing.assert_allclose(batch, scalar, rtol=3e-12, atol=1e-14)
                np.testing.assert_allclose(batch, getattr(forward, name), rtol=3e-12, atol=1e-14)
            b, s_time = median(bt), median(st)
            cases.append(
                dict(
                    operation=name,
                    size=size,
                    batch_seconds=b,
                    scalar_seconds=s_time,
                    speedup=s_time / b,
                )
            )
    result = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        repetitions=3,
        inputs="df=2, pnonc=3, t=linspace(-2,-.2,size), df bracket=(1.9,2.1)",
        comparison=(
            "One broadcast call versus n scalar calls to the same Python API; not native C/Fortran"
        ),
        cases=cases,
    )
    Path("docs/dcdflib-nc-t-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
