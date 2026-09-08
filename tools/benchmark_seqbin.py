"""SEQBIN operating-characteristic batching with numerical equivalence checks."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from benchmark_numerics import measure

from mdanderson_stats import SeqBinDesign


def main() -> None:
    probabilities = np.linspace(0, 1, 101)
    results = {}
    for name, looks in [("sequential", None), ("early_final_look", [50, 100, 150, 200])]:
        design = SeqBinDesign(
            500, prior=[0.5, 0.5], alternative="two-sided", tail_probability=0.005, looks=looks
        )

        def summary(p):
            result = design.operating_characteristics(p)
            return np.stack(
                [result.rejection_probability, result.complete, result.expected_subjects], axis=-1
            )

        results[name] = measure(
            lambda: summary(probabilities),
            lambda: np.array([summary(p) for p in probabilities]),
            "101 true probabilities from 0 to 1; maximum 500 subjects; "
            + ("analyses after every subject" if looks is None else "analyses at 50,100,150,200"),
        )
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison="One batched call versus 101 scalar calls to the same Python API; "
        "not a comparison against archived executables or calibration",
        results=results,
    )
    Path("docs/seqbin-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
