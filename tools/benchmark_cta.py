"""CTA batched calculations versus repeated scalar calls of the same API."""

import json
import platform
from pathlib import Path

import numpy as np
import scipy
from benchmark_numerics import measure

from mdanderson_stats import binomial_comparison, contingency_chi_square, diagnostic_accuracy


def main():
    tables = (np.arange(40000).reshape(10000, 2, 2) % 37 + 1).astype(float)
    results = {}
    for name, function, field in [
        ("chi_square", contingency_chi_square, "statistic"),
        ("diagnostic", diagnostic_accuracy, "standard_errors"),
        ("binomial", lambda x: binomial_comparison(x, event_index=0), "pvalue"),
    ]:
        results[name] = measure(
            lambda: getattr(function(tables), field),
            lambda: np.array([getattr(function(table), field) for table in tables]),
            f"10000 positive 2x2 integer tables; compare {field}",
        )
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "comparison": "Same Python API called on arrays versus once per table; "
        "not the Fortran executable or report formatting",
        "results": results,
    }
    Path("docs/cta-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
