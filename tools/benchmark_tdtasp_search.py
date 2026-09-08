"""Compare the bounded TDTASP search with an exhaustive forward scan."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import tdtasp_ascertainment, tdtasp_genetics, tdtasp_power, tdtasp_sample_size


def main():
    genetics = tdtasp_genetics([0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1)
    cases = []
    for eligibility in ["father", "one", "both"]:
        model = tdtasp_ascertainment(genetics, 2, eligibility=eligibility)
        timings = {}
        result = tdtasp_sample_size(model, 0.8, max_families=10000)
        for method in ["bounded", "exhaustive"]:
            elapsed = []
            for _ in range(3):
                selected = None
                start = perf_counter()
                if method == "bounded":
                    selected = tdtasp_sample_size(model, 0.8, max_families=10000).design.families
                else:
                    for n in range(1, result.design.families + 1):
                        if tdtasp_power(model, n).power >= 0.8:
                            selected = n
                            break
                elapsed.append(perf_counter() - start)
                if selected != result.design.families:
                    raise RuntimeError("searches disagree on first qualifying design")
            timings[method] = median(elapsed)
        cases.append(
            {
                "eligibility": eligibility,
                "target_power": 0.8,
                "families": result.design.families,
                "power": result.design.power,
                "search_start": result.search_start,
                "evaluations": result.evaluations,
                "seconds": timings,
                "speedup": timings["exhaustive"] / timings["bounded"],
            }
        )
    data = {
        "comparison": "Prefix-maximum upper-bound search versus ordered calls to the same Python "
        "forward power API; median of three repeats. Not a native Fortran comparison.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "haplotype_frequencies": [0.3, 0.2, 0.1, 0.4],
        "penetrance": [0.8, 0.5, 0.2],
        "recombination": 0.1,
        "mean_offspring": 2,
        "cases": cases,
    }
    Path("docs/tdtasp-search-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    for case in cases:
        print(case["eligibility"], case["families"], case["seconds"], case["speedup"])


if __name__ == "__main__":
    main()
