"""Run the two guide scenarios using the shipped Jeffreys-prior configuration."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mdanderson_stats import BlockArandDesign, simulate_blockarand_oc

design = BlockArandDesign()
start = time.perf_counter()
scenarios = []
for i, truth in enumerate(([0.3, 0.8], [0.5, 0.8])):
    result = simulate_blockarand_oc(
        truth, repetitions=500, design=design, rng=np.random.default_rng(9006 + i)
    )
    scenarios.append(
        {
            "truth": truth,
            "seed": 9006 + i,
            "repetitions": 500,
            "selection": result.selection_probability.tolist(),
            "selection_mcse": result.selection_mcse.tolist(),
            "no_selection": result.no_selection_probability,
            "mean_patients": result.mean_patients.tolist(),
            "patients_mcse": result.patients_mcse.tolist(),
            "comparisons": result.comparisons,
            "cache_hits": result.cache_hits,
        }
    )
report = {
    "design": asdict(design),
    "scenarios": scenarios,
    "elapsed_seconds": time.perf_counter() - start,
    "scope": "Guide scenarios using NumPy RNG and direct beta quadrature; not native output parity",
}
Path("docs/blockarand-pilot.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
