"""Compare PRT's count recursion with equation (4.3)'s exhaustive outcome sum."""

import itertools
import json
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import prt_predictive_risk

rng = np.random.default_rng(6902)
total = rng.beta(2, 7, 512)
future = total[:, None] * rng.uniform(0.1, 1, size=(512, 12))
start = time.perf_counter()
result = prt_predictive_risk(total, future)
dynamic_seconds = time.perf_counter() - start
start = time.perf_counter()
reference = np.zeros(13)
for outcomes in itertools.product((0, 1), repeat=12):
    reference[sum(outcomes)] += np.prod(np.where(outcomes, future, 1 - future), axis=1).mean()
enumeration_seconds = time.perf_counter() - start
error = float(np.max(abs(result.count_probability - reference)))
assert error < 1e-13
report = {
    "seed": 6902,
    "posterior_draws": 512,
    "pending_patients": 12,
    "enumerated_outcomes": 4096,
    "maximum_count_probability_difference": error,
    "dynamic_seconds": dynamic_seconds,
    "enumeration_seconds": enumeration_seconds,
    "predictive_negligible": result.predictive_negligible,
    "predictive_excessive": result.predictive_excessive,
    "count_probability": result.count_probability.tolist(),
    "scope": (
        "Synthetic aligned conditional posterior risks; not a fitted PRT model or native RNG audit"
    ),
}
Path("docs/prt-reference.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
