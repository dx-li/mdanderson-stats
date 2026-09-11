"""Reproduce the archived Case 1 calendar pilot, not an operating-characteristic study."""

import json
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import simulate_phase12_calendar

start = time.perf_counter()
r = simulate_phase12_calendar(
    [0.05, 0.1, 0.1, 0.15, 0.15, 0.2],
    [0.1, 0.2, 0.2, 0.3, 0.3, 0.5],
    draws=2000,
    warmup=1000,
    chains=4,
    rng=np.random.default_rng(8554),
)
out = {
    "scenario": "C++ SimulationCases Case 1: response and toxicity increasing",
    "seed": 8554,
    "draws": 2000,
    "warmup": 1000,
    "chains": 4,
    "enrolled": len(r.records),
    "phase_two_start": r.phase_two_start,
    "attempts": len(r.attempts),
    "blocked_attempts": int(r.attempts[:, 3].sum()),
    "reason": r.reason,
    "early_selected": r.early_selected,
    "future_selected": r.future_selected,
    "selected_eligible": r.selected_eligible,
    "stop_time": r.stop_time,
    "final_analysis_time": r.final_analysis_time,
    "analysis_max_split_rhat": [a.max_split_rhat for a in r.analyses],
    "elapsed_seconds": time.perf_counter() - start,
    "scope": (
        "Integrated workflow pilot with finite MCMC budget, not native random-stream "
        "or published operating-characteristic replication."
    ),
}
Path("docs/phase12-calendar-pilot.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
