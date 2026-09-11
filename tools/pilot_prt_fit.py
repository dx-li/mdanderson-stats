"""Inspect the PRT guide's six-dose all-censored history under the stated prior."""

import json
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import fit_prt_model, prt_isotonic_projection

survived = np.zeros((6, 6))
survived[:5, 1] = 3
survived[5, 1] = 1
survived[0, 2] = 3
survived[1, 2] = 1
start = time.perf_counter()
fit = fit_prt_model(
    survived, np.zeros_like(survived), draws=2000, warmup=1000, rng=np.random.default_rng(6911)
)
report = {
    "scenario": "Guide history: doses 2/3, completed intervals (6,5,5)/(2,1,1), no events",
    "seed": 6911,
    "prior_mean": -14,
    "increment_variance": 28,
    "chains": 4,
    "draws": 2000,
    "warmup": 1000,
    "raw_total_risk_mean": fit.conditional_toxicity[:, :, 0].mean(axis=(0, 1)).tolist(),
    "maximum_beta_split_rhat": float(np.max(fit.beta_summary.split_rhat)),
    "fit_seconds": time.perf_counter() - start,
    "likelihood_evaluations": fit.likelihood_evaluations,
}
try:
    projection = prt_isotonic_projection(fit.conditional_toxicity[:, :, :-1].reshape(-1, 6, 6))
except ArithmeticError as exc:
    report["projection_status"] = "rejected"
    report["projection_reason"] = str(exc)
else:
    report["projection_status"] = "completed"
    report["projected_mean"] = projection.probability[:, 0].mean(axis=0).tolist()
report["scope"] = "Raw posterior pilot; native implementation of projection remains unverified"
Path("docs/prt-fit-pilot.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
