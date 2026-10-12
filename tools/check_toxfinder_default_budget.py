"""Representative 60-patient execution at unchanged default sampler quality.

This deliberately takes several minutes. The explicit total budget is bounded;
no draws/warmup are weakened. The scenario/prior are diagnostic inputs, not a
calibrated clinical design or a guarantee of cost/convergence elsewhere.
"""

import json
import time

import numpy as np

from mdanderson_stats import ToxFinderPrior, read_toxfinder_tmml
from mdanderson_stats.toxfinder_study import simulate_toxfinder_study

study = read_toxfinder_tmml("""win-drug-names 'A' 'B'
pi_star .3
lambda 1
tl-d_star 1200 600
tl-nD 3
tl-D1 120 60 600 300 1200 600
nD-start 1
sim-num-points 2 2
sim-surface 0 .2 .3 .5
num-stage1 20
num-stage2 40
cohort 2
tl-alloc 1
""")
prior = ToxFinderPrior(np.ones(6), np.full(6, 0.25))
start = time.monotonic()
run = simulate_toxfinder_study(
    study, prior=prior, trials=2, seed=55, max_sampler_evaluations=200_000_000
)
assert all(t.subjects == t.posterior_subjects == 60 for t in run.simulation.trials)
assert "Allocation MAP diagnostics" in run.to_html()
result = {
    "study": study.to_tmml(),
    "prior_mean": prior.mean.tolist(),
    "prior_variance": prior.variance.tolist(),
    "seed": 55,
    "draws": run.draws,
    "warmup": run.warmup,
    "chains": run.chains,
    "seconds": time.monotonic() - start,
    "evaluations": [t.sampler_evaluations for t in run.simulation.trials],
    "patients": [t.subjects for t in run.simulation.trials],
    "final_parameter_split_rhat": [
        t.final_fit.parameter_summary.split_rhat.tolist() for t in run.simulation.trials
    ],
    "final_parameter_batch_mcse": [
        t.final_fit.parameter_summary.batch_mean_mcse.tolist() for t in run.simulation.trials
    ],
    "map_convergence_by_cohort": [
        [None if c.map_fit is None else c.map_fit.pass_converged for c in t.cohorts]
        for t in run.simulation.trials
    ],
}
print(json.dumps(result, indent=2))
