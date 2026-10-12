"""Exercise per-chain precision on a correlated posterior with pending efficacy."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats import fit_uaroet_desktop, read_uaroet_desktop_study

study = read_uaroet_desktop_study(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/uaroet-python-study.json").read_text()
)
study = replace(study, simulation_parameters={"NumChains": 4, "BurnInSamples": 1000})
fit = fit_uaroet_desktop(
    study,
    rng=np.random.default_rng(20261012),
    conduct=False,
    chunk_draws=1000,
    max_draws=20000,
    max_evaluations=8000000,
)
result = dict(
    scope="correlated two-dose posterior, one complete and one toxicity-only patient",
    precision_met=fit.precision_met,
    target=fit.target,
    draws_per_chain=fit.draws_per_chain,
    per_chain_utility_mcse_ratios=fit.utility_mcse_ratio.tolist(),
    utility_split_rhat=fit.utility_split_rhat.tolist(),
    posterior_mean_joint=fit.mean_joint.tolist(),
    work_evaluations=fit.work_evaluations,
    native_rng_equivalence=False,
    clinical_validation=False,
)
assert fit.precision_met
assert np.max(fit.utility_split_rhat) < 1.05
print(json.dumps(result, indent=2))
