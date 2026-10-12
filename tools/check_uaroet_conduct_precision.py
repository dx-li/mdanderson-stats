"""Exercise the conduct precision target with one independent-prior chain."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats import fit_uaroet_desktop, read_uaroet_desktop_study

s = read_uaroet_desktop_study(Path("tests/fixtures/uaroet-python-study.json").read_text())
s = replace(s, patients=(), simulation_parameters={"NumChains": 1, "BurnInSamples": 0})
f = fit_uaroet_desktop(
    s,
    rng=np.random.default_rng(8304),
    conduct=True,
    chunk_draws=10000,
    max_draws=100000,
    max_evaluations=8000000,
)
assert f.precision_met
print(
    json.dumps(
        dict(
            scope=(
                "one-chain direct-prior conduct precision; "
                "no posterior-data or R-hat convergence claim"
            ),
            target=f.target,
            draws=f.draws_per_chain,
            ratios=f.utility_mcse_ratio.tolist(),
            work=f.work_evaluations,
            clinical_validation=False,
        ),
        indent=2,
    )
)
