"""One exploratory pseudo-trial through the source-specified calibration driver."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mdanderson_stats import (
    UAROETDesktopScenario,
    calibrate_uaroet_prior,
    read_uaroet_desktop_study,
)

study = read_uaroet_desktop_study(
    (Path(__file__).resolve().parents[1] / "tests/fixtures/uaroet-python-study.json").read_text()
)
study = replace(study, simulation_parameters={"NumChains": 4, "BurnInSamples": 1000})
scenario = UAROETDesktopScenario(
    1, np.array([[0.7, 0.3], [0.5, 0.5]]), np.array([[0.9, 0.1], [0.7, 0.3]]), 0.2
)
fit = calibrate_uaroet_prior(
    study,
    scenario,
    repetitions=1,
    rng=np.random.default_rng(1003),
    max_draws=20000,
    max_evaluations_per_trial=8000000,
)
print(
    json.dumps(
        dict(
            scope="one exploratory pseudo-trial,100 patients per dose,zero-mean SD100 pseudo-prior",
            seed=1003,
            prior_mean=fit.prior_mean.tolist(),
            association_mean=float(fit.replicate_association.mean()),
            mcse=fit.maximum_mcse_ratios.tolist(),
            rhat=fit.maximum_utility_rhat.tolist(),
            draws=fit.draws_per_chain,
            meets_guide_replication_minimum=fit.meets_guide_replication_minimum,
            clinical_validation=False,
        ),
        indent=2,
    )
)
