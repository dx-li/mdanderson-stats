"""One bounded calibration/prior-predictive input generator for the R oracle."""

from __future__ import annotations

import csv
import resource
import time
from pathlib import Path

import numpy as np

from mdanderson_stats.cibolus import CiBolusPrior
from mdanderson_stats.cibolus_calibration import (
    calibrate_cibolus_prior,
    cibolus_prior_predictive_moments,
)

OUT = Path(__file__).resolve().parents[1] / "tests/fixtures/cibolus-calibration"
started = time.perf_counter()
coordinates = [*(f"log_alpha{i}" for i in range(6)), *(f"log_beta{i}" for i in range(5))]
prior_mean = np.array([0.0] * 6 + [-2.5, 0.5, 0.0, 0.0, -1.0])
prior_sd = np.array([0.0] * 6 + [0.7, 0.0, 0.0, 0.0, 0.0])
pseudo_prior = CiBolusPrior(prior_mean, prior_sd)
concentrations = np.array([0.2, 0.4])
bolus = np.array([0.1])
endpoints = np.array([0.25, 0.5, 0.75, 1.0])
response_mass = np.array(
    [
        [0.12, 0.10, 0.14, 0.16, 0.18, 0.30],
        [0.08, 0.10, 0.13, 0.16, 0.22, 0.31],
    ]
)
tox_risk = np.array(
    [
        [0.02, 0.03, 0.05, 0.06, 0.08, 0.10],
        [0.03, 0.04, 0.05, 0.07, 0.10, 0.13],
    ]
)
truth = np.empty((2, 1, 6, 2))
for ci in range(2):
    truth[ci, 0, :, 1] = response_mass[ci] * tox_risk[ci]
    truth[ci, 0, :, 0] = response_mass[ci] * (1.0 - tox_risk[ci])
assert np.allclose(truth.sum(axis=(-2, -1)), 1.0, atol=2e-16, rtol=0.0)

fit = calibrate_cibolus_prior(
    concentrations,
    bolus,
    endpoints,
    truth,
    repetitions=2,
    patients_per_regimen=10,
    pseudo_prior=pseudo_prior,
    resulting_sd=np.full(11, 0.7),
    pseudo_draws=600,
    pseudo_warmup=200,
    pseudo_chains=2,
    rng=np.random.default_rng(884821),
    max_total_evaluations=20_000,
    max_total_work=2_000_000,
)
with (OUT / "joint_counts.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["replicate", "regimen", "response_category", "toxicity", "count"])
    for replicate in range(fit.joint_counts.shape[0]):
        for ci in range(fit.joint_counts.shape[1]):
            for qi in range(fit.joint_counts.shape[2]):
                regimen = ci * fit.joint_counts.shape[2] + qi + 1
                for category in range(fit.joint_counts.shape[3]):
                    for toxicity in range(2):
                        writer.writerow(
                            [
                                replicate + 1,
                                regimen,
                                category,
                                toxicity,
                                int(fit.joint_counts[replicate, ci, qi, category, toxicity]),
                            ]
                        )
with (OUT / "regimens.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["regimen", "concentration", "bolus"])
    for ci, c in enumerate(concentrations):
        for qi, q in enumerate(bolus):
            writer.writerow([ci * len(bolus) + qi + 1, c, q])
with (OUT / "endpoints.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["endpoint"])
    writer.writerows((float(value),) for value in endpoints)
with (OUT / "prior.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["coordinate", "mean", "sd"])
    writer.writerows(zip(coordinates, pseudo_prior.mean, pseudo_prior.sd, strict=True))
with (OUT / "python_calibration.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["replicate", "posterior_mean_log_beta0", "sampler_mcse_log_beta0", "replicate_seed"]
    )
    for i in range(fit.repetitions):
        writer.writerow(
            [
                i + 1,
                float(fit.replicate_posterior_mean[i, 6]),
                float(fit.replicate_sampler_mcse[i, 6]),
                int(fit.replicate_seeds[i]),
            ]
        )

# Direct-prior-draw estimator compared against exact 1-D R quadrature.
prior_rng = np.random.default_rng(884822)
predictive = cibolus_prior_predictive_moments(
    pseudo_prior,
    concentrations,
    bolus,
    endpoints,
    draws=2500,
    chains=2,
    rng=prior_rng,
    max_work=5_000_000,
    max_retained_cells=2_000_000,
)
with (OUT / "python_prior_predictive.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["regimen", "metric", "mean", "variance", "ess", "n_draws"])
    source_names = predictive.source_probability_names
    for ci in range(2):
        vals = (
            predictive.source_probability_mean[ci, 0],
            predictive.source_probability_variance[ci, 0],
            predictive.source_probability_ess[ci, 0],
        )
        for mi, name in enumerate(source_names):
            writer.writerow(
                [ci + 1, name, float(vals[0][mi]), float(vals[1][mi]), float(vals[2][mi]), 5_000]
            )

elapsed = time.perf_counter() - started
usage = resource.getrusage(resource.RUSAGE_SELF)
print(f"calibration reps={fit.repetitions} counts_shape={fit.joint_counts.shape}")
print(f"elapsed_seconds={elapsed:.3f} peak_rss_bytes={usage.ru_maxrss} swaps={usage.ru_nswap}")
