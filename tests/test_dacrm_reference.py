"""DA posterior checks against exact small-state base-R integration."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.dacrm import DACRMPrior, fit_dacrm
from mdanderson_stats.hierarchical_binomial import summarize_chains


def test_sampler_against_exact_enumeration_and_gamma_integration():
    with (Path(__file__).parent / "fixtures" / "dacrm-posterior.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = list(dict.fromkeys(row["scenario"] for row in rows))
    for case_index, name in enumerate(names):
        group = [row for row in rows if row["scenario"] == name]
        first = group[0]

        def vector(key):
            return np.array([float(v) for v in first[key].split("|") if v])

        def expected(quantity):
            return np.array([float(r["value"]) for r in group if r["quantity"] == quantity])

        prior = DACRMPrior(
            vector("breaks"), vector("shape"), vector("rate"), alpha_sd=float(first["prior_sd"])
        )
        fit = fit_dacrm(
            vector("skeleton"),
            vector("doses"),
            vector("outcomes"),
            vector("times"),
            prior=prior,
            target=float(first["target"]),
            rng=np.random.default_rng(7301 + case_index),
            draws=3000,
            warmup=1000,
            chains=2,
        )

        def check(draws, truth, label, floor=0.004):
            summary = summarize_chains(draws)
            error = np.abs(summary.mean - truth)
            bound = 6 * summary.batch_mean_mcse + floor
            assert np.all(error <= bound), (name, label, summary.mean, truth, bound)
            variable = np.std(draws, axis=(0, 1)) > 0
            assert np.all(summary.split_rhat[variable] < 1.05), (name, label, summary.split_rhat)

        parameters = np.concatenate((fit.alpha_draws[..., None], fit.hazard_draws), axis=-1)
        mean = np.concatenate((expected("alpha_mean"), expected("hazard_mean")))
        sd = np.concatenate((expected("alpha_sd"), expected("hazard_sd")))
        check(parameters, mean, "parameter means")
        check(parameters**2, sd**2 + mean**2, "parameter second moments", floor=0.01)
        check(fit.dose_probability, expected("dose_mean"), "dose means")
        thresholds = np.log(-np.log(float(first["target"]))) - np.log(-np.log(vector("skeleton")))
        indicators = (fit.alpha_draws[..., None] < thresholds).astype(float)
        check(indicators, expected("overdose"), "overdose probabilities")
        if expected("pending_probability").size:
            check(
                fit.pending_probability_draws, expected("pending_probability"), "pending outcomes"
            )
