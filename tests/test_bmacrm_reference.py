"""Independent base-R integrals of the documented BMA-CRM posterior."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bmacrm import fit_bmacrm


def test_posterior_against_independent_base_r():
    with (Path(__file__).parent / "fixtures" / "bmacrm-posterior.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for name in dict.fromkeys(row["scenario"] for row in rows):
        group = [row for row in rows if row["scenario"] == name]
        k = 1 + max(int(row["model"]) for row in group)
        j = 1 + max(int(row["dose"]) for row in group)

        def matrix(key):
            return np.array([float(row[key]) for row in group]).reshape(k, j)

        fit = fit_bmacrm(
            matrix("skeleton"),
            matrix("events")[0],
            matrix("subjects")[0],
            target=float(group[0]["target"]),
            prior_sd=float(group[0]["prior_sd"]),
            model_prior=matrix("raw_prior_weight")[:, 0],
        )
        for actual, key in (
            (fit.posterior_model_weights, "posterior_weight"),
            (fit.model_log_evidence, "model_log_evidence"),
            (fit.alpha_mean, "alpha_mean"),
            (fit.alpha_sd, "alpha_sd"),
        ):
            np.testing.assert_allclose(
                actual, matrix(key)[:, 0], rtol=2e-8, atol=2e-10, err_msg=f"{name}: {key}"
            )
        for actual, key in (
            (fit.model_dose_mean, "model_dose_mean"),
            (fit.model_overdose_probability, "model_overdose"),
        ):
            np.testing.assert_allclose(
                actual, matrix(key), rtol=2e-8, atol=2e-10, err_msg=f"{name}: {key}"
            )
        np.testing.assert_allclose(
            fit.dose_mean, matrix("dose_mean")[0], rtol=2e-8, atol=2e-10, err_msg=name
        )
        np.testing.assert_allclose(
            fit.overdose_probability, matrix("overdose")[0], rtol=2e-8, atol=2e-10, err_msg=name
        )
