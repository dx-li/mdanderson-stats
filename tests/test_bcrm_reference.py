"""Independent posterior integrals for native and bounded bCRM curves."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.bcrm_model import BCRMCurve, fit_bcrm


def test_posterior_against_independent_base_r():
    with (Path(__file__).parent / "fixtures" / "bcrm-posterior.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for name in dict.fromkeys(row["scenario"] for row in rows):
        group = [r for r in rows if r["scenario"] == name]
        first = group[0]

        def values(key):
            return np.array([float(r[key]) for r in group])

        curve = BCRMCurve(
            values("skeleton"),
            alpha=float(first["alpha"]),
            lower=float(first["lower"]),
            upper=float(first["upper"]),
        )
        fit = fit_bcrm(
            curve,
            values("events"),
            values("subjects"),
            prior_bounds=(float(first["prior_lower"]), float(first["prior_upper"])),
        )
        np.testing.assert_allclose(curve.standardized_doses, values("x"), atol=2e-14)
        np.testing.assert_allclose(
            [fit.beta_mean, fit.beta_sd, *fit.beta_interval, fit.log_evidence],
            [
                float(first[k])
                for k in ("beta_mean", "beta_sd", "beta_low", "beta_high", "log_evidence")
            ],
            rtol=2e-8,
            atol=2e-10,
            err_msg=name,
        )
        np.testing.assert_allclose(fit.dose_mean, values("dose_mean"), rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(
            fit.dose_interval,
            np.column_stack((values("dose_low"), values("dose_high"))),
            rtol=2e-8,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            fit.plugin_dose_probability,
            values("plugin_probability"),
            rtol=2e-8,
            atol=2e-10,
        )
