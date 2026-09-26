"""Independent base-R min-max isotonic fits and beta prior/tail references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.mtadf import (
    double_sided_isotonic,
    mtadf_decision,
    mtadf_toxicity_prior,
)


def test_isotonic_fits_against_independent_min_max_reference():
    path = Path(__file__).parent / "fixtures" / "mtadf-isotonic-reference.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        group = [row for row in rows if row["scenario"] == scenario]
        fit = double_sided_isotonic(
            [float(row["value"]) for row in group],
            [float(row["weight"]) for row in group],
        )
        np.testing.assert_allclose(
            fit.fitted,
            [float(row["fitted"]) for row in group],
            atol=2e-15,
            rtol=2e-14,
            err_msg=scenario,
        )
        np.testing.assert_allclose(
            fit.candidate_scores[fit.split],
            float(group[0]["score"]),
            atol=2e-15,
            rtol=2e-14,
            err_msg=scenario,
        )
        assert fit.peak == int(group[0]["peak"]), scenario


def test_calibrated_prior_and_overdose_tails_against_independent_r():
    path = Path(__file__).parent / "fixtures" / "mtadf-prior-reference.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        prior = mtadf_toxicity_prior(
            toxicity_limit=float(row["phi"]),
            safety_cutoff=float(row["cutoff"]),
            margin=float(row["margin"]),
            concentration=float(row["concentration"]),
        )
        np.testing.assert_allclose(
            [prior.alpha, prior.beta],
            [float(row["alpha"]), float(row["beta"])],
            atol=2e-12,
            rtol=2e-10,
            err_msg=row["scenario"],
        )
        n, x = int(row["subjects"]), int(row["toxicities"])
        decision = mtadf_decision(
            [n],
            [x],
            [0],
            current_dose=0 if n else None,
            toxicity_limit=float(row["phi"]),
            safety_cutoff=float(row["cutoff"]),
            prior=prior,
        )
        np.testing.assert_allclose(
            decision.raw_overdose_probability[0],
            float(row["overdose"]),
            atol=2e-12,
            rtol=2e-10,
            err_msg=f"{row['scenario']}: n={n}, x={x}",
        )
