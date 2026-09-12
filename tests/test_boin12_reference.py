"""Numerical checks against the native BOIN12 reference fixtures."""

import csv
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.boin12 import BOIN12Design, rank_desirability


FIXTURES = Path(__file__).parent / "fixtures"


def _outcomes(text: str, ndoses: int = 5):
    n = np.zeros(ndoses, dtype=int)
    t = np.zeros(ndoses, dtype=int)
    e = np.zeros(ndoses, dtype=int)
    for token in text.split():
        dose, outcomes = int(token[0]), token[1:]
        n[dose - 1] += len(outcomes)
        t[dose - 1] += sum(v in "TB" for v in outcomes)
        e[dose - 1] += sum(v in "EB" for v in outcomes)
    return n, t, e


def test_posterior_fixture_matches_quasi_beta_reference():
    design = BOIN12Design(0.30, 0.35, 0.25)
    rows = list(csv.DictReader((FIXTURES / "boin12-posterior.csv").open()))
    for row in rows:
        result = design.posterior(
            [int(row["patients"])], [int(row["toxicities"])],
            [int(row["efficacies"])],
        )
        assert_allclose(result.utility_probability[0], float(row["posterior_utility_gt_benchmark"]), atol=2e-12)
        assert_allclose(result.toxicity_overdose_probability[0], float(row["posterior_tox_gt_limit"]), atol=2e-12)
        assert_allclose(result.efficacy_futility_probability[0], float(row["posterior_eff_lt_limit"]), atol=2e-12)


def test_rds_fixture_matches_all_native_rows_by_outcome_key():
    native = {}
    with (FIXTURES / "boin12-rds.csv").open() as source:
        for row in csv.DictReader(source):
            native[(int(row["Patients"]), int(row["Toxicity"]), int(row["Efficacy"]))] = row
    result = rank_desirability([0, 3, 6, 9], toxicity_limit=.35, efficacy_limit=.25)
    for n, t, e, allowed, rds in zip(
        result.patients, result.toxicities, result.efficacies, result.admissible, result.rds, strict=True
    ):
        row = native[(int(n), int(t), int(e))]
        assert bool(allowed) == (row["Admissible"] == "Admissible")
        if row["RDS_x"] == "NA":
            assert np.isnan(rds)
        else:
            assert_allclose(rds, float(row["RDS_x"]), atol=0)


def test_decision_fixture_covers_exploration_stay_and_deescalation():
    design = BOIN12Design(.30, .35, .25)
    cases = {
        "extra_exploration_at_nine": ([9, 0, 0], [0, 0, 0], [0, 0, 0], 1),
        "stay_interval_rds": ([3, 6, 3, 0, 0], [0, 0, 2, 0, 0], [0, 1, 1, 0, 0], 3),
        "deescalate_toxic_current": ([0, 0, 3, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], 3),
    }
    expected = {
        row["case"]: row for row in csv.DictReader((FIXTURES / "boin12-decisions.csv").open())
    }
    for name, (n, t, e, current) in cases.items():
        result = design.next_dose(n, t, e, current)
        row = expected[name]
        assert result.next_dose == int(row["next_dose"])
        assert result.action in {"explore_escalate", "stay", "deescalate", "escalate"}


def test_final_obd_fixture_matches_isotonic_toxicity_and_utility_selection():
    design = BOIN12Design(.30, .35, .25)
    rows = list(csv.DictReader((FIXTURES / "boin12-obd.csv").open()))
    for row in rows:
        n, t, e = _outcomes(row["outcomes"])
        result = design.select_obd(n, t, e)
        assert result.obd == int(row["obd"])
        assert result.mtd == int(row["mtd"])
        expected_fit = np.asarray([float(v) for v in row["isotonic_toxicity"].split(";")])
        assert_allclose(result.isotonic_toxicity[n > 0], expected_fit, atol=2e-12)


def test_final_selection_stops_when_all_treated_doses_are_inadmissible():
    result = BOIN12Design(.30, .35, .25).select_obd([3, 3], [3, 3], [0, 0])
    assert result.obd is None
    assert result.mtd is None
    assert not result.admissible.any()
