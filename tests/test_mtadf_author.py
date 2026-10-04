import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betainc

from mdanderson_stats.mtadf import MTADFPrior
from mdanderson_stats.mtadf_author import (
    _mtadf_author_admissible_dose_count,
    _mtadf_author_decision_with_count,
    _mtadf_author_prior,
    mtadf_author_decision,
)

_REFERENCE = Path(__file__).parent / "fixtures" / "mtadf-author-reference.csv"


def _reference_rows() -> list[dict[str, str]]:
    with _REFERENCE.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_fixed_author_prior_and_inclusive_safety_boundary():
    prior = _mtadf_author_prior()
    assert isinstance(prior, MTADFPrior)
    assert prior.alpha + prior.beta == pytest.approx(0.5)
    assert betainc(prior.alpha, prior.beta, 0.3) == pytest.approx(0.22, abs=2e-14)

    fresh_count = _mtadf_author_admissible_dose_count([0, 0], [0, 0])
    # Choose the computed author tail itself: equality is admissible in the R reference.
    baseline = mtadf_author_decision([0, 0], [0, 0], [0, 0])
    boundary = float(baseline.adjusted_overdose_probability[0])
    equality = mtadf_author_decision([0, 0], [0, 0], [0, 0], safety_cutoff=boundary)
    assert equality.admissible[0]
    assert equality.admissible_dose_count == fresh_count == 2

    floor = mtadf_author_decision([0, 0], [0, 0], [0, 0], safety_cutoff=0.01)
    assert floor.admissible_dose_count == 1
    assert floor.admissible.tolist() == [True, False]


def test_interim_uses_rightmost_unimodal_mode_and_lagged_cap_helper():
    subjects = [3, 3, 0]
    toxicities = [0, 0, 0]
    responses = [1, 1, 0]
    decision = mtadf_author_decision(subjects, toxicities, responses, current_dose=0)
    assert decision.action == "treat"
    assert decision.peak == 1
    assert decision.dose == 1

    capped = _mtadf_author_decision_with_count(
        subjects,
        toxicities,
        responses,
        current_dose=0,
        final=False,
        toxicity_limit=0.3,
        safety_cutoff=0.8,
        admissibility_count=1,
    )
    assert capped.admissible_dose_count == 3
    assert capped.admissibility_count_used == 1
    assert capped.dose == 0

    one_observed = mtadf_author_decision([3, 0, 0], [0, 0, 0], [2, 0, 0], current_dose=0)
    assert one_observed.fitted_efficacy[0] == pytest.approx(2 / 3)
    assert np.isnan(one_observed.fitted_efficacy[1:]).all()
    assert one_observed.peak == 0
    assert one_observed.dose == 1


def test_final_uses_all_doses_and_rightmost_plateau_after_epsilon_rates():
    result = mtadf_author_decision([3, 3, 0], [0, 0, 0], [0, 0, 0], final=True)
    assert result.action == "select_obd"
    # With every fitted rate zero, the reference takes the rightmost maximum,
    # including an untried dose, before applying the inclusive safety cap.
    assert result.peak == 2
    assert result.dose == 2
    assert result.fitted_efficacy.tolist() == [0.0, 0.0, 0.0]


def test_core_matches_independent_author_fit_and_decision_ledgers():
    rows = _reference_rows()
    unequal = [row for row in rows if row["label"] == "unequal_n"]
    observed = np.array([float(row["observed_rate"]) for row in unequal])
    expected = np.array([float(row["fitted_rate"]) for row in unequal])
    from mdanderson_stats.mtadf_author import _author_unimodal_fit

    assert _author_unimodal_fit(observed) == pytest.approx(expected)

    lag = next(row for row in rows if row["label"] == "toxic_first_cohort_lag")
    lag_state = ([3, 0, 0], [3, 0, 0], [3, 0, 0])
    author = _mtadf_author_decision_with_count(
        *lag_state,
        current_dose=0,
        final=False,
        toxicity_limit=0.3,
        safety_cutoff=0.8,
        admissibility_count=int(lag["initial_cap"]),
    )
    fresh = mtadf_author_decision(*lag_state, current_dose=0, toxicity_limit=0.3, safety_cutoff=0.8)
    assert author.dose == int(lag["author_next_dose"]) - 1
    assert fresh.dose == int(lag["df_isotonic_next_dose"]) - 1

    zero = next(row for row in rows if row["label"] == "zero_final_untried_tie")
    final = mtadf_author_decision([3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], final=True)
    assert final.peak == int(zero["peak_rightmost"]) - 1
    assert final.dose == int(zero["author_next_dose"]) - 1
