import numpy as np
import pytest
from scipy.special import betainc

from mdanderson_stats.mtadf import (
    MTADFPrior,
    double_sided_isotonic,
    mtadf_decision,
    mtadf_toxicity_prior,
)


def test_prior_calibration_and_weighted_double_sided_fit() -> None:
    prior = mtadf_toxicity_prior()
    assert prior.alpha + prior.beta == pytest.approx(0.5)
    assert betainc(prior.alpha, prior.beta, 0.3) == pytest.approx(0.25, abs=2e-12)

    # Unequal sample sizes pool the middle observations to their weighted mean.
    fit = double_sided_isotonic([0.1, 0.3, 0.2, 0.1], [10, 10, 5, 5])
    assert fit.candidate_fits.shape == (4, 4)
    assert fit.candidate_scores.shape == (4,)
    assert fit.candidate_fits[2] == pytest.approx([0.1, 4 / 15, 4 / 15, 0.1])
    assert fit.fitted == pytest.approx([0.1, 0.3, 0.2, 0.1])


def test_decision_uses_safety_and_observed_dose_mapping() -> None:
    prior = MTADFPrior(0.3, 0.7)
    # Gaps are valid, and the efficacy fit maps back to original indices.
    decision = mtadf_decision(
        [10, 0, 10, 0],
        [0, 0, 1, 0],
        [1, 0, 8, 0],
        current_dose=2,
        prior=prior,
        toxicity_limit=0.3,
        safety_cutoff=0.99,
    )
    assert decision.action == "treat"
    assert decision.peak == 2
    assert decision.dose == 3
    assert np.isnan(decision.fitted_efficacy[1])
    assert np.isnan(decision.fitted_efficacy[3])


def test_final_selection_excludes_untried_and_stops_when_all_unsafe() -> None:
    prior = MTADFPrior(0.1, 0.9)
    final = mtadf_decision(
        [10, 10, 0],
        [0, 0, 0],
        [4, 8, 0],
        final=True,
        prior=prior,
        toxicity_limit=0.3,
        safety_cutoff=0.99,
    )
    assert final.action == "select_obd"
    assert final.dose == 1
    assert np.isnan(final.fitted_efficacy[2])

    unsafe = mtadf_decision(
        [10, 10],
        [10, 10],
        [0, 0],
        current_dose=1,
        prior=MTADFPrior(2, 0.1),
        toxicity_limit=0.05,
        safety_cutoff=0.5,
    )
    assert unsafe.action == "stop"
    assert unsafe.dose is None
