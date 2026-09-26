import numpy as np
import pytest
from scipy.special import betainc, betaincc

from mdanderson_stats.mtadf import (
    MTADFPrior,
    double_sided_isotonic,
    mtadf_decision,
    mtadf_toxicity_prior,
)
from mdanderson_stats.mtadf_simulation import simulate_mtadf


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

    # Preserve a tiny prior beta parameter when every treated patient is toxic.
    tiny_beta = MTADFPrior(0.2, 1e-20)
    all_events = mtadf_decision(
        [10],
        [10],
        [0],
        current_dose=0,
        prior=tiny_beta,
        toxicity_limit=0.2,
        safety_cutoff=0.99,
    )
    assert all_events.raw_overdose_probability[0] == pytest.approx(
        betaincc(10.2, 1e-20, 0.2), abs=0
    )


def test_small_simulation_reproducibility_and_conservation() -> None:
    args = ([0.05, 0.2, 0.4], [0.1, 0.8, 0.3])
    first = simulate_mtadf(*args, cohorts=3, cohort_size=2, trials=4, rng=71)
    second = simulate_mtadf(*args, cohorts=3, cohort_size=2, trials=4, rng=71)
    assert first.selected_dose.tolist() == second.selected_dose.tolist()
    assert first.patients.tolist() == second.patients.tolist()
    assert np.all(first.toxicities <= first.patients)
    assert np.all(first.responses <= first.patients)
    assert np.all(first.patients.sum(axis=1) <= 6)
    assert np.sum(first.selection_probability) + first.no_selection_probability == pytest.approx(1)
    assert not first.patients.flags.writeable
