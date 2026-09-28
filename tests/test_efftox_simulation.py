import numpy as np
import pytest

from mdanderson_stats.efftox_decision import EffToxContour, efftox_decision
from mdanderson_stats.efftox_model import EffToxPrior, fit_efftox
from mdanderson_stats.efftox_simulation import simulate_efftox


def _scenario():
    doses = np.asarray([1.0, 2.0])
    prior = EffToxPrior(
        mean=[-2.0, 0.5, 1.0, 0.5, -0.1, 0.0],
        sd=np.zeros(6),
    )
    contour = EffToxContour.from_points(0.5, 0.7, 0.65, 0.25)
    truth = np.asarray(
        [
            [[0.55, 0.15], [0.20, 0.10]],
            [[0.30, 0.20], [0.35, 0.15]],
        ]
    )
    return doses, prior, contour, truth


def test_fixed_parameter_simulation_matches_existing_final_decision():
    doses, prior, contour, truth = _scenario()
    counts = np.zeros((2, 2, 2), dtype=np.int64)
    counts[0, 0, 0] = 1
    fit = fit_efftox(
        doses,
        counts,
        prior=prior,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(13),
    )
    expected = efftox_decision(
        fit,
        contour,
        efficacy_limit=0.3,
        toxicity_limit=0.4,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        starting_dose=1,
        phase="final",
        allow_untried_exploration=False,
    )

    result = simulate_efftox(
        doses,
        truth,
        prior=prior,
        contour=contour,
        efficacy_limit=0.3,
        toxicity_limit=0.4,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        starting_dose=1,
        cohorts=2,
        cohort_size=2,
        trials=8,
        draws=8,
        warmup=0,
        chains=2,
        allow_untried_exploration=False,
        rng=19,
        sampler_rng=23,
    )

    assert expected.action == "select"
    np.testing.assert_array_equal(result.selected_dose, expected.dose)
    np.testing.assert_array_equal(result.outcome_counts.sum(axis=(2, 3)), result.dose_patients)
    np.testing.assert_array_equal(result.dose_patients.sum(axis=1), result.cohorts_completed * 2)
    assert result.selection_probability.sum() == pytest.approx(1.0)
    assert result.stop_reason_probability.sum() == pytest.approx(1.0)
    assert result.allocation_probability.sum() == pytest.approx(1.0)
    for dose, assigned in enumerate(result.dose_patients.sum(axis=0)):
        if assigned:
            assert result.observed_joint_probability[dose].sum() == pytest.approx(1.0)
    # Fixed chains have undefined R-hat; the API reports it rather than
    # treating an undefined diagnostic as evidence of convergence.
    assert np.isnan(result.max_split_rhat).all()


def test_simulation_streams_are_reproducible_and_truth_must_be_jointly_normalized():
    doses, prior, contour, truth = _scenario()
    options = dict(
        prior=prior,
        contour=contour,
        efficacy_limit=0.3,
        toxicity_limit=0.4,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        cohorts=2,
        cohort_size=2,
        trials=4,
        draws=8,
        warmup=0,
        chains=2,
        allow_untried_exploration=False,
        rng=31,
        sampler_rng=37,
    )
    first = simulate_efftox(doses, truth, **options)
    second = simulate_efftox(doses, truth, **options)
    np.testing.assert_array_equal(first.outcome_counts, second.outcome_counts)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    assert first.work_estimate == second.work_estimate

    invalid = truth.copy()
    invalid[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="sum to one"):
        simulate_efftox(doses, invalid, **options)


def test_safety_stop_occurs_after_completed_cohort_and_refits_posterior():
    doses, _, contour, truth = _scenario()
    prior = EffToxPrior(
        mean=[-2.0, 0.5, 1.0, 0.5, -0.1, 0.0],
        sd=[0.2, 0.1, 0.2, 0.1, 0.05, 0.1],
    )

    result = simulate_efftox(
        doses,
        truth,
        prior=prior,
        contour=contour,
        efficacy_limit=0.3,
        toxicity_limit=0.0,
        efficacy_probability=0.1,
        toxicity_probability=0.1,
        starting_dose=1,
        cohorts=3,
        cohort_size=2,
        trials=1,
        draws=8,
        warmup=0,
        chains=2,
        allow_untried_exploration=False,
        rng=41,
        sampler_rng=43,
    )

    assert result.stop_reason == ("stop_no_admissible",)
    assert result.selected_dose[0] == 0
    assert result.cohorts_completed[0] == 1
    assert result.posterior_fit_count[0] == 2
    assert result.likelihood_evaluations[0] > 0
    assert result.early_stop_probability == 1.0
    assert result.outcome_counts.sum() == 2
