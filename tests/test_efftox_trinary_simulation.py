import numpy as np

from mdanderson_stats.efftox_simulation import simulate_efftox
from mdanderson_stats.efftox_trinary_contour import EffToxTrinaryContour
from mdanderson_stats.efftox_trinary_model import EffToxTrinaryPrior


def _contour():
    return EffToxTrinaryContour.from_points([0.2, 0.5, 0.8], [0.0, 0.1, 0.2])


def _thresholds():
    return dict(
        efficacy_limit=0.0,
        toxicity_limit=1.0,
        efficacy_probability=0.0,
        toxicity_probability=0.0,
    )


def test_fixed_trinary_simulation_preserves_cell_order_and_patient_accounting():
    doses = np.asarray([1.0, 2.0])
    truth = np.asarray([[0.50, 0.30, 0.20], [0.25, 0.45, 0.30]])
    prior = EffToxTrinaryPrior(mean=[-1.5, 0.7, 0.3, 0.8], sd=np.zeros(4))
    seed = 29
    repetitions, size = 3, 4

    expected_rng = np.random.default_rng(seed)
    expected = np.stack([expected_rng.multinomial(size, truth[1]) for _ in range(repetitions)])
    result = simulate_efftox(
        doses,
        truth,
        prior=prior,
        contour=_contour(),
        starting_dose=2,
        cohorts=1,
        cohort_size=size,
        trials=repetitions,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(seed),
        sampler_rng=np.random.default_rng(31),
        **_thresholds(),
    )

    assert result.outcome_model == "trinary"
    assert result.outcome_counts.shape == (repetitions, 2, 3)
    np.testing.assert_array_equal(result.outcome_counts[:, 1], expected)
    np.testing.assert_array_equal(result.outcome_counts[:, 0], 0)
    np.testing.assert_array_equal(result.outcome_counts.sum(axis=2), result.dose_patients)
    np.testing.assert_array_equal(result.dose_patients.sum(axis=1), [size] * repetitions)
    np.testing.assert_array_equal(result.true_joint_probabilities, truth)
    np.testing.assert_allclose(
        result.observed_joint_probability[1], expected.sum(axis=0) / (size * repetitions)
    )
    np.testing.assert_array_equal(np.isnan(result.observed_joint_probability[0]), True)


def test_nonfixed_trinary_simulation_runs_data_conditioned_posterior_fits():
    prior = EffToxTrinaryPrior(
        mean=[-1.5, 0.7, 0.3, 0.8],
        sd=[0.3, 0.1, 0.3, 0.1],
    )
    result = simulate_efftox(
        [1.0, 2.0],
        [[0.55, 0.30, 0.15], [0.25, 0.45, 0.30]],
        prior=prior,
        contour=_contour(),
        starting_dose=1,
        cohorts=1,
        cohort_size=2,
        trials=1,
        draws=8,
        warmup=0,
        chains=2,
        rng=37,
        sampler_rng=41,
        **_thresholds(),
    )

    assert result.outcome_model == "trinary"
    assert result.posterior_fit_count[0] == 2
    assert result.likelihood_evaluations[0] > 0
    assert result.outcome_counts.sum() == 2
