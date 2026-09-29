from itertools import product

import numpy as np
import pytest
from scipy.optimize import isotonic_regression

from mdanderson_stats.merit_interim_search import (
    _candidate_probabilities,
    merit_interim_sample_size,
)
from mdanderson_stats.merit_interims import MERITInterims
from mdanderson_stats.merit_search import merit_sample_size


def test_candidate_boundary_probabilities_match_exhaustive_active_arm_reference():
    toxicity = np.array([[0, 1], [1, 0], [2, 1], [1, 2]], dtype=np.int64)
    efficacy = np.array([[2, 1], [0, 2], [1, 0], [2, 2]], dtype=np.int64)
    active = np.array([[True, True], [True, False], [False, True], [False, False]])
    truth = np.array([True, False])
    got_one, got_two = _candidate_probabilities(
        toxicity,
        efficacy,
        active,
        truth,
        2,
        isotonic_toxicity=True,
        isotonic_efficacy=True,
    )
    assert got_two is not None
    for tox_max, eff_min in product(range(3), repeat=2):
        selected_true = []
        selected_bad = []
        for row in range(toxicity.shape[0]):
            doses = np.flatnonzero(active[row])
            if doses.size == 0:
                selected_true.append(False)
                selected_bad.append(False)
                continue
            tx = isotonic_regression(toxicity[row, doses].astype(float)).x
            ex = isotonic_regression(efficacy[row, doses].astype(float)).x
            selected = (tx <= tox_max) & (ex >= eff_min)
            selected_true.append(bool(np.any(selected & truth[doses])))
            selected_bad.append(bool(np.any(selected & ~truth[doses])))
        expected_one = np.mean(np.asarray(selected_true) & ~np.asarray(selected_bad))
        expected_two = np.mean(selected_true)
        assert got_one[tox_max, eff_min] == pytest.approx(expected_one, abs=1e-15)
        assert got_two[tox_max, eff_min] == pytest.approx(expected_two, abs=1e-15)


def test_no_look_search_reduces_exactly_to_existing_fixed_size_search():
    common = dict(
        toxicity_null=0.4,
        toxicity_alternative=0.2,
        efficacy_null=0.2,
        efficacy_alternative=0.4,
        doses=2,
        alpha=0.1,
        power=0.6,
        power_definition=2,
        max_patients_per_arm=30,
        trials=5000,
        correlation=0.5,
    )
    expected = merit_sample_size(**common, rng=160)
    actual = merit_interim_sample_size(interims=MERITInterims(0.2, 0.4), **common, rng=160)
    assert actual.design == expected.design
    assert actual.global_type_one_error == expected.global_type_one_error
    assert actual.global_power_one == expected.global_power_one
    assert actual.global_power_two == expected.global_power_two
    np.testing.assert_array_equal(actual.null_error, expected.null_error)
    np.testing.assert_array_equal(actual.alternative_power_one, expected.alternative_power_one)
    np.testing.assert_array_equal(actual.alternative_power_two, expected.alternative_power_two)
    np.testing.assert_array_equal(actual.mean_enrolled_null, actual.design.patients_per_arm)


def test_interim_search_returns_corner_mcse_and_reports_enrollment_savings():
    policy = MERITInterims(
        0.1,
        0.8,
        toxicity_looks=[1],
        efficacy_looks=[1],
        toxicity_cutoff=0.5,
        efficacy_cutoff=0.5,
    )
    result = merit_interim_sample_size(
        interims=policy,
        toxicity_null=0.8,
        toxicity_alternative=0.1,
        efficacy_null=0.1,
        efficacy_alternative=0.8,
        alpha=0.99,
        power=0.01,
        max_patients_per_arm=6,
        trials=500,
        correlation=0.0,
        rng=17,
    )
    assert result.design.patients_per_arm > 1
    assert result.null_error.shape == (6,)
    assert result.alternative_power_one.shape == (3,)
    np.testing.assert_allclose(
        result.null_error_mcse,
        np.sqrt(result.null_error * (1 - result.null_error) / result.trials),
    )
    assert np.any(result.mean_enrolled_null < result.design.patients_per_arm)
    assert result.max_work_estimate > 0
    assert result.max_scratch_bytes_estimate > 0


def test_preflight_and_target_agreement_happen_before_rng_is_advanced():
    generator = np.random.default_rng(21)
    before = repr(generator.bit_generator.state)
    args = dict(
        interims=MERITInterims(0.2, 0.4),
        toxicity_null=0.4,
        toxicity_alternative=0.2,
        efficacy_null=0.2,
        efficacy_alternative=0.4,
        max_patients_per_arm=10,
        trials=100,
    )
    with pytest.raises(ValueError, match="max_work"):
        merit_interim_sample_size(**args, rng=generator, max_work=1)
    assert repr(generator.bit_generator.state) == before
    with pytest.raises(ValueError, match="must equal"):
        merit_interim_sample_size(**{**args, "interims": MERITInterims(0.3, 0.4)}, rng=generator)
    assert repr(generator.bit_generator.state) == before
