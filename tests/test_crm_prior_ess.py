from math import erf, exp, pi, sqrt

import numpy as np
import pytest

from mdanderson_stats.crm_prior_ess import (
    _information_for_patient,
    _posterior_moments,
    simulate_crm_prior_ess,
)


def test_adaptive_crm_and_rao_blackwell_information_match_source_reference():
    uniforms = np.array(
        [
            0.85444291937165,
            0.675258936593309,
            0.736596974777058,
            0.389275512425229,
            0.164962356444448,
            0.875901142833754,
            0.764744409825653,
            0.38052157103084,
            0.49425824219361,
            0.886865027714521,
            0.906866500386968,
            0.646414319286123,
        ]
    )
    result = simulate_crm_prior_ess(
        [0.08, 0.18, 0.30, 0.45],
        [0.05, 0.15, 0.30, 0.50],
        0.25,
        max_patients=12,
        replications=1,
        beta_sd=1.15758369027902,
        outcome_uniforms=uniforms[None, :],
    )

    np.testing.assert_array_equal(result.dose_indices[0], [1, 2, 3, 4, 3, 2, 2, 2, 3, 3, 3, 3])
    np.testing.assert_array_equal(result.outcomes[0], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        result.beta_mean_before_patient[0],
        [
            0.0,
            0.257441717646434,
            0.491144040261997,
            0.723230484604071,
            0.0782210348396812,
            -0.325408365344774,
            -0.194896034276495,
            -0.0937433718333906,
            -0.0124765693027964,
            0.0856297776701288,
            0.168588860575462,
            0.23981108083507,
        ],
        rtol=2e-8,
        atol=2e-9,
    )
    assert result.final_beta_mean[0] == pytest.approx(0.301753310729686, abs=2e-8)
    assert result.final_beta_variance[0] == pytest.approx(0.14425292170627, abs=2e-8)
    assert result.replication_information[0] == pytest.approx(5.74380375988402, abs=2e-13)
    assert result.continuous_ess is not None
    assert result.crossing_status == "reached"
    assert not result.dose_indices.flags.writeable


def test_crm_likelihood_information_is_stable_near_skeleton_boundaries():
    assert _information_for_patient(1e-300, 0) == pytest.approx(
        4.764800544151576e-295, rel=3e-14, abs=0.0
    )
    assert _information_for_patient(1e-300, 1) == pytest.approx(
        690.7755278982137, rel=3e-14, abs=0.0
    )
    assert _information_for_patient(np.nextafter(1.0, 0.0), 0) == pytest.approx(
        5.551115123125783e-17, rel=3e-14, abs=0.0
    )
    assert _information_for_patient(np.nextafter(1.0, 0.0), 1) == pytest.approx(
        1.1102230246251565e-16, rel=3e-14, abs=0.0
    )


def test_full_and_native_moment_conventions_handle_narrow_and_diffuse_priors():
    empty_doses = np.empty(0, dtype=np.int64)
    empty_outcomes = np.empty(0, dtype=np.int8)
    skeleton = np.array([0.3])

    full_narrow = _posterior_moments(empty_doses, empty_outcomes, skeleton, 1e-4, "full")
    native_narrow = _posterior_moments(
        empty_doses, empty_outcomes, skeleton, 1e-4, "native_truncated_numerator"
    )
    assert full_narrow[0] == pytest.approx(0.0, abs=1e-16)
    assert native_narrow[0] == pytest.approx(0.0, abs=1e-16)
    assert full_narrow[1] == pytest.approx(1e-8, rel=2e-13)
    assert native_narrow[1] == pytest.approx(1e-8, rel=2e-13)

    sd = 4.0
    full = _posterior_moments(empty_doses, empty_outcomes, skeleton, sd, "full")
    native = _posterior_moments(
        empty_doses, empty_outcomes, skeleton, sd, "native_truncated_numerator"
    )
    a = 10.0 / sd
    mass = erf(a / sqrt(2.0))
    density = exp(-0.5 * a * a) / sqrt(2.0 * pi)
    expected_native_variance = sd * sd * (mass - 2.0 * a * density)
    assert full[:2] == pytest.approx((0.0, 16.0), abs=2e-12)
    assert native[0] == pytest.approx(0.0, abs=2e-12)
    assert native[1] == pytest.approx(expected_native_variance, rel=2e-9)
    assert native[1] < full[1]
