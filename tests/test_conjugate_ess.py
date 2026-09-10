import numpy as np
import pytest

from mdanderson_stats.conjugate_ess import conjugate_prior_ess


def test_native_r_conjugate_results_and_explicit_exposure_convention():
    # Unmodified BayesESS 0.1.19 ess.R/internal.R, commit 4bbf4df.
    cases = [
        ("beta_binomial", [2, 3], 5),
        ("gamma_exponential", [2, 7], 7),
        ("gamma_poisson", [2, 7], 7),
        ("dirichlet_multinomial", [1, 2, 3], 6),
        # R specifies sampling variance 12 and n0=4, hence prior variance 3.
        ("normal_normal", [0, 12, 3], 4),
        ("inverse_chi_squared_normal", [8, 3], 8),
        ("inverse_gamma_normal", [4, 6], 8),
    ]
    for model, parameters, expected in cases:
        assert conjugate_prior_ess(model, parameters, convention="native") == expected
    assert conjugate_prior_ess("gamma_exponential", [2, 7]) == 2
    # A time-unit change rescales the gamma rate, but not prior equivalent count.
    assert conjugate_prior_ess("gamma_exponential", [2, 7e100]) == 2


def test_information_updates_vectorization_and_extreme_scales():
    # Posterior effective information adds n for binomial/normal/variance models.
    for model, prior, posterior in [
        ("beta_binomial", [2, 3], [4, 6]),
        ("dirichlet_multinomial", [1, 2, 3], [2, 5, 4]),
        ("gamma_exponential", [2, 7], [7, 19]),
        ("gamma_poisson", [2, 7], [14, 12]),
        ("inverse_chi_squared_normal", [8, 3], [13, 4]),
        ("inverse_gamma_normal", [4, 6], [6.5, 19]),
        ("normal_normal", [-2, 12, 3], [1, 12, 12 / 9]),
    ]:
        assert conjugate_prior_ess(model, posterior) - conjugate_prior_ess(model, prior) == 5
    values = conjugate_prior_ess("normal_normal", [[-2, 1e-200, 2e-200], [3, 1e200, 2e200]])
    np.testing.assert_array_equal(values, [0.5, 0.5])
    assert not values.flags.writeable
    np.testing.assert_allclose(conjugate_prior_ess("dirichlet_multinomial", [0.1, 0.2, 0.3]), 0.6)
    with pytest.raises(ArithmeticError):
        conjugate_prior_ess("normal_normal", [0, 1e300, 1e-300])
    with pytest.raises(ArithmeticError):
        conjugate_prior_ess("normal_normal", [0, 1e-300, 1e300])
    with pytest.raises(ValueError):
        conjugate_prior_ess("beta_binomial", [1, 0])
