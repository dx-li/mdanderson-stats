import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_poisson


def test_original_poisson_smo_and_inverses():
    x = asypow_smo_poisson([2, 4], group_size=[1, 2])
    assert_allclose(x.null_parameters, 10 / 3, rtol=1e-14)
    assert_allclose(x.divergence_per_observation, 0.29128080454643612, rtol=1e-13)
    assert_allclose(x.power(30), 0.79441660256420221, atol=1e-13)
    assert_allclose(x.sample_size(), 30.37914057915722, rtol=1e-12)
    uncorrected = asypow_smo_poisson([2, 4], group_size=[1, 2], subtract_df=False)
    assert_allclose(uncorrected.power(30), 0.84040417524447775, atol=1e-13)
    fixed = asypow_smo_poisson([2, 4], null_means=3)
    assert fixed.degrees_of_freedom == 2
    assert_allclose(fixed.power(50, fixed.significance(50)), 0.8, atol=1e-13)
    assert_allclose(fixed.power(fixed.sample_size()), 0.8, atol=1e-13)


def test_poisson_divergence_extremes_and_close_alternatives():
    for p in [1e-300, 1.0, 1e300]:
        q = p * (1 + 1e-10)
        x = asypow_smo_poisson([p], null_means=q)
        relative = (q - p) / p
        # Twice KL = p*r^2*(1 - 2*r/3 + O(r^2)).
        assert_allclose(
            x.divergence_per_observation,
            p * relative**2 * (1 - 2 * relative / 3),
            rtol=1e-11,
            atol=5e-324,
        )
    a = asypow_smo_poisson([2, 4], group_size=[1e300, 2e300])
    b = asypow_smo_poisson([2e-200, 4e-200], group_size=[1, 2])
    assert_allclose(b.divergence_per_observation / 1e-200, a.divergence_per_observation, rtol=1e-12)
    # Unweighted first-group KL exceeds float range, but its weighted result fits.
    extreme = asypow_smo_poisson([1e308, 1], null_means=[1e-300, 1], group_size=[1e-300, 1])
    assert_allclose(extreme.divergence_per_observation, 2e8 * (608 * np.log(10) - 1), rtol=1e-12)
    assert asypow_smo_poisson([1e308, 1e308]).divergence_per_observation == 0
    assert not a.null_parameters.flags.writeable
    with pytest.raises(ValueError, match="positive"):
        asypow_smo_poisson([0, 1])
