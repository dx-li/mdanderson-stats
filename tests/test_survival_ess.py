import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln

from mdanderson_stats.survival_ess import survival_prior_ess


def test_expected_native_curvature_against_independent_integration():
    for a, b, c in [(1.2, 5, 3), (4, 2, 3), (20, 100, 0.2)]:
        # Integrate over prior lambda*b ~ Gamma(a,1), avoiding native MC error.
        def density(x):
            return np.exp((a - 1) * np.log(x) - x - gammaln(a))

        slope = quad(lambda x: x * x * (-np.expm1(-c * x / b)) * density(x), 0, np.inf)[0]
        intercept = a * (a + 1)
        dp = (a - 3) * (a - 1) ** 2
        exact = (dp + intercept) / slope
        result = survival_prior_ess(a, b, censoring_time=c)
        np.testing.assert_allclose(result.ess, exact, rtol=2e-9)
        reference_grid = np.linspace(1, 100, 50)
        expected_grid = reference_grid[np.argmin(abs(dp + intercept - reference_grid * slope))]
        assert result.grid_ess == expected_grid
        assert not result.patients.flags.writeable


def test_scale_invariance_search_limits_and_tiny_followup():
    # Original R essSurv, seed 154, 200000 prior draws: 1.428571.
    np.testing.assert_allclose(survival_prior_ess(4, 2, max_patients=4).grid_ess, 1.428571428571)
    base = survival_prior_ess(4, 2, censoring_time=3)
    for scale in [1e-200, 1e200]:
        result = survival_prior_ess(4, 2 * scale, censoring_time=3 * scale)
        np.testing.assert_allclose(result.ess, base.ess, rtol=1e-13)
    limited = survival_prior_ess(4, 1e300, censoring_time=1e-300, max_patients=10)
    assert limited.grid_ess == 10 and limited.at_search_boundary
    assert np.isfinite(limited.log_ess) and limited.log_ess > 1000
    complete = survival_prior_ess(4, 2, censoring_time=1e300)
    np.testing.assert_allclose(complete.ess, 1.45)
