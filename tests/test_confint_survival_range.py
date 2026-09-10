import numpy as np

from mdanderson_stats import confint_survival_hazard_range, confint_survival_probability


def test_confint_hazard_peak_and_two_crossings_against_independent_r():
    result = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5)
    np.testing.assert_allclose(result.peak_hazard, 0.15644553768140521, rtol=2e-5)
    np.testing.assert_allclose(result.peak.probability, 0.99997842687325389, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result.intervals, [(0.0028583338690436641, 0.8409063518812448512)], rtol=2e-9
    )
    assert not result.lower_clipped and not result.upper_clipped and not result.peak_at_boundary
    for bound in result.intervals[0]:
        np.testing.assert_allclose(
            confint_survival_probability(bound, 5, 10, 0, 0.5).probability, 0.5, atol=1e-10
        )
    refined = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5, grid_points=129)
    np.testing.assert_allclose(refined.intervals, result.intervals, rtol=2e-9)


def test_confint_hazard_range_clipping_empty_design_and_mean_target():
    clipped = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5, hazard_bounds=(0.1, 0.5))
    assert clipped.intervals == ((0.1, 0.5),)
    assert clipped.lower_clipped and clipped.upper_clipped
    descending = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5, hazard_bounds=(0.5, 2))
    assert descending.peak_at_boundary and descending.lower_clipped and not descending.upper_clipped
    np.testing.assert_allclose(descending.intervals, [(0.5, 0.8409063518812448512)], rtol=2e-9)
    assert confint_survival_hazard_range(0.01, 1, 0, 0.5, assurance=0.5).intervals == ()
    mean = confint_survival_hazard_range(5, 10, 0, 0.5, assurance=0.5, target="mean")
    np.testing.assert_allclose(mean.intervals, [(1.1974428957731611, 1000)], rtol=2e-9)
    assert mean.upper_clipped and not mean.lower_clipped


def test_confint_hazard_search_preserves_physical_time_units():
    for scale in [1e-200, 1e200]:
        result = confint_survival_hazard_range(
            5 / scale,
            10 * scale,
            0,
            0.5 / scale,
            assurance=0.5,
            hazard_bounds=(1e-4 / scale, 1000 / scale),
        )
        np.testing.assert_allclose(
            np.array(result.intervals) * scale,
            [(0.0028583338690436641, 0.8409063518812448512)],
            rtol=2e-8,
        )
        np.testing.assert_allclose(result.peak_hazard * scale, 0.15644553768140521, rtol=3e-5)
