import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import extsig


def test_original_fortran_all_orderings_and_mid_p():
    reference = {
        "difference": [
            0.031905308651956614,
            0.03174831140833668,
            0.0633899875269098,
            0.06279051884181798,
        ],
        "log_likelihood": [
            0.03843137684318817,
            0.03843137684318817,
            0.0668614379342715,
            0.0668614263465889,
        ],
        "chi_square": [
            0.031436080696427185,
            0.0314360806964132,
            0.05890861558415046,
            0.05830914689905863,
        ],
        "unpooled_z": [
            0.030565501701186427,
            0.0292638918605937,
            0.05210363659506713,
            0.0515041679099753,
        ],
        "fisher": [
            0.029659682965697244,
            0.029187223335434415,
            0.05888313802093932,
            0.05828366933584749,
        ],
    }
    for ordering, expected in reference.items():
        result = extsig(5, 23, 13, 27, ordering=ordering, nuisance="native_grid", tie_rule="native")
        observed = [
            result.one_sided,
            result.mid_p_one_sided,
            result.two_sided,
            result.mid_p_two_sided,
        ]
        assert_allclose([v.pvalue for v in observed], expected, rtol=2e-12)
        assert_allclose(result.fisher_one_sided, 0.04912, atol=5e-6)
        assert_allclose(result.fisher_two_sided, 0.07729, atol=5e-6)
        assert_allclose(result.chi_square_two_sided, 0.05250, atol=5e-6)


def test_continuous_bounds_contain_analytic_maxima():
    for ordering in ["difference", "log_likelihood", "chi_square", "unpooled_z", "fisher"]:
        result = extsig(0, 1, 2, 2, ordering=ordering)
        for maximum, expected in zip(
            [result.one_sided, result.mid_p_one_sided, result.two_sided, result.mid_p_two_sided],
            [4 / 27, 2 / 27, 0.25, 0.125],
            strict=True,
        ):
            assert maximum.lower_bound <= expected <= maximum.pvalue
            assert maximum.error_bound < 1e-8
        p = result.one_sided.nuisance_probability
        attained = p * p * (1 - p)
        assert result.one_sided.lower_bound <= attained <= result.one_sided.pvalue
    grid = extsig(0, 1, 2, 2, ordering="difference", nuisance="native_grid")
    assert grid.one_sided.pvalue < 4 / 27


def test_extreme_tail_symmetry_and_degenerate_data():
    result = extsig(0, 200, 200, 200, ordering="difference")
    expected = 2.0**-399
    assert result.two_sided.lower_bound <= expected <= result.two_sided.pvalue
    assert_allclose(result.two_sided.pvalue, expected, rtol=1e-7, atol=0)
    reversed_result = extsig(200, 200, 0, 200, ordering="difference")
    assert result.two_sided == reversed_result.two_sided
    assert reversed_result.direction == "p1 >= p2"
    null = extsig(0, 5, 0, 7, ordering="difference")
    assert null.two_sided.pvalue == 1
    assert np.isfinite(null.mid_p_two_sided.pvalue)
