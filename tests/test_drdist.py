import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import drdist


def test_native_menu_operations_and_corrected_chi_square():
    results = [
        drdist(1, 0.05, numerator_df=5, denominator_df=10),
        drdist(2, 2.5, numerator_df=5, denominator_df=10),
        drdist(3, 0.975, mean=2, sd=3),
        drdist(4, 1, mean=2, sd=3),
        drdist(5, 0.975, df=10),
        drdist(6, 2, df=10),
        drdist(7, 10, df=2),
    ]
    native = [
        3.3258352279663086,
        0.10200226306915283,
        7.879893183708191,
        0.369441419839859,
        2.228139638900757,
        0.9633060693740845,
        0.006737947463989258,
    ]
    assert_allclose(results, native, rtol=2e-5, atol=2e-6)
    # R pchisq; native overflow stub instead forces a cube-root approximation.
    assert_allclose(drdist(7, 10, df=5), 0.075235246146512197, rtol=1e-13)


def test_tail_conventions_broadcasting_and_extreme_inverse_roundtrips():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    for inverse, forward, kwargs in [
        (1, 2, dict(numerator_df=5, denominator_df=10)),
        (3, 4, dict(mean=2, sd=3)),
        (5, 6, dict(df=10)),
    ]:
        assert_allclose(drdist(forward, drdist(inverse, p, **kwargs), **kwargs), p, rtol=1e-11)
    f = drdist("inverse_f", 1e-30, numerator_df=5, denominator_df=10)
    assert_allclose(drdist("f", f, numerator_df=5, denominator_df=10), 1e-30, rtol=1e-11, atol=0)
    normal = drdist("inverse_normal", 1e-300, sd=1e-20)
    assert_allclose(drdist("normal", normal, sd=1e-20), 1e-300, rtol=1e-11, atol=0)
    assert drdist("normal", -15) > 0
    assert drdist("normal", [0, 1], mean=[[0], [1]]).shape == (2, 2)
    assert_allclose(drdist("inverse_t", [0, 1], df=10), [-np.inf, np.inf])
    assert_allclose(drdist("inverse_f", [0, 1], numerator_df=5, denominator_df=10), [np.inf, 0])


def test_normal_unit_scaling_avoids_intermediate_overflow():
    assert_allclose(drdist("normal", 1e308, mean=-1e308, sd=1e308), drdist("normal", 2), rtol=1e-14)
    p = float(drdist("normal", 2))
    assert_allclose(drdist("inverse_normal", p, mean=-1e308, sd=1e308) / 1e308, 1, rtol=1e-14)
