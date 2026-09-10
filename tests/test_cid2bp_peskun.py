import numpy as np

from mdanderson_stats import cid2bp_interval


def test_compiled_original_fortran_reference_and_boundary_adjustment():
    # Original cfpesk compiled with gfortran -std=legacy, z=1.959963984540054.
    for args, expected in [
        ((7, 12, 1, 7), [-0.031337541969674233, 0.76625276842079237]),
        # Source upper-gap branch uses .10 although nearest positive gap is .15.
        ((3, 5, 0, 4), [-0.080295502337045035, 0.91848568857132684]),
    ]:
        ci = cid2bp_interval(*args, method="peskun_native")
        np.testing.assert_allclose([ci.lower, ci.upper], expected, atol=2e-15)
    boundary = cid2bp_interval(10, 10, 0, 10, method="peskun_native")
    np.testing.assert_allclose(boundary.lower, 2 * 0.025 ** (1 / 20) - 1)
    assert boundary.upper == 1
