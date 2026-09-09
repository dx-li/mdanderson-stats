"""Independent gamma-ratio values, extreme corrections and exact input sums."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from test_cdflib_beta_support_reference import FIXTURE, independent
from test_cdflib_gamma_support_reference import EULER

from mdanderson_stats import algdiv, bcorr, gsumln

FUNCTIONS = {1: algdiv, 2: bcorr, 5: gsumln}


@pytest.mark.parametrize("case", [c for c in FIXTURE["cases"] if c["mode"] in FUNCTIONS])
def test_native_grid_against_independent_values(case):
    mode, a, b = case["mode"], case["a"], case["b"]
    function = FUNCTIONS[mode]
    if mode == 1 and a + b <= 0:
        with pytest.raises(ValueError):
            function(a, b)
        return
    expected = independent(mode, a, b)
    if math.isinf(expected):
        with pytest.raises(ArithmeticError):
            function(a, b)
        return
    result = float(function(a, b))
    np.testing.assert_allclose(result, expected, rtol=5e-14, atol=1e-323)
    if mode == 5 and expected > 0:
        assert result > 0
    if mode == 1 and a == 0:
        assert result == 0


@pytest.mark.parametrize(
    "a,b",
    [
        (-1e-100, 8.0),
        (-5e-324, 8.0),
        (-1e-10, 8.0),
        (-0.1, 8.0),
        (-1.0, 8.0),
        (np.nextafter(-8.0, 0), 8.0),
        (-7.5, 8.0),
        (-1e100, 2e100),
        (-1.0, 1e100),
        (-1e-100, 1e308),
        (-1e-10, 1e308),
        (1e-10, 1e308),
        (1e-100, 1e308),
        (1e-309, 8.0),
        (1.0, 8.0),
        (0.1, 9.0),
        (1e305, 8.0),
        (1e305, 1e305),
        (1e305, np.finfo(float).max),
    ],
)
def test_ratio_near_zero_negative_shifts_and_wide_ranges(a, b):
    np.testing.assert_allclose(
        algdiv(a, b), independent(1, float(a), float(b)), rtol=5e-14, atol=1e-323
    )


@pytest.mark.parametrize(
    "a,b",
    [
        (1.0, 1 + 2**-52),
        (1 + 2**-52, 1.0),
        (1.0, 1.25),
        (1.0, np.nextafter(1.25, 2)),
        (1.5, 1.75),
        (2.0, 2.0),
    ],
)
def test_sum_preserves_small_offsets_and_switches(a, b):
    result = float(gsumln(a, b))
    np.testing.assert_allclose(result, independent(5, a, float(b)), rtol=5e-14, atol=0)
    if a + b == 2 and (a > 1 or b > 1):
        assert result > 0  # Exact real input sum differs from its rounded float sum.


@pytest.mark.parametrize(
    "function,args", [(algdiv, (0.5, 8.0)), (bcorr, (8.0, 10.0)), (gsumln, (1.0, 1.5))]
)
def test_broadcasting_immutability_and_empty_arrays(function, args):
    a = np.full((2, 1), args[0])
    b = np.full((1, 3), args[1])
    expected = np.full((2, 3), float(function(*args)))
    result = function(a, b)
    a[:] = 99
    b[:] = 99
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    assert function(np.empty((0, 1)), np.empty((1, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        function(np.ones(2), np.ones(3))


@pytest.mark.parametrize(
    "function,a,b",
    [
        (algdiv, 0, 7.9),
        (algdiv, -8, 8),
        (algdiv, -9, 8),
        (algdiv, np.inf, 8),
        (algdiv, 1, np.nan),
        (bcorr, 7.9, 8),
        (bcorr, 8, 7.9),
        (bcorr, np.inf, 8),
        (gsumln, 0.9, 1),
        (gsumln, 1, 2.1),
        (gsumln, np.nan, 1),
    ],
)
def test_invalid_domains(function, a, b):
    with pytest.raises(ValueError):
        function(a, b)


def test_ratio_output_overflow_and_identical_gamma_values():
    with pytest.raises(ArithmeticError):
        algdiv([0, 1e308], [8, 8])
    np.testing.assert_array_equal(algdiv(0, [8, 1e100, 1e308]), [0, 0, 0])


@pytest.mark.parametrize("a,b", [(8.0, 8.0), (8.0, 1e308), (1e100, 1e308), (1e308, 1e308)])
def test_stirling_correction_symmetry_and_positivity(a, b):
    assert float(bcorr(a, b)) > 0
    assert float(bcorr(a, b)) == float(bcorr(b, a))
    np.testing.assert_allclose(bcorr(a, b), independent(2, a, b), rtol=5e-14, atol=1e-323)


@pytest.mark.parametrize("a", [-1e-100, -5e-324, 1e-100, 5e-324])
def test_tiny_ratio_against_exact_harmonic_digamma_identity(a):
    with localcontext() as ctx:
        ctx.prec = 100
        harmonic = sum((Decimal(1) / k for k in range(1, 8)), Decimal(0))
        expected = float(-Decimal.from_float(a) * (harmonic - EULER))
    assert float(algdiv(a, 8)) == pytest.approx(expected, rel=5e-15, abs=5e-324)
    assert independent(1, a, 8.0) == expected
