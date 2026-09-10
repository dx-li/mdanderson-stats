import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import ipd_cox_compare


def test_r_efron_cox_reference():
    x = ipd_cox_compare(
        [1, 2, 2, 4, 5, 7], [1, 1, 0, 1, 0, 1], [1, 2, 3, 4, 6, 8], [0, 1, 1, 0, 1, 1]
    )
    # R survival::coxph, exact time grouping, Efron ties.
    assert_allclose(x.log_hazard_ratio, -0.43030910662005284, atol=1e-10)
    assert_allclose(x.standard_error, 0.76960882544820530, atol=1e-10)
    assert_allclose(x.score_statistic, 0.31726642456256182, atol=1e-14)
    assert_allclose(x.score_pvalue, 0.57325507328419045, atol=1e-14)
    assert (x.patients, x.events) == (12, 8)
    assert x.lower < x.hazard_ratio < x.upper


def test_arm_swap_time_units_and_input_order():
    a, da = np.array([1, 2, 2, 4, 5, 7]), [1, 1, 0, 1, 0, 1]
    b, db = np.array([1, 2, 3, 4, 6, 8]), [0, 1, 1, 0, 1, 1]
    base = ipd_cox_compare(a, da, b, db)
    for scale in (1e-200, 1e200):
        x = ipd_cox_compare(b[::-1] * scale, db[::-1], a[::-1] * scale, da[::-1])
        assert_allclose(x.log_hazard_ratio, -base.log_hazard_ratio, atol=1e-12)
        assert_allclose(x.standard_error, base.standard_error, atol=1e-12)
        assert_allclose(x.score_pvalue, base.score_pvalue, atol=1e-14)
        assert_allclose(x.lower, 1 / base.upper, rtol=1e-12)


def test_nonidentifiable_and_separated_comparisons():
    with pytest.raises(ValueError, match="no events"):
        ipd_cox_compare([1, 2], [0, 0], [1, 2], [0, 0])
    with pytest.raises(ValueError, match="no finite Cox"):
        ipd_cox_compare([1, 2], [1, 1], [3, 4], [0, 0])
    # If all patients fail simultaneously, both groups inform the Efron model.
    x = ipd_cox_compare([1] * 4, [1] * 4, [1] * 4, [1] * 4)
    assert_allclose(x.log_hazard_ratio, 0, atol=1e-14)
    assert_allclose(x.score_pvalue, 1, atol=1e-14)
