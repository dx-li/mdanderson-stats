import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import proportional_density_bootstrap
from mdanderson_stats.proportional_density_bootstrap import _failure_area


def test_paper_bootstrap_against_independent_r_and_time_units():
    root = Path(__file__).parent / "fixtures"
    case = json.loads((root / "proportional-density-native.json").read_text())["cases"][0]
    reference = json.loads((root / "proportional-density-bootstrap.json").read_text())
    fit = proportional_density_bootstrap(
        case["time"], case["event"], case["treatment"], replicates=5, seed=reference["seed"]
    )
    assert fit.tau == reference["tau"]
    assert_allclose(fit.bootstrap_statistics, reference["statistics"], atol=2e-13, rtol=0)
    assert fit.failed_replicates == 0
    assert fit.pvalue == fit.pvalue_lower == fit.pvalue_upper == 5 / 6
    assert fit.monte_carlo_standard_error == pytest.approx(1 / 6)
    scaled = proportional_density_bootstrap(
        np.array(case["time"]) * 100,
        case["event"],
        case["treatment"],
        replicates=5,
        seed=reference["seed"],
    )
    assert_allclose(scaled.bootstrap_statistics / 100, fit.bootstrap_statistics, atol=1e-13, rtol=0)
    assert_allclose(scaled.statistic / 100, fit.statistic, atol=1e-13, rtol=0)
    assert scaled.pvalue == fit.pvalue
    with pytest.raises(ValueError):
        fit.bootstrap_statistics[0] = 0
    # Exact integration uses left step values, truncates at tau, and includes
    # the constant tail after the last event. Hand calculation: .25*2 + .0625*1.
    assert (
        _failure_area(np.array([1.0, 3.0]), np.array([0.5, 0.5]), np.array([0.0, 0.75]), 4.0)
        == 0.5625
    )


def test_failed_resamples_remain_in_calibration_denominator():
    fit = proportional_density_bootstrap(
        [1, 3, 2, 4], [1] * 4, [0, 0, 1, 1], replicates=100, seed=21
    )
    assert 0 < fit.failed_replicates < 100
    assert np.isnan(fit.bootstrap_statistics).sum() == fit.failed_replicates
    assert sum(n for _, n in fit.failure_reasons) == fit.failed_replicates
    assert fit.pvalue is None and fit.monte_carlo_standard_error is None
    exceed = np.sum(fit.bootstrap_statistics >= fit.statistic)
    assert fit.pvalue_lower == (1 + exceed) / 101
    assert fit.pvalue_upper == (1 + exceed + fit.failed_replicates) / 101
    with pytest.raises(ValueError, match="common follow-up"):
        proportional_density_bootstrap([1, 3, 2, 4], [1] * 4, [0, 0, 1, 1], tau=4)
