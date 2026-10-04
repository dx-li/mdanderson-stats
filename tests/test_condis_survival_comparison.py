import numpy as np
import pytest

from mdanderson_stats import condis_impute
from mdanderson_stats.condis_survival_comparison import (
    condis_survival_comparison,
    plot_condis_survival_comparison,
)


def test_comparison_uses_original_censoring_and_imputed_all_event_curve():
    fit = condis_impute([0, 2, 2, 4], [1, 0, 1, 0], interpolation="step")
    result = condis_survival_comparison(fit, risk_times=[0, 2, 4, 5])

    np.testing.assert_array_equal(result.observed_curve.time, [0, 2, 4])
    np.testing.assert_array_equal(result.observed_curve.events, [1, 1, 0])
    np.testing.assert_array_equal(result.observed_curve.censored, [0, 1, 1])
    np.testing.assert_array_equal(result.imputed_curve.events, [1, 1, 2])
    np.testing.assert_array_equal(result.imputed_curve.censored, [0, 0, 0])
    np.testing.assert_array_equal(result.censor_time, [2, 4])
    np.testing.assert_allclose(result.censor_survival, [0.5, 0.5])
    np.testing.assert_array_equal(result.observed_at_risk, [4, 3, 1, 0])
    np.testing.assert_array_equal(
        result.imputed_at_risk,
        [4, np.count_nonzero(fit.imputed_time >= 2), np.count_nonzero(fit.imputed_time >= 4), 0],
    )
    assert not result.risk_times.flags.writeable
    assert not result.observed_at_risk.flags.writeable


def test_comparison_rejects_unbounded_or_malformed_risk_times():
    fit = condis_impute([1, 2, 3], [1, 0, 1])
    with pytest.raises(ValueError, match="at most 20"):
        condis_survival_comparison(fit, risk_times=list(range(21)))
    with pytest.raises(ValueError, match="strictly increasing"):
        condis_survival_comparison(fit, risk_times=[0, 2, 2])
    with pytest.raises(ValueError, match="finite"):
        condis_survival_comparison(fit, risk_times=[0, np.inf])


def test_plot_uses_two_curve_axes_and_explicit_risk_table():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = condis_impute([1, 2, 4], [1, 0, 1])
    comparison = condis_survival_comparison(fit, risk_times=[0, 2, 4])
    curve_ax, risk_ax = plot_condis_survival_comparison(comparison)
    assert curve_ax.get_xlabel() == "Follow-up time"
    assert [line.get_label() for line in curve_ax.lines] == ["Censored", "CondiS imputed"]
    assert risk_ax.tables
    plt.close(curve_ax.figure)
