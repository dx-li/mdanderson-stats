import numpy as np
import pytest
from scipy.special import betaincc

from mdanderson_stats.bacis_trial import bacis_one_trial


def test_singleton_one_trial_rows_keep_full_precision_and_separate_report_rounding():
    result = bacis_one_trial([2], [25], draws=256, warmup=0, seed=202)
    assert result.values.shape == (10, 1)
    assert result.row_labels[0] == "Prob(p_i>phi_1)"
    assert result.row_labels[-1] == "Effective sample size"
    assert result.values[5, 0] == pytest.approx(3 / 27)
    assert result.values[6, 0] == pytest.approx(2 / 25)
    assert result.values[7, 0] == 2
    assert result.values[8, 0] == 25
    assert result.values[0, 0] == pytest.approx(betaincc(3, 24, 0.1))
    assert result.values[1, 0] == pytest.approx(betaincc(3, 24, 0.3))
    np.testing.assert_array_equal(result.report_values, np.round(result.values, 3))
    assert result.report_values[5, 0] == 0.111
    assert result.values[5, 0] != result.report_values[5, 0]
    assert not result.values.flags.writeable
    assert not result.successes.flags.writeable
    assert not result.trials.flags.writeable


def test_five_group_tiny_borrowing_trial_runs_once_and_conserves_rows():
    result = bacis_one_trial(
        [0, 1, 2, 5, 10],
        [25] * 5,
        draws=16,
        warmup=0,
        chains=2,
        seed=91,
    )
    assert result.values.shape == (10, 5)
    np.testing.assert_array_equal(result.values[7], [0, 1, 2, 5, 10])
    np.testing.assert_array_equal(result.values[8], [25] * 5)
    np.testing.assert_array_equal(result.values[3], result.fit.classification.cluster == 2)
    np.testing.assert_array_equal(result.values[4], result.fit.efficacious)
    np.testing.assert_allclose(
        result.values[9], result.equivalent_sample_size.equivalent_sample_size
    )
    assert np.isfinite(result.values).all()
