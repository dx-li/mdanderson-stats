import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import prepare_km_coordinates, reconstruct_ipd


def test_original_r_cleaning_order_and_provenance():
    # Unmodified CRAN preprocess.R with dplyr, including a dip its rule retains.
    time = [5, 0, 1, 2, 2, 2, 3, 4, 6, 7, 8, 9, 10, np.nan]
    survival = [70, 100, 94, 90, 87, 88, 20, 79, 65, 60, 56, 50, 45, 20]
    x = prepare_km_coordinates(time, survival)
    assert_array_equal(x.time, [0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_allclose(x.survival, [1, 0.94, 0.9, 0.87] + [0.2] * 8, rtol=0, atol=1e-15)
    assert_array_equal(x.source_index, [1, 2, 3, 4, 6, 7, 0, 8, 9, 10, 11, 12])
    assert x.omitted_missing == x.omitted_outliers == 1
    assert x.adjusted_survival == 7
    assert not x.baseline_added
    assert not x.time.flags.writeable
    ipd = reconstruct_ipd(x.time, x.survival, patients=100)
    assert len(ipd.time) == 100
    assert ipd.event.sum() == 80


def test_flat_curve_baseline_and_invalid_data():
    # Inclusive native fences flag every zero span, removing the initial point.
    x = prepare_km_coordinates([0, 1, 2, 3, 4], [100] * 5)
    assert x.omitted_outliers == 1
    assert x.baseline_added
    assert_array_equal(x.time, [0, 1, 2, 3, 4])
    assert x.source_index[0] == -1
    assert reconstruct_ipd(x.time, x.survival, patients=20).event.sum() == 0
    with pytest.raises(ValueError, match="infinities"):
        prepare_km_coordinates([0, 1, 2, 3, np.inf], [1] * 5, scale=1)
    with pytest.raises(ValueError, match="too few finite"):
        prepare_km_coordinates([np.nan] * 5, [100] * 5)
    with pytest.raises(ValueError, match="no in-range"):
        prepare_km_coordinates([0, 1, 2, 3, 4], [-100] * 5)
