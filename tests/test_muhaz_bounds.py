"""Independent lower/upper defaults through all bandwidth-selection workflows."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import muhaz_global, muhaz_knn, muhaz_local


@pytest.mark.parametrize("function", [muhaz_global, muhaz_local, muhaz_knn])
@pytest.mark.parametrize(
    "bounds,expected", [((1, None), (1, 12)), ((None, 15), (0, 15)), ((None, None), (0, 12))]
)
def test_independent_bound_defaults_match_explicit_interval(function, bounds, expected):
    t = np.arange(1.0, 22.0)
    kwargs = {"neighbors": 3} if function is muhaz_knn else {"bandwidths": 0.7}
    a = function(t, bounds=bounds, **kwargs)
    b = function(t, bounds=expected, **kwargs)
    assert_allclose(a.time, b.time)
    assert_allclose(a.hazard, b.hazard)


@pytest.mark.parametrize("bounds", [(np.nan, None), (None, np.inf), (13, None), [1], [[1, 2]]])
def test_invalid_partial_bounds(bounds):
    with pytest.raises(ValueError):
        muhaz_global(np.arange(1, 22), bounds=bounds, bandwidths=0.7)
