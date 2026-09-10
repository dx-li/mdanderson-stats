import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import scan_accflf, search_accflf


def test_profile_shape_search_against_independent_r_and_grid():
    case = json.loads((Path(__file__).parent / "fixtures/accflf-search.json").read_text())
    time, event = case["time"], case["event"]
    search = search_accflf(time, event, fixed_p=0)
    assert search.converged
    assert search.failed_shapes == ()
    fit = search.best
    assert_allclose(
        [fit.q, np.log(fit.sigma), fit.coefficients[0], fit.log_likelihood],
        case["r_reference"],
        atol=1e-6,
        rtol=0,
    )
    grid = scan_accflf(time, event, p=[0, 1], q=[-0.5, 0.5, 1])
    assert grid.log_likelihood.shape == (2, 3)
    assert grid.errors == (None,) * 6
    assert grid.best is grid.fits[int(np.argmax(grid.log_likelihood))]
    assert fit.log_likelihood >= np.max(grid.log_likelihood[0])
    for i, fitted in enumerate(grid.fits):
        assert fitted is not None
        assert fitted.p == grid.p[i // 3]
        assert fitted.q == grid.q[i % 3]
        assert fitted.log_likelihood == grid.log_likelihood[i // 3, i % 3]
    assert not grid.log_likelihood.flags.writeable
    limited = search_accflf(time, event, fixed_p=0, starts=[[0, 0.5]], max_evaluations=10)
    assert not limited.converged
    assert limited.runs[0].evaluations == 10
    assert limited.best.log_likelihood <= search.best.log_likelihood
