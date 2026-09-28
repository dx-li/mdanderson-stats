import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.proportional_density_full_bootstrap import (
    ProportionalDensityFullBootstrapTape,
    _full_curve_area,
    proportional_density_full_bootstrap,
)


def test_exact_right_continuous_area_and_identity_resample_tape():
    assert (
        _full_curve_area(
            np.array([1.0, 3.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 0.75]),
            4.0,
        )
        == 0.5625
    )

    root = Path(__file__).parent / "fixtures"
    case = json.loads((root / "proportional-density-native.json").read_text())["cases"][0]
    time = np.asarray(case["time"], dtype=float)
    event = np.asarray(case["event"], dtype=int)
    treatment = np.asarray(case["treatment"], dtype=int)
    support = np.unique(time[event == 1])
    event_draws = [
        np.searchsorted(support, time[(treatment == arm) & (event == 1)]) for arm in (0, 1)
    ]
    censor_draws = [
        np.arange(np.count_nonzero((treatment == arm) & (event == 0))) for arm in (0, 1)
    ]
    tape = ProportionalDensityFullBootstrapTape(
        control_event=event_draws[0][None, :],
        treatment_event=event_draws[1][None, :],
        control_censor=censor_draws[0][None, :],
        treatment_censor=censor_draws[1][None, :],
    )
    result = proportional_density_full_bootstrap(
        time, event, treatment, replicates=1, resample_tape=tape
    )
    assert result.failed_replicates == 0
    assert_allclose(result.bootstrap_statistics, [result.statistic], atol=2e-13, rtol=0)
    with pytest.raises(ValueError, match="seed cannot be supplied"):
        proportional_density_full_bootstrap(
            time, event, treatment, replicates=1, seed=1, resample_tape=tape
        )
