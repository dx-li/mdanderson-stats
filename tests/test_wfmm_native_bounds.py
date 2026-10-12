"""Native likelihood-width references distinguish root bounds and sentinels."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-bounds.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_original_conditional_width_roots_and_invalid_sentinels(case):
    trace = case["traces"][0]
    model = case["model"]
    y, x = np.array(case["y"]), np.array(model["X"])
    beta = np.array(trace["beta"])
    mle, lower, upper = (np.array(trace[k]) for k in ("mle", "lower", "upper"))
    np.testing.assert_array_equal(mle, case["output"]["omega_MLE"])
    np.testing.assert_array_equal((upper - lower) / 2, case["output"]["se_omega"])
    components = []
    offset = 0
    for count in model.get("m", []):
        z = np.array(model["Z"])[:, offset : offset + int(count)]
        components.append(z @ z.T)
        offset += int(count)
    components.extend(np.diag(c) for c in np.array(model["C"]).T)
    for j in range(y.shape[1]):
        r = y[:, j] - x @ beta[:, j]

        def nll(v):
            covariance = sum(s * q for s, q in zip(v, components, strict=True))
            return 0.5 * (np.linalg.slogdet(covariance)[1] + r @ np.linalg.solve(covariance, r))

        peak = nll(mle[:, j])
        for h in range(len(components)):
            if upper[h, j] == 0:
                # At the native floor, both returned bounds are zero. This is
                # not a confidence interval or a usable positive proposal SD.
                assert lower[h, j] == 0 and mle[h, j] == 1e-6
                continue

            def root(value):
                v = mle[:, j].copy()
                v[h] = value
                return nll(v) - peak - 1

            hi = brentq(root, mle[h, j], upper[h, j] * 1.01)
            assert upper[h, j] == pytest.approx(hi, abs=5e-5, rel=0)
            if lower[h, j] < 0:
                # When no nonnegative lower root exists, native writes -upper;
                # it doubles the scale relative to a floor-truncated interval.
                assert lower[h, j] == -upper[h, j]
                assert root(1e-6) < 0
            else:
                lo = brentq(root, 1e-6, mle[h, j])
                assert lower[h, j] == pytest.approx(lo, abs=5e-5, rel=0)
