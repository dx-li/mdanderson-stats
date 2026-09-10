import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import anovaddp_amplitude_posterior, anovaddp_curve, anovaddp_loglikelihood


def test_native_response_curves_and_likelihood():
    reference = json.loads((Path(__file__).parent / "fixtures/anovaddp-kernels.json").read_text())
    curves = anovaddp_curve(reference["parameters"], reference["time"])
    assert_allclose(curves, reference["curves"], atol=1e-14, rtol=0)
    repaired = anovaddp_curve([2, -1, 4, 3, 1, -0.7], [2, 3, 4, 5], repair_order=True)
    assert_allclose(repaired, anovaddp_curve([2, -1, 4, 3, 4, -0.7], [2, 3, 4, 5]))
    assert_allclose(
        anovaddp_curve([2, -1, 4, 1, 3, 1e200], [3, 4, 1e200]), [-1 + 4 / (1 + np.exp(2)), 3, 3]
    )
    y = np.array(reference["curves"][0]) + 0.5
    ll = anovaddp_loglikelihood(reference["parameters"][0], reference["time"], y, 0.25)
    assert ll == -len(y) / 2
    norm = anovaddp_loglikelihood(
        reference["parameters"][0], reference["time"], y, 0.25, normalized=True
    )
    assert norm == pytest.approx(float(ll) - len(y) / 2 * np.log(2 * np.pi * 0.25))
    with pytest.raises(ValueError):
        curves[0, 0] = 0


def test_correlated_amplitude_conditional_against_r():
    reference = json.loads((Path(__file__).parent / "fixtures/anovaddp-kernels.json").read_text())[
        "conditional"
    ]
    fit = anovaddp_amplitude_posterior(
        reference["parameters"],
        reference["time"],
        reference["observations"],
        prior_mean=reference["prior_mean"],
        prior_covariance=reference["prior_covariance"],
        variance=0.4,
    )
    assert_allclose(fit.mean, reference["mean"], atol=1e-13, rtol=0)
    assert_allclose(fit.covariance, reference["covariance"], atol=1e-13, rtol=0)
    assert np.all(np.linalg.eigvalsh(fit.covariance) > 0)
    assert_allclose(
        fit.design @ np.array(reference["parameters"][:3]),
        anovaddp_curve(reference["parameters"], reference["time"]),
        atol=1e-15,
        rtol=0,
    )
    with pytest.raises(ValueError, match="positive definite"):
        anovaddp_amplitude_posterior(
            reference["parameters"],
            reference["time"],
            reference["observations"],
            prior_mean=reference["prior_mean"],
            prior_covariance=np.zeros((6, 6)),
            variance=0.4,
        )
