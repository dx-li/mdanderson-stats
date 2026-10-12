"""Independent Gaussian optima and native restricted-fit reference contracts."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.wfmm_conditional import fit_wfmm_conditional_variances

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-bounds.json").read_text())[
    "cases"
]


def inputs(case):
    m = case["model"]
    t = case["traces"][0]
    counts = np.array(m.get("m", []), int)
    h = len(counts)
    v = np.array(t["theta_MOM"])
    return dict(
        coefficients=case["y"],
        fixed_design=m["X"],
        fixed_coefficients=t["beta"],
        random_design=m.get("Z"),
        random_variance=v[:h] if h else None,
        residual_variance=v[h:],
        random_strata=np.repeat(np.arange(h), counts) if h else None,
        residual_strata=np.argmax(m["C"], axis=1),
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
@pytest.mark.parametrize("profile", ["stable", "legacy_3_1"])
def test_conditional_objective_widths_and_native_restricted_optimum(case, profile):
    result = fit_wfmm_conditional_variances(**inputs(case), profile=profile)
    assert result.optimizer_success.all()
    assert max(result.projected_gradient) < 1e-5
    assert set(result.status) == {"converged"}
    m = case["model"]
    x = np.array(m["X"])
    y = np.array(case["y"])
    t = case["traces"][0]
    beta = np.array(t["beta"])
    components = []
    offset = 0
    for count in m.get("m", []):
        z = np.array(m["Z"])[:, offset : offset + int(count)]
        components.append(z @ z.T)
        offset += int(count)
    components.extend(np.diag(v) for v in np.array(m["C"]).T)
    variance = np.vstack((result.random_variance, result.residual_variance))
    native = np.array(t["mle"])
    for j in range(y.shape[1]):
        residual = y[:, j] - x @ beta[:, j]

        def nll(v):
            covariance = sum(a * b for a, b in zip(v, components, strict=True))
            return 0.5 * (
                np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(covariance, residual)
            )

        optimum = nll(variance[:, j])
        assert result.log_likelihood[j] == pytest.approx(-optimum, abs=2e-12)
        assert optimum <= nll(native[:, j]) + 1e-10
        if profile == "legacy_3_1":
            # Same restricted objective with tighter numerical optimization;
            # native Brent iteration/rounding identity is deliberately not claimed.
            assert nll(native[:, j]) - optimum < 3e-7
        for h in range(len(components)):
            if result.frozen[h, j]:
                assert result.proposal_sd[h, j] == 0
                continue
            v = variance[:, j].copy()
            v[h] = result.upper[h, j]
            assert nll(v) - optimum == pytest.approx(1, abs=2e-8)
            if result.artificial_lower[h, j]:
                assert profile == "legacy_3_1"
                assert result.lower[h, j] == -result.upper[h, j]
            elif result.lower_truncated[h, j]:
                assert profile == "stable" and result.lower[h, j] == pytest.approx(1e-6)
            else:
                v[h] = result.lower[h, j]
                assert nll(v) - optimum == pytest.approx(1, abs=2e-8)
            assert result.proposal_sd[h, j] == pytest.approx(
                0.75 * (result.upper[h, j] - result.lower[h, j])
            )


def test_fixed_beta_identity_ml_is_rss_over_n_not_profile_ml_or_reml():
    case = CASES[0]
    result = fit_wfmm_conditional_variances(**inputs(case))
    y = np.array(case["y"])
    beta = np.array(case["traces"][0]["beta"])
    expected = np.mean((y - beta) ** 2, axis=0)
    np.testing.assert_allclose(result.residual_variance[0], expected, rtol=1e-8)
    assert np.all(abs(result.residual_variance[0] - y.var(axis=0)) > 0.009)


def test_mathematical_optimizer_leaves_floor_and_matches_balanced_closed_form():
    case = CASES[-1]
    stable = fit_wfmm_conditional_variances(**inputs(case))
    legacy = fit_wfmm_conditional_variances(**inputs(case), profile="legacy_3_1")
    residual = (np.array(case["y"]) - np.array(case["traces"][0]["beta"])).reshape(4, 2, 4)
    means = residual.mean(axis=1)
    s = ((residual - means[:, None, :]) ** 2).sum(axis=(0, 1)) / 4
    q = (means**2).mean(axis=0) - s / 2
    np.testing.assert_allclose(stable.random_variance[0], q, rtol=1e-7)
    np.testing.assert_allclose(stable.residual_variance[0], s, rtol=1e-7)
    assert legacy.frozen_descent[0, 1:3].all()
    assert not legacy.positive_proposals and stable.positive_proposals
    assert not stable.artificial_lower.any()
    base = fit_wfmm_conditional_variances(**inputs(CASES[2]), profile="legacy_3_1")
    assert base.artificial_lower.any()


def test_iteration_failure_is_a_hold_not_usable_proposals():
    case = CASES[1]
    result = fit_wfmm_conditional_variances(**inputs(case), max_iterations=1)
    assert not result.optimizer_success.all()
    assert not result.positive_proposals
    assert "optimizer_hold" in result.status


def test_owned_readonly_outputs():
    result = fit_wfmm_conditional_variances(**inputs(CASES[0]))
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert value.flags.owndata and not value.flags.writeable


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "unknown"),
        ("min_variance", 0),
        ("min_variance", True),
        ("min_variance", 1e101),
        ("proposal_multiplier", 0),
        ("proposal_multiplier", False),
        ("max_iterations", True),
        ("max_iterations", 1001),
        ("fixed_coefficients", [[np.nan]]),
        ("fixed_coefficients", [[0]]),
        ("residual_variance", 0),
        ("residual_variance", 1e-7),
        ("random_variance", [[1, 2, 3, 4]]),
        ("coefficients", np.ones((500, 4))),
    ],
)
def test_invalid_and_excessive_work_rejected(key, value):
    with pytest.raises((ValueError, ArithmeticError)):
        fit_wfmm_conditional_variances(**(inputs(CASES[0]) | {key: value}))
