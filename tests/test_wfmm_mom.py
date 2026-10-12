"""Native iteration references and independent moment/GLS identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.wfmm_mom import wfmm_mom_initialization

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-mom.json").read_text())["cases"]


def inputs(case):
    model = case["model"]
    settings = case["settings"]
    counts = np.array(model.get("m", []), dtype=int)
    groups = np.repeat(np.arange(len(counts)), counts)
    return dict(
        coefficients=case["y"],
        fixed_design=model["X"],
        random_design=model.get("Z"),
        random_strata=groups if groups.size else None,
        residual_strata=np.argmax(model["C"], axis=1),
        **{
            b: settings[a]
            for a, b in [
                ("minVC", "min_variance"),
                ("VC0_thresh", "zero_threshold"),
                ("omega_MOM_convcrit", "tolerance"),
                ("omega_MOM_maxiter", "max_iterations"),
            ]
            if a in settings
        },
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case"])
def test_native_mom_outputs_and_every_traced_normal_equation(case):
    result = wfmm_mom_initialization(**inputs(case), profile="legacy_3_1")
    # Native Woodbury inverse loses precision when residual floors meet large
    # random components (condition numbers up to 1e8). Preserve tight tolerances
    # for well-conditioned designs; explicitly bound these floor cases.
    tolerance = (
        2e-6 if case["case"] in ("three_residual_strata", "two_random_residual_strata") else 1e-8
    )
    expected = np.array(case["output"]["omega_MOM"])
    actual = np.vstack((result.random_variance, result.residual_variance))
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=3e-13)
    assert result.iterations == len(case["traces"])
    # Independently reconstruct each source RHS and raw solution. These do not
    # use the production initializer's covariance/projector helpers.
    y = np.array(case["y"])
    model = case["model"]
    x, c = np.array(model["X"]), np.array(model["C"])
    projectors, components = [], []
    offset = 0
    for count in model.get("m", []):
        z = np.array(model["Z"])[:, offset : offset + int(count)]
        projectors.append(z @ np.linalg.solve(z.T @ z, z.T))
        components.append(z @ z.T)
        offset += int(count)
    h = len(components)
    for column in c.T:
        projectors.append(np.diag(column))
        components.append(np.diag(column))
    a = np.array([[np.trace(p @ q) for q in components] for p in projectors])
    mathematical_a = a.copy()
    for column in range(h + 1, len(a)):
        a[:, column] = 0
        a[column, column] = mathematical_a[column, column]
    np.testing.assert_allclose(result.normal_equations, a, atol=2e-14)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for row in case["traces"]:
        residual = y - x @ beta
        b = np.array([np.sum(residual * (p @ residual), axis=0) for p in projectors])
        np.testing.assert_allclose(row["A"], a, rtol=1e-13, atol=2e-14)
        np.testing.assert_allclose(row["b"], b, rtol=tolerance, atol=3e-12)
        raw = np.linalg.solve(a, b)
        np.testing.assert_allclose(row["solution"], raw, rtol=tolerance, atol=3e-12)
        variance = np.maximum(raw, result.min_variance)
        variance[:h][variance[:h] <= result.zero_threshold] = 0
        for j in range(y.shape[1]):
            v = sum(s * q for s, q in zip(variance[:, j], components, strict=True))
            vi = np.linalg.inv(v)
            beta[:, j] = np.linalg.solve(x.T @ vi @ x, x.T @ vi @ y[:, j])
    np.testing.assert_allclose(result.fixed_effect_estimates, beta, rtol=tolerance, atol=1e-12)
    assert result.converged == (result.maximum_change <= result.tolerance)


def test_identity_ordinary_ml_variance_and_ols_are_exact_moment_solution():
    y = np.array(CASES[0]["y"])
    result = wfmm_mom_initialization(y, np.ones((8, 1)))
    np.testing.assert_allclose(result.residual_variance[0], y.var(axis=0))
    np.testing.assert_allclose(result.fixed_effect_estimates[0], y.mean(axis=0))
    assert result.converged
    assert result.profile == "stable"
    assert result.iterations == 2


def test_budget_exhaustion_is_not_convergence():
    case = next(c for c in CASES if c["case"] == "short_iteration")
    result = wfmm_mom_initialization(**inputs(case), profile="legacy_3_1")
    assert result.iterations == 2
    assert not result.converged
    assert not result.legacy_stopping_satisfied


def test_default_retains_floored_random_components():
    case = next(c for c in CASES if c["case"] == "equal_floor_threshold")
    stable = wfmm_mom_initialization(**inputs(case))
    legacy = wfmm_mom_initialization(**inputs(case), profile="legacy_3_1")
    assert not stable.omitted_random_components.any()
    assert legacy.omitted_random_components.any()
    assert np.max(abs(stable.fixed_effect_estimates - legacy.fixed_effect_estimates)) > 1e-7


def test_outputs_are_owned_and_readonly():
    kwargs = inputs(CASES[0])
    y = np.array(kwargs.pop("coefficients"))
    result = wfmm_mom_initialization(y, **kwargs)
    saved = result.fixed_effect_estimates.copy()
    y[:] = 0
    np.testing.assert_array_equal(result.fixed_effect_estimates, saved)
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert value.flags.owndata and not value.flags.writeable


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "unknown"),
        ("min_variance", 0),
        ("min_variance", True),
        ("zero_threshold", -1),
        ("zero_threshold", False),
        ("tolerance", 0),
        ("tolerance", True),
        ("max_iterations", False),
        ("max_iterations", 1.1),
        ("max_iterations", 1001),
        ("coefficients", [[np.nan]]),
        ("fixed_design", np.zeros((8, 1))),
        ("fixed_design", np.ones((8, 2))),
        ("random_design", np.eye(8)),
        ("random_design", np.ones((8, 2))),
        ("random_strata", [0]),
        ("residual_strata", [0]),
    ],
)
def test_invalid_inputs_fail_explicitly(key, value):
    kwargs = inputs(CASES[0]) | {key: value}
    with pytest.raises((ValueError, ArithmeticError)):
        wfmm_mom_initialization(**kwargs)


def test_work_bound_precedes_dense_construction():
    with pytest.raises(ValueError, match="budget"):
        wfmm_mom_initialization(np.ones((500, 4)), np.ones((500, 1)))


def test_source_proposal_setting_multiplies_sd_and_keeps_zero():
    fixture = json.loads((Path(__file__).parent / "fixtures/wfmm-native-proposal.json").read_text())
    for case in fixture["cases"]:
        se = np.array(case["se_omega"])
        actual = np.array(case["proposal_sd"])
        np.testing.assert_array_equal(actual, case["propvar_omega"] * se)
        assert np.count_nonzero(actual == 0) == 8
        assert np.count_nonzero(actual > 0) == 24


def test_signed_native_stop_does_not_report_scientific_convergence():
    case = next(c for c in CASES if c["case"] == "signed_stop")
    result = wfmm_mom_initialization(**inputs(case), profile="legacy_3_1")
    assert result.iterations == 7
    assert result.legacy_stopping_satisfied
    assert not result.converged
    assert result.maximum_change > 0.017
    assert result.maximum_signed_change == 0


def test_stable_mixed_strata_preserve_all_trace_cross_terms():
    case = next(c for c in CASES if c["case"] == "mixed_residual_floor")
    result = wfmm_mom_initialization(**inputs(case))
    legacy = wfmm_mom_initialization(**inputs(case), profile="legacy_3_1")
    assert result.normal_equations[0, 2] == pytest.approx(5 / 3)
    assert legacy.normal_equations[0, 2] == 0
    assert result.converged
    assert result.maximum_change <= result.tolerance
