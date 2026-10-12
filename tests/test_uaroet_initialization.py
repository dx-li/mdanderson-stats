"""Diffuse-prior initialization and analytically bounded MCMC proposal rejection."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    fit_uaroet,
    uaroet_desktop_prior_coordinates,
    uaroet_logits,
    uaroet_probabilities,
)
from mdanderson_stats._uaroet_bounds import _likelihood_upper_bound
from mdanderson_stats.uaroet import _marginal_log_probabilities

JOINT = json.loads((Path(__file__).parent / "fixtures/uaroet-desktop-joint.json").read_text())


@pytest.mark.parametrize("case", JOINT["cases"])
def test_analytic_likelihood_bounds_dominate_actual_gaussian_probabilities(case):
    d, e, t = case["doses"], case["efficacy_levels"], case["toxicity_levels"]
    native = np.array(case["parameters"][:30]).reshape(2, 3, 5)
    parameters, rho = uaroet_desktop_prior_coordinates(
        np.r_[
            native[0, : e - 1, :d].ravel(), native[1, : t - 1, :d].ravel(), case["parameters"][-1]
        ],
        dose_count=d,
        efficacy_levels=e,
        toxicity_levels=t,
    )
    el, tl = uaroet_logits(parameters, dose_count=d, efficacy_levels=e, toxicity_levels=t)
    model = uaroet_probabilities(el, tl, association=rho)
    le, lt = _marginal_log_probabilities(el), _marginal_log_probabilities(tl)
    for ei, ti in np.ndindex(e, t):
        counts = np.zeros((d, e, t), dtype=int)
        counts[:, ei, ti] = 1
        partial = np.ones((d, t), dtype=int)
        with np.errstate(divide="ignore"):
            exact = float(np.log(model.joint[:, ei, ti]).sum() + lt.sum())
        marginal = _likelihood_upper_bound(le, lt, counts, partial)
        gaussian = _likelihood_upper_bound(le, lt, counts, partial, rho)
        assert exact <= gaussian + 1e-9
        assert gaussian <= marginal + 1e-9


def test_bounds_preserve_seeded_chain_and_accepted_likelihood(monkeypatch):
    import mdanderson_stats.uaroet_fit as module

    kwargs = dict(
        counts=np.array([[[2, 1], [3, 2]], [[1, 1], [4, 2]]]),
        prior_mean=[0, 0.3, 0, 0.3],
        prior_sd=[0.4] * 4,
        initial_parameters=[0, 0.3, 0, 0.3],
        initial_association=0,
        draws=32,
        warmup=16,
        chains=2,
    )
    a = fit_uaroet(**kwargs, rng=np.random.default_rng(481))
    monkeypatch.setattr(module, "_likelihood_upper_bound", lambda *args: np.inf)
    b = fit_uaroet(**kwargs, rng=np.random.default_rng(481))
    np.testing.assert_array_equal(a.parameters, b.parameters)
    np.testing.assert_array_equal(a.association, b.association)
    np.testing.assert_array_equal(a.log_likelihood, b.log_likelihood)
    assert a.work_evaluations <= b.work_evaluations


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diffuse_pseudo_prior_can_start_at_data_compatible_parameters(seed):
    fit = fit_uaroet(
        np.full((2, 2, 2), 25),
        prior_mean=np.zeros(4),
        prior_sd=np.full(4, 100),
        initial_parameters=[0, 0.1, 0, 0.1],
        initial_association=0,
        draws=16,
        warmup=16,
        chains=2,
        rng=np.random.default_rng(seed),
    )
    assert np.all(np.isfinite(fit.log_likelihood))
    assert np.all(fit.parameters[:, :, [1, 3]] >= 0)
    # Short-chain completion is deliberately not a convergence assertion.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_parameters": [0, 1]},
        {"initial_parameters": [0, -1, 0, 1]},
        {"initial_parameters": [0, np.nan, 0, 1]},
        {"initial_parameters": True},
        {"initial_association": 1},
        {"initial_association": True},
        {"initial_association": 0, "association": 0},
    ],
)
def test_invalid_start_fails_before_rng_advances(kwargs):
    rng = np.random.default_rng(73)
    before = json.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError):
        fit_uaroet(
            np.ones((2, 2, 2)),
            prior_mean=[0, 1, 0, 1],
            prior_sd=[1] * 4,
            draws=8,
            warmup=0,
            chains=2,
            rng=rng,
            **kwargs,
        )
    assert json.dumps(rng.bit_generator.state) == before
