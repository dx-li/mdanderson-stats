"""Continuous chains, per-chain diagnostics, and bounded precision failures."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.uaroet_desktop_study import read_uaroet_desktop_study
from mdanderson_stats.uaroet_fit import fit_uaroet
from mdanderson_stats.uaroet_precision import _utility_precision, fit_uaroet_desktop


def study(chains=2, burn=0):
    s = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    return replace(s, simulation_parameters={"NumChains": chains, "BurnInSamples": burn})


@pytest.mark.parametrize("association", [None, 0.0])
def test_one_chain_continuation_matches_uninterrupted(association):
    options = dict(
        counts=[[[3, 1], [2, 4]]],
        prior_mean=[0, 0],
        prior_sd=[1, 1],
        chains=1,
        warmup=0,
        association=association,
    )
    a = fit_uaroet(**options, draws=64, rng=np.random.default_rng(42))
    rng = np.random.default_rng(42)
    b = fit_uaroet(**options, draws=32, rng=rng)
    c = fit_uaroet(
        **options,
        draws=32,
        rng=rng,
        initial_parameters=b.parameters[0, -1],
        initial_association=b.association[0, -1] if association is None else None,
    )
    np.testing.assert_array_equal(
        a.parameters, np.concatenate([b.parameters, c.parameters], axis=1)
    )
    np.testing.assert_array_equal(
        a.association, np.concatenate([b.association, c.association], axis=1)
    )
    assert np.isnan(a.parameter_summary.split_rhat).all()


@pytest.mark.parametrize("chains", [1, 2, 20])
def test_supported_chain_counts(chains):
    fit = fit_uaroet(
        [[[0, 0], [0, 0]]],
        prior_mean=[0, 0],
        prior_sd=[1, 1],
        chains=chains,
        draws=8,
        rng=np.random.default_rng(22),
        association=0,
    )
    assert fit.parameters.shape == (chains, 8, 2)
    assert study(chains).simulation_parameters["NumChains"] == chains


def test_per_chain_precision_has_no_pooled_chain_advantage():
    samples = np.random.default_rng(3).normal(size=(1, 400, 2))
    expected = []
    for d in range(2):
        values = samples[0, :, d]
        expected.append(
            values.reshape(20, 20).mean(1).std(ddof=1) / np.sqrt(20) / values.std(ddof=1)
        )
    actual = _utility_precision(np.repeat(samples, 4, axis=0))
    np.testing.assert_allclose(actual, np.tile(expected, (4, 1)))
    assert np.isinf(_utility_precision(np.ones((2, 400, 2)))).all()


def test_precision_chunks_preserve_chain_and_averages():
    s = study()
    a = fit_uaroet_desktop(
        s, rng=np.random.default_rng(76), target=1e-6, chunk_draws=32, max_draws=64
    )
    b = fit_uaroet_desktop(
        s, rng=np.random.default_rng(76), target=1e-6, chunk_draws=64, max_draws=64
    )
    np.testing.assert_array_equal(a.utility_samples, b.utility_samples)
    for name in (
        "mean_joint",
        "mean_parameters",
        "probability_best",
        "toxicity_exceedance",
        "efficacy_shortfall",
    ):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-14)
    assert not a.precision_met and a.draws_per_chain == 64
    assert not a.utility_samples.flags.writeable
    assert a.monitored_doses == (0, 1)


def test_precision_success_respects_minimum_retained_draws():
    s = study(1, 64)
    fit = fit_uaroet_desktop(
        s, rng=np.random.default_rng(19), target=0.99, chunk_draws=32, max_draws=96
    )
    assert fit.precision_met and fit.draws_per_chain == 64
    assert np.isnan(fit.utility_split_rhat).all()


@pytest.mark.parametrize(
    "options",
    [
        {"target": True},
        {"target": 0},
        {"target": float("nan")},
        {"conduct": 1},
        {"chunk_draws": 7},
        {"chunk_draws": 65, "max_draws": 64},
        {"max_evaluations": 0},
        {"initial_parameters": [0, -1, 0, 1]},
        {"initial_association": 1},
        {"initial_association": [0]},
        {"initial_parameters": [0, 0]},
    ],
)
def test_invalid_options_do_not_advance_rng(options):
    rng = np.random.default_rng(72)
    before = rng.bit_generator.state
    with pytest.raises(ValueError):
        fit_uaroet_desktop(study(), rng=rng, **options)
    assert rng.bit_generator.state == before


def test_analytically_constant_utility_is_distinct_from_stuck_chain():
    s = replace(study(1), utility=np.full((2, 2), 30.0))
    fit = fit_uaroet_desktop(s, rng=np.random.default_rng(12), chunk_draws=8, max_draws=8)
    assert fit.precision_met
    np.testing.assert_array_equal(fit.utility_mcse_ratio, 0)
