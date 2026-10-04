from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import isotonic_regression

from mdanderson_stats.mtpi import MTPIDesign
from mdanderson_stats.mtpi_isotonic_posterior import mtpi_isotonic_posterior_intervals
from mdanderson_stats.tpi import TPIDesign
from mdanderson_stats.tpi_isotonic_posterior import tpi_isotonic_posterior_intervals


def test_tpi_dose_specific_posteriors_replay_including_untried_prior_draw():
    design = TPIDesign(
        target=0.3,
        prior=[(1.0, 19.0), (2.0, 8.0), (4.0, 6.0)],
    )
    seed = 2781
    draws = 96
    weights = [1.0, 2.0, 0.75]
    result = tpi_isotonic_posterior_intervals(
        design,
        [8, 4, 0],
        [1, 2, 0],
        draws=draws,
        rng=np.random.default_rng(seed),
        confidence=0.9,
        weights=weights,
        retain_draws=True,
    )

    replay_rng = np.random.default_rng(seed)
    alpha = np.array([2.0, 4.0, 4.0])
    beta = np.array([26.0, 10.0, 6.0])
    replay = np.empty((draws, 3))
    normalized_weights = np.asarray(weights) / max(weights)
    for index in range(draws):
        sample = replay_rng.beta(alpha, beta)
        replay[index] = isotonic_regression(sample, weights=normalized_weights).x
    assert result.transformed_draws is not None
    np.testing.assert_array_equal(result.transformed_draws, replay)
    np.testing.assert_allclose(result.lower, np.quantile(replay, 0.05, axis=0, method="linear"))
    np.testing.assert_allclose(result.median, np.quantile(replay, 0.5, axis=0, method="linear"))
    np.testing.assert_allclose(result.upper, np.quantile(replay, 0.95, axis=0, method="linear"))
    np.testing.assert_allclose(result.mean, replay.mean(axis=0))
    assert np.all(np.diff(replay, axis=1) >= 0)
    assert not result.transformed_draws.flags.writeable


def test_tpi_common_prior_uses_same_shared_draw_path_as_mtpi():
    tpi = TPIDesign(prior=(0.005, 0.005))
    mtpi = MTPIDesign(prior_alpha=0.005, prior_beta=0.005)
    args = ([8, 0, 5], [6, 0, 1])
    tpi_result = tpi_isotonic_posterior_intervals(
        tpi,
        *args,
        draws=64,
        rng=np.random.default_rng(84),
        weights=[1.0, 2.0, 1.0],
        retain_draws=True,
    )
    mtpi_result = mtpi_isotonic_posterior_intervals(
        mtpi,
        *args,
        draws=64,
        rng=np.random.default_rng(84),
        weights=[1.0, 2.0, 1.0],
        retain_draws=True,
    )
    for field in ("lower", "median", "upper", "mean", "weights", "transformed_draws"):
        np.testing.assert_array_equal(getattr(tpi_result, field), getattr(mtpi_result, field))


def test_tpi_shape_and_work_rejections_leave_rng_unchanged():
    design = TPIDesign(prior=[(1.0, 19.0), (2.0, 8.0), (4.0, 6.0)])
    rng = np.random.default_rng(98)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="one count per configured dose"):
        tpi_isotonic_posterior_intervals(design, [4, 3], [1, 2], draws=16, rng=rng)
    assert rng.bit_generator.state == before

    with pytest.raises(ValueError, match="max_work"):
        tpi_isotonic_posterior_intervals(
            design, [4, 3, 0], [1, 2, 0], draws=16, rng=rng, max_work=1
        )
    assert rng.bit_generator.state == before
