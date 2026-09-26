"""Focused checks for DA-CRM prior elicitation helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.dacrm_priors import dacrm_trimester_prior, dacrm_uniform_prior


def test_uniform_prior_formula_and_time_unit_rescaling():
    prior = dacrm_uniform_prior(6, intervals=3, dispersion=2)
    assert_allclose(prior.shape, [0.1, 1 / 6, 0.5])
    assert_allclose(prior.rate, [0.5, 0.5, 0.5])
    rescaled = dacrm_uniform_prior(60, intervals=3, dispersion=0.2)
    assert_allclose(rescaled.shape, prior.shape)
    assert_allclose(rescaled.rate, 10 * prior.rate)
    assert_allclose(rescaled.breaks, 10 * prior.breaks)


def test_trimester_prior_matches_piecewise_survival_calibration():
    prior = dacrm_trimester_prior(6, [0.2, 0.3, 0.5], dispersion=2)
    assert prior.shape.shape == (6,)
    interval_width = prior.breaks[1] - prior.breaks[0]
    cumulative_hazard = np.cumsum(prior.shape / prior.rate * interval_width)
    assert_allclose(np.exp(-cumulative_hazard[[1, 3, 5]]), [0.8, 0.5, 0.01])


def test_trimester_prior_rejects_unrepresentable_terminal_cumulative_probability():
    with pytest.raises(ValueError, match="increase strictly"):
        dacrm_trimester_prior(1, [0.1, 0.9, 0.0 + 1e-15], dispersion=1)
