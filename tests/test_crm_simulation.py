"""Small CRM operating-characteristic and random-stream checks."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats.crm_simulation import simulate_crm
from mdanderson_stats.dacrm import DACRMPrior


def test_bma_simulation_is_reproducible_and_conserves_patient_counts():
    settings = dict(
        skeletons=[0.1, 0.25, 0.45],
        true_toxicity=[0.05, 0.25, 0.5],
        window=1,
        accrual_rate=2,
        target=0.25,
        cohorts=2,
        cohort_size=1,
        trials=3,
        arrival="fixed",
        event_distribution="uniform",
        safety_cutoff=1,
    )
    first = simulate_crm(**settings, rng=274)
    second = simulate_crm(**settings, rng=274)
    assert_array_equal(first.patients, second.patients)
    assert_array_equal(first.toxicities, second.toxicities)
    assert_array_equal(first.selected_dose, second.selected_dose)
    assert_array_equal(first.patients.sum(axis=1), [2, 2, 2])
    assert np.all(first.toxicities <= first.patients)
    assert np.sum(first.selection_probability) + first.no_selection_probability == 1
    assert first.evaluations == first.trial_evaluations.sum()
    with pytest.raises(ValueError):
        first.patients.setflags(write=True)


def test_da_simulation_uses_an_independent_sampler_stream_for_pending_data():
    prior = DACRMPrior([0, 1], [0.5], [0.5])
    scenario_rng = np.random.default_rng(17)
    sampler_rng = np.random.default_rng(29)
    sampler_state = deepcopy(sampler_rng.bit_generator.state)
    result = simulate_crm(
        [0.1, 0.3],
        [0.1, 0.25],
        1,
        100,
        target=0.2,
        cohorts=2,
        cohort_size=1,
        trials=2,
        method="dacrm",
        da_prior=prior,
        minimum_observed=0,
        rng=scenario_rng,
        sampler_rng=sampler_rng,
        draws=8,
        warmup=0,
        chains=2,
        arrival="fixed",
        event_distribution="uniform",
        safety_cutoff=1,
    )
    assert result.patients.sum() == 4
    assert result.evaluations > 0
    assert np.all(result.trial_evaluations > 0)
    assert result.max_dose_mcse.shape == (2,)
    assert result.max_split_rhat.shape == (2,)
    assert sampler_rng.bit_generator.state != sampler_state
    expected_scenario_rng = np.random.default_rng(17)
    expected_scenario_rng.random((2, 2))
    expected_scenario_rng.random((2, 2))
    assert scenario_rng.bit_generator.state == expected_scenario_rng.bit_generator.state
