from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    U2OETCriteria,
    simulate_u2oet_trial,
    summarize_u2oet_trials,
    u2oet_patients,
    u2oet_scenario,
)


def _trial(rng, utility=None):
    probability = np.array([[0.2, 0.4], [0.6, 0.8]])
    scenario = u2oet_scenario(
        np.stack((1 - probability, probability), -1), np.broadcast_to([1, 0], (2, 2, 2))
    )
    mean = np.zeros(12)
    mean[[1, 2, 7, 8]] = 0.5
    return simulate_u2oet_trial(
        [1, 2],
        [1, 2],
        scenario,
        [[0, 0], [100, 100]] if utility is None else utility,
        prior_mean=mean,
        prior_sd=np.full(12, 0.2),
        initial=(0, 0),
        max_patients=4,
        criteria=U2OETCriteria(1, 1, min_efficacy=0.3, max_toxicity=1),
        efficacy_window=(0, 0),
        toxicity_window=(0, 0),
        draws=8,
        warmup=5,
        chains=2,
        rng=rng,
    )


def test_hand_calculated_selection_allocation_and_utility_performance():
    template = _trial(np.random.default_rng(7721))
    trials = []
    specifications = [
        ([(0, 0), (0, 0), (1, 1), (1, 1)], (1, 1), False),
        ([(0, 0), (0, 0)], None, True),
        ([(0, 1)] * 4, (0, 1), False),
    ]
    for r, (pairs, selected, early) in enumerate(specifications):
        records = [[i + 1, a + 1, b + 1, 0, 0] for i, (a, b) in enumerate(pairs)]
        patients = u2oet_patients(records, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2)
        trials.append(
            replace(
                template,
                patients=patients,
                selected=selected,
                stopped_early=early,
                data_seed=100 + 2 * r,
                posterior_seed=101 + 2 * r,
                analysis_time=2 if early else 3,
            )
        )
    result = summarize_u2oet_trials(iter(trials))
    assert result.trials == 3 and result.selected_trials == 2
    assert_allclose(result.true_utility, [[20, 40], [60, 80]], atol=1e-12)
    assert_allclose(result.selection_probability, [[0, 1 / 3], [0, 1 / 3]])
    assert_allclose(
        [result.none_probability, result.early_stop_probability, result.best_probability], 1 / 3
    )
    assert_allclose(result.best_mcse, np.sqrt(2 / 27))
    assert_allclose(result.mean_enrollment, 10 / 3)
    assert_allclose(result.enrollment_mcse, 2 / 3)
    assert_allclose(result.mean_treated, [[4 / 3, 4 / 3], [0, 2 / 3]])
    assert_allclose(result.rselect[[0, 2]], [100, 100 / 3])
    assert np.isnan(result.rselect[1])
    assert_allclose(result.rtreat, [50, 0, 100 / 3])
    assert_allclose([result.mean_rselect, result.rselect_mcse], [200 / 3, 100 / 3])
    assert_allclose([result.mean_rtreat, result.rtreat_mcse], [250 / 9, np.sqrt(17500) / 9])
    assert_allclose([result.mean_duration, result.duration_mcse], [8 / 3, 1 / 3])
    assert np.array_equal(result.true_best, [[False, False], [False, True]])
    assert not result.rtreat.flags.writeable
    with pytest.raises(ValueError, match="reuse"):
        summarize_u2oet_trials([trials[0], trials[0]])
    with pytest.raises(ValueError, match="cannot pool"):
        summarize_u2oet_trials([trials[0], replace(trials[1], design_json="different")])


def test_streamed_real_replicates_and_undefined_flat_utility_scores():
    rng = np.random.default_rng(7722)
    result = summarize_u2oet_trials(_trial(rng) for _ in range(3))
    assert result.trials == 3
    assert_allclose(result.selection_probability.sum() + result.none_probability, 1)
    assert_allclose(result.mean_treated.sum(), result.mean_enrollment)
    flat = summarize_u2oet_trials([_trial(rng, [[100, 100], [100, 100]])])
    assert np.isnan(flat.mean_rselect) and np.isnan(flat.mean_rtreat)
    assert np.isnan(flat.enrollment_mcse)
    assert_allclose(flat.true_utility, 100)
