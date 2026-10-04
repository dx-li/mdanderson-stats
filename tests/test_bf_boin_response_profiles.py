from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.bard_response import bard_response_model
from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.bf_boin_simulation import (
    _validate_bard_response_truth,
    simulate_bf_boin,
)


def _small_response_model():
    return bard_response_model(
        [0.4, 0.6],
        [[1], [2]],
        [0.25, 0.75],
        [[1.0, 2.0]],
    )


def test_profile_response_and_joint_endpoint_histories_are_aligned_and_replayable():
    model = _small_response_model()
    kwargs = dict(
        design=BFBOINDesign(n_cap=8),
        true_toxicity=[0.5, 0.5],
        true_response=model.population_response,
        cohorts=2,
        cohort_size=2,
        trials=3,
        response_model=model,
        joint_toxicity_response_probability=[[0.0, 0.4], [0.1, 0.5]],
        rng=824,
    )
    first = simulate_bf_boin(**kwargs)
    second = simulate_bf_boin(**kwargs)

    assert first.profile_index_history is not None
    assert first.factor_history is not None
    assert first.response_probability_history is not None
    assert first.sampled_response_probability_history is not None
    assert first.eliminated_history is not None
    for trial in range(3):
        indices = first.profile_index_history[trial]
        factors = first.factor_history[trial]
        doses = first.assigned_history[trial] - 1
        dlt = first.dlt_history[trial]
        assert indices.shape == factors.shape[:1] == doses.shape == dlt.shape
        np.testing.assert_array_equal(factors, model.factor_profiles[indices])
        np.testing.assert_array_equal(
            first.response_probability_history[trial],
            model.conditional_probabilities[doses, indices],
        )
        joint = np.asarray(kwargs["joint_toxicity_response_probability"])
        toxicity = np.asarray(kwargs["true_toxicity"])
        marginal = first.response_probability_history[trial]
        expected_conditional = np.where(
            dlt,
            joint[doses, indices] / toxicity[doses],
            (marginal - joint[doses, indices]) / (1 - toxicity[doses]),
        )
        np.testing.assert_allclose(
            first.sampled_response_probability_history[trial], expected_conditional
        )
        assert first.eliminated_history[trial].shape == (2,)
        assert first.eliminated_history[trial].dtype == np.bool_
        assert not first.eliminated_history[trial].flags.writeable
        np.testing.assert_array_equal(
            first.profile_index_history[trial], second.profile_index_history[trial]
        )
        np.testing.assert_array_equal(first.response_history[trial], second.response_history[trial])
        np.testing.assert_array_equal(
            first.eliminated_history[trial], second.eliminated_history[trial]
        )


def test_default_path_keeps_profile_metadata_absent_and_seeded_results_stable():
    kwargs = dict(
        design=BFBOINDesign(n_cap=4),
        true_toxicity=[0.0, 0.0],
        true_response=[0.4, 0.6],
        cohorts=2,
        cohort_size=2,
        trials=2,
        rng=19,
    )
    first = simulate_bf_boin(**kwargs)
    second = simulate_bf_boin(**kwargs)
    for name in (
        "profile_index_history",
        "factor_history",
        "response_probability_history",
        "sampled_response_probability_history",
        "eliminated_history",
    ):
        assert getattr(first, name) is None
    for name in (
        "assigned_history",
        "arrival_history",
        "assessment_history",
        "dlt_history",
        "response_history",
    ):
        for left, right in zip(getattr(first, name), getattr(second, name), strict=True):
            np.testing.assert_array_equal(left, right)


def test_invalid_model_margin_and_rare_frechet_violation_fail_before_rng():
    model = _small_response_model()
    rng = np.random.default_rng(91)
    before = rng.bit_generator.state
    common = dict(
        design=BFBOINDesign(n_cap=4),
        true_toxicity=[1e-310, 0.1],
        cohorts=1,
        cohort_size=1,
        trials=1,
        response_model=model,
        rng=rng,
    )
    with pytest.raises(ValueError, match="margins"):
        simulate_bf_boin(true_response=[0.4 + 1e-13, 0.6], **common)
    assert rng.bit_generator.state == before

    certain_response = bard_response_model([1.0, 1.0], [[1], [2]], [0.25, 0.75], [[1.0, 2.0]])
    _, _, _, valid_rare_q = _validate_bard_response_truth(
        certain_response,
        certain_response.population_response,
        np.asarray([1e-310, 0.2]),
        [[1e-310, 1e-310], [0.2, 0.2]],
    )
    assert valid_rare_q is not None
    np.testing.assert_array_equal(valid_rare_q[0], [1e-310, 1e-310])

    forged_conditional = np.array(model.conditional_probabilities, copy=True)
    forged_conditional[0, 0] += 2e-14
    forged = replace(model, conditional_probabilities=forged_conditional)
    invalid_q = np.array(model.conditional_probabilities, copy=True)
    invalid_q[0, 0] += 1e-14
    with pytest.raises(ValueError, match="Frechet"):
        _validate_bard_response_truth(
            forged,
            model.population_response,
            np.asarray([1.0, 0.1]),
            invalid_q,
        )

    with pytest.raises(ValueError, match="Frechet"):
        simulate_bf_boin(
            true_response=model.population_response,
            joint_toxicity_response_probability=[[1e-20, 1e-20], [0.0, 0.0]],
            **common,
        )
    assert rng.bit_generator.state == before
