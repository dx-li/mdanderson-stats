import numpy as np
import pytest

from mdanderson_stats.u2oet_gao2010_decision import u2oet_gao2010_decision
from mdanderson_stats.u2oet_gao2010_trial import simulate_u2oet_gao2010_trial


def _draws(*, severe: float = 0.3) -> np.ndarray:
    values = np.zeros((2, 8, 2, 2, 2, 2))
    values[..., 0, 0] = (1.0 - severe) / 2.0
    values[..., 1, 0] = (1.0 - severe) / 2.0
    values[..., 0, 1] = severe / 2.0
    values[..., 1, 1] = severe / 2.0
    return values


def test_global_safety_boundary_is_strict_and_utility_summary_is_scaled() -> None:
    result = u2oet_gao2010_decision(
        _draws(),
        [[0.0, 0.0], [1e308, 1e308]],
        toxicity_limit=0.3,
        stopping_probability=0.8,
    )
    assert result.stopped_for_global_toxicity is False
    assert result.minimum_exceedance_probability == 0.0
    assert result.selected_pair == (0, 0)  # exact row-major utility tie
    assert np.all(np.isfinite(result.mean_utility))
    assert np.all(np.isfinite(result.utility_sd))
    assert np.all(np.isfinite(result.utility_mcse))
    assert result.mean_utility[0, 0] > 4e307
    assert not result.eligible.flags.writeable


def test_interim_eligibility_uses_untried_source_upper_neighbors_only() -> None:
    posterior = np.tile(_draws(severe=0.1), (1, 1, 1, 1, 1, 1))
    result = u2oet_gao2010_decision(
        posterior,
        [[0.0, 0.0], [1.0, 1.0]],
        toxicity_limit=0.2,
        current_pair=(0, 1),
        treated=[[0, 0], [0, 1]],
    )
    assert result.eligible.tolist() == [[True, True], [False, True]]
    assert result.selected_pair == (0, 0)


def test_trial_uses_explicit_truth_replay_and_final_full_grid_selection() -> None:
    truth = np.full((2, 2, 2, 2), 0.25)
    tape = np.array([[0.0, 0.1], [0.3, 0.9], [0.6, 0.2], [0.9, 0.7]])
    result = simulate_u2oet_gao2010_trial(
        [0.0, 1.0],
        [0.0, 2.0],
        truth,
        [[0.0, 0.0], [1.0, 1.0]],
        prior_mean=np.zeros(12),
        prior_sd=np.zeros(12),
        starting=(0, 0),
        n_patients=4,
        cohort_size=2,
        efficacy_evaluability=0.5,
        toxicity_limit=0.99,
        draws=8,
        warmup=0,
        chains=2,
        fixed_association=0.0,
        outcome_uniforms=tape,
        rng=np.random.default_rng(121),
    )
    assert np.array_equal(result.outcome_uniforms, tape)
    assert result.patients.records.shape == (4, 5)
    assert result.patients.complete.sum() + result.patients.toxicity_only.sum() == 4
    assert result.posterior_fits == 2
    assert [look.patients for look in result.looks] == [2, 4]
    assert result.final_decision.eligible.all()  # final Eq. 11 is unrestricted
    assert result.selected_pair == (0, 0)


def test_invalid_sampler_start_is_rejected_before_caller_rng_advances() -> None:
    truth = np.full((2, 2, 2, 2), 0.25)
    prior_sd = np.zeros(12)
    prior_sd[5] = 0.1  # efficacy interaction coordinate is free
    starts = np.zeros(13)
    starts[5] = -2.0  # outside the valid domain on this dose grid
    rng = np.random.default_rng(44)
    reference = np.random.default_rng(44)
    with pytest.raises(ValueError, match="initial state is outside"):
        simulate_u2oet_gao2010_trial(
            [0.0, 1.0],
            [0.0, 2.0],
            truth,
            [[0.0, 0.0], [1.0, 1.0]],
            prior_mean=np.zeros(12),
            prior_sd=prior_sd,
            starting=(0, 0),
            n_patients=2,
            efficacy_evaluability=1.0,
            toxicity_limit=0.99,
            draws=8,
            warmup=0,
            chains=2,
            fixed_association=0.0,
            initial_parameters=starts,
            rng=rng,
        )
    assert rng.random() == reference.random()


def test_nested_truth_lists_are_bounded_by_the_declared_four_axes() -> None:
    truth = [[[[0.25, 0.25], [0.25, 0.25]] for _ in range(2)] for _ in range(2)]
    result = simulate_u2oet_gao2010_trial(
        [0.0, 1.0],
        [0.0, 2.0],
        truth,
        [[0.0, 0.0], [1.0, 1.0]],
        prior_mean=np.zeros(12),
        prior_sd=np.zeros(12),
        starting=(0, 0),
        n_patients=1,
        efficacy_evaluability=1.0,
        toxicity_limit=0.99,
        draws=8,
        warmup=0,
        chains=2,
        fixed_association=0.0,
        rng=np.random.default_rng(91),
    )
    assert result.patients.records.shape == (1, 5)
