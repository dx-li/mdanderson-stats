import numpy as np
import pytest

from mdanderson_stats.mtadf_author_local_simulation import (
    replay_mtadf_author_local_trial,
    simulate_mtadf_author_local,
)


def test_local_replay_floor_override_skips_posterior_fits():
    toxicities = np.array([[3, 0, 0], [3, 0, 0], [0, 0, 0]])
    responses = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    result = replay_mtadf_author_local_trial(
        toxicities,
        responses,
        sampler_seed=77,
        draws=8,
        warmup=0,
        chains=2,
    )

    assert result.admissible_count_after[0] == 1
    np.testing.assert_array_equal(result.assigned_dose, [0, 0, 0])
    np.testing.assert_array_equal(result.fits_by_cohort, [0, 0, 0])
    np.testing.assert_array_equal(result.subjects, [9, 0, 0])
    assert result.posterior_fit_count == 0
    assert result.selected_dose == 0


def test_local_seed_pair_replays_simulated_trial_and_preflight_is_rng_free():
    kwargs = dict(
        true_toxicity=[0.0, 0.0, 0.0],
        true_efficacy=[0.2, 0.65, 0.4],
        cohorts=3,
        cohort_size=2,
        trials=2,
        draws=16,
        warmup=8,
        chains=2,
    )
    master_seed = (1 << 64) - 1
    simulation = simulate_mtadf_author_local(**kwargs, rng=master_seed)
    repeated = simulate_mtadf_author_local(**kwargs, rng=master_seed)
    np.testing.assert_array_equal(simulation.trial_seeds, repeated.trial_seeds)
    np.testing.assert_array_equal(simulation.subjects, repeated.subjects)
    np.testing.assert_array_equal(simulation.selected_dose, repeated.selected_dose)
    assert simulation.trial_seeds.dtype == np.uint64
    children = np.random.SeedSequence(master_seed).spawn(2)
    expected_seeds = np.column_stack(
        (
            children[0].generate_state(2, dtype=np.uint64),
            children[1].generate_state(2, dtype=np.uint64),
        )
    )
    np.testing.assert_array_equal(simulation.trial_seeds, expected_seeds)

    outcome_seed, sampler_seed = (int(value) for value in simulation.trial_seeds[0])
    outcome_rng = np.random.default_rng(outcome_seed)
    potential_toxicities = np.column_stack(
        [outcome_rng.binomial(2, probability, size=3) for probability in kwargs["true_toxicity"]]
    )
    potential_responses = np.column_stack(
        [outcome_rng.binomial(2, probability, size=3) for probability in kwargs["true_efficacy"]]
    )
    replay = replay_mtadf_author_local_trial(
        potential_toxicities,
        potential_responses,
        sampler_seed=sampler_seed,
        cohort_size=2,
        draws=16,
        warmup=8,
        chains=2,
    )
    np.testing.assert_array_equal(replay.subjects, simulation.subjects[0])
    np.testing.assert_array_equal(replay.toxicities, simulation.toxicities[0])
    np.testing.assert_array_equal(replay.responses, simulation.responses[0])
    assert replay.selected_dose == simulation.selected_dose[0]
    assert replay.sampler_seed == sampler_seed
    assert replay.posterior_fit_count > 0

    seed_sequence = np.random.SeedSequence(382)
    state_before = seed_sequence.state.copy()
    with pytest.raises(ValueError, match="max_total_work"):
        simulate_mtadf_author_local(
            **kwargs,
            rng=seed_sequence,
            max_total_work=1,
        )
    assert seed_sequence.state == state_before
