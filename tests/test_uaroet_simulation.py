import numpy as np

from mdanderson_stats.uaroet import uaroet_parameter_names
from mdanderson_stats.uaroet_simulation import _streams, run_uaroet_trial, simulate_uaroet


def _two_dose_arguments():
    truth = np.array(
        [
            [[0.5, 0.1], [0.1, 0.3]],
            [[0.3, 0.1], [0.2, 0.4]],
        ]
    )
    return dict(
        truth=truth,
        utility=np.array([[0.2, 0.0], [1.0, 0.4]]),
        prior_mean=np.zeros(4),
        prior_sd=np.ones(4),
        look_sizes=[2, 6],
        utility_tolerance=[1000.0, 1000.0],
        starting_dose=0,
        monotone_efficacy=False,
        monotone_toxicity=False,
        association=0.0,
        draws=8,
        warmup=0,
        chains=2,
        toxicity_limit=1.0,
        p_L=0.0,
        p_U=1.0,
        good_utility_cutoff=0.0,
    )


def test_patient_randomization_uses_frozen_look_probabilities_and_replay_seeds():
    args = _two_dose_arguments()
    assert (
        len(uaroet_parameter_names(2, 2, 2, monotone_efficacy=False, monotone_toxicity=False)) == 4
    )
    allocation_tape = np.array([0.8, 0.8, 0.1, 0.9, 0.2, 0.8])
    result = run_uaroet_trial(
        **args,
        rng=23,
        outcome_uniforms=np.array([0.05, 0.55, 0.65, 0.95, 0.45, 0.85]),
        allocation_uniforms=allocation_tape,
    )
    np.testing.assert_array_equal(result.assigned_dose[:2], [0, 0])
    assert len(result.steps) == 2
    interim = result.steps[0]
    assert interim.action == "randomize"
    assert np.isfinite(interim.max_split_rhat)
    assert np.all(interim.allocation_probabilities > 0)
    for patient in range(2, 6):
        expected = np.searchsorted(
            np.cumsum(interim.allocation_probabilities), allocation_tape[patient], side="right"
        )
        assert result.assigned_dose[patient] == expected
        np.testing.assert_allclose(
            result.assignment_probabilities[patient], interim.allocation_probabilities
        )

    streams = _streams(23)
    for stream, seed in zip(streams[:3], streams[3], strict=True):
        np.testing.assert_array_equal(stream.random(5), np.random.default_rng(seed).random(5))


def test_simulation_is_reproducible_conserves_counts_and_scales_utility_safely():
    args = _two_dose_arguments()
    args["utility"] = args["utility"] * 1e300
    kwargs = {key: value for key, value in args.items() if key != "truth"}
    first = simulate_uaroet(args["truth"], trials=2, rng=91, **kwargs)
    second = simulate_uaroet(args["truth"], trials=2, rng=91, **kwargs)
    np.testing.assert_array_equal(first.dose_counts, second.dose_counts)
    np.testing.assert_array_equal(first.selected_dose, second.selected_dose)
    np.testing.assert_array_equal(first.dose_counts.sum(axis=1), first.enrolled_patients)
    assert np.isfinite(first.mean_observed_utility).all()
    assert first.mean_observed_utility.max() <= args["utility"].max()
