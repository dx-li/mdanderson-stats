import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    multc_lean_design,
    run_multc_legacy_duration,
    simulate_multc_legacy_duration,
    summarize_multc_legacy_durations,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/multc-lean-native-boundaries.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case"])
def test_summary_matches_original_native_study_aggregation(case):
    native = case["native_study_summary"]
    if not case["duration_cases"]:
        assert native["trials_consumed"] == native["seed_calls"] == 0
        assert native["sample_size_probability"] == []
        assert all(
            native[key] == 0
            for key in (
                "mean_duration",
                "mean_sample_size",
                "mean_responses",
                "mean_toxicities",
                "mean_balks",
            )
        )
        return
    trials = []
    for reference in case["duration_cases"]:
        i = reference["inputs"]
        trials.append(
            run_multc_legacy_duration(
                i["n"],
                response_stop_at=i["response"],
                nontoxicity_stop_at=i["notox"],
                joint_probabilities=i["prob"],
                mean_interarrival=i["mean"],
                response_window=i["window"],
                uniforms=i["uniform"],
                unit_exponentials=i["unit_exponential"],
            )
        )
    result = summarize_multc_legacy_durations(
        int(case["response"]["parameters"]["cap"]),
        trials,
    )
    assert result.trials == native["trials_consumed"] == 3
    assert native["seed_calls"] == 1
    for field in (
        "mean_duration",
        "mean_sample_size",
        "mean_responses",
        "mean_toxicities",
        "mean_balks",
    ):
        assert getattr(result, field) == pytest.approx(native[field], rel=3e-15, abs=1e-14)
    np.testing.assert_allclose(
        result.sample_size_probability, native["sample_size_probability"], rtol=1e-15, atol=0
    )
    for field, error in (
        ("duration", "duration_mcse"),
        ("sample_size", "sample_size_mcse"),
        ("responses", "responses_mcse"),
        ("toxicities", "toxicities_mcse"),
        ("balks", "balks_mcse"),
    ):
        expected = np.std(getattr(result, field), ddof=1) / np.sqrt(result.trials)
        assert getattr(result, error) == pytest.approx(expected, rel=1e-14, abs=1e-14)


def design(**changes):
    return multc_lean_design(
        12, (1, 1), (1, 1), historical_response=0.5, historical_toxicity=0.5, **changes
    )


def settings(**changes):
    return {
        **dict(
            joint_probabilities=[0.1, 0.1, 0.5, 0.3],
            mean_interarrival=0.2,
            response_window=2.0,
            trials=32,
            seed=717,
            max_unit_exponentials_per_trial=1000,
        ),
        **changes,
    }


def test_replays_match_every_captured_replicate_and_own_arrays():
    truth = np.array([0.1, 0.1, 0.5, 0.3])
    result = simulate_multc_legacy_duration(design(), **settings(joint_probabilities=truth))
    second = simulate_multc_legacy_duration(design(), **settings())
    np.testing.assert_array_equal(result.trial_seeds, second.trial_seeds)
    np.testing.assert_array_equal(result.summary.duration, second.summary.duration)
    truth[:] = 0.25
    np.testing.assert_array_equal(result.joint_probabilities, [0.1, 0.1, 0.5, 0.3])
    assert result.seed == 717
    for index in range(result.summary.trials):
        trial = result.replay_trial(index)
        for field in ("duration", "sample_size", "responses", "toxicities", "balks"):
            assert getattr(trial, field) == getattr(result.summary, field)[index]
    for array in (
        result.trial_seeds,
        result.joint_probabilities,
        result.summary.duration,
        result.summary.sample_size_probability,
    ):
        assert not array.flags.writeable
    assert result.summary.sample_size_probability.sum() == 1


def test_fixed_enrollment_analytic_duration_mean_and_mcse():
    result = simulate_multc_legacy_duration(
        design(response_cutoff=1, toxicity_cutoff=1),
        **settings(
            joint_probabilities=[0, 0, 0, 1],
            mean_interarrival=1.0,
            trials=4000,
            max_unit_exponentials_per_trial=11,
        ),
    )
    summary = result.summary
    assert summary.mean_sample_size == 12
    assert summary.mean_responses == summary.mean_toxicities == summary.mean_balks == 0
    assert summary.sample_size_mcse == 0
    assert summary.sample_size_probability[12] == 1
    # With no responses, the final follow-up is sum of 11 Exp(1) gaps plus 2.
    assert summary.mean_duration == pytest.approx(13, abs=5 * summary.duration_mcse)
    assert summary.duration_mcse == pytest.approx(np.sqrt(11 / 4000), rel=0.08)


def test_prior_rejection_has_zero_duration_and_a_complete_zero_enrollment_pmf():
    result = simulate_multc_legacy_duration(design(response_cutoff=0.49), **settings())
    assert result.summary.mean_sample_size == result.summary.mean_duration == 0
    assert result.summary.duration_mcse == result.summary.sample_size_mcse == 0
    assert result.summary.sample_size_probability[0] == 1
    trial = result.replay_trial(0)
    assert trial.decision == "prior_response"
    assert trial.uniforms_consumed == trial.exponentials_consumed == 0


def test_single_replicate_mcse_is_undefined():
    result = simulate_multc_legacy_duration(design(), **settings(trials=1))
    assert np.isnan(result.summary.duration_mcse)
    assert np.isnan(result.summary.sample_size_mcse)


def test_exhaustion_does_not_return_a_partial_study():
    with pytest.raises(ValueError, match="unit_exponentials exhausted"):
        simulate_multc_legacy_duration(
            design(response_cutoff=1, toxicity_cutoff=1),
            **settings(joint_probabilities=[0, 0, 0, 1], max_unit_exponentials_per_trial=10),
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"trials": True}, "trials"),
        ({"trials": 0}, "trials"),
        ({"trials": 10_001}, "trials"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"max_unit_exponentials_per_trial": 0}, "max_unit_exponentials"),
        ({"max_unit_exponentials_per_trial": 1_000_001}, "max_unit_exponentials"),
        ({"max_total_draws": 1}, "max_total_draws"),
        ({"max_storage_bytes": 1}, "max_storage_bytes"),
        (
            {
                "trials": 1,
                "max_unit_exponentials_per_trial": 1_000_000,
                "max_storage_bytes": 9_000_000,
            },
            "max_storage_bytes",
        ),
        ({"joint_probabilities": [1, 1, 1, 1]}, "joint_probabilities"),
        ({"mean_interarrival": 0}, "mean_interarrival"),
        ({"response_window": float("inf")}, "response_window"),
    ],
)
def test_invalid_settings_fail_before_rng_creation(monkeypatch, change, match):
    def forbidden(*args, **kwargs):
        pytest.fail("RNG created before validating inputs/work/storage")

    monkeypatch.setattr(np.random, "Generator", forbidden)
    with pytest.raises(ValueError, match=match):
        simulate_multc_legacy_duration(design(), **settings(**change))


def test_replay_index_and_profile_validation():
    result = simulate_multc_legacy_duration(design(), **settings(trials=1))
    for index in (-1, 1, True, 0.5):
        with pytest.raises(ValueError, match="index"):
            result.replay_trial(index)
    with pytest.raises(ValueError, match="pretrial_check=True"):
        simulate_multc_legacy_duration(design(pretrial_check=False), **settings())
    with pytest.raises(ValueError, match="MultcLeanDesign"):
        simulate_multc_legacy_duration(None, **settings())


def test_summary_input_validation_and_large_finite_duration():
    trial = simulate_multc_legacy_duration(design(), **settings(trials=1)).replay_trial(0)
    for invalid in ([], [None], "trial", [trial] * 10_001):
        with pytest.raises(ValueError):
            summarize_multc_legacy_durations(12, invalid)
    for change in (
        {"duration": float("nan")},
        {"duration": -1},
        {"responses": 100},
        {"balks": True},
    ):
        with pytest.raises(ValueError):
            summarize_multc_legacy_durations(12, [replace(trial, **change)])
    result = summarize_multc_legacy_durations(12, [replace(trial, duration=1e308)] * 3)
    assert result.mean_duration == 1e308
    assert result.duration_mcse == 0
