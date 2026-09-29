from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    BetaBinomialPosterior,
    compare_beta_binomial,
    parallel_phase12_replay,
    simulate_parallel_phase12,
)
from mdanderson_stats.parallel_phase12_oc import simulate_parallel_phase12_oc


def test_source_phase_one_rules_and_partial_cohorts():
    # Lowest arm: 2/6 remains open but neither adjacent arm is opened.
    records = [[0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    partial = parallel_phase12_replay(records[:4])
    assert partial.phase == "phase_i"
    assert_allclose(partial.probability, [1, 0, 0, 0])
    result = parallel_phase12_replay(records)
    assert result.phase == "phase_ii" and result.phase_one_enrollment == 6
    assert np.array_equal(result.admissible, [True, False, False, False])
    assert_allclose(result.probability, [1, 0, 0, 0])
    with pytest.raises(ValueError, match="ineligible"):
        parallel_phase12_replay(records + [[1, 0, 0]])
    # 0/3 at the lowest opens both adjacent arms; 2/6 at one adjacent
    # remains admissible, but prevents opening the highest combination.
    adjacent = [[0, 0, 0]] * 3 + [[1, 1, 0], [1, 0, 1], [1, 0, 0]]
    adjacent += [[1, 1, 0], [1, 0, 1], [1, 0, 0]] + [[2, 0, 0]] * 3
    result = parallel_phase12_replay(adjacent)
    assert result.phase_one_enrollment == 12
    assert np.array_equal(result.admissible, [True, True, True, False])
    assert result.escalation_cleared[1] == -1
    assert not result.records.flags.writeable
    stopped = parallel_phase12_replay([[0, 1, 0]] * 3)
    assert stopped.phase == "complete" and stopped.selected is None
    with pytest.raises(ValueError, match="after the trial stopped"):
        parallel_phase12_replay([[0, 1, 0]] * 4)


def test_beta_reference_and_simulated_histories_replay_exactly():
    reference = np.loadtxt(
        Path(__file__).parent / "fixtures/parallel-phase12-beta.csv", delimiter=","
    )
    compare = compare_beta_binomial(
        BetaBinomialPosterior(reference[:, 2], reference[:, 3]),
        BetaBinomialPosterior(reference[:, 0], reference[:, 1]),
    )
    assert_allclose(compare.treatment_greater, reference[:, 4], atol=2e-9, rtol=0)
    rng = np.random.default_rng(8512)
    for _ in range(6):
        simulated = simulate_parallel_phase12(
            [0.04, 0.09, 0.16, 0.25], [0.1, 0.2, 0.35, 0.5], rng=rng
        )
        replay = parallel_phase12_replay(simulated.records)
        assert replay.phase == "complete"
        assert replay.selected == simulated.selected and replay.reason == simulated.reason
        assert_allclose(replay.treated, simulated.treated)
        assert len(simulated.records) <= 100
    toxic = simulate_parallel_phase12([1, 1, 1, 1], [0, 0, 0, 0], rng=rng)
    assert len(toxic.records) == 3 and toxic.reason == "no admissible arms"


def test_serial_oc_report_matches_seeded_trial_replays_and_cluster_rates():
    toxicity = [0.04, 0.09, 0.16, 0.25]
    efficacy = [0.1, 0.2, 0.35, 0.5]
    report = simulate_parallel_phase12_oc(
        toxicity,
        efficacy,
        n_trials=4,
        seed=8512,
        optimal_arms=(2, 3),
    )
    trials = [
        simulate_parallel_phase12(toxicity, efficacy, rng=np.random.default_rng(int(trial_seed)))
        for trial_seed in report.per_trial_seeds
    ]
    expected_selected_counts = np.asarray(
        [sum(trial.selected == arm for trial in trials) for arm in range(4)]
    )
    np.testing.assert_array_equal(report.selected_counts, expected_selected_counts)
    assert report.no_selection_count == sum(trial.selected is None for trial in trials)
    np.testing.assert_array_equal(
        report.treated_total, np.sum([trial.treated for trial in trials], axis=0)
    )
    np.testing.assert_array_equal(
        report.toxicity_total, np.sum([trial.toxicities for trial in trials], axis=0)
    )
    np.testing.assert_array_equal(
        report.response_total, np.sum([trial.responses for trial in trials], axis=0)
    )
    assert report.total_enrollment == sum(len(trial.records) for trial in trials)
    assert report.stopping_probability.sum() == pytest.approx(1)
    assert report.selection_probability.sum() + report.no_selection_probability == pytest.approx(1)
    assert report.optimal_arms == (2, 3)
    assert report.optimal_selection_probability == pytest.approx(
        sum(trial.selected in (2, 3) for trial in trials) / len(trials)
    )
    assert report.toxicity_rate[0] == pytest.approx(
        report.toxicity_total[0] / report.treated_total[0]
    )
    arm_zero_rate = report.toxicity_rate[0]
    arm_zero_influence = np.asarray(
        [trial.toxicities[0] - arm_zero_rate * trial.treated[0] for trial in trials]
    )
    expected_rate_mcse = (
        np.std(arm_zero_influence, ddof=1)
        / np.sqrt(len(trials))
        / np.mean([trial.treated[0] for trial in trials])
    )
    assert report.toxicity_rate_mcse[0] == pytest.approx(expected_rate_mcse)
    assert not report.per_trial_seeds.flags.writeable
    assert not hasattr(report, "records")


def test_oc_undefined_rates_and_one_trial_mcse_are_explicit():
    report = simulate_parallel_phase12_oc([1, 1, 1, 1], [0, 0, 0, 0], n_trials=1, seed=4)
    assert report.no_selection_probability == 1
    assert report.stopping_reasons[report.stopping_counts.argmax()] == "no admissible arms"
    assert report.mean_phase_one_enrollment == 3
    assert report.treated_total[0] == 3
    assert np.isnan(report.toxicity_rate[1:]).all()
    assert np.isnan(report.mcse_treated).all()
    assert np.isnan(report.toxicity_rate_mcse).all()
    assert np.isnan(report.selection_mcse).all()
    assert np.isnan(report.stopping_mcse).all()
    assert np.isnan(report.admissibility_mcse).all()
    assert np.isnan(report.no_selection_mcse)
    with pytest.raises(ValueError, match="optimal_arms"):
        simulate_parallel_phase12_oc([0.1] * 4, [0.2] * 4, n_trials=1, seed=1, optimal_arms=[4])
    with pytest.raises(ValueError, match="n_trials"):
        simulate_parallel_phase12_oc([0.1] * 4, [0.2] * 4, n_trials=10_001, seed=1)
