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
