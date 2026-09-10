from collections import defaultdict
from itertools import product

import numpy as np
import pytest
from scipy.stats import beta

from mdanderson_stats import (
    MERITDesign,
    MERITInterims,
    run_merit_trial,
    simulate_merit,
    simulate_merit_interims,
)


def test_boundaries_equal_direct_posterior_events():
    policy = MERITInterims(0.2, 0.4, toxicity_looks=[2, 5, 9], efficacy_looks=[3, 5, 8])
    table = policy.boundaries(10)
    for k, n in enumerate(table.patients):
        events = np.arange(n + 1)
        expected_t = beta.sf(0.2, events + 0.1, n - events + 0.1) > 0.95
        expected_e = beta.cdf(0.4, events + 0.1, n - events + 0.1) > 0.95
        np.testing.assert_equal(
            events >= table.toxicity_stop_min[k], expected_t & table.assess_toxicity[k]
        )
        np.testing.assert_equal(
            events <= table.efficacy_stop_max[k], expected_e & table.assess_efficacy[k]
        )
    assert not table.patients.flags.writeable
    with pytest.raises(ValueError):
        policy.boundaries(9)


def test_replay_sticky_stops_and_survivor_pooling():
    design = MERITDesign(6, 2, 2)
    policy = MERITInterims(0.3, 0.5, toxicity_looks=[2], efficacy_looks=[3])
    toxicity = np.tile([1, 0], (6, 1))
    efficacy = np.tile([1, 0], (6, 1))
    result = run_merit_trial(design, policy, toxicity, efficacy)
    np.testing.assert_equal(result.patients, [[2, 3]])
    np.testing.assert_equal(result.toxicity, [[2, 0]])
    np.testing.assert_equal(result.efficacy, [[2, 0]])
    np.testing.assert_equal(result.stopped_toxicity, [[True, False]])
    np.testing.assert_equal(result.stopped_futility, [[False, True]])
    assert not result.admissible.any()
    shorter = run_merit_trial(design, policy, toxicity[:3], efficacy[:3])
    np.testing.assert_equal(shorter.patients, result.patients)
    with pytest.raises(ValueError, match="outcomes ended"):
        run_merit_trial(design, policy, toxicity[:1], efficacy[:1])
    # All post-stop potential outcomes are irrelevant, including later efficacy.
    toxicity[2:, 0] = 0
    efficacy[3:, 1] = 1
    replay = run_merit_trial(design, policy, toxicity, efficacy)
    np.testing.assert_equal(replay.patients, result.patients)
    np.testing.assert_equal(replay.efficacy, result.efficacy)
    survivor = run_merit_trial(
        MERITDesign(6, 2, 3, doses=3),
        MERITInterims(0.3, 0.5, toxicity_looks=[2]),
        np.tile([0, 1, 0], (6, 1)),
        np.tile([1, 0, 0], (6, 1)),
    )
    np.testing.assert_equal(survivor.patients, [[6, 2, 6]])
    np.testing.assert_equal(survivor.admissible, [[True, False, True]])


def test_simulation_matches_exact_stopping_paths_and_no_look_replay():
    design = MERITDesign(4, 1, 2, isotonic_toxicity=False, isotonic_efficacy=False)
    policy = MERITInterims(
        0.2,
        0.5,
        toxicity_looks=[1, 3],
        efficacy_looks=[2],
        toxicity_cutoff=0.8,
        efficacy_cutoff=0.8,
    )
    pt, pe = 0.3, 0.55
    # Independent reference enumerates all arm-level binary histories using
    # direct posterior decisions rather than the implementation's boundary table.
    states = {(0, 0, 0, False, False): 1.0}
    for step in range(1, 5):
        updated = defaultdict(float)
        for (n, t, e, stop_t, stop_e), probability in states.items():
            if stop_t or stop_e:
                updated[n, t, e, stop_t, stop_e] += probability
                continue
            for x, y in product([0, 1], repeat=2):
                mass = (pt if x else 1 - pt) * (pe if y else 1 - pe)
                nt, ne = t + x, e + y
                st = step in [1, 3] and beta.sf(0.2, nt + 0.1, step - nt + 0.1) > 0.8
                se = step == 2 and beta.cdf(0.5, ne + 0.1, step - ne + 0.1) > 0.8
                updated[step, nt, ne, st, se] += probability * mass
        states = updated
    expected_n = sum(key[0] * p for key, p in states.items())
    expected_selection = sum(
        p for (n, t, e, st, se), p in states.items() if not (st or se) and t <= 1 and e >= 2
    )
    expected_t = sum(p for (_, _, _, st, _), p in states.items() if st)
    expected_e = sum(p for (_, _, _, _, se), p in states.items() if se)
    result = simulate_merit_interims(
        design, policy, [pt, pt], [pe, pe], correlation=0, trials=50000, rng=160
    )
    np.testing.assert_allclose(result.mean_patients, expected_n, atol=0.02)
    np.testing.assert_allclose(result.selection_probability, expected_selection, atol=0.006)
    np.testing.assert_allclose(result.stopped_toxicity.mean(0), expected_t, atol=0.006)
    np.testing.assert_allclose(result.stopped_futility.mean(0), expected_e, atol=0.006)
    fixed = simulate_merit(design, [pt, pt], [pe, pe], trials=1000, rng=162)
    no_looks = simulate_merit_interims(
        design, MERITInterims(0.2, 0.5), [pt, pt], [pe, pe], trials=1000, rng=162
    )
    np.testing.assert_equal(no_looks.toxicity, fixed.toxicity)
    np.testing.assert_equal(no_looks.efficacy, fixed.efficacy)
    np.testing.assert_equal(no_looks.admissible, fixed.admissible)
