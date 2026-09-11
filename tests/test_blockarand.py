import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import (
    BlockArandDesign,
    blockarand_block,
    blockarand_decision,
    blockarand_plan,
    simulate_blockarand,
    simulate_blockarand_oc,
)


def test_paper_block_example_posterior_identity_and_stopping():
    plan = blockarand_plan(0.68, min_block=4, max_block=8)
    assert (plan.arm_zero_count, plan.size) == (4, 6)
    assignments = blockarand_block(plan, rng=np.random.default_rng(9001))
    assert np.sum(assignments == 0) == 4 and np.sum(assignments == 1) == 2
    assert not assignments.flags.writeable
    assert blockarand_plan(0.5).size == 2
    assert blockarand_plan(1).allocation_probability == 7 / 8
    assert blockarand_plan(0).allocation_probability == 1 / 8
    design = BlockArandDesign(prior=((1, 1), (1, 1)), burnin=0)
    # P(Beta(2,1)>Beta(1,2)) = 5/6, an exact polynomial integral.
    decision = blockarand_decision([1, 0], [0, 1], design=design)
    assert_allclose(decision.superiority, [5 / 6, 1 / 6], atol=1e-10, rtol=0)
    assert_allclose(decision.target_probability, np.sqrt(5) / (1 + np.sqrt(5)))
    assert not decision.stopped
    # Library final cutoff is .9, independent of the .95 early rule.
    capped = BlockArandDesign(max_patients=4, burnin=4, prior=((1, 1), (1, 1)))
    final = blockarand_decision([2, 0], [0, 2], design=capped)
    assert_allclose(final.superiority[0], 0.95, atol=1e-10)
    assert final.stopped and final.selected == 0
    # A strong asymmetric prior can stop before the first enrollment.
    prior_stop = simulate_blockarand(
        [0.5, 0.5],
        design=BlockArandDesign(prior=((100, 1), (1, 100))),
        rng=np.random.default_rng(9002),
    )
    assert prior_stop.records.shape == (0, 3) and prior_stop.decision.selected == 0


def test_exact_burnin_switch_patient_monitoring_and_operating_characteristics():
    design = BlockArandDesign(
        max_patients=19, burnin=5, max_block=7, early_cutoff=1, final_cutoff=1
    )
    trial = simulate_blockarand([0.5, 0.5], design=design, rng=np.random.default_rng(9003))
    assert len(trial.records) == 19 and trial.decision.selected is None
    assert trial.plans[0].size == 6 and trial.block_starts[1] == 5
    # Burn-in truncates its block; subsequent complete blocks have planned counts.
    ends = np.r_[trial.block_starts[1:], len(trial.records)]
    for i, (start, end, plan) in enumerate(zip(trial.block_starts, ends, trial.plans)):
        rows = trial.records[int(start) : int(end)]
        assert_allclose(rows[:, 2], i)
        if len(rows) == plan.size:
            assert np.sum(rows[:, 0] == 0) == plan.arm_zero_count
    for i in range(len(trial.records) + 1):
        rows = trial.records[:i]
        s = [np.sum((rows[:, 0] == arm) & (rows[:, 1] == 1)) for arm in (0, 1)]
        f = [np.sum((rows[:, 0] == arm) & (rows[:, 1] == 0)) for arm in (0, 1)]
        assert blockarand_decision(s, f, design=design).stopped == (i == 19)
    oc = simulate_blockarand_oc(
        [1, 0],
        repetitions=12,
        design=BlockArandDesign(max_patients=20, burnin=20),
        rng=np.random.default_rng(9004),
    )
    assert_allclose(oc.selection_probability, [1, 0])
    assert np.all(oc.enrollment.sum(axis=1) < 20)
    assert_allclose(oc.mean_patients, oc.enrollment.mean(axis=0))
    assert oc.no_selection_probability == 0 and oc.cache_hits > 0
    # Patient-wise stopping can occur before a balanced burn-in block completes.
    stopped = simulate_blockarand(
        [1, 0], design=BlockArandDesign(burnin=200), rng=np.random.default_rng(9005)
    )
    assert len(stopped.records) < stopped.plans[0].size
