import numpy as np
import pytest
from scipy.stats import binom

from mdanderson_stats import TOPBinaryDesign, run_top_binary_trial, simulate_top_binary


def test_suspend_resume_final_wait_and_time_scaling():
    design = TOPBinaryDesign(4, 0.2, 0.86, 0.95, looks=[2, 4])
    trial = run_top_binary_trial(design, np.zeros(4), [0.2, np.inf, 0.1, np.inf], 1)
    np.testing.assert_allclose(trial.enrollment_times, [0, 0, 0.2, 0.2])
    assert trial.suspension_time == pytest.approx(0.2)
    assert trial.final_followup_time == pytest.approx(1)
    assert trial.final_time == pytest.approx(1.2)
    assert trial.decision == "success"
    assert not trial.pending.any()
    assert [s.decision for s in trial.steps] == [
        "suspend",
        "continue",
        "suspend",
        "suspend",
        "suspend",
        "success",
    ]
    assert [s.time for s in trial.steps] == pytest.approx([0, 0.2, 0.2, 0.3, 1, 1.2])
    for scale in [1e-150, 1e150]:
        rescaled = run_top_binary_trial(
            design, np.zeros(4), np.array([0.2, np.inf, 0.1, np.inf]) * scale, scale
        )
        assert rescaled.final_time / scale == pytest.approx(trial.final_time)
        assert rescaled.decision == trial.decision


def test_futility_uses_only_current_observations_and_preserves_pending():
    design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
    gaps = np.r_[0, np.ones(39)]
    delays = np.full(40, np.inf)
    reference = run_top_binary_trial(design, gaps, delays, 3)
    assert reference.final_time == 10
    assert reference.enrollment_times.size == 10
    assert reference.decision == "stop_futility"
    assert reference.pending.sum() == 2
    # Two already enrolled patients respond later, and all unenrolled patients
    # would respond instantly. None of those future events may affect the stop.
    delays[8:10] = 2.5
    delays[10:] = 0
    alternative = run_top_binary_trial(design, gaps, delays, 3)
    assert alternative.steps == reference.steps
    assert np.isinf(alternative.observed_response_times).all()
    assert not alternative.enrollment_times.flags.writeable
    with pytest.raises(ArithmeticError, match="rescale"):
        run_top_binary_trial(design, np.full(40, 1e300), np.full(40, np.inf), 1)


def test_single_final_analysis_matches_exact_binomial_probability():
    design = TOPBinaryDesign(8, 0.2, 0.8, 0.5, looks=[8])
    go = int(design.boundaries().complete_go_min[0])
    result = simulate_top_binary(design, 0.4, 2, 3, trials=30000, rng=134)
    assert result.success_probability == pytest.approx(binom.sf(go - 1, 8, 0.4), abs=0.008)
    np.testing.assert_equal(result.patients, np.full(30000, 8))
    assert not result.pending.any()
    assert not result.suspension_time.any()
    assert result.final_followup_time.mean() > 0
    for probability, decision in [(0, "stop_futility"), (1, "success")]:
        extreme = simulate_top_binary(
            design, probability, 2, 3, trials=10, arrival="fixed", rng=134
        )
        np.testing.assert_equal(extreme.decision, np.full(10, decision))
