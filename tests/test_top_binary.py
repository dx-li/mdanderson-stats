import numpy as np
import pytest
from scipy.stats import beta

from mdanderson_stats import TOPBinaryDesign


def test_published_table_and_pending_followup_example():
    design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
    table = design.boundaries()
    np.testing.assert_equal(table.complete_go_min, [2, 4, 8, 12])
    np.testing.assert_equal(table.suspend_pending_min, [3, 10, 23, 1])
    assert table.futility_effective_size[1, 3] == pytest.approx(15.09, abs=0.005)
    np.testing.assert_allclose(
        table.futility_effective_size[2, 3:8], [11.28, 15.78, 20.35, 24.96, 29.61], atol=0.005
    )
    followup = np.array([85, 78, 66, 48, 32, 28, 10, 8, 5])
    result = design.evaluate_followup(11, 3, followup, 120)
    assert result.effective_sample_size == 14
    assert result.decision == "continue"
    assert result.acceptable_probability == pytest.approx(beta.sf(0.2, 3.2, 11.8))
    # Units cancel before summation; very large and small time units are valid.
    for scale in [1e-200, 1e200]:
        scaled = design.evaluate_followup(11, 3, followup * scale, 120 * scale)
        assert scaled.effective_sample_size == pytest.approx(14)
    assert not table.futility_effective_size.flags.writeable


def test_suspension_conventions_and_final_followup():
    design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
    strict = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40], suspension="strict")
    assert design.evaluate(20, 3, 10, 8).decision == "suspend"
    assert strict.evaluate(20, 3, 10, 8).decision == "stop_futility"
    assert design.evaluate(20, 4, 16, 0).decision == "continue"
    assert design.evaluate(40, 12, 1, 0.9).decision == "suspend"
    assert design.evaluate(40, 12, 0, 0).decision == "success"
    assert design.evaluate(40, 11, 0, 0).decision == "stop_futility"
    with pytest.raises(ValueError):
        design.evaluate(20, 12, 9, 0)


def test_complete_outcomes_match_existing_bop2_and_monotone_ess():
    design = TOPBinaryDesign(40, 0.2, 0.86, 0.95, looks=[10, 20, 30, 40])
    complete = design.complete_data_design()
    for n in [10, 20, 30, 40]:
        r = np.arange(n + 1)
        result = design.evaluate(n, r, 0, 0)
        np.testing.assert_allclose(result.acceptable_probability, complete.high_probability[n, r])
        np.testing.assert_equal(
            result.decision == "stop_futility",
            complete.high_probability[n, r] < 0.86 * (n / 40) ** 0.95,
        )
    root = design.boundaries().futility_effective_size[1, 3]
    result = design.evaluate(20, 3, 9, np.array([root - 11 - 1e-6, root - 11 + 1e-6]))
    np.testing.assert_equal(result.decision, ["continue", "stop_futility"])
    np.testing.assert_allclose(result.acceptable_probability, result.acceptable_cutoff, atol=1e-7)
