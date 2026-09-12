import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design, admissibility, posterior


def test_quasi_beta_posterior_matches_marginal_utility_identity():
    result = posterior([3], [0], [0], toxicity_limit=0.35, efficacy_limit=0.25)
    # With no events, the default utility events are 40% of each patient.
    assert result.utility_events[0] == pytest.approx(1.2)
    assert result.utility_mean[0] == pytest.approx(44.0)
    assert result.toxicity_overdose_probability[0] == pytest.approx(0.65**4)


def test_nonadditive_utility_requires_joint_outcome_count():
    with pytest.raises(ValueError, match="efficacy_without_toxicity"):
        posterior(
            [4],
            [1],
            [2],
            toxicity_limit=0.35,
            efficacy_limit=0.25,
            utilities=(100, 30, 60, 0),
        )
    result = posterior(
        [4],
        [1],
        [2],
        toxicity_limit=0.35,
        efficacy_limit=0.25,
        utilities=(100, 30, 60, 0),
        efficacy_without_toxicity=[1],
    )
    assert result.utility_events[0] == pytest.approx(2.2)


def test_admissibility_requires_both_safety_and_efficacy():
    result = posterior([3, 3], [3, 0], [0, 0], toxicity_limit=0.35, efficacy_limit=0.25)
    mask = admissibility(result)
    assert np.array_equal(mask, [False, True])


def test_next_dose_explores_untreated_higher_dose_after_eight_patients():
    design = BOIN12Design(0.30, 0.35, 0.25)
    decision = design.next_dose([9, 0, 0], [0, 0, 0], [4, 0, 0], current_dose=1)
    assert decision.action == "explore_escalate"
    assert decision.next_dose == 2


def test_final_obd_is_utility_maximum_at_or_below_isotonic_mtd():
    design = BOIN12Design(0.30, 0.35, 0.25)
    result = design.select_obd([3, 6, 3], [0, 1, 2], [0, 3, 1])
    assert result.mtd == 2
    assert result.obd == 2
