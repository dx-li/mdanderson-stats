import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design, admissibility, posterior, rank_desirability


def test_quasi_beta_posterior_matches_marginal_utility_identity():
    result = posterior([3], [0], [0], toxicity_limit=0.35, efficacy_limit=0.25)
    # With no events, the default utility events are 40% of each patient.
    assert result.utility_events[0] == pytest.approx(1.2)
    assert result.utility_mean[0] == pytest.approx(44.0)
    assert result.utility_probability[0] == pytest.approx(0.11340010343238265)
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
    design = BOIN12Design(0.35, 0.25)
    decision = design.next_dose([9, 0, 0], [0, 0, 0], [4, 0, 0], current_dose=1)
    assert decision.action == "explore_escalate"
    assert decision.next_dose == 2


def test_toxicity_boundary_forces_bounded_deescalation():
    design = BOIN12Design(0.35, 0.25)
    decision = design.next_dose([0, 0, 3], [0, 0, 3], [0, 0, 0], current_dose=3)
    assert decision.action == "deescalate"
    assert decision.next_dose == 2


def test_final_obd_is_utility_maximum_at_or_below_isotonic_mtd():
    design = BOIN12Design(0.35, 0.25)
    result = design.select_obd([3, 6, 3], [0, 1, 2], [0, 3, 1])
    assert result.mtd == 2
    assert result.obd == 2


def test_source_replay_uses_joint_outcomes_and_global_rds_ranks():
    design = BOIN12Design(0.35, 0.25)
    patients = np.array([3, 6, 3, 0, 0])
    toxicities = np.array([0, 1, 2, 0, 0])
    efficacies = np.array([0, 3, 1, 0, 0])
    efficacy_without_toxicity = np.array([0, 3, 1, 0, 0])
    decision = design.next_dose(
        patients,
        toxicities,
        efficacies,
        current_dose=2,
        efficacy_without_toxicity=efficacy_without_toxicity,
    )
    assert decision.next_dose == 2
    assert decision.admissible.tolist() == [True, True, True, False, False]
    assert decision.posterior.utility_probability[:3].tolist() == pytest.approx(
        [0.113400103432383, 0.286202018249043, 0.079969448125000]
    )
    table = rank_desirability([0, 3, 6, 9], toxicity_limit=0.35, efficacy_limit=0.25)
    sample_three = np.flatnonzero(table.patients == 3)
    assert table.rds[sample_three[:5]].tolist() == pytest.approx([35, 55, 76, 91, 24])


def test_eliminated_current_dose_can_move_to_safe_neighbor():
    design = BOIN12Design(0.35, 0.25)
    decision = design.next_dose(
        [3, 3, 0], [0, 0, 0], [0, 0, 0], current_dose=2, eliminated=[False, True, False]
    )
    assert decision.action == "escalate"
    assert decision.next_dose == 3


def test_rds_rejects_unbounded_case_expansion_before_allocation():
    with pytest.raises(ValueError, match="100000-case"):
        rank_desirability(range(400), toxicity_limit=0.35, efficacy_limit=0.25)


def test_rds_rejects_nonadditive_without_joint_cell_enumeration():
    with pytest.raises(ValueError, match="nonadditive"):
        rank_desirability([3], toxicity_limit=0.35, efficacy_limit=0.25, utilities=(100, 30, 60, 0))
