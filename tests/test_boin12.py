import numpy as np
import pytest

from mdanderson_stats.boin12 import (
    BOIN12Design,
    admissibility,
    boin12_tradeoff_utilities,
    posterior,
    rank_desirability,
)


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


def test_tradeoff_utilities_are_the_exact_affine_map_for_correlated_cells():
    cells = np.array([0.18, 0.32, 0.21, 0.29])  # noT/E, noT/noE, T/E, T/noE
    pi_e = cells[0] + cells[2]
    pi_t = cells[2] + cells[3]
    for weight in (0.0, 0.37, 1.0):
        utilities = boin12_tradeoff_utilities(weight)
        mapped = float(cells @ np.asarray(utilities) / 100.0)
        expected = (weight + pi_e - weight * pi_t) / (1.0 + weight)
        assert mapped == pytest.approx(expected, abs=1e-15)
    assert boin12_tradeoff_utilities(0.0) == (100.0, 0.0, 100.0, 0.0)
    assert boin12_tradeoff_utilities(1.0) == (100.0, 50.0, 50.0, 0.0)
    for invalid in (-0.01, 1.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="weight"):
            boin12_tradeoff_utilities(invalid)


def test_tradeoff_design_uses_the_existing_posterior_and_selection_path():
    mapped = boin12_tradeoff_utilities(0.4)
    tradeoff = BOIN12Design.from_tradeoff(0.35, 0.25, weight=0.4)
    explicit = BOIN12Design(0.35, 0.25, utilities=mapped)
    assert tradeoff.utilities == mapped

    counts = ([3, 6, 3], [0, 1, 2], [0, 3, 1])
    left = tradeoff.posterior(*counts)
    right = explicit.posterior(*counts)
    assert left.utility_mean == pytest.approx(right.utility_mean)
    assert left.utility_probability == pytest.approx(right.utility_probability)
    decision_left = tradeoff.next_dose(*counts, current_dose=2)
    decision_right = explicit.next_dose(*counts, current_dose=2)
    assert decision_left.action == decision_right.action
    assert decision_left.next_dose == decision_right.next_dose
    selection_left = tradeoff.select_obd(*counts)
    selection_right = explicit.select_obd(*counts)
    assert selection_left.obd == selection_right.obd
    assert selection_left.mtd == selection_right.mtd
