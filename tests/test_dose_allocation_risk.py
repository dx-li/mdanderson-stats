import numpy as np
import pytest

from mdanderson_stats.dose_allocation_risk import dose_allocation_risks


def test_strict_risk_boundaries_use_planned_enrollment_and_report_mcse() -> None:
    result = dose_allocation_risks(
        [[0, 2, 6], [0, 2, 7], [0, 2, 8], [0, 1, 9]],
        [0.1, 0.3, 0.7],
        target=0.3,
        planned_patients=10,
    )

    assert result.has_exact_target
    assert result.overdose60_trials == 3
    assert result.overdose80_trials == 1
    assert result.overdose60_probability == 0.75
    assert result.overdose80_probability == 0.25
    assert result.overdose60_mcse == pytest.approx(np.sqrt(0.75 * 0.25 / 4))
    assert result.overdose80_mcse == pytest.approx(np.sqrt(0.25 * 0.75 / 4))


def test_exact_target_gate_and_multiple_target_doses() -> None:
    unavailable = dose_allocation_risks(
        [[1, 2], [2, 1]], [0.1, 0.4], target=0.3, planned_patients=4
    )
    assert not unavailable.has_exact_target
    assert unavailable.unavailable_reason
    assert unavailable.overdose60_probability is None
    assert unavailable.overdose80_mcse is None

    available = dose_allocation_risks(
        [[0, 0, 4, 0], [0, 0, 0, 4]],
        [0.2, 0.3, 0.3, 0.8],
        target=0.3,
        planned_patients=[4, 4],
    )
    assert available.has_exact_target
    assert available.overdose60_trials == 1
    assert available.overdose80_trials == 1


def test_zero_over_target_allocations_have_zero_risk() -> None:
    result = dose_allocation_risks(
        [[1, 2, 1], [0, 4, 0]],
        [0.1, 0.2, 0.3],
        target=0.3,
        planned_patients=4,
    )
    assert result.overdose60_probability == 0.0
    assert result.overdose80_probability == 0.0
    assert result.overdose60_mcse == 0.0
    assert result.overdose80_mcse == 0.0


def test_nonmonotone_truth_uses_probability_mask_not_dose_order() -> None:
    result = dose_allocation_risks(
        [[0, 0, 7, 0]],
        [0.5, 0.3, 0.8, 0.2],
        target=0.3,
        planned_patients=10,
    )
    assert result.overdose60_trials == 1
    assert result.overdose80_trials == 0


@pytest.mark.parametrize(
    ("patients", "planned"),
    [([[1, 2], [0, 1]], [2]), ([[1, 3]], 3)],
)
def test_rejects_planning_mismatch_before_reporting(patients, planned) -> None:
    with pytest.raises(ValueError, match="planned_patients"):
        dose_allocation_risks(patients, [0.2, 0.3], target=0.3, planned_patients=planned)


def test_rejects_boolean_counts_and_oversized_matrices() -> None:
    with pytest.raises(ValueError, match="integer counts"):
        dose_allocation_risks([[True, 0]], [0.2, 0.3], target=0.3, planned_patients=1)
    with pytest.raises(ValueError, match="1000000"):
        dose_allocation_risks(
            np.zeros((10001, 100), dtype=np.int64),
            np.linspace(0.0, 1.0, 100),
            target=0.5,
            planned_patients=1,
        )
