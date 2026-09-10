from numpy.testing import assert_allclose

from mdanderson_stats import phase12_accrual_ready, phase12_phase_one


def test_source_asymmetric_opening_and_strict_six_patient_toxicity_rule():
    first = phase12_phase_one([3, 0, 0, 0, 0, 0], [0] * 6)
    assert_allclose(first.probability, [0, 0.5, 0.5, 0, 0, 0])
    assert first.admissible[0]
    expanded = phase12_phase_one(
        [3, 3, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        current_dose=1,
        opened=first.opened,
        admissible=first.admissible,
    )
    assert_allclose(expanded.probability, [0, 1, 0, 0, 0, 0])
    rejected = phase12_phase_one([6, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0])
    assert rejected.done and rejected.trial_closed and rejected.closed[0]
    # Arm 1 closed, arm 2 clears: only arm 4 opens.
    advance = phase12_phase_one(
        [3, 3, 3, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        current_dose=2,
        opened=[1, 1, 1, 0, 0, 0],
        closed=[0, 1, 0, 0, 0, 0],
        admissible=[1, 0, 0, 0, 0, 0],
    )
    assert_allclose(advance.probability, [0, 0, 0, 0, 1, 0])
    assert not advance.opened[3] and advance.opened[4]
    # The mirror case ends phase I; it does not open arm 3.
    mirror = phase12_phase_one(
        [3, 3, 3, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        current_dose=2,
        opened=[1, 1, 1, 0, 0, 0],
        admissible=[1, 1, 0, 0, 0, 0],
    )
    assert mirror.done and not mirror.trial_closed
    assert_allclose(mirror.probability, 0)


def test_phase_specific_waiting_is_at_cohort_boundaries():
    records = [[0, 0, 0, 12, 0, 8], [0, 1, 1, 7, 0, 9], [0, 2, 0, 14, 1, 4]]
    assert phase12_accrual_ready(records[:2], time=3)
    assert not phase12_accrual_ready(records, time=8)
    assert phase12_accrual_ready(records, time=9)
    assert phase12_accrual_ready(records, time=9, phase_two_start=3)
    records += [[1, 10 + i, 0, 30 + i, 0, 20 + i] for i in range(5)]
    assert phase12_accrual_ready(records[:7], time=20, phase_two_start=3)
    assert not phase12_accrual_ready(records, time=33, phase_two_start=3)
    assert phase12_accrual_ready(records, time=34, phase_two_start=3)
    assert not phase12_accrual_ready(records, time=40, phase_two_start=3, max_patients=8)
    assert not phase12_accrual_ready([], time=0, trial_closed=True)
