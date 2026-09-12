"""Reference U-BOIN stage-I decisions from independent base R boundaries."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.uboin_conduct import UBOINDesign


def test_stage_one_boundary_decisions_against_base_r():
    with (Path(__file__).parent / "fixtures/uboin-stage-one.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        n, escalate, deescalate, eliminate = (
            int(row[k]) for k in ("n", "escalate", "deescalate", "eliminate")
        )
        design = UBOINDesign(
            prior=np.full((2, 2), 0.25),
            utilities=[[30, 0], [100, 50]],
            candidate_scope="tried",
            toxicity_limit=float(row["upper"]),
            delta=0.05,
            s1=50,
            s2=60,
            max_patients=100,
        )
        cases = {0, escalate, escalate + 1, deescalate - 1, deescalate, eliminate - 1, eliminate, n}
        for m in sorted(cases & set(range(n + 1))):
            counts = np.zeros((3, 2, 2), dtype=int)
            counts[0, 1, 0] = 3
            counts[1, 1, 0] = n - m
            counts[1, 1, 1] = m
            decision = design.decision(counts, current_dose=2, stage=1)
            expected = 1 if m >= eliminate or m >= deescalate else 3 if m <= escalate else 2
            assert decision.next_dose == expected, (row, m)
            np.testing.assert_array_equal(
                decision.eliminated, [False, m >= eliminate, m >= eliminate]
            )


def test_trial_cap_precedes_reassignment_from_eliminated_current_dose():
    design = UBOINDesign(
        prior=np.full((2, 2), 0.25),
        utilities=[[30, 0], [100, 50]],
        candidate_scope="tried",
        s1=6,
        s2=12,
        max_patients=6,
    )
    counts = np.zeros((2, 2, 2), dtype=int)
    counts[0, 1, 0] = 3
    counts[1, 0, 1] = 3
    decision = design.decision(counts, current_dose=2)
    assert decision.action == "stop_max_patients"
    assert decision.selected_dose == 1
    assert not np.any(decision.allocation_probabilities)
    assert decision.next_dose is None


def test_eliminated_starting_dose_stops_before_any_patient_is_assigned():
    design = UBOINDesign(
        prior=np.full((2, 2), 0.25),
        utilities=[[30, 0], [100, 50]],
        candidate_scope="tried",
        starting_dose=2,
    )
    decision = design.decision(np.zeros((2, 2, 2)), current_dose=2, eliminated=[False, True])
    assert decision.action == "stop_safety"
    assert not np.any(decision.allocation_probabilities)
