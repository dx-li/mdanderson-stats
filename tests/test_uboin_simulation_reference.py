"""Independent base-R Gumbel cells and exact small-trial probabilities."""

import csv
from itertools import product
from math import factorial, prod
from pathlib import Path

import numpy as np

from mdanderson_stats.uboin_conduct import UBOINDesign
from mdanderson_stats.uboin_simulation import uboin_gumbel_probabilities


def rows(name):
    with (Path(__file__).parent / "fixtures" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def test_gumbel_cells_against_base_r():
    for row in rows("uboin-gumbel.csv"):
        toxicity, efficacy = float(row["tox"]), float(row["eff"])
        joint = uboin_gumbel_probabilities(
            [toxicity], [efficacy], association=float(row["association"])
        )[0]
        np.testing.assert_allclose(
            joint.ravel(),
            [float(row[k]) for k in ("p00", "p01", "p10", "p11")],
            atol=2e-15,
            rtol=2e-13,
        )
        np.testing.assert_allclose(joint.sum(axis=0), [1 - toxicity, toxicity], atol=2e-15)
        np.testing.assert_allclose(joint.sum(axis=1), [1 - efficacy, efficacy], atol=2e-15)


def test_exact_two_cohort_trial_against_independent_base_r():
    design = UBOINDesign(
        prior=np.full((2, 2), 0.25),
        utilities=[[30, 0], [100, 50]],
        candidate_scope="tried",
        s1=3,
        s2=9,
        max_patients=6,
    )
    truth = uboin_gumbel_probabilities([0.15, 0.35], [0.35, 0.65]).reshape(2, 4)
    patterns = [p for p in product(range(4), repeat=4) if sum(p) == 3]

    def weight(counts, probabilities):
        coefficient = factorial(3) / prod(factorial(v) for v in counts)
        return coefficient * prod(p**n for p, n in zip(probabilities, counts, strict=True))

    selection = np.zeros(3)
    mean_patients = np.zeros(2)
    early = 0.0
    for first in patterns:
        counts = np.zeros((2, 2, 2), dtype=int)
        counts[0] = np.array(first).reshape(2, 2)
        w1 = weight(first, truth[0])
        decision = design.decision(counts, current_dose=1)
        if decision.next_dose is None:
            selection[decision.selected_dose or 0] += w1
            mean_patients += w1 * counts.sum(axis=(1, 2))
            early += w1
            continue
        dose = decision.next_dose
        for second in patterns:
            final = counts.copy()
            final[dose - 1] += np.array(second).reshape(2, 2)
            mass = w1 * weight(second, truth[dose - 1])
            result = design.decision(
                final, current_dose=dose, stage=decision.stage, eliminated=decision.eliminated
            )
            assert not np.any(result.allocation_probabilities)
            selection[result.selected_dose or 0] += mass
            mean_patients += mass * final.sum(axis=(1, 2))
    reference = rows("uboin-small-trial.csv")[0]
    np.testing.assert_allclose(selection, [float(reference[f"selection{k}"]) for k in range(3)])
    np.testing.assert_allclose(mean_patients, [float(reference[f"patients{k}"]) for k in (1, 2)])
    np.testing.assert_allclose(early, float(reference["early"]))
    np.testing.assert_allclose(selection.sum(), 1, atol=1e-14)
