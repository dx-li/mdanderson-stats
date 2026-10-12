from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import MTPIDesign, select_mtpi_legacy_mtd


def test_unchanged_original_r_references():
    data = np.genfromtxt(
        Path(__file__).parent / "fixtures/mtpi-legacy-selection.csv", delimiter=",", names=True
    )
    for case in np.unique(data["case"]):
        rows = data[data["case"] == case]
        result = select_mtpi_legacy_mtd(
            rows["patients"],
            rows["toxicities"],
            target=rows["target"][0],
            first_excluded_dose=int(rows["first_excluded_dose"][0]),
        )
        assert result.dose == rows["selected"][0]
        np.testing.assert_allclose(result.selection_score, rows["score"], rtol=3e-14, atol=1e-16)
        np.testing.assert_array_equal(result.eligible, rows["eligible"])
        for value in (result.isotonic_mean, result.selection_score, result.eligible):
            assert not value.flags.writeable


def test_exact_rational_max_min_characterization_of_weighted_isotonic_fit():
    # Independent characterization, not PAVA: fitted_i = max_a<=i min_b>=i
    # weighted_mean(a..b). Rational arithmetic avoids any SciPy/R dependency.
    n = [2, 3, 4]
    for y in product(*(range(value + 1) for value in n)):
        a = [Fraction(value) + Fraction(1, 200) for value in y]
        b = [Fraction(total - value) + Fraction(1, 200) for total, value in zip(n, y)]
        means = [aa / (aa + bb) for aa, bb in zip(a, b)]
        weights = [(aa + bb) ** 2 * (aa + bb + 1) / (aa * bb) for aa, bb in zip(a, b)]

        def average(left, right):
            return sum(weights[j] * means[j] for j in range(left, right + 1)) / sum(
                weights[left : right + 1]
            )

        expected = [
            max(min(average(a, b) for b in range(i, 3)) for a in range(i + 1)) for i in range(3)
        ]
        actual = select_mtpi_legacy_mtd(n, y)
        np.testing.assert_allclose(actual.isotonic_mean, list(map(float, expected)), rtol=1e-14)


def test_separate_estimation_prior_safety_and_lower_untried_doses():
    result = select_mtpi_legacy_mtd([0, 0, 6, 3], [0, 0, 2, 2])
    assert np.isfinite(result.isotonic_mean).all()
    assert result.isotonic_mean[0] == result.isotonic_mean[2]
    np.testing.assert_array_equal(result.eligible, [False, False, True, True])
    # Terminal estimator must not invent a missing controller safety decision.
    assert select_mtpi_legacy_mtd([3, 3], [3, 3]).dose == 1
    assert select_mtpi_legacy_mtd([3, 3], [3, 3], no_selection=True).dose is None
    assert MTPIDesign().select_mtd([3, 3], [3, 3]).dose is None
    pooled = select_mtpi_legacy_mtd([3, 9, 6], [1, 1, 3])
    paper = MTPIDesign().select_mtd([3, 9, 6], [1, 1, 3])
    assert not np.allclose(pooled.isotonic_mean, paper.isotonic_mean)


def test_perturbation_single_dose_and_empty_ranges():
    below = select_mtpi_legacy_mtd([3, 3, 3], [0, 0, 0])
    above = select_mtpi_legacy_mtd([3, 3, 3], [3, 3, 3])
    assert below.dose == 3
    assert above.dose == 1
    np.testing.assert_allclose(
        below.selection_score - below.isotonic_mean, [0, 2e-10, 3e-10], atol=1e-18
    )
    single = select_mtpi_legacy_mtd([3, 0], [1, 0])
    assert single.dose == 1
    assert single.selection_score[0] == single.isotonic_mean[0]
    assert np.isnan(single.selection_score[1])
    for n, y, excluded in [([0, 0], [0, 0], None), ([0, 3], [0, 1], 2), ([3], [1], 1)]:
        result = select_mtpi_legacy_mtd(n, y, first_excluded_dose=excluded)
        assert result.dose is None
        assert not result.eligible.any()


@pytest.mark.parametrize(
    "patients,toxicities,kwargs",
    [
        ([], [], {}),
        ([3], [4], {}),
        ([3], [-1], {}),
        ([201], [0], {}),
        ([3, 0], [0], {}),
        ([[3]], [[0]], {}),
        ([3.1], [0], {}),
        ([3], [0], {"target": 0}),
        ([3], [0], {"target": np.nan}),
        ([3], [0], {"first_excluded_dose": 0}),
        ([3], [0], {"first_excluded_dose": 3}),
        ([3], [0], {"first_excluded_dose": 1.5}),
        ([3], [0], {"no_selection": "false"}),
    ],
)
def test_invalid_states(patients, toxicities, kwargs):
    with pytest.raises(ValueError):
        select_mtpi_legacy_mtd(patients, toxicities, **kwargs)


def test_original_r_exact_residual_tie_selects_lowest_dose():
    data = np.genfromtxt(
        Path(__file__).parent / "fixtures/mtpi-legacy-selection.csv", delimiter=",", names=True
    )
    rows = data[data["case"] == 11]
    distances = abs(rows["score"] - rows["target"])
    assert distances[0] == distances[1]
    assert rows["selected"].tolist() == [1, 1]
    result = select_mtpi_legacy_mtd([2, 2], [1, 1], target=0.5000000001)
    assert result.dose == 1
