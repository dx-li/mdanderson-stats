"""Native offspring probabilities and independent Mendelian calculations."""

import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import tdtasp_genetics, tdtasp_haplotype_frequencies

FIXTURE = json.loads((Path(__file__).parent / "fixtures/tdtasp_genetics.json").read_text())


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda c: f"p{c['penetrance']}-t{c['recombination']}"
)
def test_native_family_probabilities(case):
    result = tdtasp_genetics(
        [0.2, 0.1, 0.3, 0.4], case["penetrance"], case["recombination"], legacy_asp=True
    )
    expected = np.array(case["rows"])
    np.testing.assert_array_equal(result.father_heterozygous, expected[:, 0])
    np.testing.assert_array_equal(result.mother_heterozygous, expected[:, 1])
    np.testing.assert_array_equal(
        result.father_heterozygous.astype(int) + result.mother_heterozygous, expected[:, 2]
    )
    np.testing.assert_allclose(result.affected_probability, expected[:, 3], rtol=2e-14, atol=0)
    np.testing.assert_allclose(
        result.transmission_probability, expected[:, 4], rtol=2e-14, atol=1e-15
    )
    np.testing.assert_allclose(result.sharing_probability, expected[:, 5], rtol=2e-14, atol=1e-15)
    corrected = tdtasp_genetics([0.2, 0.1, 0.3, 0.4], case["penetrance"], case["recombination"])
    np.testing.assert_array_equal(corrected.affected_probability, result.affected_probability)
    np.testing.assert_array_equal(
        corrected.transmission_probability, result.transmission_probability
    )


def independent_family(parent_codes, theta, penetrance):
    chromosomes = [("B", "d"), ("B", "D"), ("A", "d"), ("A", "D")]
    gametes = []
    informative = []
    for first, second in (parent_codes[:2], parent_codes[2:]):
        x, y = chromosomes[first], chromosomes[second]
        informative.append(x[0] != y[0])
        probabilities = defaultdict(float)
        for gamete, weight in (
            (x, (1 - theta) / 2),
            (y, (1 - theta) / 2),
            ((x[0], y[1]), theta / 2),
            ((y[0], x[1]), theta / 2),
        ):
            probabilities[gamete] += weight
        gametes.append(probabilities)
    outcomes = []
    for f, m in product(gametes[0], gametes[1]):
        copies = (f[1] == "D") + (m[1] == "D")
        mass = gametes[0][f] * gametes[1][m] * penetrance[2 - copies]
        outcomes.append((f, m, mass))
    affected = sum(row[2] for row in outcomes)
    if not affected or not any(informative):
        return affected, 0, 0
    transmission = (
        sum(
            mass * sum(flag and child[0] == "A" for flag, child in zip(informative, (f, m)))
            for f, m, mass in outcomes
        )
        / sum(informative)
        / affected
    )
    same = 0
    for (f1, m1, p1), (f2, m2, p2) in product(outcomes, repeat=2):
        same += (
            p1
            * p2
            * sum(flag and x[0] == y[0] for flag, x, y in zip(informative, (f1, m1), (f2, m2)))
        )
    return affected, transmission, same / sum(informative) / affected**2


@pytest.mark.parametrize("theta", [0, 0.17, 0.5])
def test_independent_affected_sibling_enumeration(theta):
    penetrance = [0.8, 0.3, 0.01]
    result = tdtasp_genetics([0.2, 0.1, 0.3, 0.4], penetrance, theta)
    expected = np.array([independent_family(row, theta, penetrance) for row in result.parents])
    np.testing.assert_allclose(result.affected_probability, expected[:, 0], atol=1e-15)
    np.testing.assert_allclose(result.transmission_probability, expected[:, 1], atol=1e-15)
    np.testing.assert_allclose(result.sharing_probability, expected[:, 2], atol=2e-15)


@pytest.mark.parametrize(
    "penetrance,affected,transmission,sharing",
    [
        ([1, 1, 0], 0.75, 2 / 3, 5 / 9),
        ([1, 0, 0], 0.25, 1, 1),
        ([0.5, 0.5, 0.5], 0.5, 0.5, 0.5),
        ([0, 0, 0], 0, 0, 0),
    ],
)
def test_coupling_phase_parents(penetrance, affected, transmission, sharing):
    result = tdtasp_genetics([0.5, 0, 0, 0.5], penetrance)
    row = np.flatnonzero(np.all(result.parents == [3, 0, 3, 0], axis=1)).item()
    assert result.affected_probability[row] == pytest.approx(affected)
    assert result.transmission_probability[row] == pytest.approx(transmission)
    assert result.sharing_probability[row] == pytest.approx(sharing)


@pytest.mark.parametrize("theta", [0, 0.13, 0.5])
def test_population_random_mating(theta):
    frequencies = [0.2, 0.1, 0.3, 0.4]
    result = tdtasp_genetics(frequencies, [0.8, 0.3, 0.01], theta)
    assert result.parent_probability.sum() == pytest.approx(1)
    assert result.population_affected_probability == pytest.approx(
        0.8 * 0.5**2 + 2 * 0.3 * 0.5 * 0.5 + 0.01 * 0.5**2
    )
    assert np.all(result.sharing_probability >= 0)
    assert np.all(result.sharing_probability <= 1 + 1e-15)
    for i, row in enumerate(result.parents):
        assert result.parent_probability[i] == pytest.approx(
            np.prod(np.array(frequencies)[3 - row])
        )


@pytest.mark.parametrize("marker,disease", [(0.2, 0.7), (0.9, 0.8), (0, 1), (1, 0)])
@pytest.mark.parametrize("relative", [-1, -0.3, 0, 0.5, 1])
def test_marginals_and_disequilibrium(marker, disease, relative):
    frequencies = tdtasp_haplotype_frequencies(marker, disease, relative)
    assert frequencies.sum() == pytest.approx(1)
    assert np.all(frequencies >= 0)
    assert frequencies[:2].sum() == pytest.approx(marker)
    assert frequencies[[0, 2]].sum() == pytest.approx(disease)
    bound = (
        min(marker * (1 - disease), (1 - marker) * disease)
        if relative >= 0
        else min(marker * disease, (1 - marker) * (1 - disease))
    )
    assert frequencies[0] * frequencies[3] - frequencies[1] * frequencies[2] == pytest.approx(
        relative * bound
    )


def test_legacy_bias_and_tiny_cutoff_are_explicit():
    corrected = tdtasp_genetics([0.25] * 4, [0.8, 0.3, 0.01], 0.1)
    legacy = tdtasp_genetics([0.25] * 4, [0.8, 0.3, 0.01], 0.1, legacy_asp=True)
    assert np.max(abs(corrected.sharing_probability - legacy.sharing_probability)) > 0.01
    tiny = tdtasp_genetics([0.25] * 4, [1e-55] * 3, 0.1, legacy_asp=True)
    assert np.all(tiny.sharing_probability == 0)
    with pytest.raises(ArithmeticError, match="underflow"):
        tdtasp_genetics([0.25] * 4, [1e-200] * 3, 0.1, legacy_asp=True)
    modern = tdtasp_genetics([0.25] * 4, [1e-200] * 3, 0.1)
    informative = modern.father_heterozygous | modern.mother_heterozygous
    np.testing.assert_allclose(modern.sharing_probability[informative], 0.5)


def test_results_own_immutable_arrays():
    frequencies = np.full(4, 0.25)
    result = tdtasp_genetics(frequencies, [0.8, 0.3, 0.01])
    frequencies[0] = 0
    assert result.haplotype_frequencies[0] == 0.25
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)


@pytest.mark.parametrize(
    "options",
    [
        {"haplotype_frequencies": [0.5, 0.5]},
        {"haplotype_frequencies": [0.2] * 4},
        {"haplotype_frequencies": [0.2, 0.3, 0.6, -0.1]},
        {"haplotype_frequencies": [float("nan")] * 4},
        {"penetrance": [1, 0]},
        {"penetrance": [1, 0, -1]},
        {"penetrance": [1, 0, float("inf")]},
        {"recombination": -0.1},
        {"recombination": 0.51},
        {"recombination": float("nan")},
        {"legacy_asp": 1},
    ],
)
def test_invalid_genetic_inputs(options):
    arguments = {"haplotype_frequencies": [0.25] * 4, "penetrance": [0.8, 0.3, 0.01]}
    arguments.update(options)
    with pytest.raises(ValueError):
        tdtasp_genetics(**arguments)


@pytest.mark.parametrize(
    "values", [(-0.1, 0.2, 0.5), (0.1, 1.1, 0.5), (0.1, 0.2, 1.1), (0.1, 0.2, float("nan"))]
)
def test_invalid_population_inputs(values):
    with pytest.raises(ValueError):
        tdtasp_haplotype_frequencies(*values)
