"""Native compatibility and direct family/count joint-distribution checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import tdtasp_ascertainment, tdtasp_genetics

FIXTURE = json.loads((Path(__file__).parent / "fixtures/tdtasp_ascertainment.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_ascertainment(case):
    genetics = tdtasp_genetics(
        case["haplotype_frequencies"], case["penetrance"], case["recombination"], legacy_asp=True
    )
    result = tdtasp_ascertainment(
        genetics,
        case["mean_offspring"],
        **{
            key: case[key]
            for key in ("test", "sampling", "eligibility", "minimum_affected", "all_affected")
        },
        legacy_moments=True,
    )
    summary, families = np.array(case["summary"]), np.array(case["families"])
    assert result.population_average_truncated_mean == pytest.approx(summary[4], abs=2e-14)
    assert result.selection_probability == pytest.approx(summary[0], abs=2e-14)
    assert result.answer_probability == pytest.approx(summary[1], abs=2e-14)
    assert result.expected_heterozygous_parents == pytest.approx(summary[2] + summary[3], abs=2e-14)
    np.testing.assert_allclose(result.family_probability, families[:, 0], atol=1e-15)
    np.testing.assert_allclose(result.mean_affected_by_family, families[:, 1], atol=2e-14)
    np.testing.assert_allclose(result.contributions_by_family, families[:, 2], atol=2e-14)


def joint_reference(genetics, mean, k, sampling, eligibility, all_affected, test):
    counts = np.arange(81)
    mu = mean * genetics.affected_probability
    pmf = np.zeros((256, 81))
    pmf[:, 0] = np.exp(-mu)
    for j in range(1, 81):
        pmf[:, j] = pmf[:, j - 1] * mu / j
    joint = genetics.parent_probability[:, None] * pmf
    if sampling == "individual":
        joint *= counts
    base = joint[:, 1:].sum()
    joint[:, :k] = 0
    list_mass = joint.sum()
    father, mother = genetics.father_heterozygous, genetics.mother_heterozygous
    eligible = {"father": father, "one": father | mother, "both": father & mother}[eligibility]
    joint[~eligible] = 0
    eligible_mass = joint.sum()
    joint /= eligible_mass
    heterozygous = father.astype(int) + mother
    units = heterozygous[:, None] * np.ones((1, 81))
    if test == "tdt" and all_affected:
        units *= counts
    answer = genetics.transmission_probability if test == "tdt" else genetics.sharing_probability
    return (
        joint.sum(axis=1),
        eligible_mass / list_mass,
        float((joint * counts).sum()),
        float((joint * units).sum()),
        float((joint * units * answer[:, None]).sum() / (joint * units).sum()),
        float(np.log(base / list_mass)),
    )


@pytest.mark.parametrize("sampling", ["family", "individual"])
@pytest.mark.parametrize("eligibility", ["father", "one", "both"])
@pytest.mark.parametrize("k", [1, 2, 4, 10])
@pytest.mark.parametrize("all_affected", [False, True])
def test_direct_poisson_joint_distribution(sampling, eligibility, k, all_affected):
    genetics = tdtasp_genetics([0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1)
    result = tdtasp_ascertainment(
        genetics,
        4,
        sampling=sampling,
        eligibility=eligibility,
        minimum_affected=k,
        all_affected=all_affected,
    )
    expected = joint_reference(genetics, 4, k, sampling, eligibility, all_affected, "tdt")
    np.testing.assert_allclose(result.family_probability, expected[0], atol=3e-15)
    np.testing.assert_allclose(
        [
            result.selection_probability,
            result.expected_affected,
            result.expected_contributions,
            result.answer_probability,
            result.log_screening_for_minimum,
        ],
        expected[1:],
        atol=3e-14,
    )


@pytest.mark.parametrize("sampling", ["family", "individual"])
@pytest.mark.parametrize("eligibility", ["father", "one", "both"])
def test_asp_joint_distribution(sampling, eligibility):
    genetics = tdtasp_genetics([0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1)
    result = tdtasp_ascertainment(
        genetics, 4, test="asp", sampling=sampling, eligibility=eligibility
    )
    expected = joint_reference(genetics, 4, 2, sampling, eligibility, False, "asp")
    assert result.answer_probability == pytest.approx(expected[4], abs=2e-14)
    assert result.expected_affected == pytest.approx(expected[2], abs=2e-14)


def test_individual_sampling_changes_within_family_mean():
    genetics = tdtasp_genetics([0.25] * 4, [0.2] * 3)
    corrected = tdtasp_ascertainment(genetics, 2, sampling="individual", all_affected=True)
    legacy = tdtasp_ascertainment(
        genetics, 2, sampling="individual", all_affected=True, legacy_moments=True
    )
    assert corrected.expected_affected == pytest.approx(1.4)
    assert legacy.expected_affected == pytest.approx(0.4 / -np.expm1(-0.4))
    assert corrected.expected_contributions > legacy.expected_contributions
    np.testing.assert_allclose(corrected.family_probability, legacy.family_probability)


@pytest.mark.parametrize("sampling", ["family", "individual"])
def test_rare_ascertainment_stays_in_log_space(sampling):
    genetics = tdtasp_genetics([0.25] * 4, [1e-100] * 3)
    result = tdtasp_ascertainment(
        genetics, 1, sampling=sampling, minimum_affected=10, all_affected=True
    )
    assert result.log_list_mass < -2000
    assert result.log_screening_for_minimum > 2000
    assert result.expected_affected == pytest.approx(10)
    assert result.selection_probability == pytest.approx(0.75)
    assert result.answer_probability == pytest.approx(0.5)
    assert result.family_probability.sum() == pytest.approx(1)


def test_impossible_models_and_immutable_result():
    for frequencies, penetrance in [([1, 0, 0, 0], [0.8, 0.3, 0.1]), ([0.25] * 4, [0, 0, 0])]:
        with pytest.raises(ValueError, match="no eligible families"):
            tdtasp_ascertainment(tdtasp_genetics(frequencies, penetrance), 2)
    result = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.8, 0.3, 0.1]), 2)
    for array in (
        result.family_probability,
        result.mean_affected_by_family,
        result.contributions_by_family,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize(
    "options",
    [
        {"mean_offspring": 0},
        {"mean_offspring": 11},
        {"mean_offspring": float("nan")},
        {"test": "wrong"},
        {"sampling": "wrong"},
        {"eligibility": "wrong"},
        {"minimum_affected": 0},
        {"minimum_affected": 11},
        {"minimum_affected": True},
        {"minimum_affected": 1.5},
        {"all_affected": 1},
        {"legacy_moments": 1},
        {"test": "asp", "minimum_affected": 1},
        {"test": "asp", "all_affected": True},
    ],
)
def test_invalid_selection_inputs(options):
    arguments = {"genetics": tdtasp_genetics([0.25] * 4, [0.8, 0.3, 0.1]), "mean_offspring": 2}
    arguments.update(options)
    with pytest.raises(ValueError):
        tdtasp_ascertainment(**arguments)


@pytest.mark.parametrize("mean", [0.1, 10])
@pytest.mark.parametrize("k", [1, 10])
def test_offspring_mean_limits(mean, k):
    genetics = tdtasp_genetics([0.25] * 4, [1] * 3)
    result = tdtasp_ascertainment(
        genetics, mean, sampling="individual", minimum_affected=k, all_affected=True
    )
    expected = joint_reference(genetics, mean, k, "individual", "one", True, "tdt")
    assert result.expected_affected == pytest.approx(expected[2], abs=2e-14)
    assert result.log_screening_for_minimum == pytest.approx(expected[5], abs=2e-14)


@pytest.mark.parametrize("sampling", ["family", "individual"])
def test_recessive_families_with_zero_affected_probability(sampling):
    genetics = tdtasp_genetics([0.25] * 4, [1, 0, 0])
    result = tdtasp_ascertainment(genetics, 2, sampling=sampling, minimum_affected=2)
    expected = joint_reference(genetics, 2, 2, sampling, "one", False, "tdt")
    np.testing.assert_allclose(result.family_probability, expected[0], atol=2e-15)
    assert np.all(result.family_probability[genetics.affected_probability == 0] == 0)
