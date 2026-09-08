"""Study integration across real genetic, selection and statistical components."""

import itertools

import pytest

from mdanderson_stats import (
    format_tdtasp_study,
    tdtasp_ascertainment,
    tdtasp_fixed_sample_size,
    tdtasp_genetics,
    tdtasp_power,
    tdtasp_study,
)

FREQUENCIES = [0.3, 0.2, 0.1, 0.4]
PENETRANCE = [0.8, 0.5, 0.2]


@pytest.mark.parametrize(
    "test,sampling,eligibility,sides,legacy",
    list(
        itertools.product(
            ["tdt", "asp"],
            ["family", "individual"],
            ["father", "one", "both"],
            [1, 2],
            [False, True],
        )
    ),
)
def test_forward_study_matches_components(test, sampling, eligibility, sides, legacy):
    options = dict(
        test=test,
        sampling=sampling,
        eligibility=eligibility,
        minimum_affected=2,
        all_affected=test == "tdt",
        legacy_moments=legacy,
    )
    study = tdtasp_study(
        FREQUENCIES,
        PENETRANCE,
        2,
        0.1,
        families=50,
        alpha=0.1,
        sides=sides,
        legacy_asp=legacy,
        legacy_scale=legacy,
        legacy_two_sided=legacy,
        **options,
    )
    selection = tdtasp_ascertainment(
        tdtasp_genetics(FREQUENCIES, PENETRANCE, 0.1, legacy_asp=legacy), 2, **options
    )
    expected = tdtasp_power(
        selection, 50, 0.1, sides=sides, legacy_scale=legacy, legacy_two_sided=legacy
    )
    assert study.design.power == expected.power
    assert study.design.actual_size == expected.actual_size
    assert study.design.contribution_per_family == expected.contribution_per_family
    assert study.search is None
    assert study.fixed_design is None
    report = format_tdtasp_study(study)
    assert f"TDTASP {test.upper()} study" in report
    assert f"legacy_asp={legacy}" in report
    assert "Families screened from qualifying list: 50" in report
    assert "Fixed-observation comparison" not in report


def test_sample_size_study_matches_first_qualifying_designs(tmp_path):
    study = tdtasp_study(
        FREQUENCIES, PENETRANCE, 2, 0.1, target_power=0.8, max_families=1000, max_observations=2000
    )
    assert study.design.families == 527
    assert study.design.power == pytest.approx(0.8004379498165892)
    assert study.search is not None and study.search.design is study.design
    assert study.fixed_design is not None
    fixed = tdtasp_fixed_sample_size(study.design.ascertainment.answer_probability)
    assert study.fixed_design.observations == fixed.observations
    assert study.fixed_design.power == fixed.power
    assert tdtasp_power(study.design.ascertainment, 526).power < 0.8
    report = format_tdtasp_study(study, digits=17)
    assert "Target power: 0.80000000000000004" in report
    assert "Family search bounds (inclusive): 1, 1000" in report
    assert f"Fixed-observation comparison: {int(fixed.observations)} observations" in report
    path = tmp_path / "study.txt"
    path.write_text(report)
    assert path.read_text() == report


def test_searches_honor_separate_inclusive_bounds():
    study = tdtasp_study(
        FREQUENCIES,
        PENETRANCE,
        2,
        0.1,
        target_power=0.2,
        min_families=600,
        max_families=600,
        min_observations=1000,
        max_observations=1000,
    )
    assert study.design.families == 600
    assert study.fixed_design.observations == 1000


@pytest.mark.parametrize("options", [{}, {"families": 10, "target_power": 0.8}])
def test_ambiguous_mode_rejected(options):
    with pytest.raises(ValueError, match="exactly one"):
        tdtasp_study(FREQUENCIES, PENETRANCE, 2, **options)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"target_power": 0.8, "max_families": 1}, "family bounds"),
        ({"target_power": 0.8, "max_observations": 1}, "observation bounds"),
        ({"families": 10, "max_terms": 2}, "max_terms"),
        ({"families": -1}, "families"),
        ({"families": 10, "test": "invalid"}, "test"),
    ],
)
def test_errors_propagate_without_partial_result(options, match):
    with pytest.raises(ValueError, match=match):
        tdtasp_study(FREQUENCIES, PENETRANCE, 2, 0.1, **options)


def test_zero_families_is_forward_mode_and_no_input_mutation():
    frequencies = FREQUENCIES.copy()
    penetrance = PENETRANCE.copy()
    study = tdtasp_study(frequencies, penetrance, 2, families=0)
    frequencies[0] = 0
    penetrance[0] = 0
    assert study.design.power == study.design.actual_size == 0
    assert study.design.ascertainment.genetics.haplotype_frequencies[0] == 0.3
    assert study.design.ascertainment.genetics.penetrance[0] == 0.8


@pytest.mark.parametrize("digits", [0, 18, 1.5, True, "8"])
def test_invalid_report_precision(digits):
    study = tdtasp_study(FREQUENCIES, PENETRANCE, 2, families=0)
    with pytest.raises(ValueError, match="digits"):
        format_tdtasp_study(study, digits=digits)


def test_report_requires_study():
    with pytest.raises(TypeError, match="TDTASPStudy"):
        format_tdtasp_study(None)


def test_report_preserves_population_and_separate_parent_summaries():
    study = tdtasp_study(
        FREQUENCIES, PENETRANCE, 2, 0.1, families=50, eligibility="father", sampling="individual"
    )
    report = format_tdtasp_study(study, digits=17)
    values = dict(line.split(": ", 1) for line in report.splitlines() if ": " in line)
    assert float(values["Marker allele A frequency"]) == 0.5
    assert float(values["Disease allele D frequency"]) == 0.4
    assert float(values["Linkage disequilibrium D"]) == pytest.approx(0.1)
    father = float(values["Father heterozygosity probability among eligible families"])
    mother = float(values["Mother heterozygosity probability among eligible families"])
    assert father == pytest.approx(1)
    assert father + mother == pytest.approx(
        study.design.ascertainment.expected_heterozygous_parents
    )
    assert "repeated selection of a family is negligible" in report
