"""Native template interpretation, strict parsing and complete file-to-study flows."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    TDTASPTemplate,
    format_tdtasp_study,
    format_tdtasp_template,
    parse_tdtasp_template,
    tdtasp_study,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/tdtasp_template.json").read_text())
BASE = FIXTURE["cases"][0]["template"]


def change(text, **updates):
    lines = []
    for line in text.splitlines():
        name = line.split("=", 1)[0].strip()
        lines.append(f"{name} = {updates[name]}" if name in updates else line)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_original_reader_interpretation(case):
    s = parse_tdtasp_template(case["template"])
    p = dict(zip(FIXTURE["parameter_order"], case["parameters"], strict=True))
    np.testing.assert_allclose(
        s.haplotype_frequencies, [p[k] for k in ("AD", "Ad", "BD", "Bd")], atol=1e-15
    )
    np.testing.assert_allclose(s.penetrance, [p[k] for k in ("DD", "Dd", "dd")], atol=1e-15)
    assert s.recombination == p["theta"]
    assert s.mean_offspring == p["mean_offspring"]
    assert s.alpha == p["alpha"]
    assert s.minimum_affected == p["minimum_affected"]
    assert s.test == ("tdt" if p["test"] == 1 else "asp")
    assert s.sampling == ("family" if p["sampling"] == 2 else "individual")
    assert s.eligibility == {0: "father", 1: "one", 2: "both"}[p["eligibility"]]
    if s.test == "tdt":
        assert s.sides == p["sides"]
        assert s.all_affected == (p["all_affected"] == 1)
    else:
        assert s.sides == 1 and not s.all_affected
    if p["calculation"] == 1:
        assert s.target_power == p["power"] and s.families is None
    else:
        assert s.families == p["families"] and s.target_power is None
    restored = parse_tdtasp_template(format_tdtasp_template(s))
    assert restored == s


def test_manual_file_through_report(tmp_path):
    path = tmp_path / "problem.in"
    path.write_text(BASE)
    spec = parse_tdtasp_template(path.read_text())
    assert spec.haplotype_frequencies == pytest.approx((0.6, 0.15, 0, 0.25))
    study = spec.run(max_families=10000)
    expected = tdtasp_study(
        spec.haplotype_frequencies,
        (0.6, 0.45, 0.3),
        3,
        sampling="individual",
        all_affected=True,
        target_power=0.8,
        max_families=10000,
    )
    assert study.design.families == expected.design.families
    assert study.design.power == expected.design.power
    assert study.fixed_design.observations == expected.fixed_design.observations
    path.with_suffix(".out").write_text(format_tdtasp_study(study))
    assert "Target power:" in path.with_suffix(".out").read_text()


def test_run_compatibility_options_and_budgets():
    spec = parse_tdtasp_template(change(BASE, which_calc="p", sample_size="200", sided="2"))
    study = spec.run(legacy_asp=True, legacy_moments=True, legacy_scale=True, legacy_two_sided=True)
    a = study.design.ascertainment
    assert a.genetics.legacy_asp and a.legacy_moments
    assert study.design.legacy_scale and study.design.conditional.legacy_two_sided
    with pytest.raises(ValueError, match="max_terms"):
        spec.run(max_terms=2)
    search = parse_tdtasp_template(BASE)
    with pytest.raises(ValueError, match="family bounds"):
        search.run(max_families=1)
    with pytest.raises(ValueError, match="observation bounds"):
        search.run(max_observations=1)


def test_blank_form_and_active_placeholders():
    blank = format_tdtasp_template()
    assert len([line for line in blank.splitlines() if line and not line.startswith("#")]) == 17
    assert "mininum_n_affected =" in blank
    with pytest.raises(ValueError, match="tdt_or_asp"):
        parse_tdtasp_template(blank)


def test_comments_multiline_numbers_and_named_order():
    text = change(
        BASE,
        penetrances="(dd=.3,\n # discarded line\n Dd=.45, DD=6D-1***)",
        prevalence="(m=.75, p=.6)",
        n_offspring="3e0***",
    )
    text += "theta = 0.3 # whole line ignored, including duplicate field\n"
    assert parse_tdtasp_template(text) == parse_tdtasp_template(BASE)
    with pytest.raises(ValueError, match="theta"):
        parse_tdtasp_template(BASE.replace("theta = 0.0000", "theta = 0.0000 # ignored"))


def test_positional_lists_case_insensitive_fields_and_prefixes():
    text = change(
        BASE,
        penetrances="(.6,.45,.3)",
        prevalence="(.6,.75)",
        tdt_or_asp="T",
        family_or_individual="IN",
        how_enter_popn_freq="DISE",
        parent_hetero_requirement="O",
        include_all_affected="Y",
    )
    text = text.replace("penetrances", "PENETRANCES").replace(
        "mininum_n_affected", "minimum_n_affected"
    )
    assert parse_tdtasp_template(text) == parse_tdtasp_template(BASE)
    with pytest.raises(ValueError, match="how_enter_popn_freq"):
        parse_tdtasp_template(change(BASE, how_enter_popn_freq="d"))


def test_inactive_fields_may_be_omitted_or_unfilled():
    asp = change(
        BASE,
        tdt_or_asp="asp",
        include_all_affected="unfilled",
        mininum_n_affected="invalid",
        sided="invalid",
    )
    spec = parse_tdtasp_template(asp)
    assert spec.minimum_affected == 2 and not spec.all_affected and spec.sides == 1
    omitted = "\n".join(
        line
        for line in asp.splitlines()
        if line.split("=", 1)[0].strip()
        not in {
            "include_all_affected",
            "mininum_n_affected",
            "sided",
            "sample_size",
            "popn_haplotype_frequencies",
        }
    )
    assert parse_tdtasp_template(omitted) == spec
    one_child = parse_tdtasp_template(
        change(BASE, include_all_affected="no", mininum_n_affected="***")
    )
    assert one_child.minimum_affected == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_size", "2.00000000000000001"),
        ("sample_size", "2.1"),
        ("sample_size", "9007199254740992"),
        ("sample_size", "-1"),
        ("sample_size", "1e309"),
        ("sample_size", "nan"),
        ("mininum_n_affected", "1.9999999999999999999"),
        ("sided", "1.1"),
        ("theta", ".6"),
        ("n_offspring", "0"),
        ("significance", "0"),
        ("penetrances", "(DD=.6,Dd=.45,DD=.3)"),
        ("penetrances", "(DD=.6,.45,dd=.3)"),
        ("penetrances", "(.6,.45)"),
        ("penetrances", "(DD=.6,DD=.45,DD=.3)"),
        ("theta", "0.1 trailing"),
        ("theta", "1*2"),
        ("which_calc", "unknown"),
        ("parent_hetero_requirement", "neither"),
    ],
)
def test_invalid_active_values(field, value):
    text = change(BASE, which_calc="p", sample_size="200")
    with pytest.raises(ValueError):
        parse_tdtasp_template(change(text, **{field: value}))


@pytest.mark.parametrize(
    "suffix",
    [
        "theta = .2",
        "minimum_n_affected = 1",
        "typo = 2",
        "not an assignment",
        "prevalence = ((.2,.3))",
        "theta = )",
        "prevalence = (p=.2,",
    ],
)
def test_invalid_structure(suffix):
    with pytest.raises(ValueError):
        parse_tdtasp_template(BASE + "\n" + suffix)


def test_negative_disequilibrium_and_zero_families_supported():
    spec = parse_tdtasp_template(
        change(BASE, relative_disequilibrium="-1", which_calc="p", sample_size="0")
    )
    assert spec.families == 0
    assert spec.run().design.power == 0


def test_constructor_owns_inputs_and_roundtrip_precision():
    frequencies = [0.3123456789012345, 0.1876543210987655, 0.1, 0.4]
    spec = TDTASPTemplate(frequencies, [0.8, 0.5, 0.2], 2, families=200)
    frequencies[0] = 0
    assert spec.haplotype_frequencies[0] == 0.3123456789012345
    assert parse_tdtasp_template(format_tdtasp_template(spec)) == spec
    with pytest.raises(FrozenInstanceError):
        spec.families = 0


@pytest.mark.parametrize(
    "updates",
    [
        {"test": "asp", "minimum_affected": 2, "all_affected": False, "sides": 2},
        {"minimum_affected": 2, "all_affected": False},
        {"families": 2.0, "target_power": None},
        {"families": True, "target_power": None},
        {"target_power": None},
        {"families": 2},
        {"target_power": 0},
        {"target_power": float("nan")},
    ],
)
def test_constructor_rejects_invalid_or_unrepresentable_inputs(updates):
    spec = parse_tdtasp_template(BASE)
    with pytest.raises(ValueError):
        replace(spec, **updates)


def test_type_errors():
    with pytest.raises(TypeError, match="text"):
        parse_tdtasp_template(None)
    with pytest.raises(TypeError, match="TDTASPTemplate"):
        format_tdtasp_template({})
