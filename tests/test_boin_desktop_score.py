"""Original ranked tables and independent safety/controller conventions."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from scipy.special import betainc
from test_boin_desktop_patients import study

from mdanderson_stats import (
    boin_desktop_score_table,
    boin_desktop_score_table_html,
    read_boin_desktop_project,
)


def configured(**values):
    root = ET.fromstring(study().xml)
    node = root.find("SimDat/TrialDesign")
    for key, value in values.items():
        child = node.find(key)
        if child is None:
            child = ET.SubElement(node, key)
        child.text = str(value)
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def cases():
    groups = {}
    path = Path(__file__).parents[1] / "research/fixtures/boin-desktop/score-table.csv"
    for row in csv.DictReader(path.read_text().splitlines()):
        groups.setdefault(row["case"], []).append(row)
    return list(groups.values())


@pytest.mark.parametrize("expected", cases())
def test_original_table_global_ranks_and_display_filter(expected):
    r = expected[0]
    saved = configured(
        TargetToxicityProbability=r["target"],
        CohortSize=r["cohort"],
        MaximumSampleSize=r["maximum"],
        AcceptableToxicity_LowerBound=r["lower"],
        AcceptableToxicity_UpperBound=r["upper"],
        UseDefaultBounds="false",
    )
    table = boin_desktop_score_table(saved, profile="desktop_table_legacy")
    visible = [v for v in expected if v["visible"] == "TRUE"]
    assert len(table.rows) == len(visible)
    for actual, native in zip(table.rows, visible, strict=True):
        assert actual.patients == int(native["patients"])
        assert actual.toxicities == int(native["toxicities"].removeprefix(">= "))
        assert actual.eliminated_from == (native["rank"] == "E")
        if native["rank"] != "E":
            assert actual.rank == float(native["rank"])


def test_exact_cutoff_equality_is_only_eliminated_in_native_table():
    saved = configured(EliminateDoseThreshold=1 - 0.25**4)
    safe = boin_desktop_score_table(saved)
    legacy = boin_desktop_score_table(saved, profile="desktop_table_legacy")
    a = next(r for r in safe.rows if r.patients == 3 and r.toxicities == 3)
    b = next(r for r in legacy.rows if r.patients == 3 and r.toxicities == 3)
    assert not a.eliminated_from and b.eliminated_from


def test_safe_ranks_and_raw_probability_are_independent_formulas():
    from mdanderson_stats.boin_desktop_controller import desktop_combination_design

    saved = configured(MaximumSampleSize=12)
    design = desktop_combination_design(saved)
    table = boin_desktop_score_table(saved)
    scores = [r.allocation_score for r in table.rows if not r.eliminated_from]
    for r in table.rows:
        if r.eliminated_from:
            continue
        probability = betainc(
            r.toxicities + 0.5, r.patients - r.toxicities + 0.5, design.deescalation_boundary
        )
        probability -= betainc(
            r.toxicities + 0.5, r.patients - r.toxicities + 0.5, design.escalation_boundary
        )
        assert r.interval_probability == pytest.approx(probability, abs=2e-15)
        assert r.allocation_score == pytest.approx(probability + 0.0005 * r.patients, abs=2e-15)
        assert (
            r.rank
            == sum(s < r.allocation_score for s in scores)
            + (sum(s == r.allocation_score for s in scores) + 1) / 2
        )


@pytest.mark.parametrize("language", ["en", "zh"])
def test_score_report_captures_inputs(language):
    saved = study()
    assert (
        read_boin_desktop_project(boin_desktop_score_table_html(saved, language=language)) == saved
    )


def test_full_ranking_scope_preflight_is_not_bypassed_by_small_display_limit():
    with pytest.raises(ValueError, match="20000"):
        boin_desktop_score_table(configured(MaximumSampleSize=1000, CohortSize=1))
