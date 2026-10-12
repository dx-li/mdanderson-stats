"""Original compressed tables interpreted at thresholds and nearby valid follow-up."""

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_tite import saved_tite

from mdanderson_stats import BOINDesign, read_boin_desktop_project
from mdanderson_stats.boin_desktop_pending import (
    _rule,
    boin_desktop_pending_rule,
    boin_desktop_pending_table,
)

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/tite-table.csv")
        .read_text()
        .splitlines()
    )
)


def contains(spec, value):
    if spec.startswith(">="):
        return value >= int(spec[2:])
    if spec.startswith("<="):
        return value <= int(spec[2:])
    return value in [int(x) for x in spec.split(",")]


def predicate(spec, followup):
    if spec in ("Yes", "Yes & Eliminate"):
        return True
    if spec in ("No", "Suspend"):
        return False
    for op in (">=", "<=", ">", "<"):
        if op in spec:
            value = float(spec.split(op)[1])
            return {
                ">=": followup >= value,
                "<=": followup <= value,
                ">": followup > value,
                "<": followup < value,
            }[op]
    raise AssertionError(spec)


def expected(row, followup):
    if row["escalate"] == "Suspend":
        return "suspend"
    for column in ("escalate", "stay", "deescalate"):
        if predicate(row[column], followup):
            return "eliminate" if "Eliminate" in row[column] else column
    raise AssertionError(row)


@pytest.mark.parametrize("row", ROWS)
def test_original_pending_table(row):
    n = int(row["patients"])
    design = BOINDesign(
        target=float(row["target"]), toxic_probability=float(row["toxic_probability"])
    )
    for y in range(n + 1):
        if not contains(row["toxicities"], y):
            continue
        for m in range(n - y + 1):
            if not contains(row["pending"], m):
                continue
            rule = _rule(design, float(row["max_pending"]), n, y, m, "desktop_table_legacy")
            values = {0.0, m / 2, float(m)}
            for spec in (row["escalate"], row["stay"], row["deescalate"]):
                for op in (">=", "<=", ">", "<"):
                    if op in spec:
                        threshold = float(spec.split(op)[1])
                        values.update((threshold, threshold - 1e-8, threshold + 1e-8))
                        break
            for value in values:
                if 0 <= value <= m:
                    assert rule.decide(value) == expected(row, value)


def configured(**fields):
    root = ET.fromstring(saved_tite().xml)
    for key, value in fields.items():
        root.find("SimDat/TrialDesign/" + key).text = str(value)
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def test_safe_inclusive_boundary_differs_from_native_strict_table():
    study = configured(AcceptableToxicity_UpperBound=0.75)
    assert boin_desktop_pending_rule(study, 2, 1, 0).decide(0) == "deescalate"
    assert (
        boin_desktop_pending_rule(study, 2, 1, 0, profile="desktop_table_legacy").decide(0)
        == "stay"
    )


def test_table_preflights_before_materializing(monkeypatch):
    import mdanderson_stats.boin_desktop_pending as module

    study = configured(MaximumSampleSize=999, EarlyStop=999)
    monkeypatch.setattr(module, "_rule", lambda *a: pytest.fail("materialized before preflight"))
    with pytest.raises(ValueError, match="20000"):
        boin_desktop_pending_table(study)


def test_table_and_single_state_agree_and_ignore_inactive_simulation_settings():
    study = read_boin_desktop_project(
        saved_tite().xml.replace("<TimeToToxDistribution>eUniform</TimeToToxDistribution>", "")
    )
    table = boin_desktop_pending_table(study)
    assert len(table.rows) > 0
    for row in table.rows:
        assert row == boin_desktop_pending_rule(study, row.patients, row.toxicities, row.pending)
    with pytest.raises(ValueError, match="follow-up"):
        table.rows[0].decide(math.nan)


@pytest.mark.parametrize("n,y,m", [(0, 0, 0), (3, 2, 2), (3.0, 0, 0), (True, 0, 0)])
def test_invalid_counts_rejected(n, y, m):
    with pytest.raises(ValueError):
        boin_desktop_pending_rule(saved_tite(), n, y, m)


def test_safe_rule_uses_unrounded_imputation_threshold():
    study = saved_tite()
    design = BOINDesign(target=study.target)
    rule = boin_desktop_pending_rule(study, 6, 0, 3)
    p = (study.target / 2) / (6 - 3 + 1)
    threshold = 3 - (1 - p) / p * (design.escalation_boundary * 6)
    # This state is unconditional escalation; a different state has a finite boundary.
    assert threshold < 0 and rule.decide(0) == "escalate"
    finite = [r for r in boin_desktop_pending_table(study).rows if r.escalate_at_least is not None]
    assert finite
    for row in finite:
        q = (row.toxicities + study.target / 2) / (row.patients - row.pending + 1)
        boundary = row.escalate_at_least
        augmented = (row.toxicities + q / (1 - q) * (row.pending - boundary)) / row.patients
        assert abs(augmented - design.escalation_boundary) < 1e-14
        assert row.decide(boundary) == "escalate"
        assert row.decide(np.nextafter(boundary, -np.inf)) != "escalate"


def test_original_high_target_false_elimination_and_undefined_extra_stop():
    edge = list(
        csv.DictReader(
            (Path(__file__).parents[1] / "research/fixtures/boin-desktop/tite-table-edges.csv")
            .read_text()
            .splitlines()
        )
    )
    assert edge[0]["decision"] == "Yes & Eliminate" and edge[0]["error"] == ""
    assert "stopneed" in edge[1]["error"]
    study = configured(
        TargetToxicityProbability=0.6,
        AcceptableToxicity_LowerBound=0.36,
        AcceptableToxicity_UpperBound=0.84,
        ExtraSafe="false",
    )
    safe = boin_desktop_pending_rule(study, 3, 3, 0)
    legacy = boin_desktop_pending_rule(study, 3, 3, 0, profile="desktop_table_legacy")
    assert safe.action == "deescalate" and not safe.safety_eliminated
    assert legacy.action == "eliminate" and not legacy.safety_eliminated
    study = read_boin_desktop_project(study.xml.replace("<ExtraSafe>false", "<ExtraSafe>true"))
    assert not boin_desktop_pending_rule(study, 3, 3, 0).lowest_stop
    with pytest.raises(ValueError, match="undefined stopping flag"):
        boin_desktop_pending_table(study, profile="desktop_table_legacy")


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("profile", ["safe", "desktop_table_legacy"])
def test_pending_html_is_reopenable_and_bound_to_inputs(language, profile):
    from mdanderson_stats.boin_desktop_report import boin_desktop_pending_table_html

    study = saved_tite()
    table = boin_desktop_pending_table(study, profile=profile)
    report = boin_desktop_pending_table_html(study, table, language=language)
    assert read_boin_desktop_project(report) == study
    assert "<svg " in report and profile in report
    if profile == "desktop_table_legacy" and language == "en":
        assert "without meeting posterior safety" in report
    with pytest.raises(ValueError, match="does not match"):
        boin_desktop_pending_table_html(configured(MaxPendingOutcomePercent=65), table)
