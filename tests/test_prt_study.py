"""Saved input validation and independent native patient-count contracts."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.prt_study import (
    PRTPatient,
    PRTStudy,
    import_prt_conduct,
    import_prt_simulation,
    prt_study_counts,
    read_prt_study,
)

SETTINGS = "3\n2\n.3\n.05\n.3\n.9\n-14\n28\n"
SIMULATION = """Trial Inputs
3
2
7
90
30 60 90
Design Inputs
.3
.05
.3
.9
-14
28
MCMC Inputs
1000
2000
Simulation Inputs
.2 .4
7.5
5 1
30
"""


def test_conduct_import_ignores_seed_lines_only_when_acknowledged():
    with pytest.raises(ValueError, match="acknowledge"):
        import_prt_conduct(SETTINGS + "12345\n34568", "")
    study = import_prt_conduct(
        SETTINGS + "12345\n34568", "1 0 1\n2 2 0", allow_unused_seed_lines=True
    )
    assert study.source_notes and study.patients == (
        PRTPatient(0, 0, True),
        PRTPatient(1, 2, False),
    )
    assert read_prt_study(study.to_json()) == study


def test_simulation_roundtrip_and_partial_final_cohort():
    study = import_prt_simulation(SIMULATION)
    assert study.max_patients == 7 and study.repetitions == 30
    assert study.interval_endpoints == (30, 60, 90)
    assert study.hazard_alpha == 5 and study.hazard_beta == 1
    assert read_prt_study(study.to_json()) == study


CASES = json.loads((Path(__file__).parent / "fixtures/prt-desktop-counts.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_patient_counts_against_native(case):
    study = PRTStudy(
        case["intervals"],
        case["doses"],
        patients=tuple(PRTPatient(d - 1, c, bool(z)) for d, c, z in case["rows"]),
    )
    s, e, a, assessed, pending = prt_study_counts(study)
    np.testing.assert_array_equal(a, case["result"]["assigned"])
    np.testing.assert_array_equal(assessed, case["result"]["assessed"])
    np.testing.assert_array_equal(e.sum(0), case["result"]["events"])
    np.testing.assert_array_equal(assessed + pending.sum(0), a)
    assert np.all(s[1:] + e[1:] <= s[:-1])
    for x in (s, e, a, assessed, pending):
        assert not x.flags.writeable


@pytest.mark.parametrize("bad", ["1 0 0", "0 1 1", "3 1 1", "1 4 1", "1 1 2", "1 1", "1 nan 1"])
def test_bad_patient_records(bad):
    with pytest.raises(ValueError):
        import_prt_conduct(SETTINGS, bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intervals": 2},
        {"doses": True},
        {"prior_variance": 0},
        {"target": float("nan")},
        {"epsilon": 2},
        {"draws": 1},
        {"chains": 1},
        {"patients": (PRTPatient(2, 1, True),)},
        {"max_patients": 7},
        {"true_toxicity": (0.2, 0.4)},
    ],
)
def test_invalid_study(kwargs):
    with pytest.raises(ValueError):
        replace(PRTStudy(3, 2), **kwargs)


def test_unknown_duplicate_and_version_fields():
    text = PRTStudy(3, 2).to_json()
    for bad in [
        text.replace('"version": 1', '"version": 2'),
        text.replace('"version": 1', '"version": 1, "version": 1'),
        text.replace('"version": 1', '"version": 1, "extra": 1'),
    ]:
        with pytest.raises(ValueError):
            read_prt_study(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("doses", 10**1000),
        ("prior_mean", 10**1000),
        ("source_notes", "note"),
        ("patients", None),
        ("interval_endpoints", None),
    ],
)
def test_malformed_values_fail_cleanly(field, value):
    with pytest.raises(ValueError):
        replace(PRTStudy(3, 2), **{field: value})


def test_boolean_schema_version_rejected():
    with pytest.raises(ValueError):
        read_prt_study(PRTStudy(3, 2).to_json().replace('"version": 1', '"version": true'))
