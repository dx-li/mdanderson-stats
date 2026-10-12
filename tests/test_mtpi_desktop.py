"""Original controller/whole-trial references and independent workflow contracts."""

import csv
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    MTPIDesktopStudy,
    MTPIDesktopTableStudy,
    mtpi_desktop_monitoring_table,
    mtpi_desktop_simulation_report,
    mtpi_desktop_table_report,
    read_mtpi_desktop_project,
    read_mtpi_desktop_workbook,
    replay_mtpi_desktop_trial,
    simulate_mtpi_desktop,
)
from mdanderson_stats.mtpi_desktop_trial import _legacy_step

FIXTURES = Path(__file__).parent / "fixtures"
STUDY = MTPIDesktopStudy(0.3, 0.05, 0.05, 4, 3, 1, ((0.1, 0.3, 0.7),), 20)
DECISIONS = json.loads((FIXTURES / "mtpi-desktop-xll-decisions.json").read_text())
TRIALS = list(csv.DictReader((FIXTURES / "mtpi-desktop-r-trials.csv").read_text().splitlines()))
TABLE = json.loads((FIXTURES / "mtpi-desktop-cached-table.json").read_text())


@pytest.mark.parametrize("case", DECISIONS)
def test_original_xll_controller(case):
    study = replace(STUDY, target=case["target"])
    d, tt, excluded, stop = _legacy_step(study, case["n"], case["y"], case["d"], case["tt"], 3)
    assert (d, tt, excluded, bool(stop)) == (
        case["next"],
        case["newtt"],
        case["toxdose"],
        bool(case["stop"]),
    )


@pytest.mark.parametrize("case", TRIALS)
def test_original_r_complete_trial(case):
    truth = {"ascending": (0.1, 0.3, 0.7), "zero": (0.0, 0.0, 0.0), "one": (1.0, 1.0, 1.0)}[
        case["truth"]
    ]
    study = replace(
        STUDY,
        target=float(case["target"]),
        cohort_size=int(case["cohort"]),
        maximum_patients=int(case["cap"]),
        starting_dose=int(case["start"]),
        scenarios=(truth,),
    )
    tape = {
        "low": np.full(200, 0.01),
        "high": np.full(200, 0.99),
        "cycle": (np.arange(1, 201) * 37 % 101) / 101,
    }[case["tape"]]
    result = replay_mtpi_desktop_trial(study, 0, tape, profile="desktop_legacy")
    for key, actual in [
        ("n", result.patients),
        ("y", result.toxicities),
        ("path", result.dose_path),
    ]:
        assert list(actual) == [int(v) for v in case[key].split(";")]
    assert result.selected == (int(case["selected"]) or None)
    assert result.first_excluded_dose == int(case["toxdose"])
    assert result.next_dose_indicator == int(case["tt"])
    assert (result.stop_reason != "capacity") == bool(int(case["nodose"]))


@pytest.mark.parametrize("profile", ["safe", "desktop_legacy"])
def test_original_cached_monitoring(profile):
    result = mtpi_desktop_monitoring_table(MTPIDesktopTableStudy(), profile=profile)
    for case in TABLE["rows"]:
        assert result.action[case["y"], case["n"] - 1] == case["action"]


def test_capacity_safety_and_native_undefined_top_start():
    study = replace(STUDY, scenarios=((0.0, 0.0, 0.0),))
    safe = replay_mtpi_desktop_trial(study, 0, np.full(4, 0.5))
    legacy = replay_mtpi_desktop_trial(study, 0, np.full(6, 0.5), profile="desktop_legacy")
    assert safe.cohort_sizes == (3, 1)
    assert legacy.cohort_sizes == (3, 3)
    with pytest.raises(ValueError, match="actual capacity"):
        replay_mtpi_desktop_trial(study, 0, np.full(4, 0.5), profile="desktop_legacy")
    top = replace(study, starting_dose=3)
    replay_mtpi_desktop_trial(top, 0, np.full(4, 0.5))
    with pytest.raises(ValueError, match="undefined"):
        simulate_mtpi_desktop(top, profile="desktop_legacy")
    with pytest.raises(ValueError, match="200-patient"):
        simulate_mtpi_desktop(replace(study, maximum_patients=200), profile="desktop_legacy")
    all_toxic = replace(study, scenarios=((1.0, 1.0, 1.0),))
    safe = replay_mtpi_desktop_trial(all_toxic, 0, np.full(4, 0.5))
    assert safe.stop_reason == "stop_safety"
    assert safe.selected is None
    assert safe.patients.sum() == 3
    assert safe.eliminated.all()


@pytest.mark.parametrize("profile", ["safe", "desktop_legacy"])
def test_simulation_matches_independent_tape_replays(profile):
    study = replace(STUDY, scenarios=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.1, 0.3, 0.7)))
    result = simulate_mtpi_desktop(study, profile=profile, seed=17)
    capacity = 6 if profile == "desktop_legacy" else 4
    total = 3 * study.trials * capacity
    tape = (
        (np.random.RandomState(17).randint(0, 2**32, total, dtype=np.uint32).astype(float) + 0.5)
        / 2**32
        if profile == "desktop_legacy"
        else np.random.default_rng(17).random(total)
    )
    cursor = 0
    for i in range(3):
        counts, selections, toxicities = [], np.zeros(4), []
        for _ in range(study.trials):
            trial = replay_mtpi_desktop_trial(
                study, i, tape[cursor : cursor + capacity], profile=profile
            )
            cursor += sum(trial.cohort_sizes)
            counts.append(trial.patients)
            toxicities.append(trial.toxicities.sum())
            selections[(trial.selected or 4) - 1] += 1
        np.testing.assert_array_equal(result.mean_patients[i], np.mean(counts, axis=0))
        np.testing.assert_array_equal(result.selection_probability[i], selections / study.trials)
        assert result.pooled_toxicity_rate[i] == sum(toxicities) / np.sum(counts)
    assert cursor == result.uniforms_consumed
    assert np.allclose(result.stopping_probability.sum(axis=1), 1)
    report = mtpi_desktop_simulation_report(result)
    assert read_mtpi_desktop_project(report) == study
    assert report.count("<h2>Scenario") == 3
    assert "MCSE" in report


def test_saved_input_validation_and_report():
    for study in (STUDY, MTPIDesktopTableStudy(maximum_patients=np.int64(4))):
        assert read_mtpi_desktop_project(study.to_json()) == study
    report = mtpi_desktop_table_report(MTPIDesktopTableStudy(maximum_patients=4))
    assert read_mtpi_desktop_project(report).maximum_patients == 4
    for text in (
        "{}",
        "[]",
        STUDY.to_json().replace('"trials": 20', '"trials": 20, "trials": 21'),
        STUDY.to_json().replace('"target": 0.3', '"target": "bad"'),
    ):
        with pytest.raises(ValueError):
            read_mtpi_desktop_project(text)
    for change in (
        {"trials": True},
        {"scenarios": ((np.nan, 0.1),)},
        {"starting_dose": 4},
        {"epsilon_lower": 0},
    ):
        with pytest.raises(ValueError):
            replace(STUDY, **change)
    with pytest.raises(ValueError, match="million"):
        simulate_mtpi_desktop(replace(STUDY, trials=100000, maximum_patients=200))
    one = simulate_mtpi_desktop(replace(STUDY, trials=1))
    assert np.isnan(one.mean_patients_mcse).all()


def workbook(cells):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
            ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Example" r:id="r1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row>'
            + "".join(f'<c r="{key}">{value}</c>' for key, value in cells.items())
            + "</row></sheetData></worksheet>",
        )
    return out.getvalue()


def test_workbook_numeric_inputs_and_formula_rejection():
    values = dict(A1=4, B1=0.3, C1=0.05, D1=0.05, L1=0)
    cells = {key: f"<v>{value}</v>" for key, value in values.items()}
    assert read_mtpi_desktop_workbook(
        workbook(cells), sheet_name="Example"
    ) == MTPIDesktopTableStudy(maximum_patients=4)
    cells["B1"] = "<f>1/3</f><v>.3</v>"
    with pytest.raises(ValueError, match="non-formula"):
        read_mtpi_desktop_workbook(workbook(cells), sheet_name="Example")


def test_original_native_uniforms():
    expected = json.loads((FIXTURES / "mtpi-desktop-uniforms.json").read_text())
    actual = (
        np.random.RandomState(2).randint(0, 2**32, len(expected), dtype=np.uint32).astype(float)
        + 0.5
    ) / 2**32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    [json.loads(line) for line in (FIXTURES / "mtpi-desktop-rank.jsonl").read_text().splitlines()],
)
def test_original_native_final_rank(case):
    from mdanderson_stats.mtpi_desktop_trial import _desktop_rank

    score = np.array(case["scores"])
    eligible = np.arange(len(score)) >= case["mindose"]
    selected, adjusted = _desktop_rank(score, eligible, case["target"])
    assert selected == case["selected"] + 1
    np.testing.assert_array_equal(adjusted, case["adjusted"])


def test_original_cached_twenty_thousand_trial_workflow():
    data = json.loads((FIXTURES / "mtpi-desktop-cached-simulation.json").read_text())
    study = MTPIDesktopStudy(**data["study"])
    result = simulate_mtpi_desktop(study, profile="desktop_legacy")
    expected = data["expected"]
    np.testing.assert_allclose(
        result.selection_probability, expected["selection"], rtol=0, atol=2e-16
    )
    np.testing.assert_array_equal(result.mean_patients, expected["mean_patients"])
    np.testing.assert_array_equal(result.pooled_toxicity_rate, expected["pooled_toxicity"])
    assert result.uniforms_consumed == 577344
    assert read_mtpi_desktop_project(mtpi_desktop_simulation_report(result)) == study


def test_report_reopen_preserves_execution_settings():
    from mdanderson_stats import read_mtpi_desktop_report

    result = simulate_mtpi_desktop(STUDY, profile="desktop_legacy", seed=19)
    report = mtpi_desktop_simulation_report(result)
    reopened = read_mtpi_desktop_report(report)
    assert (reopened.study, reopened.profile, reopened.seed) == (STUDY, "desktop_legacy", 19)
    rerun = simulate_mtpi_desktop(reopened.study, profile=reopened.profile, seed=reopened.seed)
    np.testing.assert_array_equal(rerun.selection_probability, result.selection_probability)
    table = read_mtpi_desktop_report(
        mtpi_desktop_table_report(MTPIDesktopTableStudy(), profile="desktop_legacy")
    )
    assert table.seed is None and table.profile == "desktop_legacy"
    for bad in (
        report.replace('"seed": 19', '"seed": true'),
        report.replace('"profile": "desktop_legacy"', '"profile": "unknown"'),
        report.replace('id="mtpi-run"', 'id="other"'),
    ):
        with pytest.raises(ValueError):
            read_mtpi_desktop_report(bad)


@pytest.mark.parametrize("bad", [None, "bad", True, float("nan"), float("inf")])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        replace(STUDY, target=bad)


@pytest.mark.parametrize("tape", [[0.5], [float("nan")] * 6, [-0.1] * 6, [1.0] * 6, [[0.5] * 6]])
def test_invalid_tapes(tape):
    with pytest.raises(ValueError):
        replay_mtpi_desktop_trial(STUDY, 0, tape)


def test_spreadsheet_simulation_import_and_missing_sheet():
    values = dict(
        A1=20, B1=0.3, C1=3, D1=4, E1=3, F1=1, G1=1, H1=0.05, I1=0.05, L1=1, B3=0.1, C3=0.3, D3=0.7
    )
    raw = workbook({key: f"<v>{value}</v>" for key, value in values.items()})
    assert read_mtpi_desktop_workbook(raw, sheet_name="Example") == STUDY
    with pytest.raises(ValueError, match="exactly one"):
        read_mtpi_desktop_workbook(raw, sheet_name="Missing")
    for raw in (b"bad", workbook({"L1": "<v>2</v>"})):
        with pytest.raises(ValueError):
            read_mtpi_desktop_workbook(raw, sheet_name="Example")
