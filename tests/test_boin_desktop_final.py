"""Original jagged-array serialization and independently saved final-data workflows."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_patients import study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_documents import boin_desktop_combination_selection_html
from mdanderson_stats.boin_desktop_final import (
    boin_desktop_combination_counts,
    select_boin_desktop_combination_counts,
    set_boin_desktop_combination_counts,
)


def native_counts_study():
    root = ET.fromstring(study().xml)
    sim = root.find("SimDat")
    fixture = ET.fromstring(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/combo-counts.xml").read_text()
    )
    for node in fixture:
        sim.append(node)
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


def test_original_serializer_arrays_are_drug_a_rows_and_drug_b_columns():
    saved = native_counts_study()
    n, y = boin_desktop_combination_counts(saved)
    np.testing.assert_array_equal(n, [[3, 6, 9], [12, 15, 18]])
    np.testing.assert_array_equal(y, [[0, 1, 2], [3, 4, 5]])
    assert not n.flags.writeable
    updated = set_boin_desktop_combination_counts(saved, n, y)
    oldroot, newroot = ET.fromstring(saved.xml), ET.fromstring(updated.xml)
    for field in ("ComboTrialData_NEvaluable", "ComboTrialData_NDLT"):
        original = oldroot.find("SimDat/" + field)
        original.tail = None
        assert ET.tostring(original) == ET.tostring(newroot.find("SimDat/" + field))


@pytest.mark.parametrize("profile", ["safe", "desktop_legacy"])
def test_saved_final_counts_selection_and_reopenable_report(profile):
    saved = native_counts_study()
    result = select_boin_desktop_combination_counts(saved, outcomes_complete=True, profile=profile)
    assert result.selection.dose is not None
    html = boin_desktop_combination_selection_html(saved, outcomes_complete=True, profile=profile)
    assert read_boin_desktop_project(html) == saved
    assert "not simultaneous credible bands" in html
    assert not saved.patients
    with pytest.raises(ValueError, match="complete outcomes"):
        select_boin_desktop_combination_counts(saved, outcomes_complete=False)


def test_reset_final_data_preserves_patient_file_and_metadata():
    saved = native_counts_study()
    reset = set_boin_desktop_combination_counts(saved, np.zeros(saved.shape), np.zeros(saved.shape))
    assert not boin_desktop_combination_counts(reset)[0].any()
    assert reset.scenarios == saved.scenarios
    assert (
        select_boin_desktop_combination_counts(reset, outcomes_complete=True).selection.dose is None
    )


def test_ragged_and_invalid_final_counts_are_rejected():
    saved = native_counts_study()
    for n, y in (([[3]], [[0]]), ([[3, 3, 3], [3, 3, 3]], [[0, 0, 0], [0, 0, 4]])):
        with pytest.raises(ValueError):
            set_boin_desktop_combination_counts(saved, n, y)
    with pytest.raises(ValueError, match="match saved grid"):
        boin_desktop_combination_counts(
            read_boin_desktop_project(saved.xml.replace("<int>18</int>", ""))
        )


def test_standalone_contour_report_and_selection():
    saved = native_counts_study()
    saved = read_boin_desktop_project(
        saved.xml.replace("<FindContours>false", "<FindContours>true").replace(
            "</TrialDesign>",
            "<MaximumSampleSizeForSubTrials><int>12</int><int>6</int></MaximumSampleSizeForSubTrials></TrialDesign>",
        )
    )
    result = select_boin_desktop_combination_counts(saved, outcomes_complete=True)
    assert result.selection.contour is not None
    assert (
        read_boin_desktop_project(
            boin_desktop_combination_selection_html(saved, outcomes_complete=True)
        )
        == saved
    )
