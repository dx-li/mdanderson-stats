"""Original single/TITE final selectors and saved row-major data workflows."""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_patients import study

from mdanderson_stats.boin import BOINDesign
from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_single import (
    boin_desktop_single_counts,
    select_boin_desktop_single,
    set_boin_desktop_single_counts,
)

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/single-selection.csv")
        .read_text()
        .splitlines()
    )
)


@pytest.mark.parametrize("row", ROWS)
def test_original_final_selectors(row):
    design = BOINDesign(target=float(row["target"]), extra_safe=row["extra_safe"] == "TRUE")
    n = np.fromstring(row["patients"], sep=";", dtype=int)
    y = np.fromstring(row["toxicities"], sep=";", dtype=int)
    result = design.select_mtd(n, y)
    assert (result.dose or 99) == int(row["selected"])


def single_study(tite=False):
    root = ET.fromstring(study().xml)
    design = root.find("SimDat/TrialDesign")
    for field, value in {
        "SingleDrugTrial": "true",
        "TITE_BOINTrial": str(tite).lower(),
        "NumberOfDoses_DrugB": "1",
    }.items():
        design.find(field).text = value
    probabilities = root.find("SimDat/Scenarios/Scenario/TrueProbabilities")
    for child in list(probabilities)[2:]:
        probabilities.remove(child)
    return read_boin_desktop_project(ET.tostring(root, encoding="unicode"))


@pytest.mark.parametrize("tite", [False, True])
def test_saved_final_count_workflow(tite):
    saved = set_boin_desktop_single_counts(single_study(tite), [3, 6], [0, 2])
    n, y = boin_desktop_single_counts(saved)
    np.testing.assert_array_equal(n, [3, 6])
    np.testing.assert_array_equal(y, [0, 2])
    assert not n.flags.writeable
    assert read_boin_desktop_project(saved.to_html()) == saved
    flat = [int(v.text) for v in ET.fromstring(saved.xml).findall("SimDat/TrialData/int")]
    assert flat == [3, 6, 0, 2]
    assert select_boin_desktop_single(saved, outcomes_complete=True).dose == 2
    for incomplete in (False, None, 1):
        with pytest.raises(ValueError, match="complete-outcome"):
            select_boin_desktop_single(saved, outcomes_complete=incomplete)


def test_invalid_count_shape_and_values_rejected():
    saved = single_study()
    for n, y in (([3], [0]), ([3, 3], [0, 4]), ([501, 501], [0, 0])):
        with pytest.raises(ValueError):
            set_boin_desktop_single_counts(saved, n, y)
    saved = set_boin_desktop_single_counts(saved, [3, 6], [0, 2])
    with pytest.raises(ValueError, match="integers"):
        boin_desktop_single_counts(
            read_boin_desktop_project(saved.xml.replace("<int>3", "<int>3.2"))
        )


def test_native_default_bound_flag_ignores_inactive_saved_values():
    from mdanderson_stats.boin_desktop_controller import desktop_combination_design
    from mdanderson_stats.boin_desktop_single import desktop_single_design

    for saved, resolve in (
        (study(), desktop_combination_design),
        (single_study(), desktop_single_design),
    ):
        xml = saved.xml.replace(
            "<AcceptableToxicity_LowerBound>.15", "<AcceptableToxicity_LowerBound>.01"
        ).replace("<AcceptableToxicity_UpperBound>.35", "<AcceptableToxicity_UpperBound>.99")
        explicit = resolve(read_boin_desktop_project(xml))
        default = resolve(
            read_boin_desktop_project(
                xml.replace("<UseDefaultBounds>false", "<UseDefaultBounds>true")
            )
        )
        assert (explicit.safe_probability, explicit.toxic_probability) == (0.01, 0.99)
        assert (default.safe_probability, default.toxic_probability) == (0.15, 0.35)
        with pytest.raises(ValueError, match="UseDefaultBounds"):
            resolve(
                read_boin_desktop_project(
                    xml.replace("<UseDefaultBounds>false</UseDefaultBounds>", "")
                )
            )
