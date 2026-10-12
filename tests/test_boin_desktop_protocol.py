"""Bilingual methods distinguish saved conduct from simulation overrides."""

import pytest
from test_boin_desktop_patients import study

from mdanderson_stats.boin_desktop import read_boin_desktop_project
from mdanderson_stats.boin_desktop_protocol import boin_desktop_combination_protocol


@pytest.mark.parametrize("language", ["en", "zh"])
def test_methods_include_coherent_integer_table_and_capacity(language):
    saved = study()
    text = boin_desktop_combination_protocol(saved, language=language)
    assert "3 | 0 | 2 | 3" in text
    assert "12" in text and "0.25" in text
    assert "--- | --- | --- | ---\n1 |" in text
    if language == "en":
        assert "does not reproduce the simulation titration prelude" in text
        assert "not simultaneous" in text


def test_contour_lists_executed_capacities_and_distinct_runin_table():
    saved = read_boin_desktop_project(
        study()
        .xml.replace("<FindContours>false", "<FindContours>true")
        .replace(
            "</TrialDesign>",
            "<MaximumSampleSizeForSubTrials><int>12</int><int>6</int></MaximumSampleSizeForSubTrials></TrialDesign>",
        )
    )
    text = boin_desktop_combination_protocol(saved)
    assert "(12, 6)" in text and "at most 18" in text
    assert "two DLTs among six" in text
    with pytest.raises(ValueError, match="language"):
        boin_desktop_combination_protocol(saved, language="fr")
