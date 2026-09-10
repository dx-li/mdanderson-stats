"""Protocol decision tables must retain the numerical design's exact cutoffs."""

import pytest

from mdanderson_stats import BOINDesign, boin_protocol


@pytest.mark.parametrize("language", ["en", "zh"])
def test_custom_protocol_preserves_cutoffs_and_safety_table(language):
    design = BOINDesign.from_boundaries(0.6, 0.58, 0.65, extra_safe=True, bound_mtd=True)
    report = boin_protocol(design, cohorts=20, language=language)
    assert repr(design.escalation_boundary) in report
    assert repr(design.deescalation_boundary) in report
    # 29/50 is exactly on the requested inclusive escalation cutoff.
    row = next(line for line in report.splitlines() if line.startswith("| 50 |"))
    values = [int(v.strip()) for v in row.strip("|").split("|")]
    assert values[1] == 29
    assert values[2] == 33
    table = design.boundary_table(50)
    assert values[3:] == [table.eliminate_min[-1], table.lowest_stop_min[-1]]
    assert "| 1 | 0 | 1 | — | — |" in report


def test_protocol_options_and_titration_are_explicit():
    design = BOINDesign(0.3, deescalate_at_two_of_six=True, early_stop_patients=12)
    report = boin_protocol(design, titration=True, titration_cap=3)
    assert "exactly 2 DLTs among 6" in report
    assert "at least 12 patients" in report
    assert "up to dose 3" in report
    assert "second grade-2" in report
    assert "last cohort" in report
    assert "| 6 | 1 | 2 | 4 | 4 |" in report
