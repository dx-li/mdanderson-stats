"""Source console/return conventions, geometric contour metrics and true denominators."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_boin_desktop_simulation import NativeProbeGenerator, source_study
from test_boin_desktop_univariate_simulation import scenario_study

from mdanderson_stats.boin_desktop import BOINDesktopScenario
from mdanderson_stats.boin_desktop_batch import BOINDesktopBatch, simulate_boin_desktop_scenarios
from mdanderson_stats.boin_desktop_simulation import (
    BOINDesktopSimulation,
    _contour_trials,
    _single_trials,
)
from mdanderson_stats.boin_desktop_summary import summarize_boin_desktop
from mdanderson_stats.boin_desktop_tite import desktop_tite_settings

ROWS = list(
    csv.DictReader(
        (Path(__file__).parents[1] / "research/fixtures/boin-desktop/simulation-summaries.csv")
        .read_text()
        .splitlines()
    )
)


@pytest.mark.parametrize("row", ROWS)
def test_native_complete_wrapper_summaries(row):
    saved, p = source_study(
        {**row, "truth": "gradient" if row["truth"] == "target" else row["truth"]}
    )
    if row["truth"] == "target":
        p = p.copy()
        p[0, 2] = 0.25
        saved = replace(
            saved, scenarios=(BOINDesktopScenario("Target case", tuple(tuple(v) for v in p)),)
        )
    diagnostics = {}
    if row["mode"] == "single":
        values = _single_trials(
            saved,
            p,
            2,
            NativeProbeGenerator(np.random.PCG64(1)),
            "desktop_legacy",
            diagnostics=diagnostics,
        )
    else:
        values = _contour_trials(
            saved, p, 2, NativeProbeGenerator(np.random.PCG64(1)), "desktop_legacy"
        )
    result = BOINDesktopSimulation(
        saved,
        0,
        1,
        "desktop_legacy",
        *values,
        legacy_reporting_denominator=diagnostics.get("reporting_denominator"),
    )
    summary = summarize_boin_desktop(BOINDesktopBatch((result,), 1))
    if row["mode"] == "single":
        for field, actual in [
            ("console_correct", summary.exact_target_selection_probability),
            ("console_at_mtd", summary.native_console_fraction_at_exact_target),
            ("returned_correct", summary.closest_target_selection_probability),
            ("console_toxicity_stop", summary.no_selection_probability),
        ]:
            assert abs(float(row[field]) - 100 * actual) < 0.051
    else:
        assert (
            abs(float(row["contour_correct"]) - 100 * summary.native_contour_selection_probability)
            < 0.051
        )
        if row["contour_at"] == "NA":
            assert summary.native_reported_contour_regions is None
        else:
            expected = [
                float(row[field]) for field in ("contour_at", "contour_above", "contour_below")
            ]
            np.testing.assert_allclose(
                np.asarray(summary.native_reported_contour_regions) * 100, expected, atol=0.051
            )


def test_native_conventional_duration_is_a_distinct_approximation():
    study = scenario_study(True)
    batch = simulate_boin_desktop_scenarios(study, trials=3, batches=2, seed=17)
    summary = summarize_boin_desktop(batch)
    settings = desktop_tite_settings(study)
    expected = np.ceil(summary.mean_reported_patients / study.cohort_size) * (
        (study.cohort_size - 0.5) / settings.accrual_per_month + settings.window_months
    )
    assert summary.conventional_duration_approximation == expected
    assert "not a simulated" in " ".join(summary.qualifications)


def test_empty_batch_has_no_summary():
    with pytest.raises(ValueError, match="not present"):
        summarize_boin_desktop(BOINDesktopBatch((), 0))
