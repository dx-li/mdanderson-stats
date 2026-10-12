"""Request-wide preflight, scenario replay and truthful saved-study reporting."""

from dataclasses import replace

import numpy as np
import pytest
from test_boin_desktop_patients import study
from test_boin_desktop_simulation import source_study

from mdanderson_stats.boin_desktop import BOINDesktopScenario, read_boin_desktop_project
from mdanderson_stats.boin_desktop_batch import simulate_boin_desktop_scenarios
from mdanderson_stats.boin_desktop_report import boin_desktop_simulation_html
from mdanderson_stats.boin_desktop_simulation import simulate_boin_desktop_combination


def example():
    saved, _ = source_study(
        dict(mode="single", truth="gradient", runin="FALSE", titration="TRUE", cap="2")
    )
    return replace(
        saved,
        scenarios=(
            *saved.scenarios,
            BOINDesktopScenario("<unsafe>", ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))),
        ),
    )


def test_batch_seeds_individual_replay_and_weighted_aggregation():
    saved = example()
    batch = simulate_boin_desktop_scenarios(saved, batches=2, trials=3, seed=2**64 - 1)
    assert [r.seed for r in batch.results] == [2**64 - 1, 0, 1, 2]
    for run in batch.results:
        replay = simulate_boin_desktop_combination(
            saved, scenario_index=run.scenario_index, trials=3, seed=run.seed
        )
        np.testing.assert_array_equal(run.patients, replay.patients)
    pooled = batch.aggregate(0)
    assert pooled["trials"] == 6
    np.testing.assert_array_equal(
        pooled["mean_actual_patients"],
        np.concatenate([r.actual_patients for r in batch.results[:2]]).mean(axis=0),
    )
    with pytest.raises(ValueError, match="not present"):
        batch.aggregate(3)


def test_entire_request_rejected_before_first_simulation(monkeypatch):
    import mdanderson_stats.boin_desktop_batch as module

    monkeypatch.setattr(
        module,
        "simulate_boin_desktop_combination",
        lambda *a, **k: pytest.fail("simulation started before preflight"),
    )
    for kwargs in (
        {"scenario_indices": (0, 99)},
        {"batches": 100},
        {"batches": True},
        {"scenario_indices": (0, 0)},
        {"batches": 10, "trials": 10000},
    ):
        with pytest.raises(ValueError):
            simulate_boin_desktop_scenarios(example(), **kwargs)


def test_legacy_report_labels_discrepancies_and_dependence():
    saved = read_boin_desktop_project(
        study().xml.replace("Independent -- example", "&lt;unsafe&gt;")
    )
    batch = simulate_boin_desktop_scenarios(saved, trials=2, seed=9, profile="desktop_legacy")
    report = boin_desktop_simulation_html(batch)
    assert "phantom patients" in report and "not validated uncertainty estimates" in report
    assert "Mean actual DLT" in report and "Trials selecting unobserved" in report
    assert "<unsafe>" not in report and "&lt;unsafe&gt;" in report
    assert read_boin_desktop_project(report) == saved


def test_report_rejects_inconsistent_saved_fields():
    batch = simulate_boin_desktop_scenarios(example(), trials=1)
    with pytest.raises(ValueError, match="retained saved XML"):
        boin_desktop_simulation_html(batch)


def test_legacy_partial_cohort_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="whole-cohort"):
        simulate_boin_desktop_combination(
            replace(example(), maximum_patients=2), profile="desktop_legacy"
        )
