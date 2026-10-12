import copy
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from test_efftox_desktop import document
from test_efftox_desktop_controller import decision

from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_batches import (
    efftox_desktop_batches_html,
    simulate_efftox_desktop_scenarios,
    summarize_efftox_desktop_batches,
)
from mdanderson_stats.efftox_desktop_inference import EffToxDesktopInference
from mdanderson_stats.efftox_desktop_simulation import simulate_efftox_desktop


def study():
    root = ET.fromstring(document())
    simulation = root.find("SimulationParams")
    simulation.append(copy.deepcopy(simulation.find("ScenarioParams")))
    ET.SubElement(simulation, "Seed").text = "102"
    ET.SubElement(simulation, "QtyReps").text = "2"
    return read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))


def test_all_saved_scenarios_use_saved_run_settings_and_reproducible_separate_streams():
    saved = study()
    first = simulate_efftox_desktop_scenarios(saved, lambda _: decision(0))
    again = simulate_efftox_desktop_scenarios(saved, lambda _: decision(0))
    assert len(first) == 2
    assert [r.scenario_index for r in first] == [0, 1]
    assert all(len(r.selected_dose_index) == 2 for r in first)
    assert first[0].seed != first[1].seed
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a.counts, b.counts)
    report = efftox_desktop_batches_html(first)
    assert "Included scenario indices: [0, 1]; saved scenarios: 2" in report
    assert all(str(r.seed) in report for r in first)
    assert "Prior-only diagnostics for this batch" in report


def test_all_scenario_preflight_precedes_first_posterior_fit():
    saved = study()
    service = EffToxDesktopInference(
        saved, monotone_toxicity=False, seed=1, draws=8, warmup=0, chains=2, max_work=1500
    )
    # Each scenario requires 7*192=1344 nominal work; the whole request exceeds1500.
    with pytest.raises(ValueError, match="workflow exceeds"):
        simulate_efftox_desktop_scenarios(saved, service)
    assert not service.diagnostics


def test_unequal_batch_sizes_weight_trials_and_selected_trials_separately():
    saved = study()
    first = simulate_efftox_desktop(
        saved, lambda _: decision(0), scenario_index=0, trials=3, seed=1
    )
    second = simulate_efftox_desktop(
        saved, lambda _: decision(1), scenario_index=0, trials=1, seed=2
    )
    first = replace(
        first,
        selected_dose_index=np.array([0, 0, -1]),
        stopped_early=np.array([False, False, True]),
    )
    combined = summarize_efftox_desktop_batches([first, second])
    np.testing.assert_array_equal(combined.selection_probability, [0.5, 0.25])
    np.testing.assert_allclose(combined.selection_mcse, [0.25, np.sqrt(0.25 * 0.75 / 4)])
    np.testing.assert_array_equal(combined.mean_patients_per_dose, [4.5, 1.5])
    assert combined.no_selection_probability == 0.25
    assert combined.early_stop_probability == 0.25
    normalized = (first.true_utility - first.true_utility.min()) / np.ptp(first.true_utility)
    assert combined.rselect == pytest.approx((2 * normalized[0] + normalized[1]) / 3)
    assert combined.batches == (first, second)


def test_aggregation_rejects_reused_streams_mixed_scenarios_and_changed_profiles():
    results = simulate_efftox_desktop_scenarios(study(), lambda _: decision(0))
    with pytest.raises(ValueError, match="distinct"):
        summarize_efftox_desktop_batches([results[0], results[0]])
    with pytest.raises(ValueError, match="same saved study and scenario"):
        summarize_efftox_desktop_batches(results)
    changed = replace(results[0], seed=999, inference_settings=(("draws", "999"),))
    with pytest.raises(ValueError, match="different recorded inference"):
        summarize_efftox_desktop_batches([results[0], changed])


def test_real_inference_batch_report_retains_every_fit_and_settings():
    saved = study()
    service = EffToxDesktopInference(
        saved, monotone_toxicity=False, seed=99, draws=8, warmup=0, chains=2
    )
    results = simulate_efftox_desktop_scenarios(saved, service, trials=1, seed=11)
    report = efftox_desktop_batches_html(results)
    assert "Maximum split R-hat" in report
    assert "toxicity_summary_profile" in report
    assert "No-selection MCSE" in report
    assert sum(len(r.inference_diagnostics) for r in results) == len(service.diagnostics)


@pytest.mark.parametrize("batches", [[], [object()]])
def test_batch_report_requires_typed_nonempty_results(batches):
    with pytest.raises(ValueError, match="between 1 and 100"):
        efftox_desktop_batches_html(batches)
    with pytest.raises(ValueError, match="between 1 and 100"):
        summarize_efftox_desktop_batches(batches)
