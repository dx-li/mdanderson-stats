import json
from pathlib import Path

import numpy as np
import pytest
from test_efftox_desktop import document
from test_efftox_desktop_controller import decision

from mdanderson_stats.efftox_desktop import read_efftox_desktop_xml
from mdanderson_stats.efftox_desktop_controller import recommend_efftox_desktop
from mdanderson_stats.efftox_desktop_inference import EffToxDesktopInference
from mdanderson_stats.efftox_desktop_simulation import _rselect, simulate_efftox_desktop


def test_real_posterior_service_reproduces_count_keyed_draws_and_preserves_diagnostics():
    study = read_efftox_desktop_xml(document())
    service = EffToxDesktopInference(
        study, monotone_toxicity=True, seed=91, draws=32, warmup=16, chains=2
    )
    counts = study.observed_counts()
    first = service(counts)
    service(np.zeros((2, 2, 2), dtype=np.int64))
    again = service(counts)
    assert first.dose_index == again.dose_index
    np.testing.assert_array_equal(first.utility, again.utility)
    assert service.diagnostics[0] == service.diagnostics[2]
    assert service.diagnostics[0].likelihood_evaluations > 0
    assert np.isfinite(service.diagnostics[0].max_split_rhat)
    result = recommend_efftox_desktop(study, service)
    assert result.analyses >= 2 and result.baseline is not None


def test_full_pending_work_preflight_before_real_fit():
    study = read_efftox_desktop_xml(document())
    service = EffToxDesktopInference(
        study, monotone_toxicity=False, seed=1, draws=8, warmup=0, chains=2, max_work=192
    )
    with pytest.raises(ValueError, match="workflow exceeds"):
        recommend_efftox_desktop(study, service)
    assert not service.diagnostics


def test_simulation_partial_last_cohort_always_reanalyzes_and_reports_conditional_rselect():
    study = read_efftox_desktop_xml(document().replace("<MaxQtyPatients>6", "<MaxQtyPatients>5"))
    observed = []

    def analyze(counts):
        observed.append(int(counts.sum()))
        return decision(0 if counts.sum() < 5 else 1)

    result = simulate_efftox_desktop(study, analyze, scenario_index=0, trials=2, seed=100)
    assert observed == [0, 0, 3, 5, 0, 3, 5]
    np.testing.assert_array_equal(result.counts.sum(axis=(1, 2, 3)), [5, 5])
    np.testing.assert_array_equal(result.selected_dose_index, [1, 1])
    np.testing.assert_array_equal(result.mean_patients_per_dose, [5, 0])
    assert result.rselect == pytest.approx(
        (result.true_utility[1] - result.true_utility.min()) / np.ptp(result.true_utility)
    )


def test_early_stop_still_reanalyzes_and_no_selection_is_not_in_rselect_denominator():
    study = read_efftox_desktop_xml(document())
    result = simulate_efftox_desktop(
        study, lambda _: decision(None), scenario_index=0, trials=3, seed=11
    )
    assert result.counts.sum() == 0
    np.testing.assert_array_equal(result.analyses_per_trial, [2, 2, 2])
    assert result.stopped_early.all()
    assert result.rselect == 0 and result.no_selection_probability == 1
    assert _rselect(np.array([0.0, 1.0]), np.array([-1, -1, 1])) == 1
    assert _rselect(np.array([2.0, 2.0]), np.array([0, 1])) is None


def test_real_saved_study_simulation_reproducibility_and_final_fit():
    study = read_efftox_desktop_xml(document())

    def run():
        service = EffToxDesktopInference(
            study, monotone_toxicity=True, seed=199, draws=32, warmup=16, chains=2
        )
        result = simulate_efftox_desktop(study, service, scenario_index=0, trials=2, seed=12)
        return result, service

    first, service = run()
    again, _ = run()
    np.testing.assert_array_equal(first.counts, again.counts)
    np.testing.assert_array_equal(first.selected_dose_index, again.selected_dose_index)
    assert len(service.diagnostics) == 1 + first.analyses_per_trial.sum()
    assert service.diagnostics[-1].observed_patients == first.counts[-1].sum()


def test_report_preserves_inputs_prior_means_and_all_fit_diagnostics():
    from mdanderson_stats.efftox_desktop_report import (
        efftox_desktop_conduct_html,
        efftox_desktop_simulation_html,
    )

    study = read_efftox_desktop_xml(document().replace("literal date", "&lt;script&gt;"))
    service = EffToxDesktopInference(
        study, monotone_toxicity=False, seed=42, draws=8, warmup=0, chains=2
    )
    recommendation = recommend_efftox_desktop(study, service)
    report = efftox_desktop_conduct_html(study, recommendation, diagnostics=service.diagnostics)
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert "Capacity safeguard applied" in report and "Maximum split R-hat" in report
    assert "Observed-prefix baseline dose diagnostics" in report
    assert "Mean efficacy" in report and "Admissible" in report
    result = simulate_efftox_desktop(study, service, scenario_index=0, trials=1, seed=13)
    report = efftox_desktop_simulation_html(result)
    assert "Rselect conditional on selection" in report
    assert "Prior-only diagnostics" in report and "Mean efficacy" in report
    assert "All saved input settings" in report and "monotone_toxicity" in report
    assert "Inference diagnostics were not supplied" not in report


def test_calibration_newest_input_roundtrip_and_profile_cannot_change_silently():
    import xml.etree.ElementTree as ET

    from mdanderson_stats.efftox_desktop_calibration import calibrate_efftox_desktop

    root = ET.fromstring(document())
    root.find("Cohorts").clear()
    sim = root.find("SimulationParams")
    sim.find("NewestHyperParams").set("value", "input")
    for parent, child, values in [
        ("ElicitedToxProbs", "TVal", [0.1, 0.2]),
        ("ElicitedResProbs", "RVal", [0.3, 0.5]),
    ]:
        group = ET.SubElement(sim, parent)
        for value in values:
            ET.SubElement(group, child).text = str(value)
    ET.SubElement(sim, "TargetESS").text = ".9"
    study = read_efftox_desktop_xml(ET.tostring(root, encoding="unicode"))
    result = calibrate_efftox_desktop(study, monotone_toxicity=False)
    assert result.study is not None
    assert result.calibration.quadrature_converged
    assert result.study.newest_hyperparameters == "output"
    assert read_efftox_desktop_xml(result.study.xml) == result.study
    assert "PythonEffToxCalibration" in result.study.xml
    np.testing.assert_array_equal(
        result.study.saved_prior(monotone_toxicity=False).mean, result.calibration.prior.mean
    )
    with pytest.raises(ValueError, match="profile differs"):
        EffToxDesktopInference(result.study, monotone_toxicity=True, seed=1)
    EffToxDesktopInference(result.study, monotone_toxicity=False, seed=1)


@pytest.mark.parametrize(
    "case",
    json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "research/fixtures/efftox-native/current-rselect.json"
        ).read_text()
    ),
)
def test_original_rselect_arithmetic_with_explicit_undefined_range(case):
    count = case["selected_trials"]
    middle = case["utility_sum"] / count if count else case["minimum"]
    utility = np.array([case["minimum"], case["maximum"], middle])
    if case["minimum"] == case["maximum"]:
        utility[:] = case["minimum"]
    actual = _rselect(utility, np.array([2] * count + [-1, -1], dtype=np.int64))
    if isinstance(case["native_rselect"], str):
        assert actual is None
    else:
        assert actual == pytest.approx(case["native_rselect"], abs=1e-14)


def test_service_cannot_silently_use_another_design_or_closed_history():
    study = read_efftox_desktop_xml(document())
    service = EffToxDesktopInference(study, monotone_toxicity=True, seed=1)
    changed = read_efftox_desktop_xml(
        document().replace("<ResponseCutoff>0.5", "<ResponseCutoff>0.6")
    )
    with pytest.raises(ValueError, match="different study"):
        recommend_efftox_desktop(changed, service)
    closed = read_efftox_desktop_xml(document().replace("<TrialIsShutDown>0", "<TrialIsShutDown>1"))
    with pytest.raises(ValueError, match="shut down"):
        recommend_efftox_desktop(closed, service)
    assert not service.diagnostics


@pytest.mark.parametrize(
    "case",
    json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "research/fixtures/efftox-native/current-negative-slope.json"
        ).read_text()
    )["summary_branch"],
)
def test_negative_slope_summary_is_opt_in_original_instruction_compatibility(case):
    from types import SimpleNamespace

    from mdanderson_stats.efftox_decision import _tail_probability
    from mdanderson_stats.efftox_desktop_decision import _toxicity_summary

    probabilities = np.array([[[case["first_dose_toxicity"], case["raw_current_toxicity"]]]])
    logits = np.log(probabilities) - np.log1p(-probabilities)
    fit = SimpleNamespace(
        parameters=np.array([[[0, case["slope"], 0, 0, 0, 0]]]),
        toxicity_logits=logits,
        toxicity_probabilities=probabilities,
    )
    legacy_logits, legacy_probabilities = _toxicity_summary(fit, "desktop_legacy")
    assert legacy_probabilities[0, 0, 1] == case["summary_toxicity"]
    assert (
        _tail_probability(legacy_logits, case["cutoff"], upper=False)[1]
        == case["toxicity_tail_indicator"]
    )
    _, model_probabilities = _toxicity_summary(fit, "model")
    assert model_probabilities[0, 0, 1] == case["raw_current_toxicity"]
    np.testing.assert_array_equal(fit.toxicity_probabilities, probabilities)


def test_legacy_summary_does_not_change_posterior_likelihood_or_draws():
    study = read_efftox_desktop_xml(document())
    counts = study.observed_counts()
    common = dict(monotone_toxicity=False, seed=521, draws=32, warmup=16, chains=2)
    ordinary = EffToxDesktopInference(study, **common)
    legacy = EffToxDesktopInference(study, toxicity_summary_profile="desktop_legacy", **common)
    ordinary(counts)
    legacy(counts)
    a, b = ordinary.diagnostics[0], legacy.diagnostics[0]
    assert a.likelihood_evaluations == b.likelihood_evaluations
    assert a.max_split_rhat == b.max_split_rhat
    assert a.max_batch_mean_mcse == b.max_batch_mean_mcse
    assert a.mean_efficacy == b.mean_efficacy
