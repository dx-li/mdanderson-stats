"""Portable original-engine evidence needed by the native desktop XML contract."""

import json
from pathlib import Path

import pytest

from mdanderson_stats import ArandNativeGates, arand_native_batch_monitor
from mdanderson_stats.arand_native import (
    ArandNativeArmLimits,
    ArandNativeMonitor,
    arand_native_constraints,
    arand_native_select,
    arand_native_statuses,
)
from mdanderson_stats.arand_native_trial import (
    ArandNativeEvidence,
    ArandNativeRules,
    ArandNativeState,
    arand_native_decide,
)

ROOT = Path(__file__).resolve().parents[1] / "research/fixtures/arand-native"


@pytest.mark.parametrize("case", json.loads((ROOT / "per-arm-rules.json").read_text()))
def test_original_per_arm_paths(case):
    arms = case["arm_rules"]
    high = tuple(a["upper"] for a in arms)
    actual = arand_native_statuses(
        case["probabilities"],
        tuple(case["statuses"]),
        lower=tuple(a["lower"] for a in arms),
        upper=high,
        futility=case["futility"],
        futility_cutoff=tuple(a["futility_cutoff"] if a["enforce"] else 0 for a in arms),
        min_action=case["min_action"],
        max_action=case["max_action"],
    )
    assert actual == tuple(case["expected_statuses"])
    for grouped, expected in zip((False, True), case["expected_selections"], strict=True):
        actual_selection = arand_native_select(
            case["probabilities"],
            tuple(case["statuses"]),
            upper=high,
            cutoff=tuple(a["cutoff"] for a in arms),
            completed=case["completed"],
            group=grouped,
        )
        assert [actual_selection.winner, actual_selection.early] == expected


@pytest.mark.parametrize("case", json.loads((ROOT / "constraint-gates.json").read_text()))
def test_original_independent_gate_paths(case):
    c = case["input"]
    gates = ArandNativeGates(*c["gates"])
    limits = tuple(
        ArandNativeArmLimits(*a)
        for a in zip(c["arm_min"], c["arm_max"], c["group_min"], c["group_max"], strict=True)
    )
    assert arand_native_constraints(
        tuple(c["statuses"]), tuple(c["arm_count"]), limits, now=c["now"], gates=gates
    ) == tuple(case["constraints"]["statuses"])
    rules = ArandNativeRules(
        limits,
        gates=gates,
        minimum_enrollment=c["min_count"],
        maximum_enrollment=c["max_count"],
        minimum_duration=c["min_time"],
        maximum_duration=c["max_time"],
        stop_when_one_arm=False,
        lower=0,
        upper=1,
    )
    evidence = ArandNativeEvidence([0.5, 0.5])
    state = ArandNativeState(ArandNativeMonitor(0), tuple(c["statuses"]), evidence)
    decision = arand_native_decide(
        state,
        rules,
        now=c["now"],
        enrolled=tuple(c["arm_count"]),
        observed=5,
        pending=False,
        analyze=lambda _: evidence,
    )
    assert bool(decision.analysis_times) == case["minimum"]


def test_cutoff_dimensions_and_immutable_rules():
    with pytest.raises(ValueError, match="one value per arm"):
        arand_native_statuses([0.5, 0.5], (0, 0), lower=(0.1,))
    with pytest.raises(ValueError, match="boolean"):
        ArandNativeGates(arm_minimum=1)
    rules = ArandNativeRules((ArandNativeArmLimits(),) * 2, lower=[0.1, 0.2])
    assert rules.lower == (0.1, 0.2)


@pytest.mark.parametrize(
    "case",
    [
        json.loads(line)
        for line in (ROOT / "arand-batch-init-fixtures.jsonl").read_text().splitlines()
    ],
)
def test_original_batch_initialization_and_reset(case):
    for reset in (False, True):
        monitor = arand_native_batch_monitor(case["a"], case["b"], case["maximum"], reset=reset)
        assert monitor.next_enrollment == case["reset" if reset else "initial"]
        assert monitor.targets == tuple(case["increments"])
        assert monitor.batch_index == case["batch_index"]


@pytest.mark.parametrize(
    "args", [(0, 0, 10), (10, 0, 10), (1, 1000, 10), (1, -1000, 10), (1, 0, 10001)]
)
def test_undefined_native_batch_schedules_are_rejected(args):
    with pytest.raises(ValueError):
        arand_native_batch_monitor(*args)


def test_per_arm_parameter_thresholds_against_distribution_cdfs():
    import numpy as np
    from scipy.stats import beta, invgamma

    from mdanderson_stats.arand_posterior import arand_binary_posterior, arand_survival_posterior

    prior = np.array([[2.0, 3.0], [4.0, 5.0], [1.0, 2.0]])
    binary = arand_binary_posterior([0, 0, 0], [0, 0, 0], prior=prior, threshold=(0, 0.7, 1))
    np.testing.assert_allclose(
        binary.exceedance_probability, beta.sf([0, 0.7, 1], prior[:, 0], prior[:, 1]), atol=1e-14
    )
    survival = arand_survival_posterior([0, 0, 0], [0, 0, 0], prior=prior, threshold=(0, 2, 100))
    np.testing.assert_allclose(
        survival.exceedance_probability,
        invgamma.sf([0, 2, 100], prior[:, 0], scale=prior[:, 1]),
        atol=1e-14,
    )
    assert survival.threshold == (0, 2, 100)
    with pytest.raises(ValueError, match="one value per arm"):
        arand_binary_posterior([0, 0], [0, 0], prior=prior[:2], threshold=(0.5,))


def test_per_arm_study_delay_and_threshold_roundtrip():
    from mdanderson_stats.arand_native_study import (
        ArandNativeStudy,
        arand_native_study_json,
        read_arand_native_study,
        simulate_arand_native_trial,
    )

    study = ArandNativeStudy(
        rules=ArandNativeRules(
            (ArandNativeArmLimits(),) * 2,
            maximum_enrollment=4,
            lower=(0, 0),
            upper=(1, 1),
            futility_cutoff=(0, 0),
        ),
        monitor=ArandNativeMonitor(0),
        prior=((1, 1), (1, 1)),
        true_parameters=(0.4, 0.6),
        accrual_rate=2,
        binary_delay=(0, 3),
        futility_threshold=(0.1, 0.3),
        max_candidates=10,
    )
    restored = read_arand_native_study(arand_native_study_json(study))
    assert restored == study
    trial = simulate_arand_native_trial(restored, seed=123)
    for observation in trial.records:
        assert observation.record_time - observation.arrival == pytest.approx(
            (0, 3)[observation.arm]
        )


@pytest.mark.parametrize("family", ["binary", "tte"])
def test_original_managed_xml_to_python_simulation(family):
    from mdanderson_stats import read_arand_native_xml
    from mdanderson_stats.arand_native_study import simulate_arand_native_trial

    xml = (ROOT / "managed-studies" / (family + ".xml")).read_text()
    document = read_arand_native_xml(xml, max_candidates=30)
    assert document.source_xml == xml
    assert document.has_patient_history
    assert document.seed == 12345
    assert document.replications == (100,)
    assert document.arm_names == ("Arm1", "Arm2")
    study = document.studies[0]
    assert study.rules.lower == (0.05, 0.05)
    assert study.rules.upper == (0.95, 0.95)
    assert study.rules.gates.trial_minimum
    assert not study.rules.gates.arm_minimum
    assert (
        study.prior == ((2, 2), (2, 2)) if family == "binary" else study.prior == ((2, 3), (2, 3))
    )
    result = simulate_arand_native_trial(study, seed=101)
    assert len(result.records) <= 6
    assert result.final.reason != "allocate"
    # Roundtrip is the original text including its original history/metadata.
    restored = read_arand_native_xml(document.source_xml, max_candidates=30)
    assert restored == document


def test_raw_xml_missing_flags_and_actions_keep_native_defaults():
    import xml.etree.ElementTree as ET

    from mdanderson_stats import read_arand_native_xml

    root = ET.fromstring((ROOT / "managed-studies/binary.xml").read_text())
    stop = root.find("./trial_details/stopping_rule")
    stop.attrib.clear()
    for child in list(stop):
        stop.remove(child)
    document = read_arand_native_xml(ET.tostring(root, encoding="unicode"))
    rules = document.studies[0].rules
    assert not any(vars(rules.gates).values())
    assert rules.min_action == rules.max_action == 2
    assert rules.lower == (0, 0)
    assert rules.upper == (1, 1)
    assert rules.final_cutoff == (0, 0)


@pytest.mark.parametrize(
    "before,after",
    [
        ('name="Binary"', 'name="BinaryCovariates"'),
        ('prior_dist="Beta"', 'prior_dist="Normal"'),
        ('group_name="No Groups"', 'group_name="Other"'),
        ('enforce_min_pred_prob="false"', 'enforce_min_pred_prob="true"'),
        ('name="ProbSuccess"', 'name="Unknown"'),
        ('monitoring_type="continuous"', 'monitoring_type="Unknown"'),
        ('min_pats_per_arm="false"', 'min_pats_per_arm="maybe"'),
    ],
)
def test_xml_unsupported_semantics_fail(before, after):
    from mdanderson_stats import read_arand_native_xml

    xml = (ROOT / "managed-studies/binary.xml").read_text().replace(before, after)
    with pytest.raises(ValueError):
        read_arand_native_xml(xml)


def test_xml_time_units_are_not_silently_relabelled():
    from mdanderson_stats import read_arand_native_xml

    xml = (
        (ROOT / "managed-studies/binary.xml")
        .read_text()
        .replace("<time_type>Months</time_type>", "<time_type>Days</time_type>")
    )
    with pytest.raises(ValueError, match="Months"):
        read_arand_native_xml(xml)


def test_summary_counters_against_full_original_accumulation(monkeypatch):
    import numpy as np

    import mdanderson_stats.arand_native_study as module
    from mdanderson_stats import ArandNativeObservation, ArandNativeSelection, ArandNativeTrial
    from mdanderson_stats.arand_native_trial import ArandNativeDecision

    cases = [
        json.loads(line)
        for line in (ROOT / "arand-accumulation-fixtures.jsonl").read_text().splitlines()
    ]
    document = __import__("mdanderson_stats").read_arand_native_xml(
        (ROOT / "managed-studies/binary.xml").read_text()
    )
    outcomes = []
    selected = np.zeros(2, dtype=int)
    early = np.zeros(2, dtype=int)
    statuses = np.zeros((2, 10), dtype=int)
    for case in cases:
        c = case["input"]
        records = tuple(
            ArandNativeObservation(a, 0, 1, 1)
            for a, count in enumerate(c["counts"])
            for _ in range(count)
        )
        state = ArandNativeState(
            ArandNativeMonitor(0), tuple(c["statuses"]), ArandNativeEvidence([0.5, 0.5])
        )
        decision = ArandNativeDecision(
            state,
            c["duration"],
            "reference",
            c["completed"],
            ArandNativeSelection(c["winner"], c["early"]),
            None,
            None,
            (),
            (),
        )
        outcomes.append(ArandNativeTrial(records, (decision,), "binary", "mean"))
        for call in case["native"]["calls"]:
            if call[0] == "selected":
                selected[call[1]] += 1
            elif call[0] == "selected_early":
                early[call[1]] += 1
            elif call[0] == "status":
                statuses[call[1], call[2]] += 1
    sequence = iter(outcomes)
    monkeypatch.setattr(
        module, "simulate_arand_native_trial", lambda *args, **kwargs: next(sequence)
    )
    summary = module.simulate_arand_native(document.studies[0], trials=len(cases), seed=123)
    assert summary.selected == tuple(selected)
    assert summary.selected_early == tuple(early)
    np.testing.assert_array_equal(summary.final_status_counts, statuses)
    assert summary.completed == sum(c["native"]["completed"] for c in cases)
    assert summary.duration_mean == sum(c["native"]["duration_sum"] for c in cases) / len(cases)
    np.testing.assert_array_equal(summary.mean_enrollment, [2, 4])
    # Explicit Python report definition includes permanent closures as drops.
    assert summary.dropped_early == tuple(statuses[:, 1:4].sum(axis=1))
    html = module.arand_native_report(summary)
    assert "Early selection probability" in html and "Early drop probability" in html


@pytest.mark.parametrize(
    "case",
    [
        json.loads(line)["native"]
        for line in (ROOT / "desktop-report-slices.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "percentiles"
    ],
)
def test_actual_desktop_percentile_profile(case):
    import numpy as np

    from mdanderson_stats.arand_native_study import _legacy_enrollment_percentiles

    values = np.arange(case["size"], dtype=float)[::-1, None]
    actual = _legacy_enrollment_percentiles(values)
    if case["suppressed"]:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual[0], case["indices"][::-1])


def test_legacy_report_profile_is_explicit():
    from mdanderson_stats import read_arand_native_xml
    from mdanderson_stats.arand_native_study import arand_native_report, simulate_arand_native

    study = read_arand_native_xml((ROOT / "managed-studies/binary.xml").read_text()).studies[0]
    summary = simulate_arand_native(study, trials=2, seed=123)
    assert "type 7" in arand_native_report(summary)
    assert "omitted below 100 trials" in arand_native_report(
        summary, percentile_profile="legacy-5.2.2"
    )
    with pytest.raises(ValueError, match="percentile_profile"):
        arand_native_report(summary, percentile_profile="unknown")
