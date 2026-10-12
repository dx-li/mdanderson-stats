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
