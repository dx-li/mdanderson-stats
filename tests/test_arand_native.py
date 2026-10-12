"""Independent original-instruction fixtures for the advertised single endpoint."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.arand_native import (
    ArandNativeArmLimits,
    ArandNativeMonitor,
    ArandNativeObservation,
    arand_native_allocation,
    arand_native_constraints,
    arand_native_monitor,
    arand_native_select,
    arand_native_statistics,
    arand_native_statuses,
)

FIXTURES = Path(__file__).resolve().parents[1] / "research/fixtures/arand-native"


def fixtures(name):
    return [
        json.loads(line)
        for line in (FIXTURES / f"arand-{name}-fixtures.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("case", fixtures("floor"))
def test_native_floor(case):
    np.testing.assert_allclose(
        arand_native_allocation(case["probabilities"], power=case["power"], floor=case["floor"]),
        case["native"],
        atol=2e-15,
        rtol=2e-15,
    )


@pytest.mark.parametrize("case", fixtures("scheduler"))
def test_native_scheduler(case):
    c, expected = case["input"], case["native"]
    threshold = c.get("threshold", 1)
    m = ArandNativeMonitor(c["mode"], threshold, threshold, threshold, c.get("step", 1))
    result = arand_native_monitor(
        m,
        now=c["now"],
        enrolled=c["count"],
        observed=c.get("observed", 0),
        pending=c.get("pending", False),
    )
    calls = []
    if result.trial_status is not None:
        calls.append(["status", result.trial_status])
    if result.analyze:
        calls.append(["look", c["now"]])
    assert calls == expected["calls"]
    assert result.monitor.next_time == expected["next_time"]
    assert result.monitor.next_enrollment == expected["next_enrollment"]
    assert result.monitor.next_outcome == expected["next_outcome"]
    assert result.monitor.index == expected["index"]


@pytest.mark.parametrize("case", fixtures("tte-stats"))
def test_native_censor_statistics(case):
    r = ArandNativeObservation(
        0, case["arrival"], case["record_time"], int(case["stored_indicator"])
    )
    stats = arand_native_statistics((r,), now=case["now"], arms=1)
    assert stats.positive == (case["original"]["event"],)
    assert stats.exposure[0] == case["original"]["followup"]
    assert stats.observed == (int(case["now"] >= case["record_time"]),)


@pytest.mark.parametrize(
    "case",
    [c for c in fixtures("status") if isinstance(c["input"]["probabilities"][0], (int, float))],
)
def test_native_status(case):
    c = case["input"].copy()
    c["statuses"] = tuple(c["statuses"])
    assert arand_native_statuses(**c) == tuple(case["native"]["statuses"])


@pytest.mark.parametrize("group,name", [(False, "final"), (True, "groupfinal")])
def test_native_selection(group, name):
    for case in fixtures(name):
        c = case["input"].copy()
        if isinstance(c["probabilities"][0], list):
            continue  # latent multi-endpoint template is outside desktop scope
        c["statuses"] = tuple(c["statuses"])
        result = arand_native_select(**c, group=group)
        expected = case["native"][-1]
        assert result.winner == (None if expected[2] == "None" else int(expected[2]))
        assert result.early == bool(expected[3])


def test_native_arm_constraint_fixtures():
    for case in fixtures("minmax"):
        c = case["input"]
        if any(k in c for k in ("max_count", "max_time", "min_time")):
            continue  # trial controller tests cover these
        n = len(c["statuses"])
        counts = c.get("arm_count", [0] * n)
        if c.get("group_count", counts) != counts:
            continue  # only one group is exposed in the desktop
        minimum = c.get("arm_min", [0] * n)
        limits = tuple(
            ArandNativeArmLimits(
                minimum[i],
                c.get("arm_max", [100] * n)[i],
                c.get("group_min", minimum)[i],
                c.get("group_max", [100] * n)[i],
                c.get("start", [0] * n)[i],
                c.get("duration", [0] * n)[i],
            )
            for i in range(n)
        )
        assert arand_native_constraints(
            tuple(c["statuses"]), tuple(counts), limits, now=c.get("now", 5)
        ) == tuple(case["native"]["statuses"])


def test_scheduler_repeated_equality_and_exhaustion():
    m = ArandNativeMonitor(mode=1)
    for _ in range(3):
        r = arand_native_monitor(m, now=1, enrolled=0, observed=0, pending=False)
        assert r.analyze
        m = r.monitor
    m = ArandNativeMonitor(mode=2, targets=(1, 2))
    r = arand_native_monitor(m, now=3, enrolled=0, observed=0, pending=False)
    assert r.analyze and r.monitor.index == 2
    assert not arand_native_monitor(r.monitor, now=4, enrolled=0, observed=0, pending=False).analyze


@pytest.mark.parametrize(
    "p,power,floor", [([0, 0], 1, 0), ([0.1, 0.9], 1, 0.6), ([1], -1, 0), ([float("nan")], 1, 0)]
)
def test_invalid_allocation(p, power, floor):
    with pytest.raises(ValueError):
        arand_native_allocation(p, power=power, floor=floor)


@pytest.mark.parametrize("case", fixtures("integrated"))
def test_original_integrated_pipeline(case):
    from mdanderson_stats.arand_native_trial import (
        ArandNativeEvidence,
        ArandNativeRules,
        ArandNativeState,
        arand_native_decide,
    )

    c, expected = case["input"], case["original_pipeline"]
    p = ArandNativeEvidence(np.array(c.get("probabilities", [0.6, 0.4])))
    state = ArandNativeState(ArandNativeMonitor(mode=c["mode"], next_enrollment=3), (0, 0), p)
    rules = ArandNativeRules(
        (ArandNativeArmLimits(),) * 2,
        maximum_enrollment=c.get("max_count", 100),
        minimum_enrollment=c.get("min_count", 0),
        final_followup=c.get("followup", 0),
        power=1,
    )
    result = arand_native_decide(
        state,
        rules,
        now=5,
        enrolled=(2, 1),
        observed=2 if c.get("pending", False) else 3,
        pending=c.get("pending", False),
        analyze=lambda t: p,
    )
    assert result.state.statuses == tuple(expected["statuses"])
    assert result.time == expected["time"]
    assert (result.selection is not None) == expected["stopped"]
    expected_times = tuple(t[1] for t in expected["trace"] if t[0] == "posterior_service")
    assert result.analysis_times == expected_times
    if result.selection is not None:
        assert result.selection.winner == (
            None if expected["winner"] == "None" else int(expected["winner"])
        )
    else:
        np.testing.assert_allclose(result.allocation, expected["trace"][-1][2])


def test_independent_binary_replay_and_followup():
    from fractions import Fraction

    from mdanderson_stats.arand_native_trial import ArandNativeRules, arand_native_replay

    rules = ArandNativeRules(
        (ArandNativeArmLimits(),) * 2,
        maximum_enrollment=2,
        final_followup=3,
        lower=0,
        upper=1,
        stop_when_one_arm=False,
        power=1,
        floor=0,
    )
    trial = arand_native_replay(
        [1, 2, 5],
        [0.2, 0.9, 0.5],
        [[3, 3]] * 3,
        [[1, 0]] * 3,
        prior=[[1, 1], [1, 1]],
        rules=rules,
        monitor=ArandNativeMonitor(),
    )
    assert [r.arm for r in trial.records] == [0, 1]
    assert trial.final.time == 8 and trial.final.completed
    # Independent exact integral: P(Beta(2,1)>Beta(1,2)) = int 2x(2x-x²) dx.
    exact = 4 * Fraction(1, 3) - 2 * Fraction(1, 4)
    np.testing.assert_allclose(
        trial.final.state.evidence.probability_best, [float(exact), 1 - float(exact)], atol=1e-10
    )
    assert trial.final.selection.winner == 0
    assert trial.final.analysis_times == (5, 8)


def test_censor_replay_mean_median_equivalence():
    from mdanderson_stats.arand_native_trial import ArandNativeRules, arand_native_replay

    rules = ArandNativeRules(
        (ArandNativeArmLimits(),) * 2,
        maximum_enrollment=2,
        lower=0,
        upper=1,
        stop_when_one_arm=False,
    )
    common = dict(
        arrival_times=[1, 2, 4],
        uniforms=[0.1, 0.9, 0.2],
        observation_delays=[[1, 1]] * 3,
        indicators=[[0, 1]] * 3,
        rules=rules,
        monitor=ArandNativeMonitor(),
        family="exponential",
    )
    mean = arand_native_replay(prior=[[2, 3], [2, 3]], **common)
    median = arand_native_replay(
        prior=[[2, 3 * np.log(2)], [2, 3 * np.log(2)]], parameter="median", **common
    )
    assert [r.arm for r in mean.records] == [0, 1]
    np.testing.assert_allclose(
        mean.final.state.evidence.probability_best,
        median.final.state.evidence.probability_best,
        atol=2e-10,
    )
    # theta0~IG(2,4), theta1~IG(3,4): reciprocal Gamma race gives P(theta0>theta1)=11/16.
    np.testing.assert_allclose(
        mean.final.state.evidence.probability_best, [11 / 16, 5 / 16], atol=2e-10
    )


def test_closed_arms_stay_in_ranking_but_not_assignment_population():
    from mdanderson_stats.arand_native_trial import ArandNativeRules, arand_native_replay

    rules = ArandNativeRules(
        (ArandNativeArmLimits(),) * 2,
        maximum_enrollment=1,
        lower=0.4,
        upper=1,
        stop_when_one_arm=False,
    )
    trial = arand_native_replay(
        [1, 2],
        [0, 0.5],
        [[0, 0]] * 2,
        [[1, 1]] * 2,
        prior=[[1, 9], [9, 1]],
        rules=rules,
        monitor=ArandNativeMonitor(),
    )
    # Ordinary allocation visits only eligible arms, even for exactly zero uniforms.
    assert trial.decisions[0].state.statuses == (1, 0)
    assert trial.records[0].arm == 1
    assert trial.decisions[0].assignment_arms == (1,)
    assert len(trial.final.state.evidence.probability_best) == 2


def test_tape_exhaustion_rejected():
    from mdanderson_stats.arand_native_trial import ArandNativeRules, arand_native_replay

    with pytest.raises(ValueError, match="exhausted"):
        arand_native_replay(
            [1],
            [0.5],
            [[0, 0]],
            [[1, 0]],
            prior=[[1, 1], [1, 1]],
            rules=ArandNativeRules((ArandNativeArmLimits(),) * 2),
            monitor=ArandNativeMonitor(),
        )


@pytest.mark.parametrize("case", [c for c in fixtures("outcome") if c["outcome"] in (0, 1)])
def test_original_visibility(case):
    r = ArandNativeObservation(0, case["arrival"], case["event_time"], int(case["outcome"]))
    stats = arand_native_statistics((r,), now=case["now"], arms=1)
    visible = stats.positive[0] if stats.observed[0] else -1
    assert visible == case["visible"]
    assert stats.exposure[0] == case["followup"]


@pytest.mark.parametrize(
    "case", [c for c in fixtures("pending") if isinstance(c["native_observed"], int)]
)
def test_original_pending(case):
    records = tuple(
        ArandNativeObservation(0, 0, 2 if row[0] == -1 else 1, max(0, row[0]))
        for row in case["input"]
    )
    stats = arand_native_statistics(records, now=1, arms=1)
    assert sum(stats.observed) == case["native_observed"]
    assert int(sum(stats.observed) < len(records)) == case["native_pending"]


def test_zero_uniform_native_forced_runin_population():
    from mdanderson_stats.arand_native_trial import ArandNativeRules, arand_native_replay

    rules = ArandNativeRules(
        (ArandNativeArmLimits(),) * 3,
        maximum_enrollment=2,
        lower=0,
        upper=1,
        stop_when_one_arm=False,
        runin=3,
    )
    trial = arand_native_replay(
        [1, 2, 3],
        [0.1, 0, 0.5],
        [[0, 0, 0]] * 3,
        [[1, 1, 1]] * 3,
        prior=[[1, 1]] * 3,
        rules=rules,
        monitor=ArandNativeMonitor(),
    )
    assert [r.arm for r in trial.records] == [0, 0]
    assert trial.decisions[1].allocation[0] == 0
    assert trial.decisions[1].assignment_arms == (0, 1, 2)
