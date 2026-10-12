"""Current trial orchestration, calendar boundaries and failure classification."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mdanderson_stats.uaroet_desktop_simulation as sim
from mdanderson_stats import UAROETDesktopScenario, read_uaroet_desktop_study


def design(n=3, **parameters):
    s = read_uaroet_desktop_study(
        (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
    )
    p = dict(s.trial_parameters)
    p.update(NumPatients=n, CohortSize_First=1, NumTimesToShrinkUtilityCutoff=1)
    p.update(parameters)
    return replace(
        s,
        patients=(),
        trial_parameters=p,
        simulation_parameters={"NumChains": 2, "BurnInSamples": 0},
    )


def scenario():
    return UAROETDesktopScenario(1, [[0, 1], [1, 0]], [[1, 0], [0, 1]], 0)


def fake_decision(s, *, final, action=None, **kwargs):
    return SimpleNamespace(
        action=action or ("select" if final else "treat"),
        selected=0,
        reason="test",
        posterior=None,
        status=(1, 1),
    )


def test_complete_calendar_observes_all_outcomes_at_final_and_retains_only_small_looks(monkeypatch):
    views = []

    def decision(s, **kwargs):
        views.append(s)
        return fake_decision(s, **kwargs)

    monkeypatch.setattr(sim, "recommend_uaroet_desktop", decision)
    callbacks = []
    result = sim.run_uaroet_desktop_trial(
        design(MinDaysToEfficacy=2, MaxDaysToEfficacy=2, MinDaysToToxicity=1, MaxDaysToToxicity=1),
        scenario(),
        rng=np.random.default_rng(3),
        on_look=lambda t, d: callbacks.append((t, d.action)),
    )
    assert len(result.patients) == 3 and len(result.looks) == 4
    assert result.patients[0].arrival == 0
    assert result.selected == 0 and not result.stopped_early
    assert views[-1].patients == ((1, 0, 1, 0), (2, 0, 1, 0), (3, 0, 1, 0))
    assert result.followup_duration == result.patients[-1].arrival + 2
    assert result.decision_time == result.followup_duration
    assert len(callbacks) == 4
    assert all(not hasattr(look, "posterior") for look in result.looks)


def test_pending_outcomes_are_not_visible_before_their_times(monkeypatch):
    views = []

    def decision(s, **kwargs):
        views.append(s)
        return fake_decision(s, **kwargs)

    monkeypatch.setattr(sim, "recommend_uaroet_desktop", decision)
    sim.run_uaroet_desktop_trial(
        design(
            MinDaysToEfficacy=3600,
            MaxDaysToEfficacy=3600,
            MinDaysToToxicity=3600,
            MaxDaysToToxicity=3600,
            MeanDaysBetweenArrivals=0.1,
        ),
        scenario(),
        rng=np.random.default_rng(2),
    )
    assert views[1].patients == ((1, 0, -1, -1),)
    assert views[2].patients == ((1, 0, -1, -1), (2, 0, -1, -1))
    assert all(r[2:] == (1, 0) for r in views[-1].patients)


def test_interim_stop_follows_outcomes_without_final_refit(monkeypatch):
    def decision(s, **kwargs):
        result = fake_decision(s, **kwargs)
        if s.patients:
            result.action, result.selected = "stop", None
        return result

    monkeypatch.setattr(sim, "recommend_uaroet_desktop", decision)
    result = sim.run_uaroet_desktop_trial(
        design(MinDaysToEfficacy=3600, MaxDaysToEfficacy=3600, MeanDaysBetweenArrivals=0.1),
        scenario(),
        rng=np.random.default_rng(2),
    )
    assert result.stopped_early and result.selected is None
    assert len(result.patients) == 1 and not any(look.final for look in result.looks)
    assert result.followup_duration == 3600 > result.decision_time


def test_computational_hold_raises_instead_of_counting_as_stop(monkeypatch):
    monkeypatch.setattr(
        sim, "recommend_uaroet_desktop", lambda s, **k: fake_decision(s, action="hold", **k)
    )
    with pytest.raises(sim.UAROETSimulationHold, match="held at time 0.0") as exc:
        sim.run_uaroet_desktop_trial(design(), scenario(), rng=np.random.default_rng(2))
    assert exc.value.decision.reason == "test"


@pytest.mark.parametrize("size,suspended", [(1, False), (2, True)])
def test_legacy_suspension_only_after_closing_multi_patient_new_cohort(
    monkeypatch, size, suspended
):
    monkeypatch.setattr(sim, "recommend_uaroet_desktop", fake_decision)
    result = sim.run_uaroet_desktop_trial(
        design(
            CohortSize_NewDose=size,
            SuspendAccrualForFullEvalOfNewDoseCohort=1,
            MinDaysToEfficacy=10,
            MaxDaysToEfficacy=10,
            MinDaysToToxicity=10,
            MaxDaysToToxicity=10,
            MeanDaysBetweenArrivals=0.1,
        ),
        scenario(),
        rng=np.random.default_rng(2),
        profile="legacy_1_8",
    )
    if suspended:
        assert result.patients[2].arrival >= result.patients[1].completion_time
        assert result.discarded_arrivals > 0
    else:
        assert result.patients[2].arrival < result.patients[1].completion_time
        assert result.discarded_arrivals == 0
    assert result.decision_time == result.followup_duration + 1


@pytest.mark.parametrize(
    "options",
    [
        {"target": 0},
        {"maximum_rhat": 0.9},
        {"max_draws": 7},
        {"chunk_draws": 100_001},
        {"max_evaluations": 0},
        {"max_discarded_arrivals": -1},
        {"on_look": 3},
    ],
)
def test_invalid_controls_do_not_consume_rng(options):
    rng = np.random.default_rng(2)
    before = str(rng.bit_generator.state)
    with pytest.raises(ValueError):
        sim.run_uaroet_desktop_trial(design(), scenario(), rng=rng, **options)
    assert str(rng.bit_generator.state) == before


def test_safe_suspension_is_not_silently_reinterpreted():
    with pytest.raises(ValueError, match="requires the legacy"):
        sim.run_uaroet_desktop_trial(
            design(SuspendAccrualForFullEvalOfNewDoseCohort=1),
            scenario(),
            rng=np.random.default_rng(2),
        )


def test_actual_trial_small_explicit_demonstration_tolerances():
    result = sim.run_uaroet_desktop_trial(
        design(n=1),
        scenario(),
        rng=np.random.default_rng(12),
        target=0.99,
        maximum_rhat=1000,
        chunk_draws=32,
        max_draws=32,
    )
    assert len(result.patients) == 1 and result.looks[-1].final
    assert result.looks[-1].draws_per_chain == 32


def test_legacy_suspension_waits_for_last_patient_not_full_cohort(monkeypatch):
    monkeypatch.setattr(sim, "recommend_uaroet_desktop", fake_decision)
    delays = iter([(100.0, 100.0), (10.0, 10.0), (1.0, 1.0)])
    monkeypatch.setattr(sim, "uaroet_outcome_delays", lambda *a, **k: next(delays))
    result = sim.run_uaroet_desktop_trial(
        design(
            CohortSize_NewDose=2,
            SuspendAccrualForFullEvalOfNewDoseCohort=1,
            MeanDaysBetweenArrivals=0.1,
        ),
        scenario(),
        rng=np.random.default_rng(2),
        profile="legacy_1_8",
    )
    assert result.patients[1].completion_time <= result.patients[2].arrival
    assert result.patients[2].arrival < result.patients[0].completion_time
    assert result.followup_duration == 100
