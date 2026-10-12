"""Independent native DSF contracts; no original binaries needed for tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from mdanderson_stats import dose_schedule_moment_prior
from mdanderson_stats.dose_schedule_desktop_kernel import (
    dose_schedule_desktop_decision,
    dose_schedule_desktop_event_time,
    dose_schedule_desktop_lognormal,
    dose_schedule_desktop_observation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def rows(name):
    return json.loads((FIXTURES / f"dose-schedule-desktop-{name}.json").read_text())


@pytest.mark.parametrize("case", rows("prior"))
def test_original_native_prior(case):
    prior = dose_schedule_moment_prior(
        case["prob"],
        case["admins"],
        case["peak"],
        case["tail"],
        lambda1=case["nu1"],
        lambda2=case["nu2"],
    )
    expected = np.array([case["area_mean"], case["peak_mean"], case["tail_mean"]]).T.ravel()
    np.testing.assert_allclose(prior.mean, expected, atol=5e-14, rtol=1e-12)
    expected_sd = np.array(
        [
            case["area_sd"],
            [case["time_sd"]] * len(case["prob"]),
            [case["time_sd"]] * len(case["prob"]),
        ]
    ).T.ravel()
    np.testing.assert_allclose(prior.sd, expected_sd, atol=5e-14, rtol=1e-12)


@pytest.mark.parametrize("case", rows("selector"))
def test_original_native_controller(case):
    treated = np.zeros((3, 3), dtype=int)
    if case["mode"] != "first":
        treated[tuple(case["highest"])] = 1
    result = dose_schedule_desktop_decision(
        np.array(case["mean"]).reshape(3, 3),
        np.array(case["overdose"]).reshape(3, 3),
        treated,
        target=case["target"],
        upper_probability=case["cutoff"],
        final=case["mode"] == "final",
        profile="desktop_legacy",
    )
    np.testing.assert_array_equal(result.eligible, np.array(case["admissible"]).reshape(3, 3))
    assert result.pair == (None if case["selected"] == [-1, -1] else tuple(case["selected"]))


@pytest.mark.parametrize("case", [r for r in rows("observation") if r["delay"] == 14])
def test_original_observation_calls(case):
    result = dose_schedule_desktop_observation(
        outcome=case["outcome"],
        outcome_time=case["onset"],
        entry_time=case["entry"],
        study_time=case["study"],
        horizon=case["horizon"],
        low_grade_delay=99,
        profile="desktop_legacy",
    )
    calls = case["calls"]
    assert result.included == bool(calls)
    if calls:
        assert result.event == (calls[0][0] == "hazard")
        assert result.time == calls[-1][1]


@pytest.mark.parametrize("case", rows("lognormal"))
def test_native_lognormal_against_accurate_equations(case):
    mu, sigma = dose_schedule_desktop_lognormal(case["prob"], case["horizon"])
    # Native inverse-normal CDF error is below 5.5e-8; near p=.99 this
    # amplifies to 1.23e-5 in mu and 4.61e-6 in sigma.
    np.testing.assert_allclose([mu, sigma], [case["mu"], case["sigma"]], atol=1.3e-5, rtol=1e-7)
    assert ndtr((np.log(case["horizon"]) - mu) / sigma) == pytest.approx(case["prob"], abs=2e-14)
    variance = np.expm1(sigma * sigma) * np.exp(2 * mu + sigma * sigma)
    expected = (case["horizon"] / -np.log1p(-case["prob"])) ** 2
    assert variance == pytest.approx(expected, rel=1e-11)


def test_safe_defaults_keep_cutoff_and_nonmonotone_risks_separate():
    mean = np.array([[0.1, 0.2], [0.3, 0.4], [0.3, 0.4]])
    tail = np.array([[0.8, 1], [0.1, 0.1], [0.1, 0.1]])
    treated = np.zeros((3, 2), int)
    treated[0, 0] = 1
    safe = dose_schedule_desktop_decision(
        mean, tail, treated, target=0.3, upper_probability=0.8, final=True
    )
    legacy = dose_schedule_desktop_decision(
        mean, tail, treated, target=0.3, upper_probability=0.8, final=True, profile="desktop_legacy"
    )
    assert not safe.acceptable[0, 0] and legacy.acceptable[0, 0]
    assert safe.eligible[2, 1] and not legacy.eligible[2, 1]
    assert safe.eligible[1, 1] and not legacy.eligible[1, 1]


def test_safe_pending_caps_followup_and_honors_requested_delay():
    kwargs = dict(
        outcome=2, outcome_time=55, entry_time=0, study_time=65, horizon=60, low_grade_delay=20
    )
    safe = dose_schedule_desktop_observation(**kwargs)
    legacy = dose_schedule_desktop_observation(**kwargs, profile="desktop_legacy")
    assert safe.time == 60 and safe.pending and not safe.final_ready
    assert legacy.time == 65 and legacy.pending
    kwargs["study_time"] = 70
    assert dose_schedule_desktop_observation(**kwargs, profile="desktop_legacy").event
    assert not dose_schedule_desktop_observation(**kwargs).event


@pytest.mark.parametrize(
    "distribution,shape", [("exponential", 1), ("weibull", 0.4), ("weibull", 1.6), ("lognormal", 1)]
)
def test_explicit_distribution_quantiles(distribution, shape):
    assert (
        dose_schedule_desktop_event_time(0.3, 0.3, 60, distribution=distribution, shape=shape) == 60
    )
    assert (
        dose_schedule_desktop_event_time(0.1, 0, 60, distribution=distribution, shape=shape) == 60
    )
    assert dose_schedule_desktop_event_time(0.1, 1, 60, distribution=distribution, shape=shape) == 0
    times = [
        dose_schedule_desktop_event_time(u, 0.3, 60, distribution=distribution, shape=shape)
        for u in (0.001, 0.01, 0.1, 0.2, 0.299)
    ]
    assert all(0 < a < b < 60 for a, b in zip(times, times[1:]))


def study():
    from mdanderson_stats.dose_schedule_desktop_study import read_dose_schedule_desktop_study

    return read_dose_schedule_desktop_study(
        (FIXTURES / "dose-schedule-desktop-study.xml").read_text()
    )


def test_original_managed_xml_and_python_saved_roundtrip():
    from mdanderson_stats.dose_schedule_desktop_study import read_dose_schedule_desktop_study

    result = study()
    assert result.prior_probabilities == (0.2, 0.25, 0.3)
    assert result.patients[0].dose_indices == (0, 1, 0)
    assert result.scenarios[0][2][1] == pytest.approx(0.35)
    assert result.low_grade_delay == 10
    assert result.original_extra_sample_factor == 32
    assert read_dose_schedule_desktop_study(result.to_json()) == result
    xml = (FIXTURES / "dose-schedule-desktop-study.xml").read_text()
    envelope = (
        "<!--begin4DE4157E-8DBC-4c28-B3DE-807264EA0D80\n"
        + xml
        + "\nend4DE4157E-8DBC-4c28-B3DE-807264EA0D80--><html></html>"
    )
    assert read_dose_schedule_desktop_study(envelope) == result


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace('"schema":', '"schema":"duplicate", "schema":'),
        lambda s: s.replace('"horizon": 60.0', '"horizon": true'),
        lambda s: s.replace('"schedule_count": 2', '"schedule_count": 999999'),
        lambda s: s.replace('"low_grade_delay": 10.0', '"low_grade_delay": NaN'),
    ],
)
def test_saved_json_validation(change):
    from mdanderson_stats.dose_schedule_desktop_study import read_dose_schedule_desktop_study

    with pytest.raises(ValueError):
        read_dose_schedule_desktop_study(change(study().to_json()))


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("<m_numDoses>3</m_numDoses>", "<m_numDoses>999999</m_numDoses>"),
        lambda s: s.replace(
            "<m_numSchedules>2</m_numSchedules>",
            "<m_numSchedules>2</m_numSchedules><m_numSchedules>2</m_numSchedules>",
        ),
        lambda s: "<!DOCTYPE foo>" + s,
        lambda s: s.replace('FileVersion="2.2.0"', 'FileVersion="99"'),
        lambda s: s.replace("<m_numAdmins>3</m_numAdmins>", "<m_numAdmins>2</m_numAdmins>"),
    ],
)
def test_original_xml_validation(change):
    from mdanderson_stats.dose_schedule_desktop_study import read_dose_schedule_desktop_study

    with pytest.raises(ValueError):
        read_dose_schedule_desktop_study(
            change((FIXTURES / "dose-schedule-desktop-study.xml").read_text())
        )


def test_extended_schedule_capacity_preserves_hard_bounds():
    from mdanderson_stats.dose_schedule import validate_schedules

    result = validate_schedules([list(range(10 * s)) for s in range(1, 6)], 100)
    assert len(result[-1]) == 50
    with pytest.raises(ValueError):
        validate_schedules([list(range(51))], 100)


def test_analysis_native_inputs_distributions_and_report_reopen():
    from dataclasses import replace

    from mdanderson_stats.dose_schedule_desktop_report import (
        dose_schedule_desktop_analysis_report,
        dose_schedule_desktop_distribution_json,
        read_dose_schedule_desktop_report,
    )
    from mdanderson_stats.dose_schedule_desktop_workflow import (
        DoseScheduleDesktopSampler,
        analyze_dose_schedule_desktop,
    )

    result = analyze_dose_schedule_desktop(
        replace(study(), study_time=60),
        seed=751,
        sampler=DoseScheduleDesktopSampler(draws=16, warmup=8, chains=2),
        final=True,
    )
    assert result.fit.regimen_risk.shape == (2, 16, 3, 2)
    assert result.final_ready
    report = dose_schedule_desktop_analysis_report(result)
    saved = read_dose_schedule_desktop_report(report)
    assert saved.study == result.study and saved.final and saved.sampler == result.sampler
    assert "Actual treatment ledger" in report and report.count("<svg") == 6
    draws = json.loads(dose_schedule_desktop_distribution_json(result))
    np.testing.assert_array_equal(draws["regimen_risk"], result.fit.regimen_risk)


def test_simulation_all_scenarios_reproducible_and_reported():
    from dataclasses import replace

    from mdanderson_stats.dose_schedule_desktop_report import (
        dose_schedule_desktop_simulation_report,
        read_dose_schedule_desktop_report,
    )
    from mdanderson_stats.dose_schedule_desktop_workflow import (
        DoseScheduleDesktopSampler,
        simulate_dose_schedule_desktop,
    )

    base = study()
    request = replace(
        base,
        patients=(),
        maximum_patients=2,
        repetitions=2,
        scenarios=(base.scenarios[0], tuple(tuple(0.8 for _ in row) for row in base.scenarios[0])),
    )
    sampler = DoseScheduleDesktopSampler(draws=8, warmup=0, chains=2)
    one = simulate_dose_schedule_desktop(request, seed=750, sampler=sampler)
    two = simulate_dose_schedule_desktop(request, seed=750, sampler=sampler)
    np.testing.assert_array_equal(one.selection_probability, two.selection_probability)
    np.testing.assert_allclose(one.selection_probability.sum(axis=1), 1)
    assert one.selection_probability.shape == (2, 7)
    assert np.all(one.mean_enrollment.sum(axis=(1, 2)) <= 2)
    assert np.all(one.pending_final_probability == 0)
    report = dose_schedule_desktop_simulation_report(one)
    assert "Scenario 2" in report and "Selection MCSE" in report
    saved = read_dose_schedule_desktop_report(report)
    assert saved.study == request and saved.kind == "simulation"


def test_final_readiness_rejects_incomplete_censoring():
    from dataclasses import replace

    from mdanderson_stats.dose_schedule_desktop_workflow import analyze_dose_schedule_desktop

    base = study()
    patient = replace(base.patients[0], outcome_time=10)
    with pytest.raises(ValueError, match="completed"):
        analyze_dose_schedule_desktop(
            replace(base, patients=(patient,), study_time=60), seed=75, final=True
        )


def test_simulation_preflight_budget_before_sampling():
    from dataclasses import replace

    from mdanderson_stats.dose_schedule_desktop_workflow import simulate_dose_schedule_desktop

    with pytest.raises(ValueError, match="budget"):
        simulate_dose_schedule_desktop(replace(study(), patients=(), repetitions=100000), seed=75)


@pytest.mark.parametrize("case", rows("schedule"))
def test_native_schedule_generation(case):
    from dataclasses import replace

    current = replace(
        study(),
        patients=(),
        schedule_count=case["schedule"] + 1,
        scenarios=(),
        administrations_per_cycle=case["admins"],
        cycle_interval=case["interval"],
        horizon=150,
    )
    np.testing.assert_array_equal(current.schedules()[-1], case["times"])
    assert case["doses"] == [case["dose"]] * len(case["times"])


@pytest.mark.parametrize("case", rows("final-clock"))
def test_native_final_clock(case):
    from dataclasses import replace

    from mdanderson_stats.dose_schedule_desktop_workflow import _trial_final_clock

    records = [
        replace(study().patients[0], entry_time=e, outcome_time=t)
        for e, t in zip(case["entries"], case["times"], strict=True)
    ]
    duration, analysis_time = _trial_final_clock(records, study(), 1000, "desktop_legacy")
    assert duration == case["duration"] and analysis_time == case["analysis_time"]


def test_legacy_final_pending_and_full_administration_ledger():
    from dataclasses import replace

    from mdanderson_stats import DoseScheduleDesktopSampler, replay_dose_schedule_desktop_trial

    current = replace(study(), patients=(), maximum_patients=1, low_grade_fraction=1.0)
    result = replay_dose_schedule_desktop_trial(
        current,
        np.ones((3, 2)),
        arrival_uniforms=[0.5],
        event_uniforms=[0.1],
        grade_uniforms=[0.1],
        posterior_seeds=[75, 76],
        profile="desktop_legacy",
        sampler=DoseScheduleDesktopSampler(draws=8, warmup=0, chains=2),
    )
    assert result.duration == 0.01
    assert result.analysis.study.study_time == 1.01
    assert result.analysis.pending_patients == (0,)
    assert result.analysis.study.patients[0].administration_times == (0, 1, 2)
    assert not result.analysis.final_ready


def test_safe_waits_for_low_grade_qualification():
    from dataclasses import replace

    from mdanderson_stats import DoseScheduleDesktopSampler, replay_dose_schedule_desktop_trial

    current = replace(study(), patients=(), maximum_patients=1, low_grade_fraction=1.0)
    result = replay_dose_schedule_desktop_trial(
        current,
        np.ones((3, 2)),
        arrival_uniforms=[0.5],
        event_uniforms=[0.1],
        grade_uniforms=[0.1],
        posterior_seeds=[75, 76],
        sampler=DoseScheduleDesktopSampler(draws=8, warmup=0, chains=2),
    )
    assert result.duration == 60
    assert not result.analysis.pending_patients and result.analysis.final_ready
    assert result.analysis.study.patients[0].administration_times == (0,)


@pytest.mark.parametrize("entry", [0.01, 0.7, 7.1, 18.213, 99.999, 1234.56789])
def test_exact_calendar_completion_avoids_cancellation(entry):
    result = dose_schedule_desktop_observation(
        outcome=0, outcome_time=60, entry_time=entry, study_time=entry + 60, horizon=60
    )
    assert result.final_ready and result.time == 60


def test_streamed_patient_detail_tapes_and_visible_study_parameters():
    from dataclasses import replace

    from mdanderson_stats import (
        DoseScheduleDesktopSampler,
        dose_schedule_desktop_simulation_report,
        dose_schedule_desktop_trial_json,
        simulate_dose_schedule_desktop,
    )

    captured = []

    def capture(scenario, trial, result):
        captured.append((scenario, trial, json.loads(dose_schedule_desktop_trial_json(result))))

    result = simulate_dose_schedule_desktop(
        replace(study(), patients=(), maximum_patients=1, repetitions=2),
        seed=752,
        sampler=DoseScheduleDesktopSampler(draws=8, warmup=0, chains=2),
        trial_callback=capture,
    )
    assert [(s, t) for s, t, _ in captured] == [(0, 0), (0, 1)]
    for _, _, data in captured:
        assert len(data["patients"]) == len(data["assignments"]) == 1
        assert len(data["posterior_seeds"]) == 2
        assert len(data["event_uniforms"]) == 1
    report = dose_schedule_desktop_simulation_report(result)
    assert "Captured study parameters" in report and "toxicity_limit" in report
    assert "Mean toxicities" in report and "Mean total enrollment" in report


def test_exported_tapes_replay_exact_trial():
    from dataclasses import replace

    from mdanderson_stats import (
        DoseScheduleDesktopSampler,
        dose_schedule_desktop_trial_json,
        read_dose_schedule_desktop_study,
        replay_dose_schedule_desktop_trial,
    )

    current = replace(study(), patients=(), maximum_patients=2)
    first = replay_dose_schedule_desktop_trial(
        current,
        current.scenarios[0],
        arrival_uniforms=[0.2, 0.7],
        event_uniforms=[0.8, 0.2],
        grade_uniforms=[0.2, 0.4],
        posterior_seeds=[1, 2, 3],
        sampler=DoseScheduleDesktopSampler(draws=8, warmup=0, chains=2),
    )
    data = json.loads(dose_schedule_desktop_trial_json(first))
    replay = replay_dose_schedule_desktop_trial(
        read_dose_schedule_desktop_study(json.dumps(data["study"])),
        data["truth"],
        arrival_uniforms=data["arrival_uniforms"],
        event_uniforms=data["event_uniforms"],
        grade_uniforms=data["grade_uniforms"],
        posterior_seeds=data["posterior_seeds"],
        profile=data["profile"],
        sampler=DoseScheduleDesktopSampler(**data["sampler"]),
        max_evaluations=data["max_evaluations"],
        max_work=data["max_work"],
    )
    assert replay.assignments == first.assignments
    np.testing.assert_array_equal(replay.analysis.fit.regimen_risk, first.analysis.fit.regimen_risk)
