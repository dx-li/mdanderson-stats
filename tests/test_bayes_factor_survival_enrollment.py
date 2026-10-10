import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logit

import mdanderson_stats.bayes_factor_survival_enrollment as enrollment
from mdanderson_stats import (
    bayes_factor_survival,
    bayes_factor_survival_day_boundaries,
    bayes_factor_survival_enrollment_trial,
    simulate_bayes_factor_survival_enrollment,
)


def test_original_managed_control_with_external_bayes_factor_tape(monkeypatch):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/bayes-factor-tte-control.json").read_text()
    )
    for case in fixture["cases"]:
        tape = iter(case["probabilities"])
        monkeypatch.setattr(enrollment, "_monitor", lambda *args: float(logit(next(tape))))
        result = bayes_factor_survival_enrollment_trial(
            case["ledger"]["timeEnrollDays"],
            [1, 0, 3, 2],
            null_median_months=4,
            alternative_median_months=5.5,
        )
        assert result.patients == case["result"]["patients"]
        assert result.history[-1].hypothesis == case["result"]["hypothesis"]
        for actual, original in zip(result.history, case["history"], strict=True):
            assert actual.patients == original["patients"]
            assert actual.events == original["events"]
            assert actual.exposure_days == original["exposure_days"]
        assert not result.enrollment_days.flags.writeable


def test_native_schedule_zero_duration_ties_and_no_extra_final_followup(tmp_path):
    trial = bayes_factor_survival_enrollment_trial(
        [0, 0, 2, 3],
        [1, 0, 3, 2],
        null_median_months=4,
        alternative_median_months=5.5,
        inferiority_cutoff=0,
        superiority_cutoff=1,
    )
    assert [(row.patients, row.events, row.exposure_days) for row in trial.history] == [
        (2, 0, 0),
        (3, 2, 1),
        (4, 2, 2),
    ]
    assert trial.history[-1].calendar_day == 3
    assert trial.decision == "inconclusive"
    # Independent continuous-coordinate BF equals the native workflow at positive exposure.
    for row in trial.history[1:]:
        direct = bayes_factor_survival(
            row.events,
            row.exposure_days,
            null_median=4 * 30.4375,
            alternative_median_mode=5.5 * 30.4375,
            inferiority_cutoff=0,
            superiority_cutoff=1,
        )
        np.testing.assert_allclose(row.log_bayes_factor, direct.log_bayes_factor, atol=1e-12)
    saved = json.loads(trial.write_json(tmp_path / "trial.json").read_text())
    assert saved["patients"] == 4
    tied = bayes_factor_survival_enrollment_trial(
        [0, 2, 3],
        [2, 0, 0],
        null_median_months=4,
        alternative_median_months=5.5,
        inferiority_cutoff=0,
        superiority_cutoff=1,
    )
    assert tied.history[0].events == 0  # event exactly at the enrollment check
    assert tied.history[1].events == 2


def test_generated_source_exponential_truncation_order_and_replay(tmp_path):
    kwargs = dict(
        null_median_months=4,
        alternative_median_months=5.5,
        true_median_months=4,
        accrual_rate_per_month=2,
        max_patients=5,
        repetitions=3,
        seed=89,
        inferiority_cutoff=0,
        superiority_cutoff=1,
    )
    study = simulate_bayes_factor_survival_enrollment(**kwargs)
    assert study.to_json() == simulate_bayes_factor_survival_enrollment(**kwargs).to_json()
    for seed, trial in zip(study.trial_seeds, study.trials, strict=True):
        original = np.random.Generator(np.random.PCG64(seed))
        days = original.exponential([4 / np.log(2) * 30.4375, 30.4375 / 2], size=(5, 2)).astype(int)
        np.testing.assert_array_equal(trial.enrollment_days, np.r_[0, np.cumsum(days[:-1, 1])])
        np.testing.assert_array_equal(trial.event_duration_days, days[:, 0])
    np.testing.assert_array_equal(study.stopping_probability, [0, 0, 1])
    assert study.mean_patients == 5
    assert study.patient_quantile(0.9) == 5
    assert json.loads(study.write_json(tmp_path / "study.json").read_text())["seed"] == 89


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_patients": 1},
        {"seed": True},
        {"seed": -1},
        {"true_median_months": 0},
        {"accrual_rate_per_month": 0},
        {"max_total_work": 1},
        {"max_total_quadratures": 1},
        {"max_storage_bytes": 1},
    ],
)
def test_simulation_preflight(kwargs):
    controls = dict(
        null_median_months=4,
        alternative_median_months=5.5,
        true_median_months=4,
        accrual_rate_per_month=2,
    )
    controls.update(kwargs)
    with pytest.raises(ValueError):
        simulate_bayes_factor_survival_enrollment(**controls)


@pytest.mark.parametrize(
    "arrivals,durations",
    [
        ([0], [1]),
        ([1, 2], [1, 1]),
        ([0, 2, 1], [1, 1, 1]),
        ([0, 0.5], [1, 1]),
        ([0, 1], [-1, 1]),
        ([0, 1], [True, False]),
        ([0, 1], [1, 2**31 - 1]),
        (list(range(501)), [1] * 501),
        ([0, True], [1, 1]),
    ],
)
def test_native_day_tape_rejection(arrivals, durations):
    with pytest.raises(ValueError):
        bayes_factor_survival_enrollment_trial(
            arrivals, durations, null_median_months=4, alternative_median_months=5.5
        )


def test_integer_boundary_comparisons_match_direct_evidence_on_neighbor_days():
    lower, upper = bayes_factor_survival_day_boundaries(
        max_patients=7, null_median_months=4, alternative_median_months=5.5
    )
    for d in range(7):
        for boundary, decision in ((lower[d], "inferiority"), (upper[d], "superiority")):
            for day in (max(0, boundary - 1), boundary, boundary + 1):
                if day == 0 and d:
                    log_bf = enrollment._monitor(
                        d, 0, 4 * 30.4375, enrollment._parameters(4, 5.5, 0.15, 0.8)[1]
                    )
                    actual = enrollment._hypothesis(log_bf, 1, 2, 0.15, 0.8)
                    expected = 0 if decision == "inferiority" else 2
                    selected = actual == expected
                else:
                    state = bayes_factor_survival(
                        d, day, null_median=4 * 30.4375, alternative_median_mode=5.5 * 30.4375
                    )
                    selected = str(state.decision) == decision
                assert selected == (
                    day < boundary if decision == "inferiority" else day >= boundary
                )


def test_original_managed_integer_search_convention_with_independent_evidence(monkeypatch):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/bayes-factor-tte-control.json").read_text()
    )
    # Original managed integer search runs against the external BF=day/(events+1).
    # Supply its analytic continuous roots to the Python discretization only.
    roots = SimpleNamespace(
        inferiority_time=0.15 / 0.85 * np.arange(1, 5), superiority_time=4 * np.arange(1, 5)
    )
    monkeypatch.setattr(enrollment, "bayes_factor_survival_boundaries", lambda *a, **kw: roots)
    low, high = bayes_factor_survival_day_boundaries(
        max_patients=4, null_median_months=4, alternative_median_months=5.5
    )
    np.testing.assert_array_equal(low, fixture["integer_boundaries"]["inferiority"])
    np.testing.assert_array_equal(high, fixture["integer_boundaries"]["superiority"])
