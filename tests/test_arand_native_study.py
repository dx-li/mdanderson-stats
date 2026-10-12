import json
from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats.arand_native import ArandNativeArmLimits, ArandNativeMonitor
from mdanderson_stats.arand_native_study import (
    ArandNativeStudy,
    arand_native_report,
    arand_native_study_json,
    read_arand_native_study,
    simulate_arand_native,
    simulate_arand_native_trial,
)
from mdanderson_stats.arand_native_trial import ArandNativeRules


def study(**kwargs):
    return ArandNativeStudy(
        prior=((1, 1), (1, 1)),
        rules=ArandNativeRules(
            (ArandNativeArmLimits(),) * 2,
            maximum_enrollment=6,
            lower=0,
            upper=1,
            runin=5,
            power=1,
            stop_when_one_arm=False,
        ),
        monitor=ArandNativeMonitor(),
        true_parameters=(0.3, 0.7),
        accrual_rate=2,
        max_candidates=10,
        **kwargs,
    )


def test_saved_inputs_report_and_seed_replay():
    config = study()
    assert read_arand_native_study(arand_native_study_json(config)) == config
    result = simulate_arand_native(config, trials=3, seed=728)
    runs = [simulate_arand_native_trial(config, seed=s) for s in result.trial_seeds]
    counts = np.array([[sum(r.arm == arm for r in t.records) for arm in range(2)] for t in runs])
    # Forced run-in rounds 5 up to 6 and achieves 3 patients in each arm.
    np.testing.assert_array_equal(counts, [[3, 3]] * 3)
    assert result.completed == 3
    assert sum(result.selected) + result.no_selection == 3
    np.testing.assert_array_equal(result.mean_enrollment, [3, 3])
    np.testing.assert_array_equal(result.enrollment_mcse, [0, 0])
    assert all(t.final.time == t.decisions[-1].time for t in runs)
    report = arand_native_report(result)
    assert "type 7" in report and "728" not in result.study_json
    assert str(result.trial_seeds[0]) in report
    assert "&quot;schema&quot;" in report
    with pytest.raises(ValueError):
        result.mean_enrollment[0] = 1


def test_binary_delay_and_survival_censor_generator():
    config = study(binary_delay=2)
    trial = simulate_arand_native_trial(config, seed=77)
    np.testing.assert_allclose([r.record_time - r.arrival for r in trial.records], 2)
    survival = replace(config, family="exponential", true_parameters=(1, 2), censor_time=0.2)
    trial = simulate_arand_native_trial(survival, seed=77)
    assert any(r.indicator == 0 for r in trial.records)
    assert all(r.record_time - r.arrival <= 0.2 + 1e-14 for r in trial.records)
    median = replace(
        survival,
        parameter="median",
        true_parameters=tuple(np.asarray(survival.true_parameters) * np.log(2)),
        prior=((1, np.log(2)),) * 2,
    )
    other = simulate_arand_native_trial(median, seed=77)
    assert [r.arm for r in trial.records] == [r.arm for r in other.records]
    np.testing.assert_allclose(
        [r.record_time for r in trial.records], [r.record_time for r in other.records]
    )


def test_single_trial_undefined_mcse():
    result = simulate_arand_native(study(), trials=1, seed=12)
    assert np.isnan(result.enrollment_mcse).all() and np.isnan(result.duration_mcse)
    assert "nan" in arand_native_report(result)


@pytest.mark.parametrize(
    "mutation", ["version", "duplicate", "extra", "negative", "shape", "family"]
)
def test_invalid_saved_studies(mutation):
    text = arand_native_study_json(study())
    data = json.loads(text)
    if mutation == "version":
        data["version"] = 2
    elif mutation == "duplicate":
        text = text.replace('"version": 1', '"version": 1, "version": 1')
    elif mutation == "extra":
        data["study"]["secret_unknown"] = 1
    elif mutation == "negative":
        data["study"]["accrual_rate"] = -1
    elif mutation == "shape":
        data["study"]["prior"] = [[1, 1]]
    elif mutation == "family":
        data["study"]["family"] = "unknown"
    with pytest.raises(ValueError):
        read_arand_native_study(text if mutation == "duplicate" else json.dumps(data))


def test_candidate_and_work_budgets():
    with pytest.raises(ValueError, match="exhausted"):
        simulate_arand_native_trial(replace(study(), max_candidates=1), seed=1)
    with pytest.raises(ValueError, match="bounded"):
        simulate_arand_native(replace(study(), max_candidates=10000), trials=101, seed=1)


def test_poisson_duration_overshoot_identity():
    # With a candidate at zero, enrollment ~1+Poisson(.5); the first candidate
    # beyond the horizon stops the trial, so mean stopping time is .25+1/2.
    config = ArandNativeStudy(
        prior=((1, 1),),
        rules=ArandNativeRules(
            (ArandNativeArmLimits(),),
            maximum_duration=0.25,
            maximum_enrollment=100,
            lower=0,
            upper=1,
            stop_when_one_arm=False,
        ),
        monitor=ArandNativeMonitor(),
        true_parameters=(0.5,),
        accrual_rate=2,
        max_candidates=30,
    )
    result = simulate_arand_native(config, trials=512, seed=81)
    assert abs(result.mean_enrollment[0] - 1.5) < 6 * np.sqrt(0.5 / 512)
    assert abs(result.duration_mean - 0.75) < 6 * 0.5 / np.sqrt(512)
    assert result.completed == 512
