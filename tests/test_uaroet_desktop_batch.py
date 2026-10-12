"""Job partition reproducibility, merge restrictions, portable summaries and MCSE."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import mdanderson_stats.uaroet_desktop_batch as batch
from mdanderson_stats import (
    UAROETCalendarTrial,
    UAROETDesktopScenario,
    UAROETTimedPatient,
    read_uaroet_desktop_study,
)


def design():
    return replace(
        read_uaroet_desktop_study(
            (Path(__file__).parent / "fixtures/uaroet-python-study.json").read_text()
        ),
        patients=(),
    )


def scenario():
    return UAROETDesktopScenario(1, [[0.5, 0.5]] * 2, [[0.8, 0.2]] * 2, 0.2)


def fake_trial(s, scenario, *, rng, profile, **kwargs):
    dose, efficacy, toxicity = [int(v) for v in rng.integers(0, 2, size=3)]
    stopped = bool(rng.random() < 0.25)
    return UAROETCalendarTrial(
        scenario.identifier,
        profile,
        None if stopped else dose,
        stopped,
        (UAROETTimedPatient(1, dose, efficacy, toxicity, 0, 2, 3),),
        (),
        3.0,
        1.0 if stopped else 3.0,
        0,
    )


def job(**kwargs):
    return batch.simulate_uaroet_desktop(design(), scenario(), seed=92, **kwargs)


def test_split_jobs_preserve_streams_and_merge_matches_single_job(monkeypatch):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)
    all_trials, split_trials = [], []
    all_job = job(repetitions=20, on_trial=lambda i, t: all_trials.append((i, t)))
    a = job(repetitions=7, on_trial=lambda i, t: split_trials.append((i, t)))
    b = job(repetitions=13, first_trial=7, on_trial=lambda i, t: split_trials.append((i, t)))
    combined = batch.merge_uaroet_desktop_batches(b, a)
    assert all_trials == split_trials
    assert combined.totals == all_job.totals
    assert combined.squared_totals == all_job.squared_totals
    assert combined.summary() == all_job.summary()
    assert combined.trials == 20
    assert batch.read_uaroet_desktop_batch(combined.to_json()) == combined
    assert len(combined.to_csv().splitlines()) == len(combined.metrics) + 1


def test_report_mcse_matches_independent_trial_sample_standard_error(monkeypatch):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)
    trials = []
    result = job(repetitions=20, on_trial=lambda i, t: trials.append(t))
    sample = np.array([t.selected is None for t in trials], dtype=float)
    assert result.summary()["no_selection"]["mean"] == sample.mean()
    assert result.summary()["no_selection"]["mcse"] == pytest.approx(
        sample.std(ddof=1) / np.sqrt(20)
    )
    one = job(repetitions=1)
    assert all(row["mcse"] is None for row in one.summary().values())


@pytest.mark.parametrize("first,count", [(0, 1), (3, 4), (6, 5)])
def test_overlapping_streams_rejected(monkeypatch, first, count):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)
    a = job(repetitions=7)
    b = job(repetitions=count, first_trial=first)
    with pytest.raises(ValueError, match="overlapping"):
        batch.merge_uaroet_desktop_batches(a, b)


def test_different_computation_settings_cannot_merge(monkeypatch):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)
    a = job(repetitions=2)
    b = job(repetitions=2, first_trial=2, maximum_rhat=1.1)
    with pytest.raises(ValueError, match="identical"):
        batch.merge_uaroet_desktop_batches(a, b)


@pytest.mark.parametrize("mutation", ["hash", "counts", "square", "duplicate", "grid", "schema"])
def test_corrupted_jobs_rejected(monkeypatch, mutation):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)
    value = json.loads(job(repetitions=2).to_json())
    if mutation == "hash":
        value["configuration_sha256"] = "0" * 64
    elif mutation == "counts":
        value["totals"][-1] += 1
    elif mutation == "square":
        value["squared_totals"][0] = -1
    elif mutation == "duplicate":
        value["segments"] *= 2
    elif mutation == "grid":
        value["configuration"]["grid"] = [10000000, 2, 2]
    else:
        value["version"] = True
    with pytest.raises(ValueError):
        batch.read_uaroet_desktop_batch(json.dumps(value))


def test_callback_failure_does_not_return_completed_job(monkeypatch):
    monkeypatch.setattr(batch, "run_uaroet_desktop_trial", fake_trial)

    def fail(index, trial):
        raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        job(repetitions=2, on_trial=fail)
