import numpy as np
import pytest

from mdanderson_stats.dose_schedule import dose_schedule_patient_loglikelihood
from mdanderson_stats.dose_schedule_fit import fit_dose_schedule
from mdanderson_stats.dose_schedule_observation import (
    DoseScheduleToxicityEpisode,
    observe_dose_schedule_patient,
)
from mdanderson_stats.dose_schedule_prior import DoseSchedulePrior


def _observe(episodes, *, as_of, horizon=30, administrations=(2, 10, 11, 24), doses=(0, 0, 1, 1)):
    return observe_dose_schedule_patient(
        episodes,
        as_of=as_of,
        horizon=horizon,
        administration_times=administrations,
        dose_indices=doses,
    )


def test_grade2_onset_is_backdated_after_explicit_adjudication():
    episode = DoseScheduleToxicityEpisode(10, 24, True)
    before = _observe([episode], as_of=23)
    assert not before.patient.event
    assert before.patient.time == 23
    assert not before.final_ready
    assert before.pending_episode_indices == (0,)
    assert before.episode_statuses[0].adjudication_time is None
    assert before.episode_statuses[0].qualifies is None
    assert before.delivered_administration_times.tolist() == [2, 10, 11]

    after = _observe([episode], as_of=24)
    assert after.patient.event
    assert after.patient.time == 10
    assert after.final_ready
    assert after.qualifying_episode_index == 0
    assert after.patient.administration_times.tolist() == [2]
    assert after.delivered_administration_times.tolist() == [2, 10, 11, 24]
    assert after.delivered_dose_indices.dtype == np.int64
    assert not after.delivered_dose_indices.flags.writeable
    # The data likelihood sees the onset event, while later delivered doses
    # remain available in the separate actual-history ledger.
    assert np.isfinite(
        dose_schedule_patient_loglikelihood(after.patient, [0.1, 0.2], [2, 2], [8, 8])
    )
    prior = DoseSchedulePrior([-2.0, np.log(2.0), np.log(10.0)], [0.3, 0.0, 0.0], dose_count=1)
    fit = fit_dose_schedule(
        [after.patient],
        prior,
        [[0.0], [0.0, 2.0]],
        horizon=30.0,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(412),
    )
    assert np.all(np.isfinite(fit.log_likelihood))


def test_future_onsets_and_adjudications_are_masked_at_interim_time():
    earlier_input = [
        DoseScheduleToxicityEpisode(4, 18, True),
        DoseScheduleToxicityEpisode(20, 28, False),
    ]
    later_input = earlier_input + [DoseScheduleToxicityEpisode(29, 35, True)]
    first = _observe(earlier_input, as_of=12)
    second = _observe(later_input, as_of=12)
    assert first.patient.time == second.patient.time
    assert first.patient.event == second.patient.event
    np.testing.assert_array_equal(
        first.patient.administration_times, second.patient.administration_times
    )
    np.testing.assert_array_equal(first.patient.dose_indices, second.patient.dose_indices)
    assert first.observation_end == second.observation_end == 12
    assert first.episode_statuses == second.episode_statuses
    assert first.pending_episode_indices == second.pending_episode_indices == (0,)
    assert first.episode_statuses[0].onset_time == 4
    assert first.episode_statuses[0].adjudication_time is None
    assert not first.patient.event


def test_known_event_uses_earliest_onset_and_waits_for_earlier_pending_episode():
    episodes = [
        DoseScheduleToxicityEpisode(8, None, None),
        DoseScheduleToxicityEpisode(12, 13, True),
    ]
    interim = _observe(episodes, as_of=13)
    assert interim.patient.event and interim.patient.time == 12
    assert not interim.final_ready
    assert interim.pending_episode_indices == (0,)

    revised = _observe([DoseScheduleToxicityEpisode(8, 20, True), episodes[1]], as_of=20)
    assert revised.patient.event and revised.patient.time == 8
    assert revised.qualifying_episode_index == 0
    assert revised.final_ready


def test_horizon_requires_onset_adjudication_but_ignores_later_episodes():
    crossing = DoseScheduleToxicityEpisode(9, 14, True)
    pending_at_horizon = _observe([crossing], as_of=10, horizon=10)
    assert not pending_at_horizon.patient.event
    assert pending_at_horizon.patient.time == 10
    assert not pending_at_horizon.final_ready

    after_adjudication = _observe([crossing], as_of=14, horizon=10)
    assert after_adjudication.patient.event and after_adjudication.patient.time == 9
    assert after_adjudication.final_ready

    later_pending = _observe([DoseScheduleToxicityEpisode(11, None, None)], as_of=15, horizon=10)
    assert later_pending.pending_episode_indices == (0,)
    assert later_pending.final_ready


def test_onset_administration_tie_is_excluded_from_likelihood_history():
    result = observe_dose_schedule_patient(
        [DoseScheduleToxicityEpisode(10, 10, True)],
        as_of=10,
        horizon=20,
        administration_times=[2, 10],
        dose_indices=[0, 1],
    )
    assert result.patient.administration_times.tolist() == [2]
    assert result.delivered_administration_times.tolist() == [2, 10]


def test_episode_pairing_and_time_order_are_validated():
    with pytest.raises(ValueError, match="supplied together"):
        DoseScheduleToxicityEpisode(2, adjudication_time=3)
    with pytest.raises(ValueError, match="cannot precede"):
        DoseScheduleToxicityEpisode(2, adjudication_time=1, qualifies=True)
    with pytest.raises(ValueError, match="nondecreasing"):
        observe_dose_schedule_patient(
            [], as_of=2, horizon=3, administration_times=[2, 1], dose_indices=[0, 0]
        )
