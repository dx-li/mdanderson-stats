import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import U2OETCriteria, simulate_u2oet_trial, u2oet_scenario


def _prior():
    mean = np.zeros(12)
    mean[[1, 2, 7, 8]] = 0.5
    return dict(prior_mean=mean, prior_sd=np.full(12, 0.2), draws=24, warmup=12, chains=2)


def test_calendar_retains_pending_assignments_and_finishes_followup():
    scenario = u2oet_scenario(
        np.broadcast_to([0, 1], (2, 2, 2)), np.broadcast_to([1, 0], (2, 2, 2))
    )
    trial = simulate_u2oet_trial(
        [1, 2],
        [1, 2],
        scenario,
        [[10, 0], [100, 40]],
        initial=(0, 0),
        max_patients=6,
        cohort_size=3,
        mean_interarrival=1,
        efficacy_window=(1000, 1000),
        toxicity_window=(1000, 1000),
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=1),
        final_scope="tried",
        rng=np.random.default_rng(7717),
        **_prior(),
    )
    assert len(trial.patients.records) == 6 and len(trial.decisions) == 5
    assert_allclose(trial.patients.records[:3, 1:3], 1)
    assert trial.posterior_fits == 2  # same empty sufficient statistics throughout accrual
    assert not trial.stopped_early and trial.arrival_times[0] == 0
    assert np.all(np.diff(trial.arrival_times) > 0)
    assert trial.analysis_time == trial.arrival_times[-1] + 1000
    assert_allclose(trial.outcome_times, trial.arrival_times[:, None] + np.full((6, 2), 1000))
    assert trial.patients.complete.sum() == 6 and trial.patients.toxicity_only.sum() == 0
    assert trial.selected is not None and trial.patients.treated[trial.selected] > 0
    for i, decision in enumerate(trial.decisions, 1):
        assert decision.time == trial.arrival_times[i]
        assert decision.complete_patients == decision.toxicity_only_patients == 0
        assert decision.ignored_patients == i
        assert_allclose(decision.probabilities.sum(), 1)
    assert trial.data_seed != trial.posterior_seed
    assert not trial.arrival_times.flags.writeable


def test_toxicity_only_stop_does_not_enroll_or_resurrect_after_followup():
    scenario = u2oet_scenario(
        np.broadcast_to([0, 1], (2, 2, 2)), np.broadcast_to([0, 1], (2, 2, 2))
    )
    trial = simulate_u2oet_trial(
        [1, 2],
        [1, 2],
        scenario,
        [[10, 0], [100, 40]],
        initial=(0, 0),
        max_patients=6,
        mean_interarrival=1,
        efficacy_window=(1000, 1000),
        toxicity_window=(0, 0),
        criteria=U2OETCriteria(1, 1, min_efficacy=0, max_toxicity=0, toxicity_cutoff=0),
        rng=np.random.default_rng(7718),
        **_prior(),
    )
    assert trial.stopped_early and trial.selected is None
    assert len(trial.patients.records) == 1 and len(trial.decisions) == 1
    assert trial.decisions[0].toxicity_only_patients == 1
    assert trial.decisions[0].complete_patients == trial.decisions[0].ignored_patients == 0
    assert trial.decisions[0].probabilities.sum() == 0
    assert trial.stop_time == trial.decisions[0].time
    assert trial.analysis_time == 1000 and trial.patients.complete.sum() == 1
