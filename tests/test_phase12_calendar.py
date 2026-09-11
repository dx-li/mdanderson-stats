import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import phase12_accrual_ready, simulate_phase12_calendar


def _simulate(seed, **kwargs):
    return simulate_phase12_calendar(
        [0] * 6, [0] * 6, draws=16, warmup=10, chains=2, rng=np.random.default_rng(seed), **kwargs
    )


def test_rounding_blocked_attempts_and_source_incomplete_final_followup():
    source = _simulate(8551, max_patients=20)
    complete = _simulate(8551, max_patients=20, complete_followup=True)
    assert source.phase_two_start == 18
    assert len(source.records) == 20 and source.reason == "maximum enrollment"
    assert_allclose(source.records, complete.records)
    assert source.final_analysis_time == source.records[-1, 1]
    assert complete.final_analysis_time == complete.records[-1, 1] + 84
    assert source.analyses[-1].snapshot.pending_efficacy.sum() > 0
    assert_allclose(complete.analyses[-1].snapshot.pending_efficacy, 0)
    assert_allclose(complete.analyses[-1].snapshot.pending_toxicity, 0)
    assert np.any(source.attempts[:, 3] == 1)
    assert_allclose(source.records[:, 3] - source.records[:, 1], 84)
    assert_allclose(source.records[:, 5] - source.records[:, 1], 28)
    assert_allclose(source.attempts[:, 0], np.floor(source.attempts[:, 0]))
    for time, enrollment, phase, blocked in source.attempts:
        history = source.records[: int(enrollment)]
        ready = phase12_accrual_ready(
            history, time=time, phase_two_start=18 if phase else None, max_patients=20
        )
        assert ready != bool(blocked)
    assert source.data_seed != source.posterior_seed
    assert not source.records.flags.writeable


def test_phase_two_decisions_and_toxicity_stop_do_not_enroll_extra_patients():
    trial = simulate_phase12_calendar(
        [0] * 6,
        [0.4] * 6,
        max_patients=30,
        efficacy_window=5,
        toxicity_window=2,
        draws=16,
        warmup=10,
        chains=2,
        rng=np.random.default_rng(8552),
    )
    assert trial.phase_two_start is not None
    interim = [a for a in trial.analyses if a.decision is not None]
    assert interim
    for analysis in interim:
        observed = trial.records[trial.records[:, 1] <= analysis.time]
        # Enrollment times may tie: snapshot includes only patients actually
        # enrolled before the analysis, whose count is recorded in its snapshot.
        observed = observed[: int(analysis.snapshot.enrolled.sum())]
        for dose in range(6):
            rows = observed[observed[:, 0] == dose]
            for event, col in [(2, 0), (4, 2)]:
                available = rows[:, event + 1] <= analysis.time
                assert_allclose(
                    analysis.snapshot.tally[dose, col : col + 2],
                    [
                        np.sum(available & (rows[:, event] == 0)),
                        np.sum(available & (rows[:, event] == 1)),
                    ],
                )
        assert (analysis.snapshot.enrolled.sum() - trial.phase_two_start) % 5 == 0
    toxic = simulate_phase12_calendar(
        [1] * 6,
        [0] * 6,
        draws=8,
        warmup=0,
        chains=2,
        rng=np.random.default_rng(8553),
        complete_followup=True,
    )
    assert toxic.reason == "phase-I toxicity"
    assert len(toxic.records) <= 3
    assert toxic.last_fit is None and toxic.final_analysis_time is None
    assert toxic.early_selected is None and toxic.future_selected is None
