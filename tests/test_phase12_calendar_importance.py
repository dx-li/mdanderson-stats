import copy
from dataclasses import replace

import numpy as np
import pytest

import mdanderson_stats.parallel_phase12_calendar as calendar_module
from mdanderson_stats.parallel_phase12_calendar import simulate_phase12_calendar
from mdanderson_stats.parallel_phase12_summary import summarize_phase12_importance_fits


def _run(seed: int, *, backend: str, **kwargs):
    return simulate_phase12_calendar(
        [0.0] * 6,
        [0.0] * 6,
        max_patients=18,
        posterior_backend=backend,
        importance_max_integrations=100,
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def test_importance_calendar_replays_and_records_per_analysis_diagnostics():
    first = _run(20260929, backend="importance")
    second = _run(20260929, backend="importance")
    mcmc = _run(20260929, backend="mcmc", draws=16, warmup=8, chains=2)

    np.testing.assert_array_equal(first.records, second.records)
    np.testing.assert_array_equal(first.records, mcmc.records)
    assert first.data_seed == second.data_seed == mcmc.data_seed
    assert first.posterior_seed == second.posterior_seed
    # This cap is reached before the next attempted arrival can transition the
    # phase-I state into phase II, so posterior decisions cannot alter treatment.
    assert first.phase_two_start is None and len(first.records) == 18
    assert first.posterior_component_evaluations == sum(
        analysis.posterior_integrations * 67
        for analysis in first.analyses
        if analysis.posterior_integrations is not None
    )
    assert first.posterior_mode_iterations == sum(
        analysis.posterior_mode_iterations
        for analysis in first.analyses
        if analysis.posterior_mode_iterations is not None
    )
    assert first.analyses[-1].posterior_backend == "importance"
    assert first.analyses[-1].max_split_rhat is None
    assert first.analyses[-1].posterior_integrations is not None
    assert first.analyses[-1].posterior_converged in (True, False)
    assert first.analyses[-1].posterior_log_evidence is not None
    assert first.analyses[-1].log_integral_mc_se.shape == (67,)
    assert first.analyses[-1].ratio_mc_se.shape == (66,)
    assert first.last_fit is not None
    assert first.last_fit.__class__.__name__ == "Phase12ImportanceFit"
    assert first.posterior_probability_summary is not None
    assert first.posterior_probability_summary.analysis_call_count == len(first.analyses)
    assert first.posterior_refit_count >= 1
    assert all(a.posterior_backend == "mcmc" for a in mcmc.analyses)
    assert all(a.max_split_rhat is not None for a in mcmc.analyses)
    assert mcmc.posterior_probability_summary is None
    assert mcmc.posterior_refit_count == len(mcmc.analyses)


def test_importance_fit_drives_phase_two_and_cached_analysis_history(monkeypatch):
    original = calendar_module.fit_phase12_importance
    fit_tallies = []
    fits = []

    def counting_fit(tally, **kwargs):
        fit_tallies.append(np.asarray(tally).copy())
        fit = original(tally, **kwargs)
        fits.append(fit)
        return fit

    monkeypatch.setattr(calendar_module, "fit_phase12_importance", counting_fit)
    trial = simulate_phase12_calendar(
        [0.0] * 6,
        [0.2] * 6,
        max_patients=24,
        efficacy_window=1,
        toxicity_window=100,
        posterior_backend="importance",
        importance_max_integrations=100,
        rng=np.random.default_rng(3),
    )
    interim = [analysis for analysis in trial.analyses if analysis.decision is not None]
    final = [analysis for analysis in trial.analyses if analysis.decision is None]
    assert interim and final
    assert any(analysis.snapshot.pending_toxicity.sum() > 0 for analysis in interim)
    assert all(analysis.posterior_backend == "importance" for analysis in trial.analyses)

    expected_fits = []
    previous = None
    for analysis in trial.analyses:
        if previous is None or not np.array_equal(analysis.snapshot.tally, previous):
            expected_fits.append(analysis)
            previous = analysis.snapshot.tally
    assert len(expected_fits) < len(trial.analyses)
    assert len(fit_tallies) == len(expected_fits)
    for actual, expected in zip(fit_tallies, expected_fits, strict=True):
        np.testing.assert_array_equal(actual, expected.snapshot.tally)
    assert trial.posterior_component_evaluations == sum(
        analysis.posterior_integrations * 67 for analysis in expected_fits
    )
    assert trial.posterior_refit_count == len(expected_fits)
    assert trial.posterior_probability_summary is not None
    assert trial.posterior_probability_summary.analysis_call_count == len(trial.analyses)
    replayed_call_fits = []
    cached_fit = None
    fit_index = 0
    previous_tally = None
    for analysis in trial.analyses:
        if previous_tally is None or not np.array_equal(analysis.snapshot.tally, previous_tally):
            cached_fit = fits[fit_index]
            fit_index += 1
            previous_tally = analysis.snapshot.tally
        assert cached_fit is not None
        replayed_call_fits.append(cached_fit)
    expected_summary = summarize_phase12_importance_fits(replayed_call_fits)
    np.testing.assert_allclose(
        trial.posterior_probability_summary.component_means,
        expected_summary.component_means,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        trial.posterior_probability_summary.component_sample_variances,
        expected_summary.component_sample_variances,
        rtol=0,
        atol=0,
    )


def test_early_stop_keeps_latest_successful_fit_in_automatic_summaries(monkeypatch):
    original = calendar_module.fit_phase12_importance
    seed_fit = original(np.zeros((6, 4)), max_integrations=100, rng=np.random.default_rng(0))
    stopped_fit = replace(
        seed_fit,
        reference_superiority=np.full(6, 0.5),
        efficacy_probability=np.zeros(6),
        future_probability=np.zeros(6),
        pairwise_superiority=np.zeros((6, 6)),
        toxicity_probability=np.ones(6),
    )
    monkeypatch.setattr(
        calendar_module, "fit_phase12_importance", lambda *_args, **_kwargs: stopped_fit
    )
    trial = simulate_phase12_calendar(
        [0.0] * 6,
        [0.0] * 6,
        max_patients=24,
        efficacy_window=1,
        toxicity_window=100,
        posterior_backend="importance",
        importance_max_integrations=100,
        rng=np.random.default_rng(3),
    )

    assert trial.reason == "futility"
    assert trial.phase_two_start is not None
    assert trial.final_analysis_time is None
    assert trial.last_fit is stopped_fit
    assert trial.posterior_refit_count == 1
    assert trial.posterior_probability_summary is not None
    assert trial.posterior_probability_summary.analysis_call_count == 1
    np.testing.assert_array_equal(
        trial.posterior_probability_summary.component_sample_variances, np.zeros(60)
    )


def test_importance_calendar_work_preflight_does_not_advance_rng():
    rng = np.random.default_rng(42)
    state_before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="worst-case importance component work"):
        simulate_phase12_calendar(
            [0.0] * 6,
            [0.0] * 6,
            max_patients=20,
            posterior_backend="importance",
            importance_max_integrations=100,
            max_total_posterior_component_evaluations=6_700,
            rng=rng,
        )
    assert rng.bit_generator.state == state_before


def test_mcmc_calendar_default_does_not_apply_importance_preflight():
    trial = simulate_phase12_calendar(
        [0.0] * 6,
        [0.0] * 6,
        max_patients=18,
        max_total_posterior_component_evaluations=67,
        draws=16,
        warmup=8,
        chains=2,
        rng=np.random.default_rng(43),
    )
    assert trial.last_fit is not None
    assert trial.posterior_component_evaluations == 0
    assert trial.posterior_mode_iterations == 0
    assert trial.posterior_probability_summary is None
    assert trial.posterior_refit_count == 1
