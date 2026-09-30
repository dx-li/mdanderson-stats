import copy

import numpy as np
import pytest

import mdanderson_stats.parallel_phase12_calendar as calendar_module
from mdanderson_stats.parallel_phase12_calendar import simulate_phase12_calendar


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
    assert all(a.posterior_backend == "mcmc" for a in mcmc.analyses)
    assert all(a.max_split_rhat is not None for a in mcmc.analyses)


def test_importance_fit_drives_phase_two_and_cached_analysis_history(monkeypatch):
    original = calendar_module.fit_phase12_importance
    fit_tallies = []

    def counting_fit(tally, **kwargs):
        fit_tallies.append(np.asarray(tally).copy())
        return original(tally, **kwargs)

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
