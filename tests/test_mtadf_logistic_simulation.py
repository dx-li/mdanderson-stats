import numpy as np
import pytest

from mdanderson_stats.mtadf import MTADFPrior
from mdanderson_stats.mtadf_logistic_simulation import (
    MTADFLogisticSimulationConfig,
    simulate_mtadf_logistic,
    simulate_mtadf_logistic_trial,
)


def _config(model: str) -> MTADFLogisticSimulationConfig:
    return MTADFLogisticSimulationConfig(
        model=model,  # type: ignore[arg-type]
        true_toxicity=[0.05, 0.12, 0.2],
        true_efficacy=[0.15, 0.45, 0.65],
        doses=[-1.0, 0.0, 1.0],
        cohorts=3,
        cohort_size=2,
        window_length=2,
        draws=32,
        warmup=32,
        chains=2,
    )


@pytest.mark.parametrize("model", ["global", "local"])
def test_logistic_simulation_seed_pairs_replay_counts_and_conserve_cohorts(model):
    config = _config(model)
    result = simulate_mtadf_logistic(config, trials=3, rng=927)
    for index in range(3):
        replay = simulate_mtadf_logistic_trial(
            config,
            outcome_seed=int(result.trial_seeds[index, 0]),
            sampler_seed=int(result.trial_seeds[index, 1]),
        )
        np.testing.assert_array_equal(replay.subjects, result.subjects[index])
        np.testing.assert_array_equal(replay.toxicities, result.toxicities[index])
        np.testing.assert_array_equal(replay.responses, result.responses[index])
        assert replay.selected_dose == result.selected_dose[index]
        assert replay.stop_reason == result.stop_reason[index]
        assert replay.posterior_fit_count == result.posterior_fit_count[index]
        if result.subjects[index].sum() > 0:
            assert result.subjects[index].sum() % config.cohort_size == 0
        assert np.all(result.toxicities[index] <= result.subjects[index])
        assert np.all(result.responses[index] <= result.subjects[index])
    assert result.selection_probability.sum() + result.no_selection_probability == pytest.approx(1)
    assert result.subjects.shape == (3, 3)
    assert not result.subjects.flags.writeable
    assert result.trial_seeds.dtype == np.uint64


@pytest.mark.parametrize("model", ["global", "local"])
def test_logistic_simulation_prior_can_stop_before_enrollment(model):
    config = MTADFLogisticSimulationConfig(
        model=model,  # type: ignore[arg-type]
        true_toxicity=[0.1, 0.2],
        true_efficacy=[0.2, 0.4],
        doses=[0.0, 1.0],
        cohorts=2,
        window_length=2,
        prior=MTADFPrior(1000.0, 1.0),
        draws=32,
        warmup=32,
        chains=2,
    )
    result = simulate_mtadf_logistic(config, trials=2, rng=19)
    assert np.all(result.subjects == 0)
    assert np.all(result.selected_dose == -1)
    assert result.stop_reason == ("no_admissible_start", "no_admissible_start")
    assert np.all(result.posterior_fit_count == 0)
    assert result.no_selection_probability == 1.0
    assert result.early_stop_probability == 1.0


def test_local_simulation_rejects_incomplete_initial_ramp_and_aggregate_budget():
    with pytest.raises(ValueError, match="at least window_length cohorts"):
        simulate_mtadf_logistic(
            MTADFLogisticSimulationConfig(
                model="local",
                true_toxicity=[0.1, 0.2, 0.3],
                true_efficacy=[0.2, 0.4, 0.5],
                doses=[0.0, 1.0, 2.0],
                cohorts=1,
                window_length=2,
            ),
            trials=1,
            rng=5,
        )
    config = _config("global")
    with pytest.raises(ValueError, match="max_total_work"):
        simulate_mtadf_logistic(config, trials=10, rng=5, max_total_work=1)


def test_global_one_dose_simulation_ignores_unused_local_window_setting():
    config = MTADFLogisticSimulationConfig(
        model="global",
        true_toxicity=[0.1],
        true_efficacy=[0.5],
        doses=[0.0],
        cohorts=1,
        draws=32,
        warmup=32,
        chains=2,
    )
    trial = simulate_mtadf_logistic_trial(config, outcome_seed=12, sampler_seed=13)
    assert trial.subjects.tolist() == [3]
    assert trial.posterior_fit_count == 1
    assert trial.selected_dose in (-1, 0)


def test_simulation_rejects_unsupported_outcome_model():
    config = MTADFLogisticSimulationConfig(
        model="global",
        true_toxicity=[0.1, 0.2],
        true_efficacy=[0.2, 0.4],
        doses=[0.0, 1.0],
        outcome_model="unspecified_association",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="outcome_model"):
        simulate_mtadf_logistic(config, trials=1, rng=5)
