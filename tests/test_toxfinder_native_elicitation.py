"""Original stochastic-optimizer control and supplied-Gamma probability fixtures."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.toxfinder_native_elicitation import (
    _calibrate,
    _tail_probabilities,
    elicit_toxfinder_native_prior,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONTROL = json.loads((FIXTURES / "toxfinder-native-elicitor-control.json").read_text())
PROBABILITY = json.loads((FIXTURES / "toxfinder-native-elicitation-probability.json").read_text())


@pytest.mark.parametrize("row", CONTROL, ids=lambda r: r["mode"])
def test_original_full_calibrator_control(row):
    calls = []
    mode = row["mode"]

    def probabilities(value):
        la, lb = value
        if mode == "success":
            result = (0.99 + 0.002 * (la - math.log(0.12)), 0.99 + 0.002 * (lb - math.log(2)))
        elif mode == "precheck_failure":
            result = (0.0, 1.0)
        elif mode == "constant_plateau_success":
            result = (0.5, 0.5)
        elif mode == "optimizer_failure":
            result = (
                0.3 + 0.3 * ((len(calls) * 7) % 11) / 11,
                0.4 + 0.2 * ((len(calls) * 3) % 7) / 7,
            )
        else:
            result = (
                0.99 - 0.08 * math.tanh(la - math.log(0.12)) ** 2,
                0.99 - 0.08 * math.tanh(lb - math.log(2)) ** 2,
            )
        calls.append(list(value))
        return result

    if row["errors"]:
        with pytest.raises(ArithmeticError, match="no fallback"):
            _calibrate(probabilities)
    else:
        point, starts, attempts, iterations = _calibrate(probabilities)
        means = np.array([0.3 / 0.7, math.log(0.6 / (0.4 * (0.3 / 0.7))) / math.log(1400 / 1200)])
        np.testing.assert_allclose(
            means**2 / np.exp(point), np.array(row["shapes"])[[0, 3]], rtol=2e-12
        )
        np.testing.assert_allclose(
            means / np.exp(point), np.array(row["rates"])[[0, 3]], rtol=2e-12
        )
        assert attempts == len(row["attempts"])
        if mode == "constant_plateau_success":
            assert sum((p - 0.99) ** 2 for p in probabilities(point)) > 0.48
            calls.pop()  # quality check is separate from original optimizer sequence
    assert len(calls) == len(row["probability_calls"])
    np.testing.assert_allclose(
        calls, [r["log_variances"] for r in row["probability_calls"]], rtol=0, atol=2e-12
    )


@pytest.mark.parametrize("row", PROBABILITY)
def test_original_strict_probability_loop(row):
    tape = np.asarray(row["tape"])
    draws = np.tile(tape, (30_000 // len(tape), 1))
    result = _tail_probabilities(
        draws[:, 0], draws[:, 1], tuple(row["doses"]), tuple(row["thresholds"])
    )
    np.testing.assert_allclose(result, row["probabilities"], rtol=0, atol=1e-16)
    assert sum((p - 0.99) ** 2 for p in result) == pytest.approx(row["objective"], abs=2e-16)
    assert row["gamma_call_count"] == 60_000


def test_complete_seeded_native_calibration_and_quality():
    result = elicit_toxfinder_native_prior(
        [600, 1200, 1400, 2000],
        [350, 600, 700, 800],
        0.3,
        interaction_mean=[1, 1],
        interaction_variance=[3, 0.9],
        rng=np.random.default_rng(2026),
    )
    for agent in (result.agent1, result.agent2):
        assert agent.mean[0] == pytest.approx(0.3 / 0.7)
        assert agent.mean[1] == pytest.approx(8.126872809767802)
        assert np.all(agent.variance > 0)
        np.testing.assert_allclose(
            agent.assessed_target_residual, agent.assessed_tail_probability - 0.99
        )
        assert agent.assessed_objective == pytest.approx(sum(agent.assessed_target_residual**2))
        assert agent.assessed_cutoff_met == (agent.assessed_objective < 2e-6)
        assert agent.optimizer_probability_evaluations >= 9
        assert np.all(agent.assessed_probability_mcse >= 0)
    np.testing.assert_array_equal(result.prior.mean[[4, 5]], [1, 1])
    assert result.gamma_pairs_per_evaluation == 30_000
    with pytest.raises(ValueError):
        result.agent1.mean.flags.writeable = True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_probability_evaluations": 1},
        {"max_probability_evaluations": True},
        {"interaction_variance": [0, 1]},
        {"high_toxicity_probability": 0.1},
    ],
)
def test_native_calibration_rejects_invalid_or_excessive_work(kwargs):
    values = dict(
        interaction_mean=[1, 1], interaction_variance=[3, 0.9], rng=np.random.default_rng(8)
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        elicit_toxfinder_native_prior([600, 1200, 1400, 2000], [350, 600, 700, 800], 0.3, **values)


def test_full_native_calibration_with_independent_gamma_integration():
    from mdanderson_stats.toxfinder_elicitation import _adaptive_probability

    result = elicit_toxfinder_native_prior(
        [600, 1200, 1400, 2000],
        [350, 600, 700, 800],
        0.3,
        interaction_mean=[1, 1],
        interaction_variance=[3, 0.9],
        rng=np.random.default_rng(114),
    )
    for agent, ratios in (
        (result.agent1, (0.5, 2000 / 1200)),
        (result.agent2, (350 / 600, 800 / 600)),
    ):
        for j, (ratio, threshold, upper) in enumerate(zip(ratios, (0.05, 0.3), (False, True))):
            probability, error = _adaptive_probability(
                alpha_shape=float(np.float32(agent.shape[0])),
                alpha_scale=1 / agent.rate[0],
                beta_shape=float(np.float32(agent.shape[1])),
                beta_scale=1 / agent.rate[1],
                dose_ratio=ratio,
                probability=threshold,
                upper=upper,
            )
            assert (
                abs(probability - agent.assessed_tail_probability[j])
                < 6 * agent.assessed_probability_mcse[j] + error + 1 / 30_000
            )
