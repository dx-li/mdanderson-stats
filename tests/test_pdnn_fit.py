from dataclasses import replace

import numpy as np
import pytest

from mdanderson_stats import (
    PDNNConvergenceError,
    PDNNParameters,
    fit_pdnn,
    pdnn_binding_energy,
    pdnn_signal,
)
from mdanderson_stats.pdnn import _pdnn_pairs
from mdanderson_stats.pdnn_fit import _objective


def synthetic():
    rng = np.random.default_rng(7)
    sequences = np.array(["".join(row) for row in rng.choice(list("ACGT"), (600, 25))])
    groups = np.arange(600) % 6
    parameters = PDNNParameters(
        rng.normal(0.03, 0.12, (4, 4)),
        rng.normal(0.02, 0.15, (4, 4)),
        rng.uniform(0.7, 1.3, 24),
        rng.uniform(0.7, 1.3, 24),
        np.log([200, 300, 400, 600, 800, 1000]),
        np.log(100),
        np.log(10),
    )
    signal = pdnn_signal(
        pdnn_binding_energy(sequences, parameters.specific_stacking, parameters.specific_weights),
        pdnn_binding_energy(
            sequences, parameters.nonspecific_stacking, parameters.nonspecific_weights
        ),
        log_expression=parameters.log_expression[groups],
        nonspecific_amount=100,
        background=10,
    )
    return sequences, groups, parameters, signal


def test_objective_gradient_matches_independent_finite_differences():
    sequences, groups, parameters, signal = synthetic()
    vector = parameters._pack()
    observed = np.log(signal * np.linspace(0.8, 1.3, len(signal)))
    pairs = _pdnn_pairs(sequences)
    value, gradient, prediction = _objective(vector, pairs, groups, observed)
    assert value == pytest.approx(np.mean((prediction - observed) ** 2))
    numerical = np.empty_like(vector)
    for i in range(len(vector)):
        step = np.zeros_like(vector)
        step[i] = 1e-5
        numerical[i] = (
            _objective(vector + step, pairs, groups, observed)[0]
            - _objective(vector - step, pairs, groups, observed)[0]
        ) / 2e-5
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-9)


def test_fixed_energy_fit_recovers_expression_and_background():
    sequences, groups, parameters, signal = synthetic()
    initial = replace(
        parameters,
        log_expression=parameters.log_expression + 0.2,
        log_nonspecific_amount=np.log(130),
        log_background=np.log(15),
    )
    result = fit_pdnn(
        sequences,
        signal,
        groups * 10 + 5,
        initial=initial,
        fit_energies=False,
        fit_weights=False,
        tolerance=1e-10,
    )
    np.testing.assert_allclose(result.fitted_signal, signal, rtol=2e-5)
    np.testing.assert_allclose(
        result.parameters.log_expression, parameters.log_expression, atol=2e-5
    )
    assert result.parameters.log_background == pytest.approx(np.log(10), abs=2e-4)
    assert result.parameters.log_nonspecific_amount == pytest.approx(np.log(100), abs=2e-5)
    np.testing.assert_array_equal(result.probeset_ids, [5, 15, 25, 35, 45, 55])


def test_joint_fit_reduces_error_and_preserves_failed_checkpoint():
    sequences, groups, parameters, signal = synthetic()
    initial = replace(
        parameters,
        specific_stacking=parameters.specific_stacking + 0.01,
        nonspecific_stacking=parameters.nonspecific_stacking - 0.01,
        specific_weights=np.ones(24),
        nonspecific_weights=np.ones(24),
        log_expression=parameters.log_expression + 0.1,
    )
    with pytest.raises(PDNNConvergenceError) as caught:
        fit_pdnn(sequences, signal, groups, initial=initial, max_iterations=1)
    checkpoint = caught.value.result
    assert np.isfinite(checkpoint.fitted_signal).all()
    assert checkpoint.fitness < checkpoint.initial_fitness
    result = fit_pdnn(sequences, signal, groups, initial=checkpoint.parameters, tolerance=1e-8)
    assert result.converged
    assert result.fitness < 1e-6
    assert result.fitness < result.initial_fitness / 1000
    assert not result.parameters.specific_stacking.flags.writeable
    assert not result.fitted_signal.flags.writeable
