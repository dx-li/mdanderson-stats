import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    anovaddp_baseline_curves,
    anovaddp_curve,
    anovaddp_new_atom,
    anovaddp_r_outputs,
    fit_anovaddp,
    predict_anovaddp,
    read_anovaddp_data,
    write_anovaddp_prediction,
)


def test_baseline_native_rows_and_new_atom_weights():
    case = json.loads((Path(__file__).parent / "fixtures/anovaddp-baseline.json").read_text())
    curves = anovaddp_baseline_curves(case["coefficients"], case["time"])
    assert_allclose(curves, case["baseline"], atol=1e-13, rtol=0)
    # One-dimensional single-atom posterior N(1,1/2), plus a N(0,1) base option.
    for seed in (0, 2):
        fit = anovaddp_new_atom(
            [[2]],
            [[1]],
            [0],
            residual_covariance=[[1]],
            base_mean=[0],
            base_covariance=[[1]],
            concentration=1,
            seed=seed,
        )
        rng = np.random.default_rng(seed)
        u = rng.random()
        z = rng.standard_normal()
        assert_allclose(fit.weights, [0.5, 0.5])
        assert fit.cluster == (-1 if u >= 0.5 else 0)
        assert fit.coefficients[0] == pytest.approx(z if u >= 0.5 else 1 + np.sqrt(0.5) * z)


def test_complete_predictive_workflow_and_moments(tmp_path):
    x = np.array(
        [[1, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 1]]
    )
    t = np.tile([0, 1, 2, 3, 4, 6], 4)
    ids = np.repeat(np.arange(4), 6)
    th = np.tile([2, 0.5, 1.8, 1, 3, 0.2], (4, 1))
    y = np.tile(anovaddp_curve(th[0], t[:6]), 4) + 0.1 * np.cos(np.arange(24))
    fit = fit_anovaddp(
        t,
        y,
        ids,
        x,
        initial_parameters=th,
        initial_covariance=np.diag([1, 1, 1, 0.2, 0.2, 0.1]),
        base_prior=np.r_[th[0, 1:], np.zeros(30)],
        base_covariance=np.eye(35),
        iterations=12,
        burn_in=4,
        seed=6707,
    )
    prediction = predict_anovaddp(fit, x, x[:3], time=[0, 2, 4, 6], seed=67)
    assert prediction.common_effect.shape == prediction.study3.shape == (8, 4)
    assert prediction.nadir.shape == (8, 3)
    assert prediction.baseline_draws.shape == (8, 10, 4)
    assert prediction.prediction_draws.shape == (8, 3, 4)
    assert_allclose(prediction.prediction_mean, prediction.prediction_draws.mean(axis=0))
    assert_allclose(
        prediction.prediction_second_moment, (prediction.prediction_draws**2).mean(axis=0)
    )
    assert_allclose(prediction.baseline_mean, prediction.baseline_draws.mean(axis=0))
    assert_allclose(prediction.baseline_second_moment, (prediction.baseline_draws**2).mean(axis=0))
    for i in range(8):
        assert_allclose(
            prediction.common_effect[i],
            anovaddp_curve(np.r_[2, prediction.atom_draws[i, :5]], [0, 2, 4, 6], repair_order=True),
        )
    replay = predict_anovaddp(fit, x, x[:3], time=[0, 2, 4, 6], seed=67)
    assert np.array_equal(replay.nadir, prediction.nadir)
    assert np.array_equal(replay.prediction_draws, prediction.prediction_draws)
    assert not np.array_equal(prediction.common_effect, prediction.study3)
    with pytest.raises(ValueError):
        prediction.baseline_mean[0, 0] = 0

    # Exercise source-format files through fitting inputs and all output families.
    inputs = dict(
        time=t,
        observations=y,
        observations_per_subject=[6, 6, 6, 6],
        design=x,
        prediction_design=x[:3],
        initial_parameters=th,
        initial_covariance=np.diag([1, 1, 1, 0.2, 0.2, 0.1]),
        base_covariance=np.eye(35),
        base_prior=np.r_[th[0, 1:], np.zeros(30)],
    )
    paths = {}
    for name, value in inputs.items():
        path = tmp_path / (name + ".txt")
        np.savetxt(path, value)
        paths[name] = path
    data = read_anovaddp_data(**paths)
    assert np.array_equal(data.subject, ids)
    for name, value in inputs.items():
        assert np.array_equal(getattr(data, name), value)
    assert not data.subject.flags.writeable
    outputs = write_anovaddp_prediction(prediction, tmp_path / "outputs")
    expected = [
        prediction.common_effect,
        prediction.study3,
        prediction.nadir,
        prediction.baseline_mean,
        prediction.baseline_second_moment,
        prediction.prediction_mean,
        prediction.prediction_second_moment,
        prediction.time,
    ]
    for path, value in zip(outputs, expected):
        assert np.array_equal(np.loadtxt(path), value)
    assert anovaddp_r_outputs(prediction)["a02"] is prediction.baseline_second_moment
    with pytest.raises(FileExistsError):
        write_anovaddp_prediction(prediction, tmp_path / "outputs")
    np.savetxt(paths["observations_per_subject"], [6, 6, 6, 5])
    with pytest.raises(ValueError, match="subject counts"):
        read_anovaddp_data(**paths)
