import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    INTERACTION_INDEX_SOURCE_SCENARIOS,
    simulate_interaction_index_fixed_ray_study,
)


def test_source_default_study_truth_and_original_controls(tmp_path):
    study = simulate_interaction_index_fixed_ray_study()
    reference = np.genfromtxt(
        Path(__file__).parent / "fixtures/interaction-index-native/source-truth.csv",
        delimiter=",",
        names=True,
    )
    assert len(study.results) == 14
    assert study.dose_ratio == 2
    assert study.samples == 500
    assert INTERACTION_INDEX_SOURCE_SCENARIOS[6] == 5 / 3
    np.testing.assert_allclose(study.effects, reference["effect"], atol=2e-16)
    np.testing.assert_allclose(study.true_index, reference["index"], atol=1e-14)
    for row in study.results:
        assert row.monte_carlo.coefficient_draws.shape == (500, 3, 2)
        assert row.observed_effects.shape == (3, 5)
        assert not row.observed_effects.flags.writeable
        assert not row.source_monte_carlo_interval.flags.writeable
        assert np.all(row.source_monte_carlo_interval[:, 0] >= 0.0001)
        assert np.isfinite(row.observed_length_ratio).all()
    path = study.write_json(tmp_path / "scenario2.json")
    record = json.loads(path.read_text())
    assert record["source_monte_carlo_lower_floor"] == 0.0001
    assert len(record["results"]) == 14


def test_replay_small_study_and_optional_plot():
    kwargs = dict(error_sd=(0.2,), replicates=1, samples=31, effects=(0.2, 0.5, 0.8), rng=65)
    a = simulate_interaction_index_fixed_ray_study(**kwargs)
    b = simulate_interaction_index_fixed_ray_study(**kwargs)
    assert a.to_json() == b.to_json()
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    figure = a.plot()
    assert figure.axes[0].get_yscale() == "log"
    plt.close(figure)
    for index in (True, -1, 1, 0.5, "0"):
        with pytest.raises(ValueError, match="result index"):
            a.plot(index)


@pytest.mark.parametrize(
    "control",
    [
        {"error_sd": (0,)},
        {"error_sd": (np.nan,)},
        {"effects": (0.5, 0.2)},
        {"replicates": True},
        {"samples": 1},
        {"confidence": np.nan},
        {"max_total_coefficient_draws": 1},
        {"max_total_work": 1},
        {"max_storage_bytes": 1},
    ],
)
def test_controls_and_aggregate_budgets_preflight(control):
    with pytest.raises(ValueError):
        simulate_interaction_index_fixed_ray_study(**control)
