import json
import math
from pathlib import Path

import pytest

from mdanderson_stats.interaction_index_study import (
    simulate_interaction_index_three_drug_study,
)


def test_seeded_study_replay_captures_source_design_and_all_summary_cells():
    settings = {
        "interaction_indices": (0.6, 1.67),
        "error_sd": (0.1, 0.4),
        "replicates": 3,
        "rng": 417,
    }
    first = simulate_interaction_index_three_drug_study(**settings)
    second = simulate_interaction_index_three_drug_study(**settings)
    assert first == second
    assert first.seed == 417
    assert first.error_distribution == "normal"
    assert first.error_scale == "logit_effect"
    assert first.observations_per_trial == 19
    assert first.single_agent_doses[0][0] == pytest.approx(0.1)
    assert first.single_agent_doses[1][0] == pytest.approx(0.1)
    assert first.single_agent_doses[2][-1] == pytest.approx(12.0)
    assert first.combination_doses == pytest.approx((1 / 3, 2 / 3, 4 / 3))
    assert first.true_coefficients[1] == pytest.approx((math.log(2), -1))
    assert first.simulated_trials == 12
    assert len(first.cells) == 4
    assert all(0 <= cell.raw_ci_coverage <= 1 for cell in first.cells)
    assert all(0 <= cell.log_ci_coverage <= 1 for cell in first.cells)
    assert all(
        cell.fraction_log_ci_below_one
        + cell.fraction_log_ci_contains_one
        + cell.fraction_log_ci_above_one
        == pytest.approx(1.0)
        for cell in first.cells
    )
    automatic = simulate_interaction_index_three_drug_study(
        interaction_indices=(1.0,), error_sd=(0.1,), replicates=1
    )
    replay = simulate_interaction_index_three_drug_study(
        interaction_indices=(1.0,), error_sd=(0.1,), replicates=1, rng=automatic.seed
    )
    assert automatic == replay


def test_independent_base_r_reference_summaries_match_all_reported_metrics():
    fixture_path = Path(__file__).parent / "fixtures" / "interaction_index_study.json"
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))
    metric_names = (
        "mean_estimated_index",
        "raw_ci_coverage",
        "log_ci_coverage",
        "mean_raw_ci_length",
        "mean_log_ci_length_on_index_scale",
        "fraction_log_ci_below_one",
        "fraction_log_ci_contains_one",
        "fraction_log_ci_above_one",
    )
    for expected in reference["cells"]:
        study = simulate_interaction_index_three_drug_study(
            interaction_indices=(expected["interaction_index"],),
            error_sd=(reference["error_sd"],),
            replicates=reference["replicates"],
            rng=reference["seed"],
        )
        cell = study.cells[0]
        assert study.seed == reference["seed"]
        for name in metric_names:
            assert getattr(cell, name) == pytest.approx(expected[name], rel=0, abs=1e-11)


def test_source_qq_samples_and_saved_study(tmp_path):
    import numpy as np
    from scipy.stats import norm

    study = simulate_interaction_index_three_drug_study(
        interaction_indices=(5 / 3,), error_sd=(0.1,), replicates=8, rng=65, retain_samples=True
    )
    assert not study.estimated_indices.flags.writeable
    np.testing.assert_allclose(study.estimated_indices.mean(), study.cells[0].mean_estimated_index)
    saved = json.loads(study.write_json(tmp_path / "scenario1.json").read_text())
    np.testing.assert_array_equal(saved["estimated_indices"], study.estimated_indices)
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    figure = study.plot_qq()
    raw = np.asarray(figure.axes[0].collections[0].get_offsets())
    np.testing.assert_allclose(raw[:, 0], norm.ppf((np.arange(1, 9) - 0.375) / (8 + 0.25)))
    np.testing.assert_array_equal(raw[:, 1], np.sort(study.estimated_indices[0]))
    plt.close(figure)
    for bad in (True, -1, 1, 0.5):
        with pytest.raises(ValueError):
            study.plot_qq(bad)
    without = simulate_interaction_index_three_drug_study(
        interaction_indices=(1,), error_sd=(0.1,), replicates=1, rng=65
    )
    with pytest.raises(ValueError, match="retain_samples"):
        without.plot_qq()
