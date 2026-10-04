import csv

import numpy as np

from mdanderson_stats.bchm_scenarios import BCHMScenario, fit_bchm_scenarios
from mdanderson_stats.hierarchical_binomial import summarize_chains


def test_scenario_report_csv_roundtrip_and_streamed_samples(tmp_path):
    scenario = BCHMScenario(
        "small illustration",
        [1, 2],
        [10, 12],
        seed=158,
        subgroup_labels=("A", "B"),
        burn_in=8,
        iterations=16,
        draws=8,
        warmup=2,
    )
    input_path = scenario.write_input_csv(tmp_path / "input.csv")
    loaded = BCHMScenario.from_input_csv(input_path)
    assert loaded == scenario

    batch = fit_bchm_scenarios((loaded,))
    result = batch.scenarios[0]
    report = batch.report()
    assert "clustering_seed=" in report and "borrowing_seeds=" in report
    assert "similarity_batch_means_mcse=" in report
    assert "efficacy_indicator_batch_means_mcse=" in report
    assert "Native R/JAGS uses different RNG streams" in report
    saved = batch.write_report(tmp_path / "report.txt")
    assert saved.read_text(encoding="utf-8") == report

    sample_path = batch.write_samples_csv(tmp_path / "samples.csv")
    with sample_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * 2 * 8
    assert {row["subgroup"] for row in rows} == {"A", "B"}
    assert rows[0]["chain"] == "0" and rows[0]["draw"] == "0"
    assert not result.fit.borrowing[0].efficacy_indicators.flags.writeable
    for target, borrow in enumerate(result.fit.borrowing):
        indicators = borrow.efficacy_indicators
        assert indicators is not None
        expected_mcse = summarize_chains(indicators.astype(float)[:, :, None]).batch_mean_mcse[0]
        assert result.efficacy_mcse[target] == expected_mcse
    expected = np.random.default_rng(loaded.seed)
    assert result.clustering_seed == int(expected.integers(2**31))
    assert result.borrowing_seeds == tuple(int(expected.integers(2**31)) for _ in range(2))


def test_bchm_scenario_preflight_rejects_bad_input_and_duplicate_names():
    import pytest

    with pytest.raises(ValueError, match="must be integers"):
        BCHMScenario("bad", [1.5], [10], seed=1)
    scenario = BCHMScenario("same", [1], [10], seed=1, burn_in=0, iterations=8, draws=8, warmup=0)
    with pytest.raises(ValueError, match="unique"):
        fit_bchm_scenarios((scenario, scenario))
