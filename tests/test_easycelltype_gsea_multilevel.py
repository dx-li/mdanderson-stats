import numpy as np
import pytest
from scipy.special import digamma

from mdanderson_stats.easycelltype_gsea_multilevel import (
    _beta_mean_log,
    _simple_pilot,
    easycelltype_gsea,
)


def _example():
    genes = [f"g{i}" for i in range(12)]
    scores = [6.3, 5.9, 4.2, 2.8, 1.1, 0.7, -0.25, -0.8, -1.6, -2.4, -4.3, -7.7]
    ref_genes = ["g0", "g1", "g2", "g5", "g6", "g7", "g9", "g10"]
    ref_types = ["A"] * 3 + ["B"] * 3 + ["C"] * 2
    return genes, scores, ref_genes, ref_types


def test_single_set_pilot_uses_the_r_style_observed_statistic_for_zero_hits():
    weights = np.asarray([1.0, 0.0, 0.0, 0.5, 0.25])
    pilot = _simple_pilot(
        weights,
        [2],
        [0.5],
        score_type="std",
        log_weight_scale=0.0,
        single_set=True,
        n_perm=20,
        rng=np.random.default_rng(4),
    )[0]
    assert 0 <= pilot.mode_count <= 20
    assert 0 <= pilot.extreme_count <= 20
    assert pilot.p_value is not None
    assert pilot.nes is not None


def test_single_set_pilot_uses_r_zero_hit_rule_not_cpp_epsilon(monkeypatch):
    import mdanderson_stats.easycelltype_gsea_multilevel as module

    weights = np.asarray([4.0, 3.0, 2.0, 0.0, 0.0, 1.0, 2.0, 5.0])
    monkeypatch.setattr(
        module,
        "_sample_positions",
        lambda n, k, rng: np.asarray([3, 1], dtype=np.intp),
    )
    pilot = module._simple_pilot(
        weights,
        [2],
        [5.0 / 6.0],
        score_type="std",
        log_weight_scale=0.0,
        single_set=True,
        n_perm=20,
        rng=np.random.default_rng(0),
    )[0]
    assert pilot.extreme_count == 20
    assert pilot.mode_count == 20
    assert pilot.nes == pytest.approx(1.0)
    assert pilot.p_value == pytest.approx(1.0)


def test_full_gsea_replays_seeded_multilevel_and_unavailable_pilots():
    genes, scores, ref_genes, ref_types = _example()
    kwargs = dict(
        query_genes=genes,
        clusters=["cluster"] * len(genes),
        scores=scores,
        reference_genes=ref_genes,
        reference_cell_types=ref_types,
        sample_size=9,
        n_perm_simple=25,
        eps=1e-4,
        rng=1,
        max_work=1_000_000,
    )
    first = easycelltype_gsea(**kwargs)
    second = easycelltype_gsea(**kwargs)
    assert first == second
    methods = {item.inference_method for item in first.clusters[0].sets}
    assert "multilevel" in methods
    assert "unavailable" in methods
    unavailable = next(item for item in first.clusters[0].sets if item.status is not None)
    assert unavailable.status == "insufficient_directional_pilot"
    assert unavailable.p_value is None


def test_all_zero_prepared_weights_are_rejected_for_full_inference():
    genes, _, ref_genes, ref_types = _example()
    with pytest.raises(ValueError, match="all-zero prepared ranking"):
        easycelltype_gsea(
            genes,
            ["cluster"] * len(genes),
            [0.0] * len(genes),
            ref_genes,
            ref_types,
            n_perm_simple=10,
            sample_size=9,
            rng=3,
        )


def test_work_budget_and_source_beta_log_mean():
    genes, scores, ref_genes, ref_types = _example()
    with pytest.raises(ValueError, match="max_work"):
        easycelltype_gsea(
            genes,
            ["cluster"] * len(genes),
            scores,
            ref_genes,
            ref_types,
            n_perm_simple=10,
            sample_size=9,
            max_work=1,
            rng=2,
        )
    assert _beta_mean_log(3, 5) == pytest.approx(float(digamma(3) - digamma(6)))
