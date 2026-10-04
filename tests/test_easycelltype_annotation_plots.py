from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mdanderson_stats.easycelltype import EasyCellTypeLabel
from mdanderson_stats.easycelltype_annotation_plots import (
    plot_easycelltype_annotation_dots,
    plot_easycelltype_candidates,
)
from mdanderson_stats.easycelltype_gsea_labels import EasyCellTypeGSEALabel


def test_fisher_candidate_bars_show_actual_scores_and_input_order() -> None:
    labels = (
        EasyCellTypeLabel("B", "T cell", "hard_fisher", 1, 0.01, 0.02, -1.25, ("g1", "g2")),
        EasyCellTypeLabel("B", "NK", "soft_fisher", 2, 0.03, 0.04, 0.5, ("g3",)),
    )
    figure, axis = plt.subplots()
    try:
        returned = plot_easycelltype_candidates(labels, ax=axis)
        assert returned is axis
        assert [patch.get_width() for patch in axis.patches] == [-1.25, 0.5]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["B / T cell", "B / NK"]
        assert axis.get_xlabel() == "Mean marker score"
    finally:
        plt.close(figure)


def test_gsea_dot_plot_encodes_significance_core_size_and_method() -> None:
    labels = (
        EasyCellTypeGSEALabel("C1", "A", 0.01, 0.02, 1.4, "hard_enrich", ("e1", "e2", "e3")),
        EasyCellTypeGSEALabel("C2", "B", 0.0, 0.0, -1.1, "soft_enrich", None),
    )
    figure, axis = plt.subplots()
    try:
        plot_easycelltype_annotation_dots(labels, pvalue="raw", ax=axis)
        collections = axis.collections
        assert len(collections) == 2
        hard = collections[0]
        soft = collections[1]
        np.testing.assert_allclose(hard.get_offsets(), [[0.0, 0.0]])
        np.testing.assert_allclose(soft.get_offsets(), [[1.0, 1.0]])
        assert hard.get_sizes()[0] == pytest.approx(24 + 18 * np.sqrt(3))
        assert soft.get_sizes()[0] == 24
        assert axis.get_title().endswith("raw p-value)")
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["C1", "C2"]
    finally:
        plt.close(figure)


def test_fisher_dot_adjusted_p_and_score_bar_do_not_mutate_labels() -> None:
    label = EasyCellTypeLabel("C", "A", "hard_fisher", 1, 0.2, 0.05, 2.0, ("x",))
    figure, axis = plt.subplots()
    try:
        plot_easycelltype_annotation_dots((label,), pvalue="adjusted", ax=axis)
        color_value = axis.collections[0].get_array()[0]
        assert color_value == pytest.approx(-np.log10(0.05))
        assert label.pvalue == 0.2
    finally:
        plt.close(figure)


def test_plot_rejects_mixed_families_and_invalid_pvalue_mode() -> None:
    fisher = EasyCellTypeLabel("C", "A", "hard_fisher", 1, 0.1, 0.1, 1.0, ("x",))
    gsea = EasyCellTypeGSEALabel("C", "B", 0.1, 0.1, 1.0, "soft_enrich", ())
    with pytest.raises(ValueError, match="not a mixture"):
        plot_easycelltype_candidates((fisher, gsea))
    with pytest.raises(ValueError, match="pvalue"):
        plot_easycelltype_annotation_dots((fisher,), pvalue="q")
