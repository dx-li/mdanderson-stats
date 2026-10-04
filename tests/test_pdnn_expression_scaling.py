import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import pdnn_expression
from mdanderson_stats.pdnn import PDNNExpression
from mdanderson_stats.pdnn_expression_scaling import pdnn_scale_expression


def test_paper_array_mean_500_scaling_preserves_gene_identity_and_log_scale():
    expression = pdnn_expression(
        [100.0, 300.0],
        [20, 10],
        [0.0, 0.0],
        [0.0, 0.0],
        nonspecific_amount=0.0,
        background=0.0,
    )

    scaled = pdnn_scale_expression(expression)

    assert_array_equal(scaled.probeset_ids, [10, 20])
    assert_allclose(scaled.log_expression, np.log([750.0, 250.0]), rtol=1e-14)
    assert_allclose(scaled.log_scale_factor, np.log(1.25), rtol=1e-14)
    assert scaled.target_mean == 500.0
    assert_allclose(expression.fitted_signal, [100.0, 300.0])
    assert not scaled.probeset_ids.flags.writeable
    assert not scaled.log_expression.flags.writeable


def test_scaling_is_stable_for_extreme_log_expression_and_translation_invariant():
    ids = np.array([4, 9], dtype=np.int64)
    base = PDNNExpression(
        ids,
        np.array([1000.0, 1001.0]),
        np.ones(2),
        np.ones(2, bool),
        np.ones(2, int),
    )
    shifted = PDNNExpression(
        ids,
        np.array([2000.0, 2001.0]),
        np.ones(2),
        np.ones(2, bool),
        np.ones(2, int),
    )

    result = pdnn_scale_expression(base)
    translated = pdnn_scale_expression(shifted)

    assert np.all(np.isfinite(result.log_expression))
    assert_allclose(result.log_expression, translated.log_expression, rtol=0, atol=1e-13)
    assert_allclose(np.logaddexp.reduce(result.log_expression) - np.log(2), np.log(500), atol=1e-13)


def test_centered_scaling_preserves_small_differences_on_large_common_log_offset():
    ids = np.array([4, 9], dtype=np.int64)
    logs = np.array([1e16, 1e16 + 2])
    result = pdnn_scale_expression(
        PDNNExpression(ids, logs, np.ones(2), np.ones(2, bool), np.ones(2, int))
    )

    assert_allclose(
        result.log_expression,
        pdnn_scale_expression(
            PDNNExpression(ids, np.array([0.0, 2.0]), np.ones(2), np.ones(2, bool), np.ones(2, int))
        ).log_expression,
        rtol=0,
        atol=1e-13,
    )


def test_mixed_extreme_log_spread_has_the_requested_array_mean():
    ids = np.array([1, 2, 3], dtype=np.int64)
    result = pdnn_scale_expression(
        PDNNExpression(
            ids,
            np.array([-1000.0, 0.0, 1000.0]),
            np.ones(3),
            np.ones(3, bool),
            np.ones(3, int),
        )
    )

    assert np.all(np.isfinite(result.log_expression))
    assert_allclose(np.logaddexp.reduce(result.log_expression) - np.log(3), np.log(500), atol=1e-13)


def test_scaling_rejects_noncanonical_or_oversized_expression_snapshots_early():
    duplicate_ids = PDNNExpression(
        np.array([4, 4]), np.array([0.0, 1.0]), np.ones(2), np.ones(2, bool), np.ones(2, int)
    )
    with pytest.raises(ValueError, match="matching nonempty"):
        pdnn_scale_expression(duplicate_ids)

    oversized = PDNNExpression(
        np.array([], dtype=np.int64),
        np.empty(500_001),
        np.empty(0),
        np.empty(0, bool),
        np.empty(0, int),
    )
    with pytest.raises(ValueError, match="1..500000 values"):
        pdnn_scale_expression(oversized)
