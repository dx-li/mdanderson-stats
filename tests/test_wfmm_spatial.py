"""Native analysis-only references and independent spatial reconstruction checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    WFMMStudy,
    read_wfmm_study,
    run_wfmm_study,
    wfmm_inverse,
    wfmm_spatial_wavelet_basis,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-spatial.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES)
def test_native_analysis_only_and_independent_reconstruction(case):
    b = wfmm_spatial_wavelet_basis(
        case["dims"],
        levels=case["levels"],
        wavelets=case["wavelets"],
        boundaries=case["boundaries"],
        layout="rectangular" if case["rectangular"] else "pyramid",
    )
    np.testing.assert_allclose(b.custom_matrix, case["analysis"], rtol=0, atol=3e-15)
    np.testing.assert_allclose(
        b.custom_matrix @ b.synthesis_matrix, np.eye(b.time_count), rtol=0, atol=3e-11
    )
    y = np.random.default_rng(125).normal(size=(3, b.time_count))
    np.testing.assert_allclose(
        wfmm_inverse(wfmm_transform(y, b).coefficients, b), y, rtol=0, atol=3e-11
    )
    assert b.spatial_shape == tuple(case["dims"])
    assert not b.custom_matrix.flags.writeable and not b.synthesis_matrix.flags.writeable
    partitions = b.coefficient_partition
    assert np.all(np.diff(partitions) >= 0)
    expected = (
        np.prod(np.array(case["levels"]) + 1)
        if case["rectangular"]
        else 1 + (2 ** len(case["dims"]) - 1) * case["levels"][0]
    )
    assert len(np.unique(partitions)) == expected


@pytest.mark.parametrize("dimensions", [2, 3])
def test_haar_spatial_blocks_have_independent_sum_and_difference_meaning(dimensions):
    shape = (4,) * dimensions
    image = np.arange(4**dimensions, dtype=float).reshape(shape, order="F") ** 2
    b = wfmm_spatial_wavelet_basis(shape)
    actual = wfmm_transform(image.reshape(1, -1, order="F"), b).coefficients[0]
    orientations = (0, 1, 2, 3) if dimensions == 2 else (0, 2, 1, 4, 3, 6, 5, 7)
    expected = []
    for orientation in orientations:
        values = np.zeros((2,) * dimensions)
        for output in np.ndindex(values.shape):
            for offset in np.ndindex((2,) * dimensions):
                sign = (-1) ** sum(
                    offset[axis] for axis in range(dimensions) if orientation & (1 << axis)
                )
                values[output] += (
                    sign
                    * image[tuple(2 * i + j for i, j in zip(output, offset, strict=True))]
                    / np.sqrt(2**dimensions)
                )
        expected.extend(values.ravel(order="F"))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-15)


@pytest.mark.parametrize(
    "settings",
    [
        dict(shape=(4,)),
        dict(shape=(2, 2, 2, 2)),
        dict(shape=(32, 32)),
        dict(shape=(True, 4)),
        dict(shape=(4, 4), levels=0),
        dict(shape=(4, 4), levels=3),
        dict(shape=(4, 4), levels=(1, 2)),
        dict(shape=(4, 4), wavelets=("db1", "db2")),
        dict(shape=(4, 4), boundaries=("zero", "periodic")),
        dict(shape=(4, 4), levels=(1,)),
        dict(shape=(4, 4), layout="bad"),
        dict(shape=(4, 4, 4), layout="rectangular"),
        dict(shape=(4, 4), wavelets="coif5"),
        dict(shape=(4, 4), max_work=1),
        dict(shape=(8, 8, 8), wavelets="db2", levels=2),
    ],
)
def test_spatial_bounds_and_ambiguous_settings_rejected(settings):
    with pytest.raises(ValueError):
        wfmm_spatial_wavelet_basis(**settings)


@pytest.mark.parametrize(
    "shape,layout,extra",
    [
        ((4, 4), "pyramid", {}),
        ((4, 4, 4), "pyramid", {}),
        (
            (5, 3),
            "rectangular",
            dict(
                axis_wavelets=("db2", "coif1"),
                axis_boundaries=("reflection", "periodic"),
                axis_levels=(2, 1),
            ),
        ),
    ],
)
def test_spatial_saved_fit_prediction_and_reporting(shape, layout, extra):
    y = np.random.default_rng(46).normal(size=(8, int(np.prod(shape))))
    y += 5 * np.random.default_rng(411).normal(size=y.shape[1])
    study = WFMMStudy(
        y,
        np.ones((8, 1)),
        transform="spatial_wavelet",
        spatial_shape=shape,
        spatial_layout=layout,
        draws=8,
        warmup=4,
        optimizer_iterations=50,
        prediction_design=[[1]],
        **extra,
    )
    reopened = read_wfmm_study(study.to_json())
    result = run_wfmm_study(reopened)
    replay = run_wfmm_study(study)
    assert result.fit is not None
    assert result.basis.spatial_shape == shape
    assert result.summary.mean.shape == (1, y.shape[1])
    assert result.prediction.mean.shape == (1, y.shape[1])
    assert result.report_json() == replay.report_json()


@pytest.mark.parametrize(
    "extra",
    [
        dict(spatial_shape=(2, 2)),
        dict(spatial_shape=(4, 4), axis_levels=(1, 1), levels=2),
        dict(spatial_shape=(4, 4), axis_wavelets=("db1", "db1"), wavelet_name="db2"),
        dict(spatial_shape=(4, 4), axis_boundaries=("zero", "zero"), boundary="zero"),
        dict(spatial_shape=(4, 4), filter_length=4),
        dict(spatial_shape=(4, 4), highpass=1),
        dict(spatial_shape=(4, 4), axis_levels=1),
    ],
)
def test_saved_spatial_rejects_conflicting_or_unsupported_settings(extra):
    with pytest.raises(ValueError):
        WFMMStudy(np.ones((8, 16)), np.ones((8, 1)), transform="spatial_wavelet", **extra)


def test_old_schema_defaults_and_nonspatial_rejects_spatial_settings():
    old = json.loads(WFMMStudy(np.ones((8, 16)), np.ones((8, 1))).to_json())
    for field in (
        "spatial_shape",
        "spatial_layout",
        "axis_levels",
        "axis_wavelets",
        "axis_boundaries",
    ):
        old.pop(field)
    assert read_wfmm_study(json.dumps(old)).spatial_shape is None
    with pytest.raises(ValueError):
        WFMMStudy(np.ones((8, 16)), np.ones((8, 1)), spatial_shape=(4, 4))


def test_spatial_initializer_hold_is_preserved_without_sampling():
    y = np.random.default_rng(46).normal(size=(8, 32))
    y += 5 * np.random.default_rng(411).normal(size=32)
    study = WFMMStudy(
        y,
        np.ones((8, 1)),
        transform="spatial_wavelet",
        spatial_shape=(4, 4, 2),
        draws=8,
        warmup=4,
        optimizer_iterations=50,
    )
    result = run_wfmm_study(read_wfmm_study(study.to_json()))
    assert "conditional_optimizer_not_converged" in result.initialization.holds
    assert result.fit is None and result.prediction is None
    assert json.loads(result.report_json())["status"] == "held"
