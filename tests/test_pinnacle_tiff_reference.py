"""Independent tiny TIFF byte streams exercise the Pinnacle file adapter."""

import sys

import numpy as np
import pytest

pytest.importorskip("PIL")

from helpers.pinnacle_tiff_writer import write_gray_tiff

from mdanderson_stats.pinnacle_pipeline import run_pinnacle
from mdanderson_stats.pinnacle_tiff import PinnacleTiffSource


def _source_with_companion(path):
    companion = path.with_name(f"{path.stem}-companion{path.suffix}")
    companion.write_bytes(path.read_bytes())
    return PinnacleTiffSource([path, companion])


def _read_single(path):
    return next(iter(_source_with_companion(path)()))


@pytest.mark.parametrize(
    ("name", "samples", "options"),
    [
        (
            "u16-little-black",
            np.array([[0, 257], [1024, 65535]], dtype=np.uint16),
            {"byte_order": "little"},
        ),
        (
            "u16-big-black",
            np.array([[5, 257], [1024, 60000]], dtype=np.uint16),
            {"byte_order": "big"},
        ),
        (
            "u32-high-black",
            np.array([[0, 2**31 + 9], [2**32 - 1, 2**31 + 17]], dtype=np.uint32),
            {"byte_order": "little"},
        ),
        ("u8-white", np.array([[0, 7], [128, 255]], dtype=np.uint8), {"photometric": 0}),
        (
            "u8-white-deflate",
            np.array([[0, 7], [128, 255]], dtype=np.uint8),
            {"photometric": 0, "compression": 8},
        ),
        (
            "u16-white",
            np.array([[0, 257], [1024, 65535]], dtype=np.uint16),
            {"byte_order": "little", "photometric": 0},
        ),
        (
            "u16-white-deflate",
            np.array([[0, 257], [1024, 65535]], dtype=np.uint16),
            {"byte_order": "little", "photometric": 0, "compression": 8},
        ),
    ],
)
def test_stored_integer_pixels_survive_byte_order_photometric_and_deflate(
    tmp_path, name, samples, options
):
    path = tmp_path / f"{name}.tif"
    write_gray_tiff(str(path), [samples], **options)
    source = _source_with_companion(path)
    np.testing.assert_array_equal(next(iter(source())), samples)
    if options.get("photometric") == 0:
        assert source.frame_metadata[0].photometric == 0


@pytest.mark.parametrize(
    ("byte_order", "compression"),
    [("little", 1), ("little", 8), ("big", 1), ("big", 8)],
)
def test_float_decode_preserves_endian_and_deflate_values(tmp_path, byte_order, compression):
    float_pixels = np.array([[0.25, 1.5], [3.125, 8.0]], dtype=np.float32)
    float_path = tmp_path / f"float32-{byte_order}-{compression}.tif"
    write_gray_tiff(
        str(float_path),
        [float_pixels],
        byte_order=byte_order,
        sample_format=3,
        compression=compression,
    )
    np.testing.assert_array_equal(_read_single(float_path), float_pixels)


@pytest.mark.parametrize(
    ("dtype", "sample_format", "byte_order", "compression"),
    [
        (np.int16, 2, byte_order, compression)
        for byte_order in ("little", "big")
        for compression in (1, 8)
    ]
    + [
        (np.int32, 2, byte_order, compression)
        for byte_order in ("little", "big")
        for compression in (1, 8)
    ],
)
def test_signed_samples_preserve_raw_values_or_reject_nonnative_libtiff(
    tmp_path, dtype, sample_format, byte_order, compression
):
    maximum = np.iinfo(dtype).max
    pixels = np.array([[0, 257], [1024, maximum]], dtype=dtype)
    path = tmp_path / f"signed-{np.dtype(dtype).itemsize}-{byte_order}-{compression}.tif"
    write_gray_tiff(
        str(path),
        [pixels],
        byte_order=byte_order,
        sample_format=sample_format,
        compression=compression,
    )
    uses_nonnative_libtiff = compression == 8 and byte_order != sys.byteorder
    if uses_nonnative_libtiff:
        with pytest.raises(ValueError, match="signed.*byte order|byte order.*signed"):
            _source_with_companion(path)
    else:
        source = _source_with_companion(path)
        np.testing.assert_array_equal(next(iter(source())), pixels)


def test_orientation_and_full_pipeline_match_direct_arrays(tmp_path):
    oriented = np.arange(8 * 16, dtype=np.uint16).reshape(8, 16)
    orientation_path = tmp_path / "orientation-6.tif"
    write_gray_tiff(str(orientation_path), [oriented], orientation=6)
    normalized = np.rot90(oriented, k=3)
    orientation_source = _source_with_companion(orientation_path)
    np.testing.assert_array_equal(next(iter(orientation_source())), normalized)
    metadata = orientation_source.frame_metadata[0]
    assert metadata.orientation == 6
    assert metadata.stored_shape == oriented.shape
    assert metadata.shape == normalized.shape

    first = np.ones((8, 8), dtype=np.uint16)
    second = first.copy()
    first[3, 3] = 10
    second[3, 3] = 12
    first_path, second_path = tmp_path / "first.tif", tmp_path / "second.tif"
    write_gray_tiff(str(first_path), [first], byte_order="little")
    write_gray_tiff(str(second_path), [second], byte_order="big", compression=8)
    tiff_source = PinnacleTiffSource([first_path, second_path])

    options = dict(
        filter_length=2,
        levels=1,
        threshold_multiplier=0,
        threshold_quantile=0.8,
        suppression_radius=1,
        peak_radius=0,
        background="none",
        normalization="none",
    )
    tiff_result = run_pinnacle(tiff_source, **options)
    array_result = run_pinnacle(lambda: iter((first, second)), **options)
    np.testing.assert_array_equal(tiff_result.average_image, array_result.average_image)
    np.testing.assert_array_equal(tiff_result.peaks.coordinates, array_result.peaks.coordinates)
    np.testing.assert_array_equal(tiff_result.quantification.raw, array_result.quantification.raw)


def test_big_endian_white_is_zero_16_bit_is_supported_or_rejected_clearly(tmp_path):
    samples = np.array([[0, 257], [1024, 65535]], dtype=np.uint16)
    path = tmp_path / "u16-white-big.tif"
    companion = tmp_path / "u16-white-big-companion.tif"
    write_gray_tiff(str(path), [samples], byte_order="big", photometric=0)
    write_gray_tiff(str(companion), [samples], byte_order="big", photometric=0)
    try:
        source = PinnacleTiffSource([path, companion])
    except ValueError as exc:
        assert "cannot inspect TIFF" in str(exc)
    else:
        np.testing.assert_array_equal(next(iter(source())), samples)


def test_multipage_tiff_requires_and_honors_explicit_frame_selection(tmp_path):
    first = np.arange(64, dtype=np.uint16).reshape(8, 8)
    second = first + 500
    path = tmp_path / "multipage.tif"
    write_gray_tiff(str(path), [first, second], byte_order="big")

    single_page = tmp_path / "single-page.tif"
    write_gray_tiff(str(single_page), [first])
    with pytest.raises(ValueError, match="frame_indices"):
        PinnacleTiffSource([path, single_page])

    selected = PinnacleTiffSource([path, path], frame_indices=[0, 1])
    images = iter(selected())
    np.testing.assert_array_equal(next(images), first)
    np.testing.assert_array_equal(next(images), second)
    canonical = path.expanduser().resolve(strict=True)
    assert selected.image_ids == (f"{canonical}#frame=0", f"{canonical}#frame=1")


@pytest.mark.parametrize(
    "samples",
    [
        np.full((8, 8), -1.0, dtype=np.float32),
        np.full((8, 8), np.nan, dtype=np.float32),
    ],
)
def test_nonfinite_or_negative_float_pixels_are_rejected(tmp_path, samples):
    path = tmp_path / "invalid-float.tif"
    write_gray_tiff(str(path), [samples], sample_format=3)
    companion = tmp_path / "invalid-float-companion.tif"
    write_gray_tiff(str(companion), [np.ones((8, 8), dtype=np.float32)], sample_format=3)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        next(iter(PinnacleTiffSource([path, companion])()))


def test_decode_workspace_budget_is_preflighted(tmp_path):
    path = tmp_path / "workspace.tif"
    write_gray_tiff(str(path), [np.ones((8, 8), dtype=np.uint16)])
    companion = tmp_path / "workspace-companion.tif"
    write_gray_tiff(str(companion), [np.ones((8, 8), dtype=np.uint16)])
    with pytest.raises(ValueError, match="decode workspace estimate exceeds max_decode_bytes"):
        PinnacleTiffSource([path, companion], max_decode_bytes=3071)
