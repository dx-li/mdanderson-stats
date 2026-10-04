"""Replayable, bounded grayscale TIFF input for the Pinnacle workflow."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .pinnacle import _MAX_IMAGES, _MAX_PIXELS

_MAX_DECODE_BYTES = 1024 * 1024 * 1024
_DEFAULT_DECODE_BYTES = 512 * 1024 * 1024
_SUPPORTED_MODES = {"L", "I", "I;16", "I;16B", "I;16L", "F"}


def _pillow_image_module():
    try:
        import PIL
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PinnacleTiffSource requires Pillow; install mdanderson-stats[image]"
        ) from exc
    version_match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", PIL.__version__)
    if version_match is None or not (12, 3) <= tuple(map(int, version_match.groups())) < (13, 0):
        raise ImportError(
            "PinnacleTiffSource supports Pillow >=12.3,<13; install mdanderson-stats[image]"
        )
    return Image


class _PillowImage(Protocol):
    size: tuple[int, int]
    mode: str
    tag_v2: Mapping[int, object]

    def seek(self, frame: int) -> None: ...

    def load(self) -> object: ...


@dataclass(frozen=True)
class PinnacleTiffFrame:
    """Immutable metadata for one selected TIFF page."""

    path: Path
    frame_index: int
    selected_explicitly: bool
    image_id: str
    shape: tuple[int, int]
    stored_shape: tuple[int, int]
    mode: str
    bits_per_sample: int
    sample_format: int
    photometric: int
    orientation: int
    byte_order: str
    decoder_float_byteswap: bool


def _tag_ints(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        values = (value,)
    if not values or any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ValueError(f"TIFF {name} tag must contain integers")
    return tuple(int(item) for item in values)


def _frame_metadata(
    image: _PillowImage, path: Path, frame: int, *, selected_explicitly: bool
) -> PinnacleTiffFrame:
    """Read and validate TIFF headers without decoding pixel data."""
    size = image.size
    mode = image.mode
    tags = image.tag_v2
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"{path} frame {frame} is not a supported scalar grayscale TIFF")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError(f"{path} frame {frame} has invalid dimensions")
    width, height = size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width * height > _MAX_PIXELS
    ):
        raise ValueError(f"{path} frame {frame} exceeds the {_MAX_PIXELS}-pixel limit")

    bits = _tag_ints(tags.get(258, 8), "BitsPerSample")
    samples = _tag_ints(tags.get(277, 1), "SamplesPerPixel")
    sample_format = _tag_ints(tags.get(339, 1), "SampleFormat")
    photometric = _tag_ints(tags.get(262, 0), "PhotometricInterpretation")
    orientation = _tag_ints(tags.get(274, 1), "Orientation")
    prefix = getattr(tags, "prefix", None)
    byte_order = prefix.decode("ascii") if isinstance(prefix, bytes) else prefix
    if len(samples) != 1 or samples[0] != 1:
        raise ValueError(f"{path} frame {frame} must have exactly one grayscale sample per pixel")
    if len(bits) != 1 or bits[0] not in (8, 16, 32):
        raise ValueError(f"{path} frame {frame} must use 8-, 16-, or 32-bit pixels")
    if len(sample_format) != 1 or sample_format[0] not in (1, 2, 3):
        raise ValueError(f"{path} frame {frame} has an unsupported TIFF SampleFormat")
    if len(photometric) != 1 or photometric[0] not in (0, 1):
        raise ValueError(f"{path} frame {frame} must use WhiteIsZero or BlackIsZero grayscale")
    if len(orientation) != 1 or orientation[0] not in range(1, 9):
        raise ValueError(f"{path} frame {frame} has an invalid TIFF orientation")
    if byte_order not in ("II", "MM"):
        raise ValueError(f"{path} frame {frame} has an unsupported TIFF byte order")
    if mode == "F" and (bits[0] != 32 or sample_format[0] != 3):
        raise ValueError(f"{path} frame {frame} has inconsistent floating-point TIFF metadata")
    if mode in ("I;16", "I;16B", "I;16L") and bits[0] != 16:
        raise ValueError(f"{path} frame {frame} has inconsistent 16-bit TIFF metadata")
    if mode == "L" and (bits[0] != 8 or sample_format[0] != 1):
        raise ValueError(f"{path} frame {frame} has inconsistent 8-bit TIFF metadata")
    if mode == "I" and (bits[0] not in (16, 32) or sample_format[0] not in (1, 2)):
        raise ValueError(f"{path} frame {frame} has inconsistent integer TIFF metadata")
    stored_width = _tag_ints(tags.get(256, width), "ImageWidth")
    stored_height = _tag_ints(tags.get(257, height), "ImageLength")
    if len(stored_width) != 1 or len(stored_height) != 1:
        raise ValueError(f"{path} frame {frame} has invalid stored dimensions")
    expected_shape = (
        (stored_width[0], stored_height[0])
        if orientation[0] in (5, 6, 7, 8)
        else (stored_height[0], stored_width[0])
    )
    if (height, width) != expected_shape:
        raise ValueError(f"{path} frame {frame} has inconsistent orientation dimensions")
    decoder_float_byteswap = False
    tiles: Sequence[object] = getattr(image, "tile", ())
    tile = tiles[0] if len(tiles) == 1 else None
    args = getattr(tile, "args", None)
    rawmode = args[0] if isinstance(args, tuple) and args else None
    is_libtiff = getattr(tile, "codec_name", None) == "libtiff"
    tiff_big_endian = byte_order == "MM"
    host_big_endian = sys.byteorder == "big"
    if mode == "F" and bits[0] == 32 and sample_format[0] == 3 and is_libtiff:
        expected_rawmode = "F;32BF" if tiff_big_endian else "F;32F"
        decoder_float_byteswap = rawmode == expected_rawmode and tiff_big_endian != host_big_endian
    if (
        mode == "I"
        and sample_format[0] == 2
        and bits[0] in (16, 32)
        and is_libtiff
        and tiff_big_endian != host_big_endian
    ):
        raise ValueError(
            f"{path} frame {frame} uses a non-native signed-integer libtiff byte order "
            "unsupported by Pillow's decoder"
        )
    return PinnacleTiffFrame(
        path=path,
        frame_index=frame,
        selected_explicitly=selected_explicitly,
        image_id=f"{path}#frame={frame}",
        shape=(height, width),
        stored_shape=(stored_height[0], stored_width[0]),
        mode=mode,
        bits_per_sample=bits[0],
        sample_format=sample_format[0],
        photometric=photometric[0],
        orientation=orientation[0],
        byte_order=byte_order,
        decoder_float_byteswap=decoder_float_byteswap,
    )


class PinnacleTiffSource:
    """A replayable, one-frame-at-a-time source accepted by :func:`run_pinnacle`.

    Paths are processed in the order supplied. A single-frame TIFF needs no
    selector; a multi-frame TIFF is rejected unless its zero-based page is
    explicitly selected in ``frame_indices``. Repeated paths can select
    different pages. ``image_ids`` records canonical paths and page indices.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        frame_indices: Sequence[int | None] | None = None,
        max_images: int = _MAX_IMAGES,
        max_decode_bytes: int = _DEFAULT_DECODE_BYTES,
    ) -> None:
        Image = _pillow_image_module()

        if isinstance(paths, (str, bytes, Path)) or not isinstance(paths, Sequence):
            raise ValueError("paths must be a finite ordered sequence of TIFF paths")
        if not 2 <= len(paths) <= _MAX_IMAGES:
            raise ValueError(f"paths must contain between 2 and {_MAX_IMAGES} TIFF files")
        if (
            isinstance(max_images, bool)
            or not isinstance(max_images, int)
            or not 2 <= max_images <= _MAX_IMAGES
        ):
            raise ValueError(f"max_images must be an integer in [2,{_MAX_IMAGES}]")
        if (
            isinstance(max_decode_bytes, bool)
            or not isinstance(max_decode_bytes, int)
            or not 1 <= max_decode_bytes <= _MAX_DECODE_BYTES
        ):
            raise ValueError(f"max_decode_bytes must be an integer in [1,{_MAX_DECODE_BYTES}]")
        if frame_indices is None:
            selectors: tuple[int | None, ...] = (None,) * len(paths)
        else:
            if isinstance(frame_indices, (str, bytes)) or not isinstance(frame_indices, Sequence):
                raise ValueError("frame_indices must be an ordered sequence or None")
            if len(frame_indices) != len(paths):
                raise ValueError("frame_indices must have one selector per path")
            selectors = tuple(frame_indices)
            for selector in selectors:
                if selector is not None and (
                    isinstance(selector, bool)
                    or not isinstance(selector, int)
                    or not 0 <= selector < _MAX_IMAGES
                ):
                    raise ValueError(
                        f"frame selectors must be None or integers in [0,{_MAX_IMAGES})"
                    )
        if len(paths) > max_images:
            raise ValueError("TIFF source exceeds max_images")

        canonical = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
        frames: list[PinnacleTiffFrame] = []
        estimated_bytes = 0
        aligned_shape: tuple[int, int] | None = None
        selected_pairs: set[tuple[Path, int]] = set()
        for path, selector in zip(canonical, selectors, strict=True):
            if not path.is_file():
                raise ValueError(f"TIFF path is not a regular file: {path}")
            try:
                with path.open("rb") as stream, Image.open(stream, formats=["TIFF"]) as image:
                    if selector is None:
                        try:
                            image.seek(1)
                        except EOFError:
                            selected_frame = 0
                            image.seek(0)
                        else:
                            raise ValueError(
                                f"{path} contains multiple frames; provide a frame_indices entry"
                            )
                    else:
                        selected_frame = selector
                        try:
                            image.seek(selected_frame)
                        except EOFError as exc:
                            raise ValueError(
                                f"frame selector {selected_frame} is outside {path}"
                            ) from exc
                    metadata = _frame_metadata(
                        image,
                        path,
                        selected_frame,
                        selected_explicitly=selector is not None,
                    )
            except (OSError, SyntaxError) as exc:
                raise ValueError(f"cannot inspect TIFF {path}: {exc}") from exc
            if aligned_shape is None:
                aligned_shape = metadata.shape
            elif metadata.shape != aligned_shape:
                raise ValueError("all selected TIFF frames must have identical dimensions")
            pair = (path, metadata.frame_index)
            if pair in selected_pairs:
                raise ValueError("the same TIFF frame cannot be selected more than once")
            selected_pairs.add(pair)
            pixels = metadata.shape[0] * metadata.shape[1]
            # Covers decoder storage, Pillow's array-interface byte buffer,
            # the owned decoded array, float64 conversion, and the previous
            # yielded/cast frame that can remain live while the next loads.
            estimated_bytes = max(estimated_bytes, pixels * 48)
            frames.append(metadata)
        if estimated_bytes > max_decode_bytes:
            raise ValueError(
                "TIFF decode workspace estimate exceeds max_decode_bytes "
                f"({estimated_bytes} > {max_decode_bytes})"
            )
        self._frames = tuple(frames)
        self.max_decode_bytes = max_decode_bytes
        self.estimated_decode_bytes = estimated_bytes

    @property
    def frame_metadata(self) -> tuple[PinnacleTiffFrame, ...]:
        return self._frames

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(frame.image_id for frame in self._frames)

    def __call__(self) -> Iterator[NDArray[np.generic]]:
        """Return a fresh iterator; each file closes before its frame is yielded."""
        return self._read_images()

    def _read_images(self) -> Iterator[NDArray[np.generic]]:
        Image = _pillow_image_module()

        for expected in self._frames:
            with expected.path.open("rb") as stream, Image.open(stream, formats=["TIFF"]) as image:
                if not expected.selected_explicitly:
                    try:
                        image.seek(1)
                    except EOFError:
                        image.seek(0)
                    else:
                        raise ValueError(
                            f"TIFF gained additional frames after preflight: {expected.path}"
                        )
                try:
                    image.seek(expected.frame_index)
                except EOFError as exc:
                    raise ValueError(
                        f"TIFF frame disappeared after preflight: {expected.image_id}"
                    ) from exc
                actual = _frame_metadata(
                    image,
                    expected.path,
                    expected.frame_index,
                    selected_explicitly=expected.selected_explicitly,
                )
                if actual != expected:
                    raise ValueError(f"TIFF metadata changed after preflight: {expected.image_id}")
                image.load()
                pixels: NDArray[np.integer | np.floating] = np.array(image, copy=True)
            # Pillow 12.3 decodes some 8-bit WhiteIsZero scalar TIFFs through
            # the inverting L;I raw mode. Restore the stored samples. Other
            # supported depths use non-inverting raw modes in this decoder.
            if expected.photometric == 0 and expected.bits_per_sample == 8:
                pixels = np.asarray(255 - pixels, dtype=pixels.dtype)
            if (
                expected.decoder_float_byteswap
                and pixels.dtype.kind == "f"
                and pixels.dtype.itemsize == 4
            ):
                # Pillow's libtiff path returns native-order bytes, but the
                # unchanged TIFF-order float rawmode interprets them as file order.
                pixels = pixels.byteswap()
            if (
                expected.bits_per_sample == 32
                and expected.sample_format == 1
                and pixels.dtype.kind == "i"
                and pixels.dtype.itemsize == 4
            ):
                pixels = pixels.view(np.uint32)
            if pixels.ndim != 2 or pixels.shape != expected.shape or pixels.dtype.kind not in "iuf":
                raise ValueError(f"TIFF decoder returned an unsupported array: {expected.image_id}")
            if not np.all(np.isfinite(pixels)) or np.any(pixels < 0):
                raise ValueError(f"TIFF pixels must be finite and nonnegative: {expected.image_id}")
            yield pixels
