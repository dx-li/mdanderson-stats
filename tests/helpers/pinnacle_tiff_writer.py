"""Write tiny uncompressed grayscale TIFFs without a TIFF library."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def write_gray_tiff(
    path: str,
    pages: Sequence[NDArray[np.generic]],
    *,
    byte_order: str = "little",
    orientation: int = 1,
    photometric: int = 1,
    sample_format: int = 1,
    compression: int = 1,
) -> None:
    """Write a classic TIFF with one uncompressed strip per grayscale page.

    Supported samples are unsigned 8/16/32-bit integers, signed 16/32-bit
    integers, and 32-bit IEEE floats; supported compression is none or TIFF
    Deflate. All fixture layout and
    pixel bytes are built with ``struct``/``zlib``; Pillow is deliberately not
    used as the writer oracle.
    """
    if not pages:
        raise ValueError("at least one TIFF page is required")
    if byte_order not in {"little", "big"}:
        raise ValueError("byte_order must be 'little' or 'big'")
    if orientation not in range(1, 9):
        raise ValueError("orientation must be a TIFF value from 1 through 8")
    if photometric not in {0, 1}:
        raise ValueError("photometric must be WhiteIsZero (0) or BlackIsZero (1)")
    if sample_format not in {1, 2, 3}:
        raise ValueError("sample_format must be unsigned (1), signed (2), or float (3)")
    if compression not in {1, 8}:
        raise ValueError("compression must be none (1) or Deflate (8)")

    endian = "<" if byte_order == "little" else ">"
    page_data: list[tuple[int, int, int, bytes]] = []
    for page in pages:
        values = np.asarray(page)
        if values.ndim != 2 or min(values.shape, default=0) < 1:
            raise ValueError("each page must be a nonempty two-dimensional grayscale array")
        height, width = (int(values.shape[0]), int(values.shape[1]))
        if sample_format == 3:
            if values.dtype.kind != "f" or values.dtype.itemsize != 4:
                raise ValueError("float sample format requires float32 arrays")
            bits, pack_code = 32, "f"
            pixels = struct.pack(
                endian + f"{values.size}{pack_code}", *values.astype(np.float32).ravel()
            )
        elif sample_format == 2:
            if values.dtype.kind != "i" or values.dtype.itemsize not in (2, 4):
                raise ValueError("signed sample format requires int16 or int32 arrays")
            bits = values.dtype.itemsize * 8
            pack_code = "h" if bits == 16 else "i"
            pixels = struct.pack(endian + f"{values.size}{pack_code}", *values.ravel())
        else:
            if values.dtype.kind not in "ui" or values.dtype.itemsize not in (1, 2, 4):
                raise ValueError("unsigned sample format requires 8/16/32-bit integer arrays")
            if values.dtype.kind == "i" and np.any(values < 0):
                raise ValueError("unsigned TIFF samples cannot be negative")
            bits = values.dtype.itemsize * 8
            pack_code = {8: "B", 16: "H", 32: "I"}[bits]
            pixels = struct.pack(
                endian + f"{values.size}{pack_code}", *values.astype(np.uint64).ravel()
            )
        if compression == 8:
            pixels = zlib.compress(pixels)
        page_data.append((height, width, bits, pixels))

    tags_per_page = 12
    ifd_size = 2 + tags_per_page * 12 + 4
    first_ifd = 8
    data_start = first_ifd + ifd_size * len(page_data)
    strip_offsets: list[int] = []
    next_offset = data_start
    for _, _, _, pixels in page_data:
        strip_offsets.append(next_offset)
        next_offset += len(pixels)

    header = (b"II" if byte_order == "little" else b"MM") + struct.pack(
        endian + "HI", 42, first_ifd
    )
    output = bytearray(header)
    for index, (height, width, bits, pixels) in enumerate(page_data):
        ifd_offset = first_ifd + index * ifd_size
        assert len(output) == ifd_offset
        entries = [
            (256, 4, 1, width),
            (257, 4, 1, height),
            (258, 3, 1, bits),
            (259, 3, 1, compression),
            (262, 3, 1, photometric),
            (273, 4, 1, strip_offsets[index]),
            (274, 3, 1, orientation),
            (277, 3, 1, 1),
            (278, 4, 1, height),
            (279, 4, 1, len(pixels)),
            (284, 3, 1, 1),
            (339, 3, 1, sample_format),
        ]
        output.extend(struct.pack(endian + "H", len(entries)))
        for tag, field_type, count, value in entries:
            if field_type == 3:
                output.extend(struct.pack(endian + "HHI", tag, field_type, count))
                output.extend(struct.pack(endian + "H", value))
                output.extend(b"\x00\x00")
            else:
                output.extend(struct.pack(endian + "HHII", tag, field_type, count, value))
        next_ifd = ifd_offset + ifd_size if index + 1 < len(page_data) else 0
        output.extend(struct.pack(endian + "I", next_ifd))
    for _, _, _, pixels in page_data:
        output.extend(pixels)

    with open(path, "wb") as target:
        target.write(output)
