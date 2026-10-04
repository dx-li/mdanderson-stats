# Independent TIFF input fixtures

`tests/helpers/pinnacle_tiff_writer.py` writes tiny classic grayscale TIFFs
directly from the TIFF header, IFD tags, and pixel bytes using Python's
`struct` and `zlib`; Pillow is not used to create the reference files. The
focused test covers little- and big-endian 16-bit pixels, unsigned 32-bit
values above the signed-int32 range, little-endian WhiteIsZero 8/16-bit stored
samples with raw and Deflate routes, signed 16/32-bit samples, float32 in both
byte orders with and without Deflate compression, TIFF orientation 6, and
explicit selection of pages in a two-frame file. Big-endian
WhiteIsZero 16-bit TIFF is separately recorded as unsupported by the current
Pillow backend (it is absent from Pillow's open-mode table), independent of
compression.

Expected arrays are the original stored numeric samples, transformed once by
the TIFF orientation rule where needed. A pipeline comparison runs those
decoded arrays both through the TIFF source and directly through
`run_pinnacle`, checking that file input does not change the existing
statistical results. Multi-frame selection uses explicit zero-based frame
indices per input path; the default path is expected to reject multipage files
rather than silently choosing a page.

The tests also reject negative/nonfinite decoded values at the source boundary
and verify the reader's documented per-frame decompression-workspace limit
before decoding.

These fixtures test grayscale pixel decoding only. They do not test native
Pinnacle project files, resampling, image registration, or any change to the
Pinnacle statistical calculations.

Validation on 2026-10-03: the focused reference file passed 25 tests in 2.33 s
(wrapper elapsed 2.647 s; child maximum RSS 154,058,752 bytes, about 146.92 MiB;
zero swaps). The Orientation-6 uint16 fixture also confirmed that opening the
TIFF through a binary stream avoids Pillow's path-mmap orientation sizing issue
and yields the expected 16-by-8 array and rotated first row.
