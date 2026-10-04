# Pinnacle exact peak-selection workflow

## Scope

The existing Pinnacle Python kernel detects local maxima and quantifies supplied
coordinates with the documented spot window, background and normalization
options. This addition makes the already-supported coordinate input usable as
an explicit manual-selection workflow: callers can create an ordered immutable
selection, add or remove exact coordinates, re-run quantification, and export
the measurements as a Python CSV.

## Contract and evidence boundary

The 2008 Pinnacle paper specifies peak quantification around selected locations
using the maximum intensity in a square neighborhood, followed by background
correction and normalization (see the source crosswalk in `docs/pinnacle.md` and
`docs/pinnacle-sources.json`). The current Python quantification API already
implements these computations and validates coordinate bounds against the
image and selected region. This workflow delegates to that implementation
rather than duplicating the measurement algorithm.

The detailed GUI manual itself is not present in the available local cache;
tracked audit notes describe its optional per-gel denoising and background
settings, but do not preserve peak-edit interaction rules or an export schema.
Accordingly, selection semantics here are stated Python conventions: exact
zero-based original-image `(row, column)` coordinates, input order preserved,
duplicates rejected, and moving a point performed by remove-then-add. There is
no inferred snapping-to-maximum behavior or native project/report format.

## Outputs and bounds

The immutable analysis couples the exact selection and its `PinnacleQuantification`
result. CSV output contains one row for every image/peak pair, the selected
coordinates, all four measurement stages, the per-image normalization factor,
and effective background/normalization settings. It uses numeric input-order
image indices so no filename identity is invented. Export is UTF-8 and written
to a temporary sibling before atomic replacement.

The workflow streams the iterable into the existing bounded quantifier; it
does not stack images. The existing caps on images, pixels, quantification
work, and image-by-peak outputs remain authoritative, and the exporter applies
the same two-million-cell output bound. This adds no new statistical
calculation and does not establish GUI, project-file, native report, or hidden
executable equivalence. Statistical coverage is unchanged; the gap addressed is
an explicit user-controlled analysis workflow and portable result export.

## Functional workflow crosswalk

The usable path is now:

1. Read supported TIFF frames through `PinnacleTiffSource` or supply aligned
   grayscale arrays.
2. Run the existing `run_pinnacle` pipeline to average, denoise and detect an
   initial set of locations; detected coordinates are available at
   `analysis.peaks.coordinates`.
3. Initialize `PinnaclePeakSelection` with those or other caller-selected
   coordinates, then explicitly add or remove exact points.
4. Stream aligned images through the existing local peak-window, background
   and normalization calculations with `quantify_selected_peaks`.
5. Write selected measurements and settings as UTF-8 CSV.

The existing numerical source checks remain the reference for measurement
behavior: `tests/test_pinnacle_reference.py` compares detection and
quantification against independent R fixtures, while
`tests/test_pinnacle_tiff_reference.py` checks decoded samples and agreement
with the array-based pipeline. This batch adds tests for exact coordinate
selection/editing, order-preserving re-quantification, image-dimension matching
and exported measurement/settings fields. The manual's click-to-snap and native
export conventions remain unknown, but there is no additional statistical
calculation missing from this declared Python workflow. Unsupported TIFF
encodings and native project/report compatibility are input/interchange gaps,
not unimplemented measurement formulas.
