# Pinnacle peak selection and re-quantification

Use `PinnaclePeakSelection` when you want to keep, add, or remove exact peak
locations before running the existing quantification method. Coordinates are
zero-based `(row, column)` indices in the original image. The selection uses
each coordinate exactly as entered; it does not move a point to a nearby local
maximum. Repeated coordinates are rejected. To move a selection, remove the old
coordinate and add the new one.

To edit automatic detections, initialize the selection from
`run_pinnacle(...).peaks.coordinates`; those coordinates already use original
image dimensions. The example below uses explicit locations to keep the
measurement reproducible without depending on a detector threshold.

The measurement rules remain those of `pinnacle_quantify`: a local maximum
within the requested square, an explicit background method, and an explicit
normalization. The selected order is retained in every output matrix and in the
CSV export. Input images are consumed one at a time; no image stack is created.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mdanderson_stats.pinnacle_peak_selection import (
    PinnaclePeakSelection,
    quantify_selected_peaks,
    write_pinnacle_selection_csv,
)

first = np.full((8, 9), 2.0)
second = np.full((8, 9), 4.0)
first[3, 2], first[5, 6] = 11.0, 8.0
second[3, 2], second[5, 6] = 15.0, 12.0

selection = PinnaclePeakSelection.from_coordinates(
    [[3, 2], [5, 6]], first.shape, region=(1, 7, 1, 8)
)
selection = selection.remove_peak((5, 6)).add_peak((4, 6))

# A one-pixel peak window makes the selected pixel itself the raw measurement.
analysis = quantify_selected_peaks(
    (image for image in (first, second)),
    selection,
    peak_radius=0,
    background="none",
    normalization="none",
)
assert analysis.quantification.raw[:, 0].tolist() == [11.0, 15.0]
assert analysis.quantification.coordinates.tolist() == [[3, 2], [4, 6]]

with TemporaryDirectory() as directory:
    output = write_pinnacle_selection_csv(analysis, Path(directory) / "selected-peaks.csv")
    assert output.is_file()
```

The UTF-8 CSV is a Python export, not a Pinnacle project or native report. It is
long-form: one row per image and selected peak, with indices, coordinates,
raw/background/corrected/normalized measurements, normalization factor, and
the effective quantification settings. Images are identified by zero-based
input order; callers should retain their own mapping from image indices to
source files.

Selections must be within both the image bounds and the chosen half-open
region. Quantification applies the existing image, image-count, pixel-work,
peak-count and output-cell limits. A nonempty selection is required for this
workflow. Native peak snapping, duplicate handling, GUI edits, project files
and report serialization are not inferred from this API.
