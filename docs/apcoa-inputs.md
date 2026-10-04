# Labeled input preparation for aPCoA

The core `adjusted_pcoa` calculation accepts a numeric distance matrix and
already encoded numeric covariates. `read_apcoa_distance_csv` and
`read_apcoa_metadata_csv` provide a bounded standard-library path from the
application's labeled CSV/TSV inputs to that calculation. TSV is selected
explicitly with `delimiter="\t"`; the reader does not guess delimiters.

The distance table has a blank top-left header cell, sample IDs across the
header, then one row per sample with its ID in the first column. Rows can arrive
in any order; they are reordered to header order. IDs must be unique and row
and column ID sets must match exactly. The metadata table has sample IDs in its
first column and unique column names in its header. Metadata IDs are aligned to
the distance header, also by exact matching. The readers allow 2–2,000 samples,
IDs and column names up to 256 characters, at most 256 metadata columns, a
128 MiB distance file, and a 16 MiB metadata file. The core retains its own
matrix and calculation limits.

Select nuisance covariates explicitly. Numeric columns must contain finite
numbers and are retained at their supplied scale, including constant or
redundant columns for the core's SVD rank policy. For a categorical column,
provide an ordered level tuple and an explicit reference level. The adapter
creates one indicator column for every non-reference level, in the specified
order, and rejects values absent from that declared level list. Column names
are written as `column[level]` and retained with the encoding provenance.
The `main_group` column is aligned and returned separately for plotting; it is
never silently added to the adjustment matrix. A caller may also explicitly
select that column as a nuisance covariate; doing so removes its encoded linear
contribution from the adjusted ordination.

This is an explicit Python encoding convention, not arbitrary R formula
emulation. In particular, aPCoA's core does not center covariates by default,
so the chosen reference coding and the `intercept` option can change the
projection. The source R function uses its formula/model-matrix and QR rules;
this helper does not claim to reproduce every R contrast, interaction, formula
transform, or rank-pivot choice. Set `intercept=True` only when the constant
direction should be included in the Python projection. See
[`docs/apcoa.md`](apcoa.md) for the matrix equations and numerical conventions.
The reviewed native formula and row-matching path is in
[`aPCoA.R`](../research/raw/aPCoA/aPCoA/R/aPCoA.R); the application CSV controls
are recorded in [`app.html`](../research/raw/aPCoA/app.html).

```python
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from mdanderson_stats.apcoa_inputs import (
    CategoricalEncoding,
    prepare_apcoa_input,
    read_apcoa_distance_csv,
    read_apcoa_metadata_csv,
)

with TemporaryDirectory() as folder:
    distance_path = Path(folder) / "distance.csv"
    metadata_path = Path(folder) / "metadata.csv"
    with distance_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["", "s1", "s2", "s3", "s4"],
            ["s1", 0, 1, 3, 4],
            ["s2", 1, 0, 2, 3],
            ["s3", 3, 2, 0, 1],
            ["s4", 4, 3, 1, 0],
        ])
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerows([
            ["sample_id", "age", "batch", "treatment"],
            ["s1", 20, "A", "control"],
            ["s2", 35, "B", "control"],
            ["s3", 42, "A", "drug"],
            ["s4", 55, "C", "drug"],
        ])

    distance = read_apcoa_distance_csv(distance_path)
    metadata = read_apcoa_metadata_csv(metadata_path)
    prepared = prepare_apcoa_input(
        distance,
        metadata,
        numeric_covariates=("age",),
        categorical_covariates={
            "batch": CategoricalEncoding(("A", "B", "C"), reference="A")
        },
        main_group="treatment",
    )
    result = prepared.fit(components=None)
    assert result.covariate_rank <= len(prepared.covariate_columns)
    assert prepared.group_labels == ("control", "control", "drug", "drug")
```
