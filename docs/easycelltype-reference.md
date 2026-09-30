# EasyCellType local reference tables

The path-based loader selects database, species and tissue rows from a
caller-provided author-format CSV or gzip CSV. For the packaged author snapshots,
use the [bundled-reference loader](easycelltype-builtin-reference.md).
This small example uses a temporary synthetic table, so it can run from an
installed package without private or external data:

```python
import csv
import tempfile
from pathlib import Path

from mdanderson_stats.easycelltype_gsea import easycelltype_gsea_es
from mdanderson_stats.easycelltype_reference import easycelltype_reference

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "markers.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["celltype", "spe", "organ", "entrezid"])
        writer.writerows(
            [
                ["T cell", "Human", "Blood", "7157"],
                ["T cell", "Human", "Blood", "1956"],
                ["B cell", "Human", "Blood", "7422"],
                ["T cell", "Mouse", "Blood", "9"],
            ]
        )

    reference = easycelltype_reference(
        path,
        database="cellmarker",
        species="Human",
        tissues=["Blood"],
        source_version="synthetic example",
    )
    result = easycelltype_gsea_es(
        ["7157", "1956", "7422", "x"],
        ["cluster"] * 4,
        [3.0, 2.0, 1.0, 0.0],
        reference.genes,
        reference.cell_types,
    )
    print(reference.selected_rows, result.clusters[0].sets[0].cell_type)
```

The loader requires columns in this exact order: `celltype`, `spe`, `organ`,
`entrezid`. It preserves row order and duplicate associations. Empty `tissues`
or `None` selects all organs for the chosen species; a blank source `organ`
value is retained and can be selected with `tissues=[""]`. Every requested
tissue must occur for that species.

The returned reference records requested and selected tissues, row counts,
source filename, SHA-256 and caller-supplied version/provenance. Its identifiers
remain in the EntrezID namespace; symbol conversion is left to a caller with an
explicit, versioned mapping. Input limits are 100 MB on disk, 32 MB expanded
text, 65,536 characters per physical line and 200,000 rows. See the
[source audit](../research/easycelltype-reference-audit.md) and the main
[EasyCellType guide](easycelltype.md) for scope and method details.

For a locally obtained EasyCellType 1.5.4 `R/sysdata.rda`, the repository includes
an optional one-time exporter using base R:

```sh
Rscript --vanilla tools/reference_easycelltype_data.R path/to/sysdata.rda local-reference-csv
```

First verify the input against the SHA-256 in the source audit. The exporter
writes `cellmarker.csv`, `clustermole.csv` and `panglao.csv`, plus small filtered
reference files. Pass the desired CSV to `easycelltype_reference` and record its
version and source in the metadata arguments. Python reference loading and
annotation do not require R. Use each reference table under its applicable
terms. The packaged references are fixed exports of the pinned author snapshot;
this optional exporter is for preparing local files from an original R asset.
