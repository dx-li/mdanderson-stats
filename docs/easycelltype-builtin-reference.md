# Bundled EasyCellType marker references

`easycelltype_builtin_reference` loads one of the pinned CellMarker, Clustermole,
or PanglaoDB tables and selects rows for a species and optional tissues. The
returned immutable reference can be passed directly to the existing Fisher or
GSEA workflows:

```python
from mdanderson_stats import easycelltype_builtin_reference, easycelltype_fisher

reference = easycelltype_builtin_reference(
    "cellmarker", "Human", tissues=("Kidney",)
)
result = easycelltype_fisher(
    query_genes=("915", "916", "917"),
    clusters=("cluster-1",) * 3,
    scores=(1.0, 0.9, 0.8),
    reference_genes=reference.genes,
    reference_cell_types=reference.cell_types,
)
assert reference.source_name == "cellmarker.csv.gz"
print(reference.selected_rows, result.clusters[0].cluster)
```

The identifiers are Entrez IDs. This loader does not map symbols, download
updates, or start R. Symbol conversion still requires a separately versioned
mapping chosen by the caller. The bundled files are the tables embedded in
EasyCellType 1.5.4; that snapshot did not record the individual upstream
database release versions. They are reproducible snapshots, not claims to be
the current live databases. See the [source and validation audit](../research/easycelltype-builtin-reference-audit.md)
for hashes and attribution details.

Filtering preserves source row order and duplicates because reference row
multiplicity is part of the existing Fisher/GSEA input contract. `None` or an
empty tissue sequence selects all organs for the species. The selected
reference records both requested tissues and the actual selected source organs.
The underlying reader bounds compressed and expanded input size, line length,
and row count; the bundled tables are also checked against their pinned
compressed-file hashes and source row counts before return.
