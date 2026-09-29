# EasyCellType local reference selection audit

## Source contract

The author snapshot is EasyCellType 1.5.4 at commit
`e85e8187c540f66994b5ca12fe95f5d9eb95f1f5`. Its `R/easyct.R` source accepts
`cellmarker`, `clustermole` and `panglao`, restricts the selected table to
`Human` or `Mouse`, then optionally filters `organ` by the requested tissue
values. The table columns used by that wrapper are `celltype`, `spe`, `organ`
and `entrezid`. Species and tissue selection preserve source row order and
repeated associations; Fisher/GSEA subsequently consume these gene/type
association rows.

The author `R/sysdata.rda` has SHA-256
`845023954bf3fb6d7426bd7544e095cb7306b42eec5912f9733f27daac8be77b`. This
batch does not bundle or redistribute its marker tables. The loader accepts a
caller-owned CSV or gzip-compressed CSV in the four-column author format and
does not assert that an arbitrary matching table is an official source
snapshot. The caller may record a version and provenance string; the returned
object also records the exact input-file SHA-256, total row count and selected
row count.

## Python contract and limits

`easycelltype_reference(path, database=..., species=..., tissues=...)`
streams one input file and returns an immutable `EasyCellTypeReference` with
parallel `genes` and `cell_types` tuples that can be passed directly to
`easycelltype_fisher` or `easycelltype_gsea_es`. `requested_tissues` records a
deduplicated caller filter (`None` means all); `selected_tissues` records
actual retained source organs in first-occurrence order. Tissue validity is
checked against the selected species, and any unknown requested name raises.
Empty tissue sequences, like `None`, mean all source organs.
If an input table contains a blank `organ` value, it is retained when no
tissue filter is supplied and can be selected explicitly with `tissues=("",)`;
this preserves the source table's row inclusion/filter behavior.

Identifiers are returned as strings in the source EntrezID namespace. No
symbol-to-Entrez mapping is attempted. The R wrapper delegates that conversion
to species-specific OrgDb packages using `multiVals="first"`; this project
does not infer a version or silently apply a local mapping. Callers with symbol
queries must supply and record their own versioned conversion first.

The loader limits an input to 100,000,000 on-disk bytes, 32,000,000 expanded
UTF-8 bytes, 65,536 characters per physical line, 200,000 rows and 1,000
tissue filters. It computes a bounded streaming hash pass and parses the
selected file serially, retaining only the selected species/tissue association
tuples. These limits protect both compressed and plain CSV input from oversized
records. These are Python safety bounds and provenance conveniences; native
app table versions and symbol mapping remain separate questions.

## Source validation

The base-R exporter in `tools/reference_easycelltype_data.R` checks exact
column names, row counts, absence of missing values and Human/Mouse species
before exporting. Blank organ strings are present in the original data and
are distinct from missing values. The source tables have 49,149 CellMarker,
172,003 Clustermole and 15,067 Panglao rows. Their Human/Mouse counts are
31,739/17,410, 166,235/5,768 and 7,697/7,370, respectively.

Python preserves those counts and exactly matches the gene/type vectors from
six independent R species/first-tissue selections, including source order and
duplicates. Four focused behavior tests, the standalone synthetic guide,
Ruff and mypy pass. The source comparison took 1.817 seconds, peaked at
125.69 MiB resident memory and reported zero swaps. Full association tables
and locally generated reference outputs remain ignored research inputs.
