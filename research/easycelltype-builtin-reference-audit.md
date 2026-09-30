# EasyCellType bundled-reference audit

The bundled tables are deterministic gzip exports of the author package's
`R/sysdata.rda` from EasyCellType 1.5.4, commit
[`e85e8187c540f66994b5ca12fe95f5d9eb95f1f5`](https://github.com/rx-li/EasyCellType/tree/e85e8187c540f66994b5ca12fe95f5d9eb95f1f5).
The pinned R object is 839,601 bytes with SHA-256
`845023954bf3fb6d7426bd7544e095cb7306b42eec5912f9733f27daac8be77b`; the
package `DESCRIPTION` declares Artistic-2.0. Conversion retained the exact
four-column author schema (`celltype,spe,organ,entrezid`), values, duplicate
rows, and order, then compressed with an empty embedded filename and zero mtime.
No upstream database version more specific than the EasyCellType snapshot was
recorded, so this code does not label the files as current database releases.

| Source table | Rows | Uncompressed CSV SHA-256 | Bundled gzip SHA-256 |
| --- | ---: | --- | --- |
| CellMarker | 49,149 | `297a0cf666066426c69d50184d86eb12ff975c7d6d5e125a62467a6d99a6baee` | `c6c1892ddeca843eac137a1b8152cfa00cb806fb1dd647aa6ebc730a03fc6835` |
| Clustermole | 172,003 | `6499b6b861e4ee482e6610ce92445b918e3635a7d3bb3c8c6ae4d529dfceb329` | `cc3b54728a1bdeb66cfba458766162c40cfbec907843ce11444984b8d1e9f408` |
| PanglaoDB | 15,067 | `6e6d06b7031cc6e791f2a40ec61425d43d26e37c67fc1f15387c63286eb15a0b` | `c0c26e2ff7513f9a9f813c2fdc943445d306bbab877746ecf4e6c2d3247c7dc0` |

The loader checks the bundled gzip digest and full source-row count after using
the common bounded streaming selector. Species is restricted to Human or Mouse;
tissue names are validated against the chosen species. It returns Entrez IDs and
does not perform symbol mapping. Existing caller-supplied reference loading and
the existing reference-selection audit retain the row/filter comparisons against
the author R tables; this addition adds fixed-snapshot identity checks and a
CellMarker Kidney Fisher annotation using the returned reference object.

Attribution and licensing:

- EasyCellType 1.5.4 declares Artistic-2.0 in its
  [DESCRIPTION](https://github.com/rx-li/EasyCellType/blob/e85e8187c540f66994b5ca12fe95f5d9eb95f1f5/DESCRIPTION).
- CellMarker is attributed to Zhang et al. (2019),
  [CellMarker](https://doi.org/10.1093/nar/gky900); Hu et al. (2023)
  describe the later [CellMarker 2.0](https://doi.org/10.1093/nar/gkac947). The particular
  embedded release and a data-specific license were not recorded in the
  EasyCellType snapshot.
- Clustermole is attributed to Dolgalev (2021),
  [Clustermole repository](https://github.com/igordot/clustermole), whose R
  package declares MIT with a `LICENSE` file. Clustermole combines several
  upstream marker resources; EasyCellType did not record their individual
  versions in this table.
- PanglaoDB is attributed to Franzén, Gan, and Björkegren (2019),
  [PanglaoDB paper](https://doi.org/10.1093/database/baz046). The official
  [PanglaoDB FAQ](https://panglaodb.se/faq.html) describes downloadable marker
  data. The snapshot does not record a database release identifier or a
  separate license for its embedded table; the paper citation alone does not
  establish the table's license.

The packaged asset is derived from the pinned EasyCellType package's
Artistic-2.0-declared data object. Citations above preserve known upstream
provenance; no paper or website license is inferred to govern a data table when
the pinned snapshot does not record that fact.


## Focused validation

Nine focused checks across the bundled loader, local-reference loader and Fisher
annotation pass, including annotation from the real CellMarker Kidney subset.
Ruff check/format and targeted mypy pass in the implementation checkout. That
sequence peaked at 137.30 MiB resident memory and reported zero swaps.

An independent streaming check verified the original R object's pinned hash,
each plain CSV hash, each packaged gzip hash, deterministic gzip headers, and
byte equality of every decompressed asset with its cached author CSV export.
The three tables contain 236,219 association rows in 968,453 compressed bytes.
This comparison took 0.018 seconds, peaked at 29.77 MiB and reported zero swaps.
The earlier local-reference audit provides the original R row/filter comparison;
this addition preserves those same table bytes and annotation algorithms.
