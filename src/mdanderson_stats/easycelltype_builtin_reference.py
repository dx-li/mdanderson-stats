"""Access the pinned, bundled EasyCellType marker-reference tables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from importlib import resources
from typing import Literal

from mdanderson_stats.easycelltype_reference import (
    EasyCellTypeReference,
    easycelltype_reference,
)

_DATABASE_FILES = {
    "cellmarker": "cellmarker.csv.gz",
    "clustermole": "clustermole.csv.gz",
    "panglao": "panglao.csv.gz",
}
_SOURCE_ROWS = {"cellmarker": 49_149, "clustermole": 172_003, "panglao": 15_067}
_BUNDLED_SHA256 = {
    "cellmarker": "c6c1892ddeca843eac137a1b8152cfa00cb806fb1dd647aa6ebc730a03fc6835",
    "clustermole": "cc3b54728a1bdeb66cfba458766162c40cfbec907843ce11444984b8d1e9f408",
    "panglao": "c0c26e2ff7513f9a9f813c2fdc943445d306bbab877746ecf4e6c2d3247c7dc0",
}
_SOURCE_CSV_SHA256 = {
    "cellmarker": "297a0cf666066426c69d50184d86eb12ff975c7d6d5e125a62467a6d99a6baee",
    "clustermole": "6499b6b861e4ee482e6610ce92445b918e3635a7d3bb3c8c6ae4d529dfceb329",
    "panglao": "6e6d06b7031cc6e791f2a40ec61425d43d26e37c67fc1f15387c63286eb15a0b",
}
_SOURCE_SYSdata_SHA256 = "845023954bf3fb6d7426bd7544e095cb7306b42eec5912f9733f27daac8be77b"
_AUTHOR_COMMIT = "e85e8187c540f66994b5ca12fe95f5d9eb95f1f5"


def easycelltype_builtin_reference(
    database: Literal["cellmarker", "clustermole", "panglao"],
    species: Literal["Human", "Mouse"],
    *,
    tissues: Sequence[str] | None = None,
) -> EasyCellTypeReference:
    """Select species/tissue rows from the packaged EasyCellType 1.5.4 data.

    The packaged tables are deterministic gzip-encoded exports of the author
    package's ``R/sysdata.rda`` at commit ``e85e8187``. Database versions are
    those embedded in that snapshot; the author package did not record the
    upstream release identifiers separately. Rows are parsed through the same
    bounded streaming reader as caller-supplied references. No network request
    or R runtime is used.

    Gene identifiers remain EntrezIDs. Symbol conversion requires a separate,
    explicit versioned mapping supplied by the caller.
    """
    if not isinstance(database, str) or database not in _DATABASE_FILES:
        raise ValueError("database must be 'cellmarker', 'clustermole', or 'panglao'")
    filename = _DATABASE_FILES[database]
    asset = resources.files("mdanderson_stats").joinpath("data", "easycelltype", filename)
    source_hash = _SOURCE_CSV_SHA256[database]
    provenance = (
        f"EasyCellType 1.5.4 author snapshot at commit {_AUTHOR_COMMIT}; "
        f"R/sysdata.rda SHA-256 {_SOURCE_SYSdata_SHA256}; source CSV SHA-256 "
        f"{source_hash}; converted to deterministic gzip CSV for this package. "
        "The specific upstream database release was not recorded by EasyCellType."
    )
    with resources.as_file(asset) as local_path:
        selected = easycelltype_reference(
            local_path,
            database=database,
            species=species,
            tissues=tissues,
            source_version=f"EasyCellType 1.5.4 ({_AUTHOR_COMMIT})",
            source_provenance=provenance,
        )
    if selected.source_sha256 != _BUNDLED_SHA256[database]:
        raise RuntimeError(f"bundled {database} reference checksum does not match its manifest")
    if selected.source_rows != _SOURCE_ROWS[database]:
        raise RuntimeError(f"bundled {database} reference row count does not match its manifest")
    # A zipped-wheel extraction may have a temporary path prefix in its name.
    # Keep immutable metadata stable across source trees, wheels, and sdists.
    return replace(selected, source_name=filename)
