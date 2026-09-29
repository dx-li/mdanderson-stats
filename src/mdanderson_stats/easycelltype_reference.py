"""Read bounded, versioned EasyCellType cell-marker associations."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_DATABASES = frozenset({"cellmarker", "clustermole", "panglao"})
_SPECIES = frozenset({"Human", "Mouse"})
_MAX_REFERENCE_ROWS = 200_000
_MAX_SOURCE_BYTES = 100_000_000
_MAX_EXPANDED_BYTES = 32_000_000
_MAX_PHYSICAL_LINE_CHARS = 65_536
_MAX_TISSUE_FILTERS = 1_000


@dataclass(frozen=True, slots=True)
class EasyCellTypeReference:
    """Immutable selected association rows ready for Fisher or GSEA inputs.

    ``requested_tissues`` records the caller's filter (``None`` means all
    tissues); ``selected_tissues`` lists the source organs actually retained
    (including a blank source-organ value when present),
    in their first-occurrence order. ``genes`` and ``cell_types`` preserve
    source-row order and multiplicity.
    """

    database: str
    species: str
    requested_tissues: tuple[str, ...] | None
    selected_tissues: tuple[str, ...]
    gene_namespace: Literal["EntrezID"]
    genes: tuple[str, ...]
    cell_types: tuple[str, ...]
    source_name: str
    source_sha256: str
    source_rows: int
    selected_rows: int
    source_version: str | None
    source_provenance: str | None

    def __post_init__(self) -> None:
        if self.database not in _DATABASES or self.species not in _SPECIES:
            raise ValueError("reference must use a supported database and species")
        if self.gene_namespace != "EntrezID":
            raise ValueError("EasyCellType references use the EntrezID namespace")
        if len(self.genes) != len(self.cell_types) or not self.genes:
            raise ValueError("reference genes and cell_types must be nonempty parallel tuples")
        if len(self.genes) > _MAX_REFERENCE_ROWS:
            raise ValueError("reference exceeds the 200,000-row limit")
        if any(not isinstance(item, str) or not item for item in self.genes):
            raise ValueError("reference identifiers must be nonempty strings")
        if any(not isinstance(item, str) or not item for item in self.cell_types):
            raise ValueError("reference identifiers must be nonempty strings")
        if any(not isinstance(tissue, str) for tissue in self.selected_tissues):
            raise ValueError("selected_tissues must contain strings")
        if self.selected_rows != len(self.genes) or self.source_rows < self.selected_rows:
            raise ValueError("reference row-count metadata is inconsistent")
        if not self.source_name:
            raise ValueError("source_name must be nonempty")
        if self.source_version is not None and not self.source_version:
            raise ValueError("source_version must be nonempty or None")
        if self.source_provenance is not None and not self.source_provenance:
            raise ValueError("source_provenance must be nonempty or None")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")


def easycelltype_reference(
    path: str | Path,
    *,
    database: Literal["cellmarker", "clustermole", "panglao"],
    species: Literal["Human", "Mouse"],
    tissues: Sequence[str] | None = None,
    source_version: str | None = None,
    source_provenance: str | None = None,
) -> EasyCellTypeReference:
    """Select association rows from a caller-supplied EasyCellType CSV.

    Database and species names follow the pinned author wrapper exactly. A
    CSV must have the author table columns ``celltype``, ``spe``, ``organ`` and
    ``entrezid`` in that order. A missing or empty ``tissues`` sequence selects
    every source organ for the requested species. An empty-string filter value
    selects rows whose source ``organ`` field is blank. Requested values are
    validated against the set of organs for that species before a reference
    is returned; an unknown value raises, even when another requested organ is
    valid. Filtering preserves row order and duplicate associations, as the
    EasyCellType Fisher and GSEA inputs consume ordered rows. Symbols are not
    mapped: the input table is EntrezID-based, so symbol queries need a
    caller-provided, versioned conversion.

    The selected CSV or ``.csv.gz`` is streamed once after a bounded checksum
    pass. This accepts caller-provided author-format tables; it does not claim
    that an arbitrary table is an official snapshot or bundle any database.
    ``source_version`` and ``source_provenance`` are recorded as caller input.
    Input limits are 100 MB compressed/on disk, 32 MB expanded UTF-8 text,
    65,536 characters per physical line and 200,000 rows.
    """
    if not isinstance(database, str) or database not in _DATABASES:
        raise ValueError("database must be 'cellmarker', 'clustermole', or 'panglao'")
    if not isinstance(species, str) or species not in _SPECIES:
        raise ValueError("species must be 'Human' or 'Mouse'")
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"EasyCellType reference file does not exist: {source_path}")
    if source_version is not None and (not isinstance(source_version, str) or not source_version):
        raise ValueError("source_version must be a nonempty string or None")
    if source_provenance is not None and (
        not isinstance(source_provenance, str) or not source_provenance
    ):
        raise ValueError("source_provenance must be a nonempty string or None")
    if tissues is None:
        requested = None
    else:
        if isinstance(tissues, (str, bytes)):
            raise ValueError("tissues must be a sequence of organ names, not a scalar string")
        if not isinstance(tissues, Sequence):
            raise ValueError("tissues must be a finite sequence of organ names")
        if len(tissues) > _MAX_TISSUE_FILTERS:
            raise ValueError("tissues may contain at most 1000 organ names")
        raw_tissues = tuple(tissues)
        if any(not isinstance(tissue, str) for tissue in raw_tissues):
            raise ValueError("tissues must contain strings; use '' for a blank source organ")
        unique = tuple(dict.fromkeys(raw_tissues))
        requested = unique or None
    requested_set = None if requested is None else set(requested)

    initial_stat = source_path.stat()
    if initial_stat.st_size > _MAX_SOURCE_BYTES:
        raise ValueError("reference file exceeds the 100,000,000-byte limit")
    digest = hashlib.sha256()
    with source_path.open("rb") as binary:
        hashed_bytes = 0
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            hashed_bytes += len(block)
            if hashed_bytes > _MAX_SOURCE_BYTES:
                raise ValueError("reference file exceeds the 100,000,000-byte limit")
            digest.update(block)

    species_organs: set[str] = set()
    selected_organs: dict[str, None] = {}
    selected_genes: list[str] = []
    selected_types: list[str] = []
    gene_pool: dict[str, str] = {}
    type_pool: dict[str, str] = {}
    source_rows = 0
    with source_path.open("rb") as raw_file:
        binary_source = (
            gzip.GzipFile(fileobj=raw_file, mode="rb")
            if source_path.name.endswith(".gz")
            else raw_file
        )
        with io.TextIOWrapper(binary_source, encoding="utf-8", newline="") as text:
            expanded_bytes = 0

            def bounded_lines():
                nonlocal expanded_bytes
                while True:
                    line = text.readline(_MAX_PHYSICAL_LINE_CHARS + 1)
                    if not line:
                        return
                    if len(line) > _MAX_PHYSICAL_LINE_CHARS:
                        raise ValueError(
                            "reference file has a physical line longer than 65,536 characters"
                        )
                    expanded_bytes += len(line.encode("utf-8"))
                    if expanded_bytes > _MAX_EXPANDED_BYTES:
                        raise ValueError(
                            "reference file exceeds the 32,000,000-byte expanded limit"
                        )
                    yield line

            reader = csv.DictReader(bounded_lines())
            expected_columns = ["celltype", "spe", "organ", "entrezid"]
            if reader.fieldnames != expected_columns:
                raise ValueError(
                    f"reference file has an unexpected CSV schema: {reader.fieldnames!r}"
                )
            for row_number, row in enumerate(reader, start=2):
                source_rows += 1
                if source_rows > _MAX_REFERENCE_ROWS:
                    raise ValueError("reference file exceeds the 200,000-row limit")
                if None in row or any(row.get(name) is None for name in expected_columns):
                    raise ValueError(f"reference file has a malformed row at line {row_number}")
                cell_type, row_species, organ, gene = (
                    row["celltype"],
                    row["spe"],
                    row["organ"],
                    row["entrezid"],
                )
                if not cell_type or not row_species or not gene:
                    raise ValueError(
                        f"reference file has a missing identifier at line {row_number}"
                    )
                if row_species not in _SPECIES:
                    raise ValueError(
                        f"reference file has an unsupported species at record {row_number}: "
                        f"{row_species!r}"
                    )
                if row_species != species:
                    continue
                species_organs.add(organ)
                if requested_set is not None and organ not in requested_set:
                    continue
                if len(selected_genes) >= _MAX_REFERENCE_ROWS:
                    raise ValueError("selected reference exceeds the 200,000-row limit")
                selected_organs.setdefault(organ, None)
                selected_genes.append(gene_pool.setdefault(gene, gene))
                selected_types.append(type_pool.setdefault(cell_type, cell_type))

    final_stat = source_path.stat()
    if (
        initial_stat.st_dev,
        initial_stat.st_ino,
        initial_stat.st_size,
        initial_stat.st_mtime_ns,
    ) != (final_stat.st_dev, final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns):
        raise RuntimeError("reference file changed while it was being read")

    if requested_set is not None:
        unknown = requested_set.difference(species_organs)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown {species} tissue/organ(s) for {database}: {names}")
    if not selected_genes:
        raise ValueError(f"no {species} association rows selected from {database}")
    return EasyCellTypeReference(
        database=database,
        species=species,
        requested_tissues=requested,
        selected_tissues=tuple(selected_organs),
        gene_namespace="EntrezID",
        genes=tuple(selected_genes),
        cell_types=tuple(selected_types),
        source_name=source_path.name,
        source_sha256=digest.hexdigest(),
        source_rows=source_rows,
        selected_rows=len(selected_genes),
        source_version=source_version,
        source_provenance=source_provenance,
    )
