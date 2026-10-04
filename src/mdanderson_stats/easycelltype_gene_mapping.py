"""Offline Human/Mouse gene identifier mappings for EasyCellType inputs.

The packaged mapping is an explicitly selected Bioconductor 3.18 snapshot.
Its stable candidate ordering is ascending OrgDb ``_id``; it does not claim
to reproduce the unspecified duplicate order of the author's AnnotationDbi
``multiVals='first'`` query.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Literal

from numpy.typing import ArrayLike

from mdanderson_stats.easycelltype import _labels

Species = Literal["Human", "Mouse"]
Direction = Literal["symbol_to_entrez", "entrez_to_symbol"]
UnmappedPolicy = Literal["keep", "drop", "error"]

_MANIFEST = "gene-mapping-bioconductor-3.18.json"
_SPECIES_PREFIX = {"Human": "Hs", "Mouse": "Mm"}
_MAX_ROWS = 200_000


@dataclass(frozen=True, slots=True)
class EasyCellTypeGeneMapping:
    """Aligned mapping results and the explicit policy used for unmapped IDs.

    ``mapped_ids`` and ``candidate_ids`` always align with ``input_ids``.
    ``retained_indices``/``retained_ids`` describe the rows passed onward
    under the requested unmapped policy. Duplicate input rows are preserved.
    """

    input_ids: tuple[str, ...]
    mapped_ids: tuple[str | None, ...]
    candidate_ids: tuple[tuple[str, ...], ...]
    retained_indices: tuple[int, ...]
    species: Species
    direction: Direction
    unmapped_policy: UnmappedPolicy
    source_version: str
    source_order: str

    @property
    def retained_ids(self) -> tuple[str | None, ...]:
        """Mapped output IDs aligned to the retained original rows."""
        return tuple(self.mapped_ids[i] for i in self.retained_indices)

    @property
    def ambiguous_indices(self) -> tuple[int, ...]:
        """Input positions with more than one supported target identifier."""
        return tuple(i for i, values in enumerate(self.candidate_ids) if len(values) > 1)

    @property
    def unmapped_indices(self) -> tuple[int, ...]:
        """Input positions with no target identifier in this snapshot."""
        return tuple(i for i, values in enumerate(self.candidate_ids) if not values)


@lru_cache(maxsize=2)
def _mapping_table(species: Species, direction: Direction) -> dict[str, tuple[str, ...]]:
    prefix = _SPECIES_PREFIX[species]
    asset_name = f"org.{prefix}.eg.db-3.18.0-symbol-entrez.tsv.gz"
    asset = resources.files("mdanderson_stats").joinpath("data", "easycelltype", asset_name)
    manifest_asset = resources.files("mdanderson_stats").joinpath("data", "easycelltype", _MANIFEST)
    manifest = json.loads(manifest_asset.read_text(encoding="utf-8"))
    package = manifest["packages"][species]
    candidates: dict[str, list[str]] = {}
    row_count = 0
    with resources.as_file(asset) as path, path.open("rb") as binary:
        digest = hashlib.sha256()
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != package["sha256"]:
            raise RuntimeError(
                f"bundled {species} gene mapping checksum does not match its manifest"
            )
        binary.seek(0)
        with gzip.GzipFile(fileobj=binary, mode="rb") as compressed:
            import io

            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text, delimiter="\t")
                if reader.fieldnames != ["symbol", "entrezid"]:
                    raise RuntimeError("bundled gene mapping has an unexpected column header")
                source_key, target_key = (
                    ("symbol", "entrezid")
                    if direction == "symbol_to_entrez"
                    else ("entrezid", "symbol")
                )
                for row in reader:
                    source = row[source_key]
                    target = row[target_key]
                    if not source or not target:
                        raise RuntimeError("bundled gene mapping contains an empty identifier")
                    candidates.setdefault(source, []).append(target)
                    row_count += 1
    if row_count != package["pair_count"]:
        raise RuntimeError(f"bundled {species} gene mapping row count does not match its manifest")
    return {key: tuple(values) for key, values in candidates.items()}


def easycelltype_gene_mapping(
    identifiers: ArrayLike,
    species: Species,
    *,
    direction: Direction = "symbol_to_entrez",
    unmapped: UnmappedPolicy = "drop",
) -> EasyCellTypeGeneMapping:
    """Map symbol/Entrez identifiers from a versioned offline snapshot.

    The author wrapper maps Human or Mouse symbols to Entrez IDs and drops
    unmapped rows before analysis. ``unmapped='drop'`` exposes that behavior
    through ``retained_indices`` while keeping an aligned audit trail. Use
    ``keep`` to retain unmapped rows as ``None`` or ``error`` to reject them.
    If a key has several targets, the first is selected in ascending OrgDb
    ``_id`` order; every candidate remains available in ``candidate_ids``.
    This is a deterministic Python convention, not a claim about native
    AnnotationDbi duplicate ordering.
    """
    if species not in _SPECIES_PREFIX:
        raise ValueError("species must be 'Human' or 'Mouse'")
    if direction not in ("symbol_to_entrez", "entrez_to_symbol"):
        raise ValueError("direction must be 'symbol_to_entrez' or 'entrez_to_symbol'")
    if unmapped not in ("keep", "drop", "error"):
        raise ValueError("unmapped must be 'keep', 'drop', or 'error'")
    labels = _labels(identifiers, "identifiers")
    if len(labels) > _MAX_ROWS:
        raise ValueError("identifiers may contain at most 200,000 rows")
    table = _mapping_table(species, direction)
    all_candidates = tuple(table.get(label, ()) for label in labels)
    missing = tuple(i for i, options in enumerate(all_candidates) if not options)
    if missing and unmapped == "error":
        raise ValueError(
            f"{len(missing)} identifiers are unmapped; first input index is {missing[0]}"
        )
    mapped = tuple(options[0] if options else None for options in all_candidates)
    retained = (
        tuple(i for i, value in enumerate(mapped) if value is not None)
        if unmapped == "drop"
        else tuple(range(len(labels)))
    )
    return EasyCellTypeGeneMapping(
        tuple(labels),
        mapped,
        all_candidates,
        retained,
        species,
        direction,
        unmapped,
        "Bioconductor 3.18; org.Hs.eg.db/org.Mm.eg.db 3.18.0",
        "ascending OrgDb _id; selected first candidate in this explicit Python snapshot order",
    )
