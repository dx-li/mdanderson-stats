"""Export compact ordered OrgDb identifier pairs without loading SQLite tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sqlite3
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source_root: Path, output: Path) -> None:
    manifest: dict[str, object] = {
        "annotation_release": "Bioconductor 3.18",
        "database_source_date": "2023-Sep11",
        "license": "Artistic-2.0",
        "order": "ascending gene_info._id",
        "package_version": "3.18.0",
        "packages": {},
    }
    packages: dict[str, object] = {}
    for species, prefix in (("Human", "Hs"), ("Mouse", "Mm")):
        archive = source_root / f"org.{prefix}.eg.db_3.18.0.tar.gz"
        database = source_root / f"org.{prefix}.eg.db/inst/extdata/org.{prefix}.eg.sqlite"
        if not archive.is_file() or not database.is_file():
            raise FileNotFoundError(f"missing Bioconductor 3.18 source for {species}")
        output.mkdir(parents=True, exist_ok=True)
        target = output / f"org.{prefix}.eg.db-3.18.0-symbol-entrez.tsv.gz"
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            with target.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
                ) as compressed:
                    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
                    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                    writer.writerow(("symbol", "entrezid"))
                    count = 0
                    query = (
                        "SELECT gene_info.symbol, genes.gene_id "
                        "FROM gene_info JOIN genes USING (_id) "
                        "ORDER BY gene_info._id"
                    )
                    for symbol, entrez_id in connection.execute(query):
                        writer.writerow((symbol, entrez_id))
                        count += 1
                    text.flush()
                    text.detach()
        finally:
            connection.close()
        packages[species] = {
            "db_file": database.name,
            "mapping_file": target.name,
            "author": "Marc Carlson",
            "pair_count": count,
            "compressed_bytes": target.stat().st_size,
            "sha256": _sha256(target),
            "sqlite_sha256": _sha256(database),
            "archive_sha256": _sha256(archive),
            "archive_url": (
                "https://bioconductor.org/packages/3.18/data/annotation/src/contrib/"
                f"org.{prefix}.eg.db_3.18.0.tar.gz"
            ),
            "package_license": "Artistic-2.0",
            "database_source_date": "2023-Sep11",
        }
    manifest["packages"] = packages
    (output / "gene-mapping-bioconductor-3.18.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="directory with extracted OrgDb packages")
    parser.add_argument("output", type=Path, help="destination data directory")
    args = parser.parse_args()
    export(args.source_root, args.output)


if __name__ == "__main__":
    main()
