"""Snapshot the public catalog and discover version-specific download metadata.

Raw responses stay local; the package includes a concise factual inventory.
Run from the repository root. Requests have timeouts; errors are never converted
into empty catalogs. Existing implementation tracking is preserved on refresh.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen

BASE = "https://biostatistics.mdanderson.org/SoftwareDownload/"
RAW = Path("research/raw")
CATALOG = Path("src/mdanderson_stats/catalog.json")


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("#", "mailto:")):
                self.links.append(urljoin(BASE, href))


def fetch(url: str, path: Path) -> bytes:
    with urlopen(url, timeout=60) as response:
        body = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def detail(row: dict) -> dict:
    software_id = row["SoftwareId"]
    url = row["Url"] if row["IsOnline"] else BASE + f"SingleSoftware/Index/{software_id}"
    retrieval_error = None
    body = ""
    if not row["IsOnline"]:
        try:
            body = fetch(url, RAW / f"detail-{software_id}.html").decode()
        except HTTPError as error:
            if error.code not in (404, 500):
                raise
            retrieval_error = f"HTTP {error.code} retrieving {url}"
    versions = sorted(set(re.findall(r'"nVersionId":(\d+)', body)))
    downloads = []
    for version in versions:
        content = fetch(
            BASE + f"SingleSoftware/GetDownloadable?nVersionId={version}",
            RAW / f"downloads-{software_id}-{version}.json",
        )
        for item in json.loads(content)["data"]:
            downloads.append(
                {
                    "id": item["DownloadableId"],
                    "version_id": int(version),
                    "filename": item["DownloadablePath"],
                    "notes": item["DownloadableDesc"],
                }
            )
    links = Links()
    links.feed(row.get("FullDesc") or "")
    return {
        "id": software_id,
        "name": row["SoftwareName"].strip(),
        "kind": "online" if row["IsOnline"] else "desktop",
        "detail_url": url,
        "application_url": row.get("Url"),
        "references": sorted(set(links.links)),
        "downloads": downloads,
        "retrieval_error": retrieval_error,
        "status": "pending",
        "implemented_features": [],
        "validation": [],
    }


def main() -> None:
    rows = []
    snapshots = []
    for online in (False, True):
        url = BASE + f"api/Softwares/GetAll?IsOnline={str(online).lower()}"
        body = fetch(url, RAW / f"catalog-{'online' if online else 'desktop'}.json")
        rows.extend(json.loads(body)["data"])
        snapshots.append({"url": url, "sha256": hashlib.sha256(body).hexdigest()})
    ids = [row["SoftwareId"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate software identifiers in catalog")
    previous = {}
    if CATALOG.exists():
        previous = {r["id"]: r for r in json.loads(CATALOG.read_text())["entries"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = list(pool.map(detail, rows))
    for entry in entries:
        if entry["id"] in previous:
            for key in ("status", "implemented_features", "validation"):
                entry[key] = previous[entry["id"]][key]
    result = {
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_snapshots": snapshots,
        "entries": sorted(entries, key=lambda row: row["id"]),
    }
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Recorded {len(entries)} entries and {sum(len(e['downloads']) for e in entries)} downloads"
    )


if __name__ == "__main__":
    main()
