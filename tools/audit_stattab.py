"""Inventory the pinned STATTAB archive without treating shared names as coverage."""

import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256 as CDFLIB_SHA256
from audit_cdflib90 import F95_ROOT, fortran_inventory

ARCHIVE = Path("research/raw/STATTAB/STATTAB      _V1.3.zip")
ARCHIVE_SHA256 = "0325f15818176418774216e546e0f61267cfd228cef184e63101c68e2b7e76f9"
SOURCE_ROOT = "stattab13/SOURCE/"


def read_archive():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("STATTAB archive hash mismatch")
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or archive.testzip() is not None:
            raise RuntimeError("Duplicate or corrupt STATTAB archive members")
        contents = {i.filename: archive.read(i) for i in archive.infolist() if not i.is_dir()}
    if len(contents) != 34:
        raise RuntimeError("STATTAB membership changed")
    return contents


def source_order(contents):
    order = re.findall(r"\b(\w+\.f90)\b", contents[SOURCE_ROOT + "compile.stattab"].decode())
    expected = {Path(n).name for n in contents if n.endswith(".f90")}
    if len(order) != len(expected) or set(order) != expected:
        raise RuntimeError("STATTAB build order does not cover each source exactly once")
    return order


def main():
    contents = read_archive()
    cdf_archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(cdf_archive.read_bytes()).hexdigest() != CDFLIB_SHA256:
        raise RuntimeError("CDFLIB90 comparison archive changed")
    with tarfile.open(cdf_archive) as archive:
        cdf = {
            Path(m.name).name: archive.extractfile(m).read()
            for m in archive.getmembers()
            if m.isfile() and m.name.startswith(F95_ROOT + "source/") and m.name.endswith(".f90")
        }
    members = []
    for name, content in contents.items():
        if (Path("research/raw/STATTAB/source") / name).read_bytes() != content:
            raise RuntimeError(f"Extracted bytes differ: {name}")
        p = Path(name)
        row = dict(path=name, size=len(content), sha256=hashlib.sha256(content).hexdigest())
        if p.suffix == ".f90":
            role = "source"
            row["declarations"] = fortran_inventory(content.decode("latin1"), fixed=False)
            source = re.sub(r"&\s*\n\s*&?", " ", content.decode("latin1"))
            attributes = re.findall(
                r"^\s*(?:TYPE|REAL|INTEGER|LOGICAL|CHARACTER)[^\n]*\bPUBLIC\s*::\s*([^\n]+)",
                source,
                re.I | re.M,
            )
            row["declarations"]["attribute_public_names"] = sorted(
                {
                    declaration.split("=", 1)[0].strip().lower()
                    for group in attributes
                    for declaration in group.split(",")
                }
            )
            if p.name in cdf:
                same = content == cdf[p.name]
                row["cdflib90_comparison"] = dict(
                    status="byte_identical" if same else "changed_source_requires_contract_review",
                    sha256=hashlib.sha256(cdf[p.name]).hexdigest(),
                )
            else:
                row["cdflib90_comparison"] = dict(status="additional_stattab_source")
            scope = "Native application/support evidence; Python STATTAB mapping pending"
        elif p.name in {"Makefile", "COMPILE.IT", "compile.stattab"}:
            role, scope = (
                "build",
                "All 22 source files compile unchanged; Python build will replace",
            )
        elif p.name in {"stattab", "stattab.exe"}:
            role, scope = (
                "executable",
                "Historical platform binaries; rebuilt source used as reference",
            )
        elif p.name == ".log":
            role, scope = (
                "build_log",
                "Interrupted TeX run with no pages produced; no additional interface",
            )
        elif p.suffix in {".pdf", ".ps", ".tex"} or p.name in {"HOWTOGET", "INSTALL", "LEGALITIES"}:
            role, scope = (
                "documentation",
                "Manual, acquisition/build instructions and retained terms",
            )
        else:
            raise RuntimeError(f"Unknown STATTAB archive role: {name}")
        row.update(role=role, scope=scope)
        members.append(row)
    notice = contents["stattab13/LEGALITIES"]
    if Path("notices/mdanderson-stattab-LEGALITIES.txt").read_bytes() != notice:
        raise RuntimeError("STATTAB notice must retain exact archived bytes")
    report = dict(
        archive_url=(
            "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/STATTAB/"
            "STATTAB%20%20%20%20%20%20_V1.3.zip"
        ),
        archive_sha256=ARCHIVE_SHA256,
        catalog_id=23,
        status="partial_discrete_probability_terms",
        implemented_scope={
            "description": "Direct discrete terms; full application workflow remains pending",
            "evidence": [
                "src/mdanderson_stats/stattab_probability.py",
                "tests/test_stattab_probability.py",
                "docs/stattab-probability.md",
                "docs/stattab-probability-benchmark.json",
            ],
        },
        version_reconciliation=(
            "Catalog/download directory says 1.3; manual and unchanged compiled program banner "
            "say Version 2.0: March, 2002. Pinned archive bytes define the reference."
        ),
        regular_file_count=len(members),
        role_counts=dict(Counter(m["role"] for m in members)),
        source_order=source_order(contents),
        comparison_limits=(
            "Byte identity supports reuse of prior numerical evidence for two modules only. "
            "Changed source and application behavior require separate validation; "
            "declaration inventories are not compiler export proofs."
        ),
        members=members,
    )
    if any(not Path(p).is_file() for p in report["implemented_scope"]["evidence"]):
        raise RuntimeError("Missing STATTAB discrete-probability evidence")
    Path("docs/stattab-archive.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["regular_file_count", "role_counts"]}, indent=2))


if __name__ == "__main__":
    main()
