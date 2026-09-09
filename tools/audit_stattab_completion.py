"""Reconcile the pinned STATTAB application, manual, and Python delivery evidence."""

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from audit_stattab import ARCHIVE_SHA256, SOURCE_ROOT, read_archive

import mdanderson_stats as package

REQUIREMENTS = {
    "twelve_families_and_42_groups": ["tests/test_stattab_results.py"],
    "independent_discrete_terms_and_extremes": ["tests/test_stattab_probability.py"],
    "requests_complements_reuse_tables_and_isolation": ["tests/test_stattab_session.py"],
    "menu_help_all_list_actions_reports_and_eof": ["tests/test_stattab_console.py"],
    "file_modes_confirmation_cancellation_and_ownership": ["tests/test_stattab_files.py"],
    "native_sessions_and_explicit_defect_repairs": ["tests/test_stattab_reference.py"],
    "manual_worked_examples_and_error_recovery": ["tests/test_stattab_manual.py"],
    "shared_source_versions_and_public_interfaces": ["docs/stattab-shared-source.json"],
    "documented_semantics_and_manual_errata": ["docs/stattab-completion.md"],
    "numerical_batch_performance": [
        "docs/stattab-probability-benchmark.json",
        "docs/stattab-results-benchmark.json",
        "docs/stattab-sessions-benchmark.json",
    ],
    "module_entry_point_and_packaging": ["src/mdanderson_stats/stattab.py", "pyproject.toml"],
    "retained_terms": ["notices/mdanderson-stattab-LEGALITIES.txt", "THIRD_PARTY_NOTICES.md"],
}
SECTIONS = [
    (
        "Identity, terms and references",
        1,
        4,
        "Exact notice retained; modified Python work identified",
    ),
    (
        "Introduction and calculation/input conventions",
        5,
        6,
        "All twelve families and request grammar",
    ),
    ("Annotated run and t table", 7, 12, "Replayed through console, including help/report/reuse"),
    ("Discrete count inversions", 12, 14, "Continuous solution and validated integer neighbors"),
    (
        "Error notification",
        14,
        15,
        "Unattainable target rejected without replacing previous result",
    ),
    ("Binomial table", 15, 17, "Eleven rows checked against exact combinatorial probabilities"),
    (
        "P-value interpretation",
        17,
        20,
        "Both tails; normal/t two-sided and chi-square/F upper tail",
    ),
    ("Distribution definitions", 20, 25, "Corrected formulas and parameterization in Python help"),
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    contents = read_archive()
    inventory = json.loads(Path("docs/stattab-archive.json").read_text())
    shared = json.loads(Path("docs/stattab-shared-source.json").read_text())
    require(shared["archive_sha256"] == ARCHIVE_SHA256, "Shared audit provenance differs")
    require(shared["compiler_import_count"] == 259, "Public interface review changed")
    require({r["path"] for r in inventory["members"]} == set(contents), "Archive coverage differs")
    for record in shared["shared_modules"]:
        native = contents[SOURCE_ROOT + record["module"] + ".f90"]
        require(hashlib.sha256(native).hexdigest() == record["stattab_sha256"], "Source differs")
    main_source = contents[SOURCE_ROOT + "stattab_main.f90"].decode("latin1")
    routines = re.findall(r"^\s*(?:SUBROUTINE|FUNCTION)\s+(\w+)", main_source, re.I | re.M)
    mapping = {
        "display_help": "stattab_help",
        "parse_parameters": "parse_stattab_request",
        "solve_distribution": "STATTABSession.execute / run_stattab",
        "check_status_value": "validated numerical results and Python exceptions",
        "display_status_error": "run_stattab invalid-request reporting",
    }
    for name in package.STATTAB_DISTRIBUTIONS:
        mapping["solve_" + name] = "stattab_solve: " + name
    require(set(routines) == set(mapping), "Main procedure mapping incomplete")
    require(len(package.STATTAB_DISTRIBUTIONS) == 12, "Family coverage changed")
    require(
        sum(len(d.groups) for d in package.STATTAB_DISTRIBUTIONS.values()) == 42,
        "Computed-group coverage changed",
    )
    for name in [
        "stattab_solve",
        "stattab_help",
        "run_stattab",
        "parse_stattab_request",
        "STATTABSession",
        "format_stattab_result",
        "stattab_open_file",
        "stattab_report_file_dialogue",
        "stattab_binomial_term",
        "stattab_negative_binomial_term",
        "stattab_poisson_term",
    ]:
        require(callable(getattr(package, name, None)), "Missing Python interface: " + name)
    for paths in REQUIREMENTS.values():
        require(all(Path(p).is_file() for p in paths), "Missing completion evidence")
    require(
        contents["stattab13/LEGALITIES"]
        == Path("notices/mdanderson-stattab-LEGALITIES.txt").read_bytes(),
        "Notice changed",
    )
    with tempfile.TemporaryDirectory(prefix="stattab-documents-") as tmp:
        work = Path(tmp)
        for suffix in ["pdf", "ps", "tex"]:
            (work / ("manual." + suffix)).write_bytes(
                contents[f"stattab13/DOC/stattab90_doc.{suffix}"]
            )
        subprocess.run(
            [
                "gs",
                "-q",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                "-sOutputFile=" + str(work / "from-ps.pdf"),
                str(work / "manual.ps"),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        extracted = []
        for name in ["manual", "from-ps"]:
            subprocess.run(
                ["pdftotext", "-layout", str(work / (name + ".pdf")), str(work / (name + ".txt"))],
                check=True,
                capture_output=True,
                timeout=30,
            )
            extracted.append((work / (name + ".txt")).read_text())
        require(all(len(t.split("\f")) - 1 == 25 for t in extracted), "Manual page count changed")
        # Font extraction can differ in ligatures; retain each extraction hash rather than
        # falsely claiming binary or visual identity of different document formats.
        text_equal = re.sub(r"\s", "", extracted[0]) == re.sub(r"\s", "", extracted[1])

        def normalize(t):
            return re.sub(r"\s", "", unicodedata.normalize("NFKC", t))

        different_pages = [
            i + 1
            for i, (a, b) in enumerate(
                zip(extracted[0].split("\f")[:-1], extracted[1].split("\f")[:-1], strict=True)
            )
            if normalize(a) != normalize(b)
        ]
        require(different_pages == [22, 23], "Unreviewed PDF/PostScript text differences")
        text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in extracted]
    dispositions = {
        "source": "Mapped shared modules, descriptors, file helpers and all 17 main procedures",
        "build": "Native build scripts replaced by pyproject.toml and Python installation",
        "executable": "Historical binaries inventoried; rebuilt source is the native reference",
        "build_log": "Interrupted TeX run with no pages; no additional runtime contract",
        "documentation": "Manual reviewed; old build instructions superseded; terms retained",
    }
    report = dict(
        archive_sha256=ARCHIVE_SHA256,
        catalog_id=23,
        status="complete_with_documented_python_semantics",
        requirements=REQUIREMENTS,
        evidence_limits=(
            "Evidence paths and source coverage are structural checks. Numerical and workflow "
            "claims require tests, native audits and benchmarks, not this manifest alone."
        ),
        main_procedure_mapping=mapping,
        documents=[
            dict(section=n, first_page=a, last_page=b, disposition=d) for n, a, b, d in SECTIONS
        ],
        pdf_and_postscript_pages=25,
        pdf_and_postscript_text_equal_ignoring_whitespace=text_equal,
        extracted_text_sha256=text_hashes,
        differences_after_whitespace_and_ligature_normalization=different_pages,
        document_comparison_review=(
            "23 pages match after whitespace and Unicode compatibility normalization. "
            "Pages 22/23 differ in formula extraction order and prime glyph extraction; "
            "rendered PDF and converted PostScript pages were visually compared and agree."
        ),
        visually_reviewed_formula_pages=[21, 22, 23, 24, 25],
        members=[
            dict(
                path=m["path"],
                sha256=m["sha256"],
                role=m["role"],
                disposition=dispositions[m["role"]],
            )
            for m in inventory["members"]
        ],
    )
    for m in report["members"]:
        require(
            hashlib.sha256(contents[m["path"]]).hexdigest() == m["sha256"], "Member hash differs"
        )
    Path("docs/stattab-completion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Reconciled 34 archive members, 17 main procedures, 12 families, 42 groups, 25 manual pages"
    )


if __name__ == "__main__":
    main()
