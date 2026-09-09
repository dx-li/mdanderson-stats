"""Compile the unchanged STATTAB application and record bounded complete sessions."""

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from audit_stattab import ARCHIVE_SHA256, SOURCE_ROOT, read_archive, source_order

# Source menu order, input parameters, and one position for every supported WHICH.
FAMILIES = [
    ("beta", ["0.5", ".", "2", "2", "0.5", "."], [4, 0, 2, 3]),
    ("binomial", ["2", "4", "0.5", ".", "0.6875", "."], [4, 0, 1, 2]),
    ("neg_binomial", ["2", "3", "0.5", ".", "0.5", "."], [4, 0, 1, 2]),
    ("chisq", ["2", "2", "0.5", "."], [2, 0, 1]),
    ("nc_chisq", ["4", "2", "2", "0.5", "."], [3, 0, 1, 2]),
    ("f", ["1", "4", "4", "0.5", "."], [3, 0]),
    ("nc_f", ["2", "4", "4", "2", "0.5", "."], [4, 0, 3]),
    ("gamma", ["1", "2", "3", "0.5", "."], [3, 0, 2, 1]),
    ("normal", ["1", "0", "1", "0.75", "."], [3, 0, 1, 2]),
    ("poisson", ["2", "3", "0.5", "."], [2, 0, 1]),
    ("t", ["1", "5", "0.8", "."], [2, 0, 1]),
    ("nc_t", ["1", "5", "0.5", "0.6", "."], [3, 0, 1, 2]),
]


def cases():
    records = []

    def add(name, family, body, **metadata):
        menu = 1 + next(i for i, (n, _, _) in enumerate(FAMILIES) if n == family)
        records.append(
            dict(name=name, family=family, input=f"\nn\n{menu}\n{body}\n\n0\n", **metadata)
        )

    for family, parameters, positions in FAMILIES:
        for which, position in enumerate(positions, 1):
            row = parameters.copy()
            row[position] = "?"
            add(f"{family}_which_{which}", family, " ".join(row), which=which)
        row = parameters.copy()
        row[-2:] = [".", "?"]
        add(f"{family}_upper_query", family, " ".join(row), which=1)
        row = parameters.copy()
        row[0] = "T"
        row[-2:] = ["?", "."]
        values = "0.25 0.5 1" if family == "beta" else "0.5 1 2"
        add(f"{family}_table", family, " ".join(row) + f"\n1\n3\n{values}\n8", which=1)

    for family, lines in {
        "binomial": ["0.5 4.75 0.5 . ? .", "2.5 4.75 0.5 . ? .", "4 4 1 . ? ."],
        "neg_binomial": ["0.5 3.75 0.5 . ? .", "2.5 3.75 0.5 . ? .", "0 3 1 . ? ."],
        "poisson": ["0.5 3 ? .", "2.5 3 ? .", "0 0 ? ."],
        "normal": ["-1 0 1 ? .", "0 0 1 ? .", "9 0 1 ? ."],
        "t": ["-1 5 ? .", "0 5 ? ."],
    }.items():
        for i, row in enumerate(lines):
            add(f"{family}_boundary_{i}", family, row, which=1)
    special = [
        ("reuse", "normal", "1 0 1 ? .\n2 = = ? ."),
        ("reuse_before_value", "normal", "= 0 1 ? ."),
        ("poisson_reuse_tail", "poisson", "2 3 ? .\n? = = ."),
        ("table_then_scalar", "normal", "T 0 1 ? .\n1\n3\n0 1 2\n8\n3 0 1 ? .\n8"),
        ("gamma_reuse", "gamma", "1 2 3 ? .\n2 = = ? ."),
        ("complement_coordinate", "beta", ". 0.25 2 2 ? ."),
        ("complement_chance", "binomial", "2 4 . 0.25 ? ."),
        ("bad_double_complement", "beta", "0.5 0.5 2 2 ? ."),
        ("bad_missing_complement", "beta", ". . 2 2 ? ."),
        ("bad_two_queries", "normal", "? 0 1 ? ."),
        ("bad_no_query", "normal", "1 0 1 0.75 ."),
        ("bad_short_input", "normal", "1 0 ? ."),
        ("bad_extra_input", "normal", "1 0 1 ? . 9 9"),
        ("bad_required_omission", "normal", ". 0 1 ? ."),
        ("bad_two_tables", "normal", "T T 1 ? ."),
        ("bad_numeric_overflow", "normal", "1e999 0 1 ? ."),
        ("bad_domain", "normal", "1 0 -1 ? ."),
        ("unsupported_f_df", "f", "1 ? 4 0.5 ."),
        ("unsupported_nc_f_df", "nc_f", "2 ? 4 2 0.5 ."),
        ("help", "normal", "HELP\n"),
        ("substring_help", "normal", "SHELPER\n"),
        ("comment", "normal", "1 0 1 ? . # note"),
    ]
    for name, family, body in special:
        add(name, family, body)
    records.append(
        dict(name="report_file", family="normal", input="\ny\nreport.txt\n9\n1 0 1 ? .\n\n0\n")
    )
    records.append(dict(name="parameter_eof", family="normal", input="\nn\n9\n"))
    return records


def main():
    contents = read_archive()
    order = source_order(contents)
    flags = ["-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    records = []
    with tempfile.TemporaryDirectory(prefix="stattab-reference-") as directory:
        work = Path(directory)
        for name in order:
            (work / name).write_bytes(contents[SOURCE_ROOT + name])
        compilation = subprocess.run(
            ["gfortran", *flags, *order, "-o", "stattab"],
            cwd=work,
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
        for case in cases():
            with tempfile.TemporaryDirectory(dir=work) as case_directory:
                try:
                    run = subprocess.run(
                        [str(work / "stattab")],
                        input=case["input"],
                        text=True,
                        capture_output=True,
                        cwd=case_directory,
                        timeout=3,
                    )
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError(f"STATTAB session timed out: {case['name']}") from error
                report = Path(case_directory, "report.txt")
                records.append(
                    dict(
                        **case,
                        returncode=run.returncode,
                        stdout=run.stdout,
                        stderr=re.sub(
                            r"0x[0-9a-fA-F]+", "<address>", run.stderr.replace(directory, "<build>")
                        ),
                        report=report.read_text() if report.exists() else None,
                    )
                )
    report = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_hashes={n: hashlib.sha256(contents[SOURCE_ROOT + n]).hexdigest() for n in order},
        flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        compile_stdout=compilation.stdout,
        compile_stderr=compilation.stderr,
        source_modifications="None; all 22 source files unchanged",
        interpretation=(
            "Exit zero means the session ended, not that its numerical answers succeeded. "
            "Failed calls may print uninitialized values; tests and method notes classify "
            "usable references and independently established defects."
        ),
        cases=records,
    )
    Path("tests/fixtures/stattab.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {len(records)} complete STATTAB sessions")
    for case in records:
        if case["returncode"] != 0:
            print(case["name"], case["returncode"], case["stderr"].splitlines()[0])


if __name__ == "__main__":
    main()
